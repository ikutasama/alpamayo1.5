# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grounded CoC Reward — verify reasoning against obstacle ground truth.

This replaces the keyword-based CoC quality scoring with a reward that
checks whether the Chain-of-Causation reasoning correctly identifies
real scene elements from obstacle.offline data.

Key improvements over keyword-based reward:
  1. **Object grounding**: Did the CoC correctly identify nearby obstacles?
  2. **Hallucination penalty**: Does the CoC mention objects NOT in the scene?
  3. **Threat assessment**: Does the CoC correctly identify the highest-threat object?
  4. **Spatial accuracy**: Does the CoC correctly describe spatial relationships?
  5. **Decision-GT alignment**: Does the CoC decision match the GT trajectory behavior?

This creates genuine reward variance because different completions will
identify different objects, make different spatial claims, and propose
different decisions — all of which can be verified against ground truth.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from rl.rewards.reward_types import RewardComponents


# ---------------------------------------------------------------------------
# Object type mention patterns (more specific than generic keywords)
# ---------------------------------------------------------------------------
OBJECT_TYPE_PATTERNS: dict[str, list[str]] = {
    "vehicle": [
        r"\b(?:car|vehicle|truck|bus|suv|van|sedan|pickup)\b",
    ],
    "pedestrian": [
        r"\b(?:pedestrian|person|people|walker|jogger|foot)\b",
    ],
    "cyclist": [
        r"\b(?:cyclist|bicycle|bicyclist|biker|motorcycle|motorcyclist|scooter)\b",
    ],
}

SPATIAL_PATTERNS: dict[str, list[str]] = {
    "left": [r"\bleft\b", r"\bleft[- ]side\b"],
    "right": [r"\bright\b", r"\bright[- ]side\b"],
    "ahead": [r"\bahead\b", r"\bin front\b", r"\bfront\b"],
    "behind": [r"\bbehind\b", r"\brear\b", r"\bback\b"],
    "approaching": [r"\bapproach(?:ing)?\b", r"\bcoming (?:towards|toward)\b", r"\bclosing in\b"],
    "crossing": [r"\bcross(?:ing)?\b", r"\bcut(?:ting)? in\b"],
}

DECISION_PATTERNS: dict[str, list[str]] = {
    "stop": [
        r"\bstop(?:ping)?\b",
        r"\bcome to (?:a )?(?:complete )?stop\b",
        r"\bhalt\b",
    ],
    "yield": [
        r"\byield(?:ing)?\b",
        r"\bgive way\b",
        r"\bwait\b",
        r"\blet (?:it |them )?(?:pass|go)\b",
    ],
    "slow_down": [
        r"\bslow(?:ing)? down\b",
        r"\bdecelerat(?:e|ing)\b",
        r"\breduce speed\b",
        r"\bbrak(?:e|ing)\b",
    ],
    "maintain": [
        r"\bmaintain(?:ing)? (?:speed|course)\b",
        r"\bkeep(?:ing)? (?:speed|going|moving)\b",
        r"\bcontinue\b",
        r"\bsteady\b",
    ],
    "accelerate": [
        r"\baccelerat(?:e|ing)\b",
        r"\bspeed(?:ing)? up\b",
        r"\bproceed\b",
        r"\bmove forward\b",
    ],
    "nudge_left": [
        r"\b(?:nudge|shift|move|steer) (?:to the )?left\b",
        r"\bleft lane change\b",
    ],
    "nudge_right": [
        r"\b(?:nudge|shift|move|steer) (?:to the )?right\b",
        r"\bright lane change\b",
    ],
}


def _has_pattern(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches in the text."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _count_matches(text: str, patterns: list[str]) -> int:
    """Count how many distinct patterns match."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def extract_gt_decision(
    gt_future_xyz: torch.Tensor,
    gt_future_rot: torch.Tensor | None = None,
) -> dict[str, float]:
    """Extract the driving decision from ground truth trajectory.

    Classifies the GT trajectory into semantic driving decisions based
    on speed profile, lateral movement, and heading changes.

    Args:
        gt_future_xyz: Ground truth future positions, shape [T, 3] or [1, 1, T, 3].
        gt_future_rot: Optional ground truth rotations.

    Returns:
        Dict mapping decision names to confidence scores in [0, 1].
    """
    # Flatten to [T, 3]
    xyz = gt_future_xyz
    while xyz.dim() > 2:
        xyz = xyz[0]

    if xyz.shape[0] < 2:
        return {"stop": 1.0, "yield": 0.0, "slow_down": 0.0,
                "maintain": 0.0, "accelerate": 0.0,
                "nudge_left": 0.0, "nudge_right": 0.0}

    # Compute step-wise speeds
    dxy = xyz[1:, :2] - xyz[:-1, :2]
    speeds = torch.linalg.norm(dxy, dim=-1)

    speed_start = speeds[:max(3, len(speeds) // 4)].mean().item()
    speed_end = speeds[-max(3, len(speeds) // 4):].mean().item()
    speed_max = speeds.max().item()
    speed_min = speeds.min().item()
    total_displacement = torch.linalg.norm(xyz[-1, :2] - xyz[0, :2]).item()

    # Lateral movement
    lateral_delta = (xyz[-1, 1] - xyz[0, 1]).item()

    # Heading change (if rotation available)
    heading_change = 0.0
    if gt_future_rot is not None:
        rot = gt_future_rot
        while rot.dim() > 3:
            rot = rot[0]
        if rot.shape[0] >= 2:
            heading = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
            heading_change = (heading[-1] - heading[0]).item()

    # Classify decisions
    decisions: dict[str, float] = {}

    # Stop: very low final speed
    if speed_end < 0.3 and speed_start > 0.5:
        decisions["stop"] = 0.9
    elif speed_end < 0.15:
        decisions["stop"] = 1.0
    else:
        decisions["stop"] = 0.0

    # Slow down: significant speed decrease but not full stop
    if speed_start > 0.3:
        speed_ratio = speed_end / max(speed_start, 0.01)
        if speed_ratio < 0.7 and speed_end > 0.3:
            decisions["slow_down"] = min(1.0, (1.0 - speed_ratio) * 2.0)
        else:
            decisions["slow_down"] = 0.0
    else:
        decisions["slow_down"] = 0.0

    # Yield: slow down + low displacement (waiting for something)
    if decisions.get("slow_down", 0) > 0.3 and total_displacement < 5.0:
        decisions["yield"] = 0.6
    elif speed_end < 0.5 and speed_start < 1.0 and total_displacement < 3.0:
        decisions["yield"] = 0.5
    else:
        decisions["yield"] = 0.0

    # Maintain: stable speed, mostly straight
    if speed_start > 0.3:
        speed_var = speeds.std().item() / max(speeds.mean().item(), 0.01)
        if speed_var < 0.2 and abs(lateral_delta) < 1.0:
            decisions["maintain"] = max(0.0, 1.0 - speed_var * 3.0)
        else:
            decisions["maintain"] = 0.0
    else:
        decisions["maintain"] = 0.0

    # Accelerate: significant speed increase
    if speed_start > 0.1:
        accel_ratio = speed_end / max(speed_start, 0.01)
        if accel_ratio > 1.3:
            decisions["accelerate"] = min(1.0, (accel_ratio - 1.0))
        else:
            decisions["accelerate"] = 0.0
    else:
        if speed_end > 0.5:
            decisions["accelerate"] = 0.5
        else:
            decisions["accelerate"] = 0.0

    # Nudge left/right: lateral movement
    if lateral_delta < -0.8:  # Negative y = left in typical ego frame
        decisions["nudge_left"] = min(1.0, abs(lateral_delta) / 3.0)
    else:
        decisions["nudge_left"] = 0.0

    if lateral_delta > 0.8:
        decisions["nudge_right"] = min(1.0, abs(lateral_delta) / 3.0)
    else:
        decisions["nudge_right"] = 0.0

    return decisions


def extract_coc_decisions(coc_text: str) -> dict[str, float]:
    """Extract driving decisions expressed in CoC text.

    Args:
        coc_text: The Chain-of-Causation reasoning text.

    Returns:
        Dict mapping decision names to confidence scores.
    """
    decisions: dict[str, float] = {}
    for decision_name, patterns in DECISION_PATTERNS.items():
        hits = _count_matches(coc_text, patterns)
        if hits > 0:
            # Sigmoid-like: 1 hit = 0.7, 2 = 0.9, 3+ = 1.0
            decisions[decision_name] = min(0.7 + 0.15 * (hits - 1), 1.0)
        else:
            decisions[decision_name] = 0.0
    return decisions


def extract_coc_object_mentions(coc_text: str) -> dict[str, bool]:
    """Extract which object types the CoC mentions.

    Args:
        coc_text: The CoC reasoning text.

    Returns:
        Dict mapping object type names to whether they are mentioned.
    """
    mentions: dict[str, bool] = {}
    for obj_type, patterns in OBJECT_TYPE_PATTERNS.items():
        mentions[obj_type] = _has_pattern(coc_text, patterns)
    return mentions


def extract_coc_spatial_claims(coc_text: str) -> dict[str, bool]:
    """Extract spatial relationship claims from CoC text.

    Args:
        coc_text: The CoC reasoning text.

    Returns:
        Dict mapping spatial relationship names to whether they are claimed.
    """
    claims: dict[str, bool] = {}
    for spatial_rel, patterns in SPATIAL_PATTERNS.items():
        claims[spatial_rel] = _has_pattern(coc_text, patterns)
    return claims


def compute_grounded_coc_reward(
    coc_text: str,
    scene_facts: dict[str, Any],
    gt_future_xyz: torch.Tensor,
    gt_future_rot: torch.Tensor | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> RewardComponents:
    """Compute grounded CoC reward verified against obstacle and GT data.

    This is the core improvement over keyword-based reward. It creates
    genuine reward variance by checking verifiable claims.

    Reward dimensions:
      1. **Object grounding**: Does CoC correctly identify nearby objects?
      2. **Hallucination Penalty**: Does CoC mention objects NOT in scene?
      3. **Spatial Accuracy**: Are spatial claims correct?
      4. **Decision Alignment**: Does CoC decision match GT behavior?
      5. **Threat Identification**: Does CoC identify high-threat objects?

    Args:
        coc_text: Extracted CoC reasoning text.
        scene_facts: Ground truth scene facts from extract_scene_facts_from_obstacles().
        gt_future_xyz: Ground truth future trajectory.
        gt_future_rot: Optional ground truth rotations.
        weights: Optional per-dimension weights.

    Returns:
        RewardComponents with aggregated reward and detailed metrics.
    """
    if weights is None:
        weights = {
            "object_grounding": 0.30,
            "hallucination_penalty": 0.15,
            "spatial_accuracy": 0.15,
            "decision_alignment": 0.30,
            "threat_identification": 0.10,
        }

    metrics: dict[str, float] = {}

    # ---- Dimension 1: Object Grounding ----
    # Did the CoC correctly mention objects that ARE in the scene?
    coc_mentions = extract_coc_object_mentions(coc_text)
    gt_types = {
        "vehicle": scene_facts.get("has_vehicle_nearby", False),
        "pedestrian": scene_facts.get("has_pedestrian_nearby", False),
        "cyclist": scene_facts.get("has_cyclist_nearby", False),
    }

    correct_mentions = 0
    total_gt_objects = 0
    for obj_type, in_scene in gt_types.items():
        if in_scene:
            total_gt_objects += 1
            if coc_mentions.get(obj_type, False):
                correct_mentions += 1

    if total_gt_objects > 0:
        object_recall = correct_mentions / total_gt_objects
    else:
        # No objects in scene — mentioning any object is hallucination
        object_recall = 1.0 if not any(coc_mentions.values()) else 0.5

    metrics["object_recall"] = float(object_recall)
    metrics["correct_object_mentions"] = float(correct_mentions)
    metrics["total_gt_objects"] = float(total_gt_objects)

    # ---- Dimension 2: Hallucination Penalty ----
    # Penalize mentioning objects NOT in the scene
    false_mentions = 0
    for obj_type, mentioned in coc_mentions.items():
        if mentioned and not gt_types.get(obj_type, False):
            false_mentions += 1

    # Precision: fraction of mentions that are correct
    total_mentions = sum(1 for v in coc_mentions.values() if v)
    if total_mentions > 0:
        object_precision = (total_mentions - false_mentions) / total_mentions
    else:
        object_precision = 0.5  # Neutral when no mentions (neither good nor bad)

    hallucination_score = max(0.0, 1.0 - false_mentions * 0.3)
    metrics["object_precision"] = float(object_precision)
    metrics["false_object_mentions"] = float(false_mentions)
    metrics["hallucination_score"] = float(hallucination_score)

    # ---- Dimension 3: Spatial Accuracy ----
    spatial_claims = extract_coc_spatial_claims(coc_text)
    spatial_checks = 0
    spatial_correct = 0

    # Check left/right claims
    if spatial_claims.get("left", False):
        spatial_checks += 1
        if scene_facts.get("object_on_left", False):
            spatial_correct += 1
    if spatial_claims.get("right", False):
        spatial_checks += 1
        if scene_facts.get("object_on_right", False):
            spatial_correct += 1
    if spatial_claims.get("ahead", False):
        spatial_checks += 1
        if scene_facts.get("object_ahead", False):
            spatial_correct += 1
    if spatial_claims.get("approaching", False):
        spatial_checks += 1
        if scene_facts.get("approaching_object", False):
            spatial_correct += 1

    if spatial_checks > 0:
        spatial_accuracy = spatial_correct / spatial_checks
    else:
        spatial_accuracy = 0.5  # Neutral

    metrics["spatial_accuracy"] = float(spatial_accuracy)
    metrics["spatial_checks_performed"] = float(spatial_checks)

    # ---- Dimension 4: Decision Alignment ----
    gt_decisions = extract_gt_decision(gt_future_xyz, gt_future_rot)
    coc_decisions = extract_coc_decisions(coc_text)

    # Check alignment between CoC decisions and GT decisions
    decision_scores = []
    for decision_name in ("stop", "yield", "slow_down", "maintain", "accelerate",
                          "nudge_left", "nudge_right"):
        gt_conf = gt_decisions.get(decision_name, 0.0)
        coc_conf = coc_decisions.get(decision_name, 0.0)

        if gt_conf > 0.3:
            # This is a real GT decision — did CoC identify it?
            alignment = 1.0 - abs(gt_conf - coc_conf)
            decision_scores.append(alignment)
        elif coc_conf > 0.5:
            # CoC claims a decision that GT doesn't support — mild penalty
            decision_scores.append(max(0.0, 1.0 - coc_conf * 0.5))

    if decision_scores:
        decision_alignment = float(np.mean(decision_scores))
    else:
        decision_alignment = 0.3  # No decisions expressed

    metrics["decision_alignment"] = float(decision_alignment)

    # Also compute the primary GT decision for logging
    primary_gt_decision = max(gt_decisions, key=lambda k: gt_decisions[k])
    metrics["gt_primary_decision"] = float(
        ["stop", "yield", "slow_down", "maintain", "accelerate", "nudge_left", "nudge_right"]
        .index(primary_gt_decision)
    )
    metrics["gt_decision_confidence"] = float(gt_decisions[primary_gt_decision])

    # ---- Dimension 5: Threat Identification ----
    high_threat = scene_facts.get("high_threat_objects", [])
    closest_dist = scene_facts.get("closest_distance", float("inf"))

    if len(high_threat) > 0:
        # Check if CoC mentions threat-related concepts
        threat_words = ["danger", "risk", "hazard", "threat", "close", "near",
                       "careful", "caution", "attention", "watch"]
        coc_lower = coc_text.lower()
        threat_mentioned = any(w in coc_lower for w in threat_words)

        # Check if CoC mentions the closest object type
        closest_type = scene_facts.get("closest_object_type", "")
        type_mentioned = False
        if closest_type:
            for obj_type, patterns in OBJECT_TYPE_PATTERNS.items():
                if any(pt in closest_type.lower() for pt in [obj_type]):
                    if coc_mentions.get(obj_type, False):
                        type_mentioned = True
                        break

        threat_score = (float(threat_mentioned) * 0.5 + float(type_mentioned) * 0.5)
    else:
        # No high-threat objects — mentioning threats is wrong
        threat_words = ["danger", "risk", "hazard", "threat"]
        coc_lower = coc_text.lower()
        threat_mentioned = any(w in coc_lower for w in threat_words)
        threat_score = 0.7 if not threat_mentioned else 0.3

    metrics["threat_score"] = float(threat_score)
    metrics["num_high_threat_objects"] = float(len(high_threat))
    metrics["closest_obstacle_distance"] = float(closest_dist) if np.isfinite(closest_dist) else 999.0

    # ---- Anti-template diversity signal ----
    # Penalize very short or formulaic CoC
    word_count = len(coc_text.strip().split())
    unique_words = len(set(coc_text.lower().split()))
    if word_count > 0:
        unique_ratio = unique_words / word_count
    else:
        unique_ratio = 0.0

    # Very low unique ratio suggests repetitive template
    diversity_score = min(1.0, unique_ratio / 0.7)  # 0.7+ unique ratio is good
    metrics["coc_word_count"] = float(word_count)
    metrics["coc_unique_ratio"] = float(unique_ratio)
    metrics["diversity_score"] = float(diversity_score)

    # ---- Aggregate Reward ----
    grounded_reward = (
        weights.get("object_grounding", 0.30) * object_recall
        + weights.get("hallucination_penalty", 0.15) * hallucination_score
        + weights.get("spatial_accuracy", 0.15) * spatial_accuracy
        + weights.get("decision_alignment", 0.30) * decision_alignment
        + weights.get("threat_identification", 0.10) * threat_score
    )

    # Apply diversity bonus/penalty
    diversity_bonus = 0.05 * (diversity_score - 0.5)  # [-0.025, +0.025]
    grounded_reward += diversity_bonus

    # Word count gate: very short CoC (<10 words) gets penalized
    if word_count < 5:
        grounded_reward *= 0.3
    elif word_count < 10:
        grounded_reward *= 0.6

    grounded_reward = float(np.clip(grounded_reward, 0.0, 1.0))
    metrics["grounded_coc_reward"] = float(grounded_reward)

    return RewardComponents(
        reward=grounded_reward,
        metrics=metrics,
    )
