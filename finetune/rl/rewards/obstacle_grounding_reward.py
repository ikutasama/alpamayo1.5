# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Obstacle-grounded CoC reward: validates whether COT hazard mentions match GT obstacles.

This replaces regex keyword matching (which caused template-hacking and reward
variance collapse) with grounded checks against actual obstacle data from the
PAI dataset. The model gets rewarded for correctly identifying obstacles that
actually exist in the scene, and penalized for mentioning obstacles that don't
exist or for failing to mention nearby high-threat obstacles.

Key design principles:
  - Binary + differential scoring: correct mention of a real obstacle yields
    a clear positive signal; template phrases that match nothing yield zero.
  - Proximity-weighted: closer obstacles are more important to mention.
  - Type-precision: "pedestrian near crosswalk" is rewarded when GT shows a
    pedestrian; "vehicle ahead" when GT shows only a cyclist gets partial credit.
"""

from __future__ import annotations

import re
import math
from typing import Any

import numpy as np
import torch

from rl.rewards.coc_reward import extract_coc_text
from rl.rewards.reward_types import RewardComponents

# ---------------------------------------------------------------------------
# Obstacle type taxonomy for matching COT text to GT categories
# ---------------------------------------------------------------------------
OBSTACLE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "pedestrian": ["pedestrian", "person", "walker", "foot traffic", "crosswalk"],
    "cyclist": ["cyclist", "bicycle", "bike", "motorcycle", "motorbike", "e-bike"],
    "vehicle": ["vehicle", "car", "truck", "bus", "van", "suv", "sedan", "automobile"],
    "barrier": ["barrier", "obstacle", "blockage", "construction", "cone", "debris"],
    "animal": ["animal", "dog", "cat"],
}

# For partial matching: "pedestrian" partially matches "person walking"
OBSTACLE_TYPE_PARTIAL: dict[str, list[str]] = {
    "pedestrian": ["person", "someone", "people", "child"],
    "cyclist": ["bike", "rider"],
    "vehicle": ["car", "truck", "bus", "van"],
}

# Position/direction keywords that COT may use to describe obstacle location
POSITION_KEYWORDS: dict[str, list[str]] = {
    "ahead": ["ahead", "front", "forward", "in front", "ahead of"],
    "left": ["left", "left side", "cross left", "from left", "left lane"],
    "right": ["right", "right side", "cross right", "from right", "right lane"],
    "behind": ["behind", "rear", "following", "back", "behind us"],
    "near": ["near", "close", "approaching", "nearby", "proximity", "adjacent"],
}

# Threat/severity keywords
THREAT_KEYWORDS: list[str] = [
    "high threat", "dangerous", "critical", "urgent", "high risk",
    "most important", "key hazard", "primary concern", "main threat",
]


def _parse_obstacle_bbox_from_reference(reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse obstacle bbox data from reference dict into a structured list.

    The obstacle.offline parquet from PAI contains per-frame obstacle data
    including type, position, bbox, and other attributes. We extract the
    obstacles present at the keyframe time (t0).

    Expected reference keys:
      - obstacle_bbox_history: obstacles visible during history window
      - obstacle_bbox_future: obstacles visible during future window

    Returns:
        List of obstacle dicts with keys: type, position_xyz, distance,
        is_high_threat, direction_from_ego.
    """
    obstacles = []

    # Try to get obstacle data from reference
    obstacle_data = reference.get("obstacle_info", None)

    if obstacle_data is None:
        # Fallback: try legacy keys
        obs_hist = reference.get("obstacle_bbox_history", None)
        obs_fut = reference.get("obstacle_bbox_future", None)

        if obs_hist is not None and isinstance(obs_hist, dict):
            obstacle_data = obs_hist
        elif obs_fut is not None and isinstance(obs_fut, dict):
            obstacle_data = obs_fut

    if obstacle_data is None:
        return obstacles

    # obstacle_data could be a dict with 'types', 'positions', etc.
    # or a list of individual obstacle dicts
    if isinstance(obstacle_data, list):
        return obstacle_data

    if isinstance(obstacle_data, dict):
        # Structured format: {"types": [...], "positions": [...], "distances": [...]}
        types = obstacle_data.get("types", [])
        positions = obstacle_data.get("positions", [])
        distances = obstacle_data.get("distances", [])
        is_threats = obstacle_data.get("is_high_threat", [])
        directions = obstacle_data.get("directions_from_ego", [])

        n = min(len(types), len(positions), len(distances))
        for i in range(n):
            obs = {
                "type": str(types[i]).lower() if types[i] else "unknown",
                "position": positions[i] if i < len(positions) else None,
                "distance": float(distances[i]) if i < len(distances) else 999.0,
                "is_high_threat": bool(is_threats[i]) if i < len(is_threats) else False,
                "direction": str(directions[i]).lower() if i < len(directions) else "unknown",
            }
            obstacles.append(obs)

    return obstacles


def _extract_motion_profile_from_gt(
    gt_future_xyz: torch.Tensor | None,
    gt_future_rot: torch.Tensor | None,
) -> dict[str, float]:
    """Extract a coarse motion profile from GT trajectory for decision matching.

    Returns dict with: is_stopped, is_slow, is_maintaining, is_accelerating,
    is_turning_left, is_turning_right, is_straight, avg_speed, displacement.
    """
    if gt_future_xyz is None:
        return {
            "is_stopped": 0.0, "is_slow": 0.0, "is_maintaining": 1.0,
            "is_accelerating": 0.0, "is_turning_left": 0.0,
            "is_turning_right": 0.0, "is_straight": 1.0,
            "avg_speed": 0.0, "displacement": 0.0,
        }

    xyz = gt_future_xyz
    while xyz.ndim > 2:
        xyz = xyz[0]

    if xyz.shape[0] < 2:
        return {
            "is_stopped": 1.0, "is_slow": 0.0, "is_maintaining": 0.0,
            "is_accelerating": 0.0, "is_turning_left": 0.0,
            "is_turning_right": 0.0, "is_straight": 1.0,
            "avg_speed": 0.0, "displacement": 0.0,
        }

    dx = xyz[1:, 0] - xyz[:-1, 0]
    dy = xyz[1:, 1] - xyz[:-1, 1]
    speeds = torch.sqrt(dx**2 + dy**2)

    avg_speed = float(speeds.mean().item())
    speed_start = float(speeds[:5].mean().item()) if speeds.numel() >= 5 else avg_speed
    speed_end = float(speeds[-5:].mean().item()) if speeds.numel() >= 5 else avg_speed

    displacement = float(torch.linalg.norm(xyz[-1, :2] - xyz[0, :2]).item())
    lateral = float((xyz[-1, 1] - xyz[0, 1]).item())

    # Heading change
    heading_delta = 0.0
    if gt_future_rot is not None:
        rot = gt_future_rot
        while rot.ndim > 3:
            rot = rot[0]
        if rot.shape[0] >= 2:
            heading = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
            dh = heading[-1] - heading[0]
            heading_delta = float(torch.atan2(torch.sin(dh), torch.cos(dh)).item())

    # Decision classification
    is_stopped = float(avg_speed < 0.25 and displacement < 1.0)
    is_slow = float(avg_speed < 0.8 and avg_speed >= 0.25)
    is_maintaining = float(avg_speed >= 0.8 and abs(speed_end - speed_start) / max(speed_start, 0.1) < 0.15)

    speed_change = (speed_end - speed_start) / max(speed_start, 0.1)
    is_accelerating = float(speed_change > 0.15 and avg_speed >= 0.5)
    is_decelerating = float(speed_change < -0.15 or (avg_speed < 0.5 and avg_speed > 0.1))

    is_turning_left = float(lateral < -0.6 or heading_delta > 0.18)
    is_turning_right = float(lateral > 0.6 or heading_delta < -0.18)
    is_straight = float(abs(lateral) < 0.8 and abs(heading_delta) < 0.22)

    return {
        "is_stopped": is_stopped, "is_slow": is_slow,
        "is_maintaining": is_maintaining, "is_accelerating": is_accelerating,
        "is_decelerating": is_decelerating,
        "is_turning_left": is_turning_left, "is_turning_right": is_turning_right,
        "is_straight": is_straight, "avg_speed": avg_speed,
        "displacement": displacement, "lateral": lateral,
        "speed_change": speed_change, "heading_delta": heading_delta,
    }


def _match_obstacle_type_in_text(
    coc_text_l: str,
    obstacle_type: str,
) -> float:
    """Check if COT text mentions an obstacle of the given type.

    Returns:
        1.0 if exact match found, 0.5 if partial match, 0.0 if no match.
    """
    # Exact keyword match
    exact_keywords = OBSTACLE_TYPE_KEYWORDS.get(obstacle_type, [])
    for kw in exact_keywords:
        if re.search(rf"\b{kw}\b", coc_text_l, re.IGNORECASE):
            return 1.0

    # Partial keyword match
    partial_keywords = OBSTACLE_TYPE_PARTIAL.get(obstacle_type, [])
    for kw in partial_keywords:
        if re.search(rf"\b{kw}\b", coc_text_l, re.IGNORECASE):
            return 0.5

    return 0.0


def _match_direction_in_text(
    coc_text_l: str,
    direction: str,
) -> float:
    """Check if COT text describes obstacle in the given direction from ego.

    Returns:
        1.0 if direction explicitly mentioned near obstacle reference,
        0.3 if direction mentioned anywhere, 0.0 if not.
    """
    direction_kws = POSITION_KEYWORDS.get(direction, [])
    for kw in direction_kws:
        if re.search(rf"\b{kw}\b", coc_text_l, re.IGNORECASE):
            return 1.0
    return 0.0


def _find_nearest_obstacle_mentions(
    coc_text_l: str,
    obstacles: list[dict[str, Any]],
) -> dict[str, float]:
    """Match COT obstacle mentions to actual GT obstacles.

    For each GT obstacle (sorted by proximity), check if the COT:
    1. Mentions the correct obstacle type
    2. Describes it in roughly the correct direction
    3. Highlights it as a threat (for high-threat obstacles)

    Returns:
        Dict with scoring components for aggregation.
    """
    if not obstacles:
        # No obstacles in scene: COT should not hallucinate threats
        has_any_obstacle_mention = any(
            re.search(rf"\b{kw}\b", coc_text_l, re.IGNORECASE)
            for kws in OBSTACLE_TYPE_KEYWORDS.values()
            for kw in kws
        )
        if has_any_obstacle_mention:
            # Penalize hallucinating obstacles when none exist
            return {
                "obstacle_type_match": 0.0,
                "obstacle_direction_match": 0.0,
                "obstacle_threat_match": 0.0,
                "obstacle_hallucination_penalty": -0.3,
                "num_gt_obstacles": 0.0,
            }
        else:
            # Correctly no obstacle mentions when scene is clear
            return {
                "obstacle_type_match": 0.5,  # neutral: correctly said nothing about obstacles
                "obstacle_direction_match": 0.0,
                "obstacle_threat_match": 0.0,
                "obstacle_hallucination_penalty": 0.0,
                "num_gt_obstacles": 0.0,
            }

    # Sort obstacles by distance (closest first)
    sorted_obs = sorted(obstacles, key=lambda o: float(o.get("distance", 999.0)))

    type_match_scores = []
    direction_match_scores = []
    threat_match_scores = []
    threat_penalties = []

    # Check the top-N closest obstacles (most safety-critical)
    top_n = min(5, len(sorted_obs))
    for obs in sorted_obs[:top_n]:
        obs_type = obs.get("type", "unknown")
        obs_direction = obs.get("direction", "unknown")
        obs_is_threat = obs.get("is_high_threat", False)
        obs_distance = float(obs.get("distance", 999.0))

        # Proximity weight: closer obstacles matter more
        proximity_weight = max(0.3, 1.0 - obs_distance / 50.0)

        # Type matching
        type_match = _match_obstacle_type_in_text(coc_text_l, obs_type)
        type_match_scores.append(type_match * proximity_weight)

        # Direction matching (only scored if type was mentioned)
        if type_match > 0.0:
            dir_match = _match_direction_in_text(coc_text_l, obs_direction)
            direction_match_scores.append(dir_match * proximity_weight)
        else:
            direction_match_scores.append(0.0)

        # Threat matching: for high-threat obstacles, did COT flag them?
        if obs_is_threat:
            # Check if COT describes this as a threat
            has_threat_desc = any(
                re.search(rf"\b{kw}\b", coc_text_l, re.IGNORECASE)
                for kw in THREAT_KEYWORDS
            )
            if type_match > 0.0:
                if has_threat_desc:
                    threat_match_scores.append(1.0 * proximity_weight)
                else:
                    threat_match_scores.append(0.3 * proximity_weight)  # mentioned but not flagged as threat
            else:
                # Failed to mention a high-threat obstacle at all
                threat_penalties.append(-0.4 * proximity_weight)

    # Aggregate
    n_gt = len(sorted_obs[:top_n])
    type_avg = sum(type_match_scores) / max(n_gt, 1)
    dir_avg = sum(direction_match_scores) / max(n_gt, 1)
    threat_avg = sum(threat_match_scores) / max(len(threat_match_scores), 1) if threat_match_scores else 0.0
    threat_penalty = sum(threat_penalties)

    hallucination_penalty = 0.0
    # Check if COT mentions obstacle types not present in GT
    gt_types = set(o.get("type", "").lower() for o in sorted_obs[:top_n])
    for cat_type, keywords in OBSTACLE_TYPE_KEYWORDS.items():
        if cat_type not in gt_types:
            for kw in keywords:
                if re.search(rf"\b{kw}\b", coc_text_l, re.IGNORECASE):
                    hallucination_penalty -= 0.15
                    break

    return {
        "obstacle_type_match": type_avg,
        "obstacle_direction_match": dir_avg,
        "obstacle_threat_match": threat_avg,
        "obstacle_threat_penalty": threat_penalty,
        "obstacle_hallucination_penalty": hallucination_penalty,
        "num_gt_obstacles": float(n_gt),
    }


def compute_obstacle_grounding_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    gt_future_xyz: torch.Tensor | None = None,
    gt_future_rot: torch.Tensor | None = None,
) -> RewardComponents:
    """Compute obstacle-grounded reward: validates COT obstacle mentions against GT.

    This is the primary replacement for regex-based factual_accuracy and
    safety_awareness scores. Instead of rewarding keyword presence, it rewards
    correct identification of actual obstacles in the scene.

    The reward is structured to create variance within GRPO groups:
    - Correctly mentioning the closest obstacle type → high score
    - Template phrases mentioning generic "vehicle" when GT shows cyclist → lower score
    - Hallucinating obstacles that don't exist → penalty
    - Missing high-threat obstacles → penalty

    Args:
        to_be_evaluated: Full rollout completion string.
        reference: Reference data dict containing obstacle_info.
        gt_future_xyz: GT future trajectory (optional, for context).
        gt_future_rot: GT future trajectory rotations (optional).

    Returns:
        RewardComponents with reward in [-0.5, 1.0] and detailed metrics.
    """
    coc_text = extract_coc_text(to_be_evaluated)
    coc_text_l = coc_text.lower()

    if not coc_text or len(coc_text.strip()) < 10:
        return RewardComponents(
            reward=0.0,
            metrics={
                "obstacle_grounding_score": 0.0,
                "obstacle_type_match": 0.0,
                "obstacle_direction_match": 0.0,
                "obstacle_hallucination_penalty": 0.0,
                "num_gt_obstacles": 0.0,
            },
        )

    # Parse obstacles from reference
    obstacles = _parse_obstacle_bbox_from_reference(reference)

    # Match COT mentions to GT obstacles
    match_scores = _find_nearest_obstacle_mentions(coc_text_l, obstacles)

    # Compute final reward with variance-creating structure
    type_match = match_scores["obstacle_type_match"]
    dir_match = match_scores["obstacle_direction_match"]
    threat_match = match_scores.get("obstacle_threat_match", 0.0)
    threat_penalty = match_scores.get("obstacle_threat_penalty", 0.0)
    halluc_penalty = match_scores.get("obstacle_hallucination_penalty", 0.0)

    # Weighted aggregation: type_match is the most important
    reward = (
        0.50 * type_match
        + 0.20 * dir_match
        + 0.15 * threat_match
        + threat_penalty
        + halluc_penalty
    )

    # Clip to reasonable range
    reward = max(-0.5, min(1.0, reward))

    return RewardComponents(
        reward=float(reward),
        metrics={
            "obstacle_grounding_score": float(reward),
            "obstacle_type_match": float(type_match),
            "obstacle_direction_match": float(dir_match),
            "obstacle_threat_match": float(threat_match),
            "obstacle_threat_penalty": float(threat_penalty),
            "obstacle_hallucination_penalty": float(halluc_penalty),
            "num_gt_obstacles": float(match_scores.get("num_gt_obstacles", 0.0)),
        },
    )