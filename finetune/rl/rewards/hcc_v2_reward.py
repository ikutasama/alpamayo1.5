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

"""Improved HCC-RM v2 — Grounded Reward with Obstacle Verification.

Replaces the keyword-based CoC reward with grounded verification
against obstacle.offline data and GT trajectory decisions.

Key changes from the original HCC-RM:
  1. Layer 1 (Scene): Now uses obstacle data for factual verification
  2. Layer 2 (CoC Quality): Replaced keyword matching with grounded checks
  3. Layer 3 (RAA): Replaced CoC-vs-predicted with CoC-vs-GT-decision
  4. Layer 4 (Trajectory): Unchanged (ADE + comfort)
  5. Added reward variance protection and anti-template signals
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rl.rewards.coc_reward import extract_coc_sections, extract_coc_text


_REQUIRED_HCC_V2_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "coc_quality_weight",
    "raa_weight",
]


def _get_hcc_v2_cfg(config: object | None) -> dict[str, float]:
    """Extract HCC-RM v2 parameters from TOML config."""
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "HCC-RM v2 config not found. "
            f"Required: {_REQUIRED_HCC_V2_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_HCC_V2_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(f"Missing keys for HCC-RM v2: {missing}")

    return {
        "traj_l2_weight": float(reward_cfg["traj_l2_weight"]),
        "comfort_weight": float(reward_cfg["comfort_weight"]),
        "coc_quality_weight": float(reward_cfg["coc_quality_weight"]),
        "raa_weight": float(reward_cfg["raa_weight"]),
        "scene_threshold": float(reward_cfg.get("scene_threshold", 0.2)),
        "ade_threshold": float(reward_cfg.get("ade_threshold", 2.0)),
        "scene_reward_weight": float(reward_cfg.get("scene_reward_weight", 0.15)),
        "use_obstacle_reward": bool(reward_cfg.get("use_obstacle_reward", True)),
        "anti_template_weight": float(reward_cfg.get("anti_template_weight", 0.05)),
    }


def compute_hcc_v2_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Compute the improved HCC-RM v2 reward.

    This version uses grounded obstacle verification instead of keyword matching.

    Args:
        to_be_evaluated: Full rollout completion string.
        reference: Reference data dict (must include scene_facts if available).
        tokenizer: Text tokenizer.
        traj_tokenizer: Trajectory tokenizer.
        config: Cosmos-RL config.
        model_config: Model configuration.

    Returns:
        Tuple of (final_reward, reward_dict).
    """
    from cosmos_rl.utils.logging import logger

    from rl.rewards.comfort_reward import compute_comfort
    from rl.rewards.grounded_coc_reward import (
        compute_grounded_coc_reward,
        extract_coc_decisions,
        extract_gt_decision,
    )
    from rl.rewards.traj_reward import calculate_ade
    from rl.utils.trajectory_decode import decode_rollout_trajectory

    w = _get_hcc_v2_cfg(config)

    # ---- Decode predicted trajectory ----
    gt_fut_xyz = reference["ego_future_xyz"]
    gt_fut_rot = reference.get("ego_future_rot", None)
    predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
        to_be_evaluated,
        reference["ego_history_xyz"],
        reference["ego_history_rot"],
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        model_config=model_config,
    )

    # ---- Extract CoC text ----
    sections = extract_coc_sections(to_be_evaluated)
    coc_text = str(sections["reasoning"]).strip()
    format_score = float(sections.get("format_score", 0.0))
    word_count = len(coc_text.split())

    # Print CoC for debugging (using print not logger to ensure visibility)
    print(f"\n[HCC-v2] CoC: {coc_text}", flush=True)
    print(f"[HCC-v2] word_count={word_count} format_score={format_score:.2f}", flush=True)

    # ---- Get scene facts (pre-computed from obstacle data) ----
    scene_facts = reference.get("scene_facts", None)

    if scene_facts is None:
        # No obstacle data available — use minimal defaults
        scene_facts = {
            "has_vehicle_nearby": False,
            "has_pedestrian_nearby": False,
            "has_cyclist_nearby": False,
            "closest_object_type": None,
            "closest_distance": float("inf"),
            "object_on_left": False,
            "object_on_right": False,
            "object_ahead": False,
            "approaching_object": False,
            "num_obstacles": 0,
            "high_threat_objects": [],
        }
        print("[HCC-v2] WARNING: No scene_facts available", flush=True)
    else:
        print(f"[HCC-v2] scene_facts: num_obstacles={scene_facts.get('num_obstacles', 0)} "
              f"vehicle={scene_facts.get('has_vehicle_nearby')} "
              f"ped={scene_facts.get('has_pedestrian_nearby')} "
              f"closest_dist={scene_facts.get('closest_distance', 'N/A')}", flush=True)

    # ============================================================
    # Layer 1: Scene Understanding (Grounded)
    # ============================================================
    has_real_obstacles = scene_facts.get("num_obstacles", 0) > 0
    if w.get("use_obstacle_reward", True) and has_real_obstacles:
        # Use grounded CoC reward with obstacle verification
        grounded = compute_grounded_coc_reward(
            coc_text, scene_facts, gt_fut_xyz[0] if gt_fut_xyz.dim() == 3 else gt_fut_xyz,
            gt_fut_rot[0] if gt_fut_rot is not None and gt_fut_rot.dim() == 4 else gt_fut_rot,
        )
        s1_scene = grounded.reward
        grounded_metrics = grounded.metrics
    else:
        # Fallback: use the original keyword-based scene score
        # but still compute grounded metrics for logging
        from rl.rewards.coc_reward import compute_coc_quality
        coc_scores_legacy = compute_coc_quality(to_be_evaluated)
        s1_scene = coc_scores_legacy.get("coc_factual", 0.0)
        grounded_metrics = {"object_recall": 0.0, "decision_alignment": 0.0}

    # ============================================================
    # Layer 2: Decision-GT Alignment (replaces old RAA)
    # ============================================================
    # Instead of checking CoC vs predicted trajectory (circular),
    # check CoC decisions vs GT trajectory decisions
    gt_decisions = extract_gt_decision(gt_fut_xyz, gt_fut_rot)
    coc_decisions = extract_coc_decisions(coc_text)

    # Compute alignment between CoC-stated decisions and GT
    decision_alignment_scores = []
    gt_active_decisions = {k: v for k, v in gt_decisions.items() if v > 0.3}

    if gt_active_decisions:
        for dec_name, gt_conf in gt_active_decisions.items():
            coc_conf = coc_decisions.get(dec_name, 0.0)
            # How well does CoC identify this GT decision?
            alignment = 1.0 - abs(gt_conf - coc_conf)
            decision_alignment_scores.append(alignment)

        # Penalize CoC decisions that contradict GT
        for dec_name, coc_conf in coc_decisions.items():
            if coc_conf > 0.5 and gt_decisions.get(dec_name, 0.0) < 0.2:
                decision_alignment_scores.append(max(0.0, 0.5 - coc_conf * 0.3))

    if decision_alignment_scores:
        s2_decision = float(max(0.0, min(1.0, sum(decision_alignment_scores) / len(decision_alignment_scores))))
    else:
        s2_decision = 0.2 if word_count < 10 else 0.3

    # ============================================================
    # Layer 3: CoC Quality (lightweight — mostly anti-template)
    # ============================================================
    # Instead of keyword-matching quality, focus on:
    # - Is the CoC substantive (not empty/too short)?
    # - Does it have unique content (not just template phrases)?
    unique_words = len(set(coc_text.lower().split()))
    unique_ratio = unique_words / max(word_count, 1)

    # Format compliance
    format_quality = format_score

    # Length quality (sweet spot: 20-30 words)
    if word_count < 10:
        length_quality = 0.2
    elif word_count < 15:
        length_quality = 0.3 + 0.3 * (word_count - 10) / 5.0
    elif word_count < 20:
        length_quality = 0.6 + 0.4 * (word_count - 15) / 5.0
    elif word_count <= 30:
        length_quality = 1.0  # Sweet spot
    elif word_count <= 40:
        length_quality = 1.0 - 0.5 * (word_count - 30) / 10.0
    elif word_count <= 50:
        length_quality = 0.5 - 0.4 * (word_count - 40) / 10.0
    else:
        length_quality = 0.1  # Severely penalize 50+ words

    # Specificity bonus: reward CoC that mention specific scene details
    # Check for specific action verbs, object types, spatial descriptions
    specificity_keywords = [
        # Specific actions
        "nudge", "merge", "yield", "brake", "accelerate", "decelerate",
        "turn", "curve", "intersection", "crosswalk", "stop sign",
        # Specific objects
        "vehicle", "car", "truck", "bus", "motorcycle", "pedestrian", "cyclist",
        "traffic light", "signal", "lane",
        # Spatial descriptions
        "left", "right", "ahead", "behind", "beside", "approaching",
        # Specific scenarios
        "cut-in", "blocking", "entering", "crossing", "parked"
    ]
    
    coc_lower = coc_text.lower()
    specificity_count = sum(1 for kw in specificity_keywords if kw in coc_lower)
    # Bonus for using 3+ specific keywords, max bonus at 6+
    specificity_bonus = min(1.0, max(0.0, (specificity_count - 2) / 4.0))

    # Diversity quality
    diversity_quality = min(1.0, unique_ratio / 0.65)
    
    # CoC quality: format + length + diversity + specificity
    s3_coc = (
        0.15 * format_quality +
        0.35 * length_quality +
        0.30 * diversity_quality +
        0.20 * specificity_bonus
    )

    # ============================================================
    # Anti-conservative penalty: force CoC to mention obstacles
    # ============================================================
    # If scene has vehicles/pedestrians but CoC doesn't mention them,
    # mildly penalize to encourage obstacle awareness (gentle, not dominating)
    anti_conservative_penalty = 0.0
    # Use consistent threshold: at least 1 obstacle for anti-conservative check
    # (was >5 which made penalty almost never fire)
    has_real_obstacles_for_penalty = scene_facts.get("num_obstacles", 0) > 1
    has_vehicle = scene_facts.get("has_vehicle_nearby", False)
    has_pedestrian = scene_facts.get("has_pedestrian_nearby", False)
    object_recall = grounded_metrics.get("object_recall", 0.0)
    
    if has_real_obstacles_for_penalty and (has_vehicle or has_pedestrian):
        # Scene has obstacles, check if CoC mentions them
        if object_recall < 0.3:
            # CoC barely mentions obstacles → small penalty
            anti_conservative_penalty = -0.05 * (1.0 - object_recall)
        elif object_recall < 0.6:
            # Partial mention → tiny penalty
            anti_conservative_penalty = -0.02 * (1.0 - object_recall)

    # ============================================================
    # Layer 4: Trajectory Quality (ADE + Comfort, unchanged)
    # ============================================================
    l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])
    if not (isinstance(l2_dist, (int, float)) and math.isfinite(l2_dist)):
        l2_dist = 999.0

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
    if not (isinstance(comfort_score, (int, float)) and math.isfinite(comfort_score)):
        comfort_score = 0.0
    comfort_norm = comfort_score - 1.0

    # Continuous trajectory reward (range [0,1] from exp(-l2/2))
    s4_traj = float(math.exp(-l2_dist / 2.0))

    # Comfort is handled SEPARATELY as a penalty (not mixed into traj normalization)
    # comfort_score ∈ [0,1] → comfort_norm ∈ [-1,0]
    # We convert it to a penalty in [-1, 0] that subtracts from final reward
    comfort_penalty_weight = w.get("comfort_weight", 0.1)
    comfort_norm = comfort_score - 1.0  # ∈ [-1, 0]
    # min(0, comfort_norm) ensures comfort only penalizes (never boosts)
    # Result ∈ [-1, 0]: bad comfort → -1, perfect comfort → 0
    comfort_contribution = comfort_penalty_weight * min(0.0, comfort_norm)

    # Normalize traj to [-1, 1] like other layers: s4_traj ∈ [0,1] → s4_norm ∈ [-1,1]
    # Bad trajectory (large l2) → s4_traj≈0 → s4_norm≈-1 (strong negative signal)
    # Good trajectory (small l2) → s4_traj≈1 → s4_norm≈+1
    s4_norm = 2.0 * s4_traj - 1.0

    # ============================================================
    # Aggregate: Weighted additive formula
    # ============================================================
    scene_w = w.get("scene_reward_weight", 0.15)
    coc_w = w["coc_quality_weight"]
    raa_w = w["raa_weight"]  # Now used for decision alignment
    traj_w = max(0.0, 1.0 - scene_w - coc_w - raa_w)

    # All layers normalized to [-1, 1] range for symmetric gradient signal
    s1_norm = 2.0 * s1_scene - 1.0
    s2_norm = 2.0 * s2_decision - 1.0
    s3_norm = 2.0 * s3_coc - 1.0

    final_reward = (
        scene_w * s1_norm
        + coc_w * s3_norm         # CoC quality (anti-template)
        + raa_w * s2_norm         # Decision alignment (replaces old RAA)
        + traj_w * s4_norm        # Trajectory quality (now symmetric [-1,1])
    )

    # Format bonus/penalty
    format_bonus = 0.03 * (2.0 * format_score - 1.0)
    final_reward += format_bonus
    
    # Apply anti-conservative penalty
    final_reward += anti_conservative_penalty
    
    # Apply comfort penalty (separate from traj, only penalizes bad comfort)
    final_reward += comfort_contribution
    
    # Hard penalty for extremely long CoC (>80 words)
    # Apply hard penalty for excessive length (>50 words)
    length_hard_penalty = 0.0
    if word_count > 50:
        length_hard_penalty = -0.3 * min((word_count - 50) / 30, 1.0)
        final_reward += length_hard_penalty

    if not (isinstance(final_reward, (int, float)) and math.isfinite(final_reward)):
        final_reward = 0.0
    final_reward = float(max(-1.0, min(1.0, final_reward)))

    # Print reward breakdown for debugging
    penalty_str = ""
    if abs(anti_conservative_penalty) > 0.001:
        penalty_str = f" penalty={anti_conservative_penalty:+.3f}"
    if word_count > 80:
        penalty_str += f" length_penalty={length_hard_penalty:+.3f}"
    print(f"[HCC-v2] Layers: scene={s1_scene:.3f}(w={scene_w:.2f}) "
          f"decision={s2_decision:.3f}(w={raa_w:.2f}) "
          f"coc_q={s3_coc:.3f}(w={coc_w:.2f}) "
          f"traj={s4_norm:.3f}(w={traj_w:.2f}) "
          f"l2={l2_dist:.2f} spec={specificity_bonus:.2f}{penalty_str} → R={final_reward:.4f}", flush=True)

    # ============================================================
    # Build reward dict for logging
    # ============================================================
    reward_dict = {
        # Layer scores
        "scene_understanding": float(s1_scene),
        "decision_alignment": float(s2_decision),
        "coc_quality": float(s3_coc),
        "traj_quality": float(s4_traj),
        "traj_quality_norm": float(s4_norm),
        # Format
        "format_score": float(format_score),
        "cot_word_count": float(word_count),
        "coc_unique_ratio": float(unique_ratio),
        "diversity_score": float(diversity_quality),
        "specificity_bonus": float(specificity_bonus),
        # Trajectory
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        # Grounded metrics
        "object_recall": float(grounded_metrics.get("object_recall", 0.0)),
        "grounded_coc_reward": float(grounded_metrics.get("grounded_coc_reward", 0.0)),
        # Anti-conservative penalty
        "anti_conservative_penalty": float(anti_conservative_penalty),
        # GT decision info
        "gt_decision_alignment": float(s2_decision),
        # Final
        "reward": float(final_reward),
    }

    # Add detailed grounded metrics if available
    for key in ("correct_object_mentions", "false_object_mentions",
                "hallucination_score", "spatial_accuracy",
                "threat_score", "num_high_threat_objects",
                "closest_obstacle_distance"):
        if key in grounded_metrics:
            reward_dict[key] = float(grounded_metrics[key])

# Log
    logger.warning(
        f"[HCC-v2] scene={s1_scene:.3f} decision={s2_decision:.3f} "
        f"coc_q={s3_coc:.3f} traj={s4_norm:.3f} "
        f"l2={l2_dist:.2f} fmt={format_score:.2f} "
        f"words={word_count} unique={unique_ratio:.2f} "
        f"R={final_reward:.4f}"
    )

    return final_reward, reward_dict
