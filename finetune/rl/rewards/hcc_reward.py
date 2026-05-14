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

"""Hierarchical Causal-Consistent Reward Model (HCC-RM).

Aggregates four layers of reward signals into a single scalar reward:
  Layer 1 — Scene Understanding (factual accuracy gating)
  Layer 2 — Causal Coherence (reasoning quality)
  Layer 3 — Reasoning-Action Alignment (trajectory follows reasoning?)
  Layer 4 — Trajectory Quality (ADE + comfort)

This is the main entry point for HCC-RM reward computation.
"""

from __future__ import annotations

from typing import Any

import torch

from rl.rewards.coc_reward import compute_coc_quality

_REQUIRED_HCC_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "coc_quality_weight",
    "raa_weight",
]


def _get_hcc_cfg(config: object | None) -> dict[str, float]:
    """Extract HCC-RM parameters from Cosmos TOML [custom.alpamayo.reward].

    Args:
        config: Cosmos-RL config object with TOML settings.

    Returns:
        Dict with reward weights and thresholds.

    Raises:
        ValueError: If required keys are missing.
    """
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "HCC-RM reward config not found in TOML. "
            f"Required keys under [custom.alpamayo.reward]: {_REQUIRED_HCC_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_HCC_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(
            f"Missing key(s) in [custom.alpamayo.reward] for HCC-RM: {missing}"
        )

    return {
        "traj_l2_weight": float(reward_cfg["traj_l2_weight"]),
        "comfort_weight": float(reward_cfg["comfort_weight"]),
        "coc_quality_weight": float(reward_cfg["coc_quality_weight"]),
        "raa_weight": float(reward_cfg["raa_weight"]),
        "scene_threshold": float(reward_cfg.get("scene_threshold", 0.3)),
        "ade_threshold": float(reward_cfg.get("ade_threshold", 3.0)),
        "coc_weights": reward_cfg.get("coc_dim_weights", None),
    }


def compute_hcc_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Compute the full HCC-RM hierarchical reward.

    Layer structure (gated):
      R = s1 * (α·s2 + β·s3 + γ·s4)   if s1 > τ_scene
      R = -1.0                          otherwise

    Where:
      s1 = scene understanding (factual accuracy from CoC scorer)
      s2 = causal coherence (CoC quality aggregate)
      s3 = reasoning-action alignment (RAA)
      s4 = trajectory quality (ADE + comfort, normalized)

    Args:
        to_be_evaluated: The full rollout completion string.
        reference: Reference data dict containing ground truth.
        tokenizer: Text tokenizer.
        traj_tokenizer: Trajectory tokenizer.
        config: Cosmos-RL config object.
        model_config: Model configuration.

    Returns:
        Tuple of (final_reward, reward_dict) where reward_dict contains
        all individual component scores for logging.
    """
    from cosmos_rl.utils.logging import logger

    from rl.rewards.comfort_reward import compute_comfort
    from rl.rewards.raa_reward import compute_raa_reward
    from rl.rewards.traj_reward import calculate_ade
    from rl.utils.trajectory_decode import decode_rollout_trajectory

    w = _get_hcc_cfg(config)

    # ---------- Decode trajectory from completion ----------
    gt_fut_xyz = reference["ego_future_xyz"]
    predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
        to_be_evaluated,
        reference["ego_history_xyz"],
        reference["ego_history_rot"],
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        model_config=model_config,
    )

    # ============================================================
    # Layer 1: Scene Understanding (Factual Accuracy Gating)
    # ============================================================
    coc_scores = compute_coc_quality(to_be_evaluated, weights=w.get("coc_weights"))
    s1_scene = coc_scores["coc_factual"]  # Use factual accuracy as scene understanding

    # If scene understanding is too low, gate the entire reward
    scene_threshold = w["scene_threshold"]
    if s1_scene < scene_threshold:
        logger.debug(
            f"[HCC-RM] Scene understanding below threshold "
            f"({s1_scene:.3f} < {scene_threshold:.3f}), reward clamped to -1.0"
        )
        reward_dict = {
            "scene_understanding": float(s1_scene),
            "coc_quality": float(coc_scores.get("coc_quality", 0.0)),
            "coc_factual": float(coc_scores.get("coc_factual", 0.0)),
            "coc_coherence": float(coc_scores.get("coc_coherence", 0.0)),
            "coc_safety": float(coc_scores.get("coc_safety", 0.0)),
            "coc_completeness": float(coc_scores.get("coc_completeness", 0.0)),
            "raa_score": 0.0,
            "traj_L2": 999.0,
            "comfort_reward": 0.0,
            "reward": -1.0,
            "reward_type": "hcc_gated_scene",
        }
        return -1.0, reward_dict

    # ============================================================
    # Layer 2: Causal Coherence (Reasoning Quality)
    # ============================================================
    s2_coc = coc_scores.get("coc_quality", 0.0)
    # Normalize to [-1, 1] range: 0→0.0, 1→1.0
    s2_coc_normalized = 2.0 * s2_coc - 1.0  # [0,1] → [-1,1]

    # ============================================================
    # Layer 3: Reasoning-Action Alignment
    # ============================================================
    raa_reward_val, raa_info = compute_raa_reward(
        to_be_evaluated, predicted_fut_xyz, predicted_fut_rot, weight=1.0
    )
    s3_raa = raa_reward_val  # Already in [0, 1], but can be low
    s3_raa_normalized = 2.0 * s3_raa - 1.0  # [0,1] → [-1,1]

    # ============================================================
    # Layer 4: Trajectory Quality (ADE + Comfort)
    # ============================================================
    l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
    comfort_score_norm = comfort_score - 1.0  # Center around 0

    # ADE-based trajectory quality normalized to [-1, 1]
    ade_threshold = w["ade_threshold"]
    if l2_dist < ade_threshold:
        s4_traj = - (l2_dist / ade_threshold)  # [0, -1]
        s4_comfort = comfort_score_norm  # Already centered
    else:
        s4_traj = -1.0
        s4_comfort = -1.0

    # Combined trajectory quality
    tw = w["traj_l2_weight"]
    cw = w["comfort_weight"]
    tw_cw_sum = tw + cw
    if tw_cw_sum > 0:
        s4_combined = (tw * s4_traj + cw * s4_comfort) / tw_cw_sum
    else:
        s4_combined = s4_traj

    # ============================================================
    # Layer Aggregation with hierarchical gating
    # ============================================================
    coc_w = w["coc_quality_weight"]
    raa_w = w["raa_weight"]
    traj_w = max(0.0, 1.0 - coc_w - raa_w)

    raw_reward = (
        coc_w * s2_coc_normalized
        + raa_w * s3_raa_normalized
        + traj_w * s4_combined
    )

    # ADE gating: if trajectory is terrible, still penalize
    if l2_dist >= ade_threshold:
        # Only reasoning quality and RAA can save it partially
        raw_reward = min(raw_reward, -0.8)

    # Apply scene understanding as a global multiplier
    final_reward = s1_scene * raw_reward

    # Clamp to reasonable range
    final_reward = float(max(-1.0, min(1.0, final_reward)))

    logger.debug(
        f"[HCC-RM] s1(scene)={s1_scene:.3f} s2(coc)={s2_coc_normalized:.3f} "
        f"s3(raa)={s3_raa_normalized:.3f} s4(traj)={s4_combined:.3f} "
        f"→ R={final_reward:.4f}"
    )

    reward_dict = {
        "scene_understanding": float(s1_scene),
        "coc_quality": float(coc_scores.get("coc_quality", 0.0)),
        "coc_factual": float(coc_scores.get("coc_factual", 0.0)),
        "coc_coherence": float(coc_scores.get("coc_coherence", 0.0)),
        "coc_safety": float(coc_scores.get("coc_safety", 0.0)),
        "coc_completeness": float(coc_scores.get("coc_completeness", 0.0)),
        "raa_score": float(s3_raa),
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "reward": float(final_reward),
        "reward_type": "hcc",
        **{
            k: float(v) if isinstance(v, (float, int)) else v
            for k, v in raa_info.items()
            if not k.startswith("raa_score")
        },
    }

    return final_reward, reward_dict
