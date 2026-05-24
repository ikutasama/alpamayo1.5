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

import math
import re
from typing import Any

import torch

from rl.rewards.coc_reward import extract_coc_sections
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


def _compute_length_bonus(word_count: int) -> float:
    """Compute CoC length bonus/penalty to encourage 20-40 words.

    Args:
        word_count: Number of words in the CoC text.

    Returns:
        Bonus in [-0.40, +0.50]. Positive for >=20 words, negative for shorter.
    """
    if word_count < 10:
        return -0.40
    if word_count < 15:
        t = (word_count - 10.0) / 5.0
        return -0.40 + t * 0.60
    if word_count < 25:
        t = (word_count - 15.0) / 10.0
        return 0.20 + t * 0.30
    if word_count <= 40:
        t = (word_count - 25.0) / 15.0
        return 0.50 - 0.10 * t
    if word_count <= 60:
        return 0.40
    return 0.40


_FACT_PATTERNS: dict[str, list[str]] = {
    "decelerate": [
        r"\bslow(?:ing)? down\b",
        r"\bdecelerate\b",
        r"\bbrak(?:e|ing)\b",
        r"\breduce speed\b",
        r"\byield\b",
        r"\bstop(?:ping)?\b",
    ],
    "accelerate": [
        r"\bspeed up\b",
        r"\baccelerat(?:e|ing)\b",
        r"\bincrease speed\b",
        r"\bproceed\b",
        r"\bmove forward\b",
    ],
    "maintain_speed": [
        r"\bmaintain\b",
        r"\bkeep\b",
        r"\bsteady\b",
        r"\bcontinue\b",
        r"\bfollow\b",
        r"\bconstant speed\b",
    ],
    "turn_left": [
        r"\bturn left\b",
        r"\bleft turn\b",
        r"\bsteer left\b",
        r"\bchange.*left\b",
        r"\bshift.*left\b",
    ],
    "turn_right": [
        r"\bturn right\b",
        r"\bright turn\b",
        r"\bsteer right\b",
        r"\bchange.*right\b",
        r"\bshift.*right\b",
    ],
    "straight": [
        r"\bstraight\b",
        r"\bkeep lane\b",
        r"\bstay in lane\b",
        r"\blane keeping\b",
        r"\bmaintain lane\b",
    ],
    "scene_observation": [
        r"\bobserve\b",
        r"\bvisible\b",
        r"\bscene\b",
        r"\bcurrent(?:ly)?\b",
        r"\bahead\b",
        r"\bfront\b",
        r"\blane\b",
        r"\broad\b",
    ],
    "safety_rationale": [
        r"\bsafe\b",
        r"\brisk\b",
        r"\bhazard\b",
        r"\bcautious\b",
        r"\bcollision\b",
        r"\bdistance\b",
        r"\bclearance\b",
    ],
}


def _has_any(text_l: str, key: str) -> bool:
    return any(re.search(pattern, text_l) for pattern in _FACT_PATTERNS[key])


def _motion_profile(
    xyz: torch.Tensor,
    rot: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize a trajectory into verifiable longitudinal/lateral facts."""
    if xyz.dim() == 3:
        xyz = xyz[0]
    if rot is not None and rot.dim() == 4:
        rot = rot[0]

    if xyz.shape[0] < 2:
        return {
            "decelerate": 0.0,
            "accelerate": 0.0,
            "maintain_speed": 1.0,
            "turn_left": 0.0,
            "turn_right": 0.0,
            "straight": 1.0,
            "displacement": 0.0,
            "lateral_delta": 0.0,
        }

    dxy = xyz[1:, :2] - xyz[:-1, :2]
    speeds = torch.linalg.norm(dxy, dim=-1)
    speed_start = speeds[: min(5, speeds.numel())].mean()
    speed_end = speeds[-min(5, speeds.numel()) :].mean()
    denom = torch.clamp(speed_start, min=0.1)
    speed_change = float(((speed_end - speed_start) / denom).item())
    displacement = float(torch.linalg.norm(xyz[-1, :2] - xyz[0, :2]).item())
    lateral_delta = float((xyz[-1, 1] - xyz[0, 1]).item())

    heading_delta = 0.0
    if rot is not None and rot.numel() > 0:
        heading = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
        dh = heading[-1] - heading[0]
        heading_delta = float(torch.atan2(torch.sin(dh), torch.cos(dh)).item())

    abs_lat = abs(lateral_delta)
    abs_heading = abs(heading_delta)
    return {
        "decelerate": float(speed_change < -0.12 or speed_end.item() < 0.25),
        "accelerate": float(speed_change > 0.12),
        "maintain_speed": float(abs(speed_change) <= 0.18 and speed_end.item() >= 0.25),
        "turn_left": float(lateral_delta < -0.6 or heading_delta > 0.18),
        "turn_right": float(lateral_delta > 0.6 or heading_delta < -0.18),
        "straight": float(abs_lat <= 0.8 and abs_heading <= 0.22),
        "displacement": displacement,
        "lateral_delta": lateral_delta,
        "speed_change": speed_change,
    }


def _compute_grounded_fact_score(
    coc_text: str,
    reference: dict[str, Any],
    gt_future_xyz: torch.Tensor,
    gt_future_rot: torch.Tensor | None,
) -> tuple[float, dict[str, float]]:
    """Reward CoT facts that are checkable from the current sample.

    This intentionally avoids rewarding length. It scores whether the CoT
    mentions the ego future behavior and available road/scene facts in a way
    that can be verified from labels or metadata.
    """
    text_l = coc_text.lower()
    profile = _motion_profile(gt_future_xyz, gt_future_rot)

    expected: list[tuple[str, float]] = []
    for key in ("decelerate", "accelerate", "maintain_speed", "turn_left", "turn_right", "straight"):
        if profile.get(key, 0.0) > 0.5:
            expected.append((key, 1.0))

    # Always require some explicit scene/risk grounding. If map/road geometry is
    # available in the sample, lane/road mentions become directly checkable.
    expected.append(("scene_observation", 0.6))
    if reference.get("egomotion_lanelines", None) is not None or reference.get(
        "egomotion_road_boundaries", None
    ) is not None:
        expected.append(("scene_observation", 0.4))
    expected.append(("safety_rationale", 0.5))

    total_w = sum(w for _, w in expected) or 1.0
    matched_w = sum(w for key, w in expected if _has_any(text_l, key))
    coverage = matched_w / total_w

    # Penalize unsupported directional/action claims. The trajectory reward still
    # anchors to GT; this prevents CoT from saying "left" while the sample is right.
    contradictions = 0.0
    opposing = (("turn_left", "turn_right"), ("accelerate", "decelerate"))
    for pos, neg in opposing:
        if _has_any(text_l, pos) and profile.get(pos, 0.0) < 0.5 and profile.get(neg, 0.0) > 0.5:
            contradictions += 1.0
        if _has_any(text_l, neg) and profile.get(neg, 0.0) < 0.5 and profile.get(pos, 0.0) > 0.5:
            contradictions += 1.0
    if profile.get("straight", 0.0) > 0.5:
        contradictions += float(_has_any(text_l, "turn_left"))
        contradictions += float(_has_any(text_l, "turn_right"))
    if profile.get("maintain_speed", 0.0) > 0.5:
        contradictions += float(_has_any(text_l, "accelerate"))
        contradictions += float(_has_any(text_l, "decelerate"))
    contradiction_penalty = min(0.35, contradictions * 0.18)
    score = max(0.0, min(1.0, coverage - contradiction_penalty))

    return score, {
        "grounded_fact_score": float(score),
        "grounded_fact_coverage": float(coverage),
        "grounded_fact_contradictions": float(contradictions),
        "gt_decelerate": float(profile.get("decelerate", 0.0)),
        "gt_accelerate": float(profile.get("accelerate", 0.0)),
        "gt_maintain_speed": float(profile.get("maintain_speed", 0.0)),
        "gt_turn_left": float(profile.get("turn_left", 0.0)),
        "gt_turn_right": float(profile.get("turn_right", 0.0)),
        "gt_straight": float(profile.get("straight", 0.0)),
    }


def _check_coc_traj_consistency(
    coc_text: str,
    predicted_fut_xyz: torch.Tensor,
    predicted_fut_rot: torch.Tensor,
) -> float:
    """Penalize when CoC action description contradicts the actual trajectory.

    Args:
        coc_text: The extracted CoC reasoning text.
        predicted_fut_xyz: Predicted trajectory XYZ.
        predicted_fut_rot: Predicted trajectory rotations.

    Returns:
        Consistency penalty in [-0.15, 0.0]. Negative = penalty, 0 = no penalty.
    """
    if not coc_text or len(coc_text.strip()) < 10:
        return 0.0

    coc_text_l = coc_text.lower()
    import re as _re

    coc_lon_decel = float(any(_re.search(p, coc_text_l) for p in [
        r"\bslow down\b", r"\bdecelerate\b", r"\bbrake\b", r"\breduce speed\b",
        r"\byield\b", r"\bstop\b(?! at| for)", r"\bwait\b",
    ]))
    coc_lon_accel = float(any(_re.search(p, coc_text_l) for p in [
        r"\bspeed up\b", r"\baccelerate\b", r"\bincrease speed\b", r"\bproceed\b",
        r"\bgo\b(?!odbye)", r"\badvance\b", r"\bresume\b", r"\bcontinue\b",
    ]))
    coc_lat_left = float(any(_re.search(p, coc_text_l) for p in [
        r"\bturn left\b", r"\bleft turn\b", r"\bsteer left\b", r"\bchange.*left\b",
    ]))
    coc_lat_right = float(any(_re.search(p, coc_text_l) for p in [
        r"\bturn right\b", r"\bright turn\b", r"\bsteer right\b", r"\bchange.*right\b",
    ]))

    heading = torch.atan2(predicted_fut_rot[..., 1, 0], predicted_fut_rot[..., 0, 0])
    dx = predicted_fut_xyz[1:, 0] - predicted_fut_xyz[:-1, 0]
    dy = predicted_fut_xyz[1:, 1] - predicted_fut_xyz[:-1, 1]
    speeds = torch.sqrt(dx**2 + dy**2)
    speed_start = speeds[:5].mean() if len(speeds) >= 5 else speeds.mean()
    speed_end = speeds[-5:].mean() if len(speeds) >= 5 else speeds.mean()
    total_dh = (heading[1:] - heading[:-1]).sum().item()
    total_dy = (predicted_fut_xyz[-1, 1] - predicted_fut_xyz[0, 1]).item()

    traj_lon_accel = 0.0
    traj_lon_decel = 0.0
    if speed_start > 0.1:
        speed_change = (speed_end - speed_start) / speed_start
        traj_lon_accel = float(max(0, speed_change > 0.1))
        traj_lon_decel = float(max(0, speed_change < -0.1))

    traj_lat_left = max(0.0, min(1.0, -total_dy / 3.5)) if total_dy < -0.5 else 0.0
    traj_lat_right = max(0.0, min(1.0, total_dy / 3.5)) if total_dy > 0.5 else 0.0

    violations = 0.0
    if coc_lon_decel > 0.3 and traj_lon_accel > 0.3:
        violations += 1.0
    if coc_lon_accel > 0.3 and traj_lon_decel > 0.3:
        violations += 1.0
    if coc_lat_left > 0.3 and traj_lat_right > 0.3:
        violations += 1.0
    if coc_lat_right > 0.3 and traj_lat_left > 0.3:
        violations += 1.0

    return -min(0.20, violations * 0.08)


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
    # Layer 1: Grounded Scene Facts (no hard length gate)
    # ============================================================
    coc_scores = compute_coc_quality(to_be_evaluated, weights=w.get("coc_weights"))
    sections = extract_coc_sections(to_be_evaluated)

    _coc_text = str(sections["reasoning"]).strip()
    _word_count = len(_coc_text.split())

    print(f"\n[CoC-Debug] word_count={_word_count} factual={coc_scores.get('coc_factual',0):.3f} coherence={coc_scores.get('coc_coherence',0):.3f} safety={coc_scores.get('coc_safety',0):.3f} completeness={coc_scores.get('coc_completeness',0):.3f}\n{_coc_text[:600]}\n[CoC-Debug-End]", flush=True)

    consistency_penalty = _check_coc_traj_consistency(
        _coc_text, predicted_fut_xyz[0], predicted_fut_rot[0]
    )
    format_score = float(sections.get("format_score", 0.0))
    ego_future_rot = reference.get("ego_future_rot", None)
    gt_future_for_facts = gt_fut_xyz[0] if gt_fut_xyz.dim() == 3 else gt_fut_xyz
    gt_rot_for_facts = None
    if isinstance(ego_future_rot, torch.Tensor):
        gt_rot_for_facts = ego_future_rot[0] if ego_future_rot.dim() == 4 else ego_future_rot
    grounded_fact_score, grounded_info = _compute_grounded_fact_score(
        _coc_text,
        reference,
        gt_future_for_facts,
        gt_rot_for_facts,
    )

    s1_scene = 0.45 * coc_scores["coc_factual"] + 0.55 * grounded_fact_score

    # ============================================================
    # Layer 2: Causal Coherence (Reasoning Quality)
    # ============================================================
    s2_coc = coc_scores.get("coc_quality", 0.0)
    # Normalize to [-1, 1] range: 0→0.0, 1→1.0
    s2_coc_normalized = 2.0 * s2_coc - 1.0  # [0,1] → [-1,1]
    logger.info(
        f"[CoC-Detail] factual={coc_scores.get('coc_factual', 0):.3f} "
        f"coherence={coc_scores.get('coc_coherence', 0):.3f} "
        f"safety={coc_scores.get('coc_safety', 0):.3f} "
        f"completeness={coc_scores.get('coc_completeness', 0):.3f} "
        f"coc_quality={s2_coc:.3f} -> s2_norm={s2_coc_normalized:.3f}"
    )

    # ============================================================
    # Layer 3: Reasoning-Action Alignment
    # ============================================================
    raa_reward_val, raa_info = compute_raa_reward(
        to_be_evaluated, predicted_fut_xyz, predicted_fut_rot, weight=1.0
    )
    s3_raa = raa_reward_val  # Already in [0, 1], low for empty/generic intent.
    s3_raa_normalized = 2.0 * s3_raa - 1.0  # [0,1] → [-1,1]

    # ============================================================
    # Layer 4: Trajectory Quality (ADE + Comfort)
    # ============================================================
    l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])
    if not (isinstance(l2_dist, (int, float)) and math.isfinite(l2_dist)):
        logger.warning(f"[HCC-Reward] ADE returned non-finite={l2_dist}, using fallback 999.0")
        l2_dist = 999.0

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
    if not (isinstance(comfort_score, (int, float)) and math.isfinite(comfort_score)):
        logger.warning(f"[HCC-Reward] comfort_score non-finite={comfort_score}, using 0")
        comfort_score = 0.0
    comfort_score_norm = comfort_score - 1.0  # Center around 0

    # Continuous trajectory reward: exp(-ade/2) in [0, 1]
    s4_traj = float(__import__('math').exp(-l2_dist / 2.0))
    s4_comfort = comfort_score_norm

    tw = w.get("traj_l2_weight", 0.5)
    cw = w.get("comfort_weight", 0.1)
    tw_cw_sum = tw + cw
    if tw_cw_sum > 0:
        s4_combined = (tw * s4_traj + cw * s4_comfort) / tw_cw_sum
    else:
        s4_combined = s4_traj

    s4_combined = max(0.0, s4_combined)

    # Additive reward formula: all layers contribute directly
    coc_w = w["coc_quality_weight"]
    raa_w = w["raa_weight"]
    scene_w = w.get("scene_reward_weight", 0.3)
    traj_layer_w = max(0.0, 1.0 - scene_w - coc_w - raa_w)

    format_w = 0.05
    final_reward = (
        scene_w * (2.0 * s1_scene - 1.0)
        + coc_w * s2_coc_normalized
        + raa_w * s3_raa_normalized
        + traj_layer_w * s4_combined
        + format_w * (2.0 * format_score - 1.0)
        + consistency_penalty
    )

    if not (isinstance(final_reward, (int, float)) and math.isfinite(final_reward)):
        logger.warning(f"[HCC-Reward] final_reward non-finite={final_reward}, clamping to 0")
        final_reward = 0.0

    final_reward = float(max(-1.0, min(1.0, final_reward)))

    logger.warning(
        f"[HCC-RM] s1(scene)={s1_scene:.3f} s2(coc)={s2_coc_normalized:.3f} "
        f"s3(raa)={s3_raa_normalized:.3f} s4(traj)={s4_combined:.3f} "
        f"s4_l2={l2_dist:.3f} s4_comf={comfort_score:.3f} "
        f"facts={grounded_fact_score:.3f} fmt={format_score:.3f} "
        f"cons_pen={consistency_penalty:.3f} "
        f"R_final={final_reward:.4f}"
    )

    reward_dict = {
        "scene_understanding": float(s1_scene),
        "coc_quality": float(coc_scores.get("coc_quality", 0.0)),
        "coc_factual": float(coc_scores.get("coc_factual", 0.0)),
        "coc_coherence": float(coc_scores.get("coc_coherence", 0.0)),
        "coc_safety": float(coc_scores.get("coc_safety", 0.0)),
        "coc_completeness": float(coc_scores.get("coc_completeness", 0.0)),
        "format_score": float(format_score),
        "cot_word_count": float(_word_count),
        "raa_score": float(s3_raa),
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "reward": float(final_reward),
        "consistency_penalty": float(consistency_penalty),
        **grounded_info,
        **{
            k: float(v) if isinstance(v, (float, int)) else 0.0
            for k, v in raa_info.items()
            if not k.startswith("raa_score") and k != "raa_empty_coc"
        },
    }

    return final_reward, reward_dict
