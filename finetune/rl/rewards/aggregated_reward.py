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


"""Aggregated reward computation for Alpamayo Cosmos-RL training.

Supports two reward modes:
  1. **HCC-RM** (default): Hierarchical Causal-Consistent Reward Model
     — 4-layer hierarchical reward with CoC quality, RAA, and trajectory scoring.
  2. **Legacy**: Original trajectory-only reward (ADE + comfort).

Mode is auto-detected from the TOML config: if ``coc_quality_weight`` and
``raa_weight`` are present, HCC-RM is used. Otherwise falls back to legacy.
"""

from __future__ import annotations

from typing import Any

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
]

_HCC_REWARD_KEYS: list[str] = [
    "coc_quality_weight",
    "raa_weight",
]

_OPTIONAL_REWARD_DEFAULTS: dict[str, float | bool] = {
    "enable_coc_reward": False,
    "coc_weight": 0.0,
    "enable_consistency_reward": False,
    "consistency_weight": 0.0,
    "enable_risk_reward": False,
    "risk_weight": 0.0,
}


def _get_reward_cfg(config: object | None) -> dict[str, float | bool]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward].

    Auto-detects HCC-RM vs legacy mode based on present keys.
    """
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "Reward config not found in TOML. "
            f"Required keys under [custom.alpamayo.reward]: {_REQUIRED_REWARD_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_REWARD_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(f"Missing key(s) in [custom.alpamayo.reward]: {missing}")

    cfg: dict[str, float | bool] = {k: float(reward_cfg[k]) for k in _REQUIRED_REWARD_KEYS}
    for key, default in _OPTIONAL_REWARD_DEFAULTS.items():
        value = reward_cfg.get(key, default)
        cfg[key] = bool(value) if isinstance(default, bool) else float(value)
    return cfg


def _is_hcc_enabled(config: object | None) -> bool:
    """Check whether HCC-RM mode is enabled via TOML config.

    HCC-RM is enabled when both ``coc_quality_weight`` and ``raa_weight``
    are present in the reward config AND ``enable_hcc`` is not explicitly
    set to false.
    """
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError):
        return False

    # Check for explicit disable flag
    explicit_enable = reward_cfg.get("enable_hcc", None)
    if explicit_enable is not None:
        return bool(explicit_enable)

    # Auto-detect: HCC keys present → enabled
    has_hcc_keys = all(k in reward_cfg for k in _HCC_REWARD_KEYS)
    return has_hcc_keys


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Compute the aggregated reward for a single rollout against reference data.

    Automatically selects HCC-RM or legacy reward based on TOML configuration.
    """
    if _is_hcc_enabled(config):
        from rl.rewards.hcc_reward import compute_hcc_reward

        return compute_hcc_reward(
            to_be_evaluated,
            reference,
            tokenizer=tokenizer,
            traj_tokenizer=traj_tokenizer,
            config=config,
            model_config=model_config,
        )

    # ---- Legacy reward path (trajectory-only ADE + comfort) ----
    from rl.rewards.consistency_reward import compute_reasoning_action_consistency
    from rl.rewards.coc_reward import compute_coc_reward
    from rl.rewards.comfort_reward import compute_comfort
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    from rl.rewards.risk_reward import compute_risk_reward
    from rl.rewards.traj_reward import calculate_ade
    from rl.utils.trajectory_decode import decode_rollout_trajectory

    w = _get_reward_cfg(config)

    gt_fut_xyz = reference["ego_future_xyz"]
    predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
        to_be_evaluated,
        reference["ego_history_xyz"],
        reference["ego_history_rot"],
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        model_config=model_config,
    )

    l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
    comfort_score = comfort_score - 1.0

    # Gated reward: ADE must be below threshold, otherwise fixed penalty.
    ade_threshold = 3.0
    if l2_dist < ade_threshold:
        final_reward = (
            -w["traj_l2_weight"] * (l2_dist / ade_threshold) + w["comfort_weight"] * comfort_score
        )
    else:
        final_reward = -1.0
    reward_dict: dict[str, float] = {
        "traj_L2": float(l2_dist),
        "traj_reward": float(-(l2_dist / ade_threshold) if l2_dist < ade_threshold else -1.0),
        "comfort_reward": float(comfort_score),
        "reward": float(final_reward),
        "reward_type": "legacy",
    }

    if bool(w["enable_coc_reward"]):
        coc = compute_coc_reward(to_be_evaluated, reference)
        final_reward += float(w["coc_weight"]) * coc.reward
        reward_dict.update(coc.to_dict(prefix="coc"))

    if bool(w["enable_consistency_reward"]):
        consistency = compute_reasoning_action_consistency(
            to_be_evaluated,
            predicted_fut_xyz,
            reference,
        )
        final_reward += float(w["consistency_weight"]) * consistency.reward
        reward_dict.update(consistency.to_dict(prefix="consistency"))

    if bool(w["enable_risk_reward"]):
        risk = compute_risk_reward(predicted_fut_xyz, reference)
        final_reward += float(w["risk_weight"]) * risk.reward
        reward_dict.update(risk.to_dict(prefix="risk"))

    reward_dict["reward"] = float(final_reward)
    logger.debug(f"[compute_reward] Final reward (legacy): {final_reward}")

    return reward_dict["reward"], reward_dict
