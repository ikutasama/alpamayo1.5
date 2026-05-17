# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning-action consistency rewards for Alpamayo RL."""

from __future__ import annotations

from typing import Any

import torch

from rl.rewards.coc_reward import extract_coc_sections
from rl.rewards.reward_types import RewardComponents


def _future_displacement(predicted_fut_xyz: torch.Tensor) -> tuple[float, float, float]:
    """Return longitudinal/lateral displacement and mean step length."""
    fut = predicted_fut_xyz
    while fut.ndim > 3:
        fut = fut[0]
    if fut.ndim == 3:
        fut = fut[0]
    if fut.shape[0] < 2:
        return 0.0, 0.0, 0.0
    delta = fut[-1, :2] - fut[0, :2]
    steps = torch.linalg.norm(fut[1:, :2] - fut[:-1, :2], dim=-1)
    return float(delta[0]), float(delta[1]), float(steps.mean())


def compute_reasoning_action_consistency(
    to_be_evaluated: str,
    predicted_fut_xyz: torch.Tensor,
    reference: dict[str, Any] | None = None,
) -> RewardComponents:
    """Score coarse agreement between textual intent and predicted trajectory."""
    del reference
    reasoning = str(extract_coc_sections(to_be_evaluated)["reasoning"]).lower()
    dx, dy, mean_step = _future_displacement(predicted_fut_xyz)

    checks: list[float] = []
    if "left" in reasoning:
        checks.append(float(dy > 0.05))
    if "right" in reasoning:
        checks.append(float(dy < -0.05))
    if any(w in reasoning for w in ("straight", "keep", "maintain")):
        checks.append(float(abs(dy) < max(0.5, abs(dx) * 0.35 + 1e-6)))
    if any(w in reasoning for w in ("slow", "decelerate", "brake", "stop", "yield")):
        checks.append(float(mean_step < 1.5))
    if "accelerate" in reasoning:
        checks.append(float(mean_step > 0.2))

    if not checks:
        reward = 0.0
        coverage = 0.0
    else:
        reward = sum(checks) / len(checks)
        coverage = 1.0

    return RewardComponents(
        reward=float(reward),
        metrics={
            "coverage": float(coverage),
            "num_checks": float(len(checks)),
            "dx": float(dx),
            "dy": float(dy),
            "mean_step": float(mean_step),
        },
    )
