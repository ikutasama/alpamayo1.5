# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight risk proxy rewards for Alpamayo RL."""

from __future__ import annotations

from typing import Any

import torch

from rl.rewards.reward_types import RewardComponents


def compute_risk_reward(
    predicted_fut_xyz: torch.Tensor,
    reference: dict[str, Any] | None = None,
) -> RewardComponents:
    """Compute simple trajectory validity proxies.

    This first version intentionally stays conservative: it only penalizes NaN/Inf
    and extreme jumps. Richer lane/obstacle rewards can be added once the exact
    tensor layouts are confirmed on the server dataset.
    """
    del reference
    valid = torch.isfinite(predicted_fut_xyz).all()
    reward = 0.0
    max_step = 0.0
    if not bool(valid):
        reward = -1.0
    else:
        fut = predicted_fut_xyz
        while fut.ndim > 3:
            fut = fut[0]
        if fut.ndim == 3:
            fut = fut[0]
        if fut.shape[0] > 1:
            steps = torch.linalg.norm(fut[1:, :2] - fut[:-1, :2], dim=-1)
            max_step = float(steps.max())
            if max_step > 15.0:
                reward = -0.5

    return RewardComponents(
        reward=float(reward),
        metrics={
            "valid": float(bool(valid)),
            "max_step": float(max_step),
        },
    )
