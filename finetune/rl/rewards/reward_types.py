# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared reward dataclasses for Alpamayo RL rewards."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardComponents:
    """Structured reward components returned by Alpamayo reward helpers."""

    reward: float
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self, prefix: str | None = None) -> dict[str, float]:
        """Return metrics with an optional prefix and include the component reward."""
        if prefix:
            out = {f"{prefix}_reward": float(self.reward)}
            out.update({f"{prefix}_{k}": float(v) for k, v in self.metrics.items()})
            return out
        out = {"reward": float(self.reward)}
        out.update({k: float(v) for k, v in self.metrics.items()})
        return out
