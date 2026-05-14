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

"""Adaptive KL divergence scheduler for PGMO-GRPO.

Dynamically adjusts the KL penalty coefficient based on reasoning-action
consistency metrics. When consistency is high (policy is trustworthy),
KL penalty is reduced to encourage exploration. When consistency drops,
KL penalty is increased to stabilize training.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class AdaptiveKLScheduler:
    """Dynamically adjusts kl_beta based on reasoning-action alignment history.

    The schedule follows:
        β_t = β_base · exp(-λ · s̄_align,t)

    where s̄_align,t is the average RAA score over the recent history window,
    and λ controls the decay rate.

    When RAA is high → β decreases → more exploration.
    When RAA is low  → β increases → more conservative updates.
    """

    def __init__(
        self,
        base_beta: float = 0.04,
        lambda_decay: float = 2.0,
        min_beta: float = 0.001,
        max_beta: float = 0.2,
        window_size: int = 100,
        warmup_steps: int = 50,
        smoothing: float = 0.9,
    ) -> None:
        """Initialize the adaptive KL scheduler.

        Args:
            base_beta: Baseline KL coefficient (when RAA ≈ 0.5).
            lambda_decay: Decay rate for the exponential schedule.
                Higher values make β more sensitive to RAA changes.
            min_beta: Minimum allowed KL coefficient.
            max_beta: Maximum allowed KL coefficient.
            window_size: Number of recent RAA scores to track.
            warmup_steps: Steps to use fixed base_beta before adaptation.
            smoothing: EMA smoothing factor for RAA trend.
        """
        self.base_beta = base_beta
        self.lambda_decay = lambda_decay
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.smoothing = smoothing

        self._raa_history: Deque[float] = deque(maxlen=window_size)
        self._beta_history: Deque[float] = deque(maxlen=window_size)
        self._ema_raa: float | None = None
        self._step_count: int = 0
        self._current_beta: float = base_beta

    def update(self, raa_score: float) -> float:
        """Update the scheduler with a new RAA score and return new beta.

        Args:
            raa_score: The current step's average Reasoning-Action Alignment score.

        Returns:
            Updated kl_beta value.
        """
        self._step_count += 1
        self._raa_history.append(raa_score)

        # Warmup: use fixed base_beta
        if self._step_count <= self.warmup_steps:
            self._current_beta = self.base_beta
            self._beta_history.append(self._current_beta)
            return self._current_beta

        # Compute smoothed RAA
        if self._ema_raa is None:
            self._ema_raa = raa_score
        else:
            self._ema_raa = (
                self.smoothing * self._ema_raa + (1.0 - self.smoothing) * raa_score
            )

        # Compute window average for more stable signal
        if len(self._raa_history) >= 10:
            window_avg = np.mean(list(self._raa_history)[-min(50, len(self._raa_history)):])
        else:
            window_avg = raa_score

        # Blend EMA and window average
        smoothed_raa = 0.5 * self._ema_raa + 0.5 * window_avg

        # Compute adaptive beta
        # High RAA (close to 1) → low beta → more exploration
        # Low RAA (close to 0) → high beta → more conservative
        self._current_beta = self.base_beta * np.exp(
            -self.lambda_decay * (smoothed_raa - 0.5)
        )

        # Clamp to allowed range
        self._current_beta = float(
            np.clip(self._current_beta, self.min_beta, self.max_beta)
        )

        self._beta_history.append(self._current_beta)
        return self._current_beta

    def get_current_beta(self) -> float:
        """Get the current kl_beta value."""
        return self._current_beta

    def get_stats(self) -> dict[str, float]:
        """Get statistics about the scheduler state."""
        recent_raa = (
            float(np.mean(list(self._raa_history)[-50:]))
            if len(self._raa_history) >= 10
            else 0.5
        )
        return {
            "adaptive_kl_beta": self._current_beta,
            "adaptive_kl_raa_ema": float(self._ema_raa or 0.5),
            "adaptive_kl_raa_mean": recent_raa,
            "adaptive_kl_step": float(self._step_count),
        }

    def reset(self) -> None:
        """Reset the scheduler state."""
        self._raa_history.clear()
        self._beta_history.clear()
        self._ema_raa = None
        self._step_count = 0
        self._current_beta = self.base_beta
