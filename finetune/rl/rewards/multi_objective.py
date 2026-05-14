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

"""Multi-objective reward decomposition and Pareto-optimal weighting for PGMO-GRPO.

Decomposes the scalar reward into multiple objective dimensions and
identifies Pareto-optimal samples within GRPO's group of generations.
"""

from __future__ import annotations

from typing import Any

import torch


def identify_pareto_optimal(
    objectives: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Identify Pareto-optimal samples from a set of multi-objective scores.

    A sample i is Pareto-optimal if no other sample j dominates it.
    Sample j dominates i if:
      - objectives[j, k] >= objectives[i, k] for all k
      - objectives[j, k] > objectives[i, k] for at least one k

    Args:
        objectives: Tensor of shape [N, M] where N is number of samples
            and M is number of objectives. Higher values are better.
        epsilon: Numerical tolerance for dominance comparison.

    Returns:
        Boolean tensor of shape [N] where True indicates Pareto-optimal.
    """
    N, M = objectives.shape
    is_pareto = torch.ones(N, dtype=torch.bool, device=objectives.device)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Check if j dominates i
            j_better_or_equal = objectives[j] >= (objectives[i] - epsilon)
            j_strictly_better = objectives[j] > (objectives[i] + epsilon)
            if j_better_or_equal.all() and j_strictly_better.any():
                is_pareto[i] = False
                break

    return is_pareto


def compute_pareto_weights(
    objectives: torch.Tensor,
    base_weight: float = 1.5,
    use_distance_weighting: bool = True,
) -> torch.Tensor:
    """Compute Pareto-optimality-based sample weights.

    Pareto-optimal samples receive higher weights (up to base_weight * 2).
    Non-Pareto samples receive weight 1.0.

    With distance weighting enabled, Pareto samples closer to the
    ideal point (maximum of each objective) receive additional weight.

    Args:
        objectives: Tensor of shape [N, M] — multi-objective scores.
        base_weight: Base weight multiplier for Pareto-optimal samples.
        use_distance_weighting: Whether to additionally weight by distance
            to the ideal point.

    Returns:
        Weight tensor of shape [N].
    """
    N = objectives.shape[0]
    if N <= 1:
        return torch.ones(N, device=objectives.device)

    # Identify Pareto frontier
    is_pareto = identify_pareto_optimal(objectives)

    # Base weights
    weights = torch.where(is_pareto, base_weight, 1.0).float()

    # Distance-based weighting within Pareto frontier
    if use_distance_weighting and is_pareto.sum() > 1:
        ideal_point = objectives.max(dim=0).values  # [M]
        nadir_point = objectives.min(dim=0).values  # [M]
        point_range = ideal_point - nadir_point
        point_range = torch.where(point_range < 1e-6, 1.0, point_range)

        # Normalized distance to ideal point
        distances = torch.norm(
            (ideal_point - objectives) / point_range, dim=1
        )  # [N]

        # Pareto samples closer to ideal get higher weight
        pareto_indices = is_pareto.nonzero(as_tuple=True)[0]
        pareto_distances = distances[pareto_indices]

        if pareto_distances.max() > 1e-6:
            pareto_dist_weights = 1.0 / (pareto_distances + 1e-6)
            pareto_dist_weights = pareto_dist_weights / pareto_dist_weights.sum()
            # Boost: base_weight + (1 - normalized weight) * base_weight
            # So the closest to ideal gets up to 2*base_weight
            weights[pareto_indices] = base_weight * (
                1.0 + (1.0 - pareto_dist_weights / pareto_dist_weights.max())
            )

    return weights


def decompose_reward_dict(
    reward_info: dict[str, float],
    *,
    objective_keys: list[str] | None = None,
) -> torch.Tensor:
    """Decompose a reward info dict into a multi-objective tensor.

    Args:
        reward_info: Dict from reward computation containing per-dimension scores.
        objective_keys: List of keys to extract as objectives.
            Default: ["coc_quality", "raa_score", "traj_L2", "comfort_reward"].

    Returns:
        Tensor of shape [M] where M = len(objective_keys).
        For minimization objectives (traj_L2), values are negated so
        higher is always better.
    """
    if objective_keys is None:
        objective_keys = ["coc_quality", "raa_score", "traj_L2", "comfort_reward"]

    values = []
    for key in objective_keys:
        val = reward_info.get(key, 0.0)
        # Negate minimization objectives
        if key in ("traj_L2",):
            val = -float(val)
        values.append(float(val))

    return torch.tensor(values)


def compute_multi_objective_advantages(
    objectives_batch: torch.Tensor,  # [N, M]
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-dimension advantages from a batch of multi-objective scores.

    For each objective dimension independently, computes Z-score normalized
    advantages. This preserves the multi-dimensional structure of the reward.

    Args:
        objectives_batch: Tensor of shape [N, M] — scores for N samples
            across M objectives.
        normalize: Whether to Z-score normalize within each dimension.

    Returns:
        Advantage tensor of shape [N, M].
    """
    if normalize and objectives_batch.shape[0] > 1:
        mean = objectives_batch.mean(dim=0, keepdim=True)
        std = objectives_batch.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-6, 1.0, std)
        advantages = (objectives_batch - mean) / std
    else:
        advantages = objectives_batch

    return advantages


def compute_weighted_scalar_advantage(
    advantages: torch.Tensor,  # [N, M]
    weights: torch.Tensor,      # [M] — per-dimension importance weights
    pareto_weights: torch.Tensor | None = None,  # [N] — per-sample Pareto weights
) -> torch.Tensor:
    """Combine multi-dimensional advantages into a weighted scalar advantage.

    Args:
        advantages: Per-dimension advantages, shape [N, M].
        weights: Per-dimension importance weights, shape [M].
        pareto_weights: Optional per-sample Pareto weights, shape [N].
            If provided, multiplies element-wise after aggregation.

    Returns:
        Scalar advantage tensor of shape [N].
    """
    # Weighted sum across objectives
    scalar_adv = (advantages * weights.unsqueeze(0)).sum(dim=1)  # [N]

    # Apply Pareto weighting if provided
    if pareto_weights is not None:
        scalar_adv = scalar_adv * pareto_weights

    return scalar_adv
