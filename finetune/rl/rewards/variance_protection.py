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

"""Reward variance protection and rank-based advantage computation.

Addresses the core problem: when keyword-based rewards give similar scores
to all completions in a GRPO group, advantages collapse to ~0 and the
model receives no gradient signal.

Solutions:
  1. **Rank-based advantages**: Use rank within group instead of raw reward
  2. **Minimum variance floor**: If group std < threshold, amplify differences
  3. **Completion diversity bonus**: Reward unique completions
  4. **Skip-on-collapse**: Optionally skip gradient update when variance is too low
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_rank_advantages(
    rewards: list[float],
    group_indices: list[int] | None = None,
    *,
    tie_method: str = "average",
) -> list[float]:
    """Compute rank-based advantages within GRPO groups.

    Instead of using raw reward values (which may have collapsed variance),
    converts rewards to ranks within each group. This preserves relative
    ordering even when absolute rewards are similar.

    Args:
        rewards: Raw reward values for all samples.
        group_indices: Group ID for each sample. If None, treats all as one group.
        tie_method: How to handle ties: "average" (default), "min", "max".

    Returns:
        Rank-based advantages normalized to zero mean, unit variance.
    """
    n = len(rewards)
    if n == 0:
        return []

    if group_indices is None:
        group_indices = [0] * n

    # Group samples by their prompt group
    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(group_indices):
        groups.setdefault(gid, []).append(i)

    advantages = [0.0] * n

    for gid, indices in groups.items():
        if len(indices) < 2:
            # Single sample group — use raw reward as advantage
            advantages[indices[0]] = float(rewards[indices[0]])
            continue

        group_rewards = [rewards[i] for i in indices]

        # Compute ranks (1-based, higher reward = higher rank)
        sorted_indices = np.argsort(group_rewards)
        ranks = np.zeros(len(indices), dtype=np.float64)

        if tie_method == "average":
            # Average rank for ties
            reward_arr = np.array(group_rewards)
            for rank_pos, sort_idx in enumerate(sorted_indices):
                ranks[sort_idx] = rank_pos + 1
            # Average ranks for tied values
            unique_vals = np.unique(reward_arr)
            for val in unique_vals:
                tied_mask = reward_arr == val
                if tied_mask.sum() > 1:
                    tied_ranks = ranks[tied_mask]
                    ranks[tied_mask] = tied_ranks.mean()
        else:
            for rank_pos, sort_idx in enumerate(sorted_indices):
                ranks[sort_idx] = rank_pos + 1

        # Normalize to zero mean, unit variance within group
        rank_mean = ranks.mean()
        rank_std = ranks.std()
        if rank_std > 1e-6:
            normalized = (ranks - rank_mean) / rank_std
        else:
            # All same rank → zero advantage (no signal)
            normalized = np.zeros_like(ranks)

        for local_idx, global_idx in enumerate(indices):
            advantages[global_idx] = float(normalized[local_idx])

    return advantages


def amplify_low_variance_advantages(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    *,
    min_std: float = 0.1,
    amplify_factor: float = 2.0,
    group_indices: list[int] | None = None,
) -> torch.Tensor:
    """Amplify advantages when group variance is below threshold.

    When all completions in a GRPO group get similar rewards, the
    standard normalization produces near-zero advantages. This function
    detects such groups and amplifies the small differences.

    Args:
        advantages: Already-computed advantages (from rank or raw).
        rewards: Original raw rewards for variance checking.
        min_std: Minimum standard deviation threshold.
        amplify_factor: How much to amplify when std < min_std.
        group_indices: Group IDs for per-group checking.

    Returns:
        Potentially amplified advantages.
    """
    if group_indices is None:
        # Global check
        std = rewards.std().item()
        if std < min_std and std > 1e-8:
            ratio = min_std / std
            amplify = min(ratio, amplify_factor)
            return advantages * amplify
        return advantages

    # Per-group check
    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(group_indices):
        groups.setdefault(gid, []).append(i)

    result = advantages.clone()
    for gid, indices in groups.items():
        if len(indices) < 2:
            continue
        group_r = rewards[indices]
        group_std = group_r.std().item()
        if group_std < min_std and group_std > 1e-8:
            ratio = min(min_std / group_std, amplify_factor)
            for idx in indices:
                result[idx] *= ratio

    return result


def compute_completion_diversity(
    completions: list[str],
    group_indices: list[int] | None = None,
) -> list[float]:
    """Compute per-completion diversity scores within groups.

    Measures how unique each completion is compared to others in the
    same group using character n-gram Jaccard distance.

    Args:
        completions: List of completion texts.
        group_indices: Group ID for each completion.

    Returns:
        Per-completion diversity scores in [0, 1].
    """
    n = len(completions)
    if group_indices is None:
        group_indices = [0] * n

    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(group_indices):
        groups.setdefault(gid, []).append(i)

    diversity = [0.5] * n  # Default neutral

    for gid, indices in groups.items():
        if len(indices) < 2:
            continue

        # Compute n-gram sets for each completion
        ngram_sets = []
        for idx in indices:
            text = completions[idx].lower()
            ngrams = set()
            for n_size in (3, 4, 5):
                for i in range(len(text) - n_size + 1):
                    ngrams.add(text[i:i + n_size])
            ngram_sets.append(ngrams)

        # Compute average Jaccard distance to other completions
        for local_i, global_i in enumerate(indices):
            distances = []
            for local_j, global_j in enumerate(indices):
                if local_i == local_j:
                    continue
                set_i = ngram_sets[local_i]
                set_j = ngram_sets[local_j]
                if len(set_i | set_j) > 0:
                    jaccard_dist = 1.0 - len(set_i & set_j) / len(set_i | set_j)
                else:
                    jaccard_dist = 1.0
                distances.append(jaccard_dist)

            if distances:
                diversity[global_i] = float(np.mean(distances))

    return diversity


def should_skip_update(
    rewards: list[float],
    group_indices: list[int] | None = None,
    *,
    min_group_std: float = 0.01,
    min_mean_advantage_abs: float = 0.001,
) -> tuple[bool, dict[str, float]]:
    """Check whether the reward variance is too low for meaningful gradient updates.

    Args:
        rewards: Raw rewards for the batch.
        group_indices: Group IDs.
        min_group_std: Minimum acceptable per-group std.
        min_mean_advantage_abs: Minimum acceptable mean absolute advantage.

    Returns:
        Tuple of (should_skip, diagnostics).
    """
    if not rewards:
        return True, {"reason_code": 1.0, "mean_std": 0.0}

    if group_indices is None:
        group_indices = [0] * len(rewards)

    groups: dict[int, list[int]] = {}
    for i, gid in enumerate(group_indices):
        groups.setdefault(gid, []).append(i)

    stds = []
    for gid, indices in groups.items():
        if len(indices) < 2:
            continue
        group_r = np.array([rewards[i] for i in indices])
        stds.append(float(group_r.std()))

    if not stds:
        return True, {"reason_code": 2.0, "mean_std": 0.0}

    mean_std = float(np.mean(stds))
    min_std = float(np.min(stds))

    should_skip = mean_std < min_group_std
    diagnostics = {
        "mean_group_std": mean_std,
        "min_group_std": min_std,
        "num_groups": len(groups),
        "skip_update": float(should_skip),
    }

    return should_skip, diagnostics
