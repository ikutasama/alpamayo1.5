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

"""Contrastive Reasoning-Action Learning (CRAL) loss for PGMO-GRPO.

Encourages the model to align reasoning latent representations with
trajectory quality, learning that good reasoning should produce good
trajectories. This is the contrastive component of PGMO-GRPO.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_contrastive_loss(
    hidden_states: torch.Tensor,           # [B, D] — pooled last-layer hidden states
    coc_quality_scores: torch.Tensor,       # [B] — CoC quality per sample
    raa_scores: torch.Tensor,               # [B] — Reasoning-Action Alignment per sample
    traj_scores: torch.Tensor,              # [B] — Trajectory quality per sample
    temperature: float = 0.07,
    weight: float = 0.05,
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute contrastive loss between reasoning-aligned and misaligned samples.

    Positive pairs: samples where CoC quality AND RAA score are both high.
    Negative pairs: samples where CoC quality is high BUT RAA is low
                    (i.e., good reasoning → bad trajectory mismatch).

    Args:
        hidden_states: Pooled hidden representations, shape [B, D].
        coc_quality_scores: CoC quality scores, shape [B].
        raa_scores: Reasoning-action alignment scores, shape [B].
        traj_scores: Trajectory quality scores, shape [B].
        temperature: Softmax temperature.
        weight: Loss weight coefficient.
        margin: Margin for distinguishing positive vs negative.

    Returns:
        Scalar contrastive loss value, or 0.0 if insufficient samples.
    """
    B = hidden_states.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=hidden_states.device)

    # Normalize hidden states
    hidden_states = F.normalize(hidden_states, p=2, dim=-1)

    # Compute combined quality score
    combined_quality = (coc_quality_scores + raa_scores) / 2.0

    # Identify positive and negative samples
    # Positive: top 50% by combined quality
    quality_median = combined_quality.median()
    pos_mask = combined_quality >= quality_median
    neg_mask = ~pos_mask

    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()

    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.0, device=hidden_states.device)

    # Build contrastive pairs: each positive vs all negatives
    pos_h = hidden_states[pos_mask]  # [n_pos, D]
    neg_h = hidden_states[neg_mask]  # [n_neg, D]

    # Compute similarity matrix
    # pos_sim: [n_pos, n_pos] — similarities among positives (should be high)
    # neg_sim: [n_pos, n_neg] — similarities with negatives (should be low)
    pos_sim = torch.matmul(pos_h, pos_h.T) / temperature   # [n_pos, n_pos]
    neg_sim = torch.matmul(pos_h, neg_h.T) / temperature   # [n_pos, n_neg]

    # InfoNCE-style loss: each positive sample is the anchor
    # L = -log( sum(exp(pos_sim)) / (sum(exp(pos_sim)) + sum(exp(neg_sim))) )
    total_loss = torch.tensor(0.0, device=hidden_states.device)
    n_valid = 0

    for i in range(n_pos):
        # Exclude self-similarity
        pos_scores = torch.cat([pos_sim[i, :i], pos_sim[i, i+1:]])  # [n_pos-1]
        neg_scores = neg_sim[i]                                      # [n_neg]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue

        numerator = torch.exp(pos_scores).sum()
        denominator = numerator + torch.exp(neg_scores).sum()

        if denominator > 0:
            total_loss -= torch.log(numerator / denominator + 1e-8)
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=hidden_states.device)

    return weight * (total_loss / n_valid)


def compute_contrastive_loss_simple(
    hidden_states: torch.Tensor,
    rewards: torch.Tensor,
    temperature: float = 0.1,
    weight: float = 0.05,
) -> torch.Tensor:
    """Simplified contrastive loss using reward ranking.

    Splits samples into high-reward (top 50%) and low-reward (bottom 50%)
    and pushes their representations apart while pulling high-reward
    representations together.

    Args:
        hidden_states: Pooled hidden representations, shape [B, D].
        rewards: Reward values, shape [B].
        temperature: Softmax temperature.
        weight: Loss weight coefficient.

    Returns:
        Scalar contrastive loss.
    """
    B = hidden_states.shape[0]
    if B < 4:
        return torch.tensor(0.0, device=hidden_states.device)

    hidden_states = F.normalize(hidden_states, p=2, dim=-1)

    # Sort by reward
    sorted_indices = torch.argsort(rewards)
    n_high = B // 2

    high_idx = sorted_indices[-n_high:]    # Top 50%
    low_idx = sorted_indices[:n_high]       # Bottom 50%

    high_h = hidden_states[high_idx]  # [n_high, D]
    low_h = hidden_states[low_idx]    # [n_high, D]

    # Similarity matrix
    sim_matrix = torch.matmul(high_h, high_h.T) / temperature  # [n_high, n_high]

    # Self-similarity diagonal
    diag_mask = torch.eye(n_high, device=hidden_states.device, dtype=torch.bool)

    # InfoNCE: each sample vs all other high-reward samples
    pos_sim = sim_matrix[~diag_mask].view(n_high, n_high - 1)
    high_low_sim = torch.matmul(high_h, low_h.T) / temperature

    total_loss = torch.tensor(0.0, device=hidden_states.device)
    for i in range(n_high):
        num = torch.exp(pos_sim[i]).sum()
        denom = num + torch.exp(high_low_sim[i]).sum()
        total_loss -= torch.log(num / (denom + 1e-8) + 1e-8)

    return weight * (total_loss / n_high)
