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

"""Group Sequence Policy Optimization (GSPO) loss computation.

Implements the sequence-level importance ratio and clipping mechanism
from the GSPO paper (Zheng et al., 2025, arXiv:2507.18071).

Key differences from GRPO:
  1. Sequence-level importance ratio instead of token-level:
     w_i = π_θ(y_i|x) / π_old(y_i|x) = exp(Σ_t log π_θ - Σ_t log π_old)
  2. Sequence-level clipping (not per-token)
  3. Normalization only over unclipped samples
  4. No per-token averaging in the objective

Reference: "Group Sequence Policy Optimization" by Zheng et al. (Qwen Team, Alibaba)
Used to train Qwen3 models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_gspo_loss(
    current_per_token_logps: torch.Tensor,  # [N_tokens] — log π_θ per token
    old_per_token_logps: torch.Tensor,       # [N_tokens] — log π_old per token
    ref_per_token_logps: torch.Tensor | None,  # [N_tokens] — log π_ref per token
    advantages: torch.Tensor,                 # [N_tokens] — per-token advantages
    cu_seqlens: torch.Tensor,                 # [B+1] — cumulative sequence lengths
    config: object,
    logprob_masks: torch.Tensor,              # [B, L] — valid token mask
    *,
    dp_group=None,
    ddp_comm=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GSPO loss with sequence-level importance ratios.

    GSPO Objective:
      L = -1/G * Σ_i w_i * A_i / N_unclipped

    where:
      w_i = exp(Σ_t log π_θ(y_{i,t}|x, y_{i,<t}) - Σ_t log π_old(y_{i,t}|x, y_{i,<t}))
          = sequence-level importance ratio
      A_i = normalized group advantage (same as GRPO)
      Clipping: w_i is clipped to [1-ε, 1+ε] at sequence level
      N_unclipped = number of samples where clipping did not apply

    Args:
        current_per_token_logps: Log probabilities under current policy, shape [N_tokens]
        old_per_token_logps: Log probabilities under old policy, shape [N_tokens]
        ref_per_token_logps: Log probabilities under reference policy, shape [N_tokens] or None
        advantages: Per-token advantages, shape [N_tokens]
        cu_seqlens: Cumulative sequence lengths, shape [B+1]
        config: Cosmos-RL training config
        logprob_masks: Valid token mask, shape [B, L]
        dp_group: Data-parallel process group for all-reduce
        ddp_comm: DDP communicator

    Returns:
        Tuple of (loss, per_token_loss, kl_loss)
    """
    # Extract config parameters
    epsilon_low = getattr(config.train.train_policy, "epsilon_low", 0.2)
    epsilon_high = getattr(config.train.train_policy, "epsilon_high", 0.28)
    kl_beta = getattr(config.train.train_policy, "kl_beta", 0.0)

    B = cu_seqlens.shape[0] - 1  # Number of sequences

    if B == 0:
        return (
            torch.tensor(0.0, device=current_per_token_logps.device),
            torch.tensor(0.0, device=current_per_token_logps.device),
            torch.tensor(0.0, device=current_per_token_logps.device),
        )

    # ---- Step 1: Compute sequence-level log-probabilities ----
    # Sum per-token log-probs within each sequence
    seq_current_logps = torch.zeros(B, device=current_per_token_logps.device)
    seq_old_logps = torch.zeros(B, device=current_per_token_logps.device)
    seq_advantages = torch.zeros(B, device=current_per_token_logps.device)
    seq_lengths = torch.zeros(B, device=current_per_token_logps.device)

    for i in range(B):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_current_logps[i] = current_per_token_logps[start:end].sum()
        seq_old_logps[i] = old_per_token_logps[start:end].sum()
        # Advantage per sequence: take the mean (or first) since GRPO assigns
        # the same advantage to all tokens in a sequence
        seq_advantages[i] = advantages[start:end].mean()
        seq_lengths[i] = end - start

    # ---- Step 2: Compute sequence-level importance ratios ----
    # w_i = exp(Σ_t log π_θ - Σ_t log π_old) = π_θ(y_i) / π_old(y_i)
    log_ratio = seq_current_logps - seq_old_logps  # [B]
    ratio = torch.exp(log_ratio)  # [B] — sequence-level importance ratio

    # ---- Step 3: Sequence-level clipping ----
    # GSPO clips the importance ratio at the sequence level
    # This is the KEY difference from GRPO which clips per-token
    ratio_clipped = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

    # ---- Step 4: Compute sequence-level GSPO loss ----
    # Unclipped loss: -ratio * advantage
    loss_unclipped = -ratio * seq_advantages  # [B]

    # Clipped loss: -clipped_ratio * advantage
    loss_clipped = -ratio_clipped * seq_advantages  # [B]

    # Take the maximum (more conservative = more negative = larger loss)
    # This is equivalent to min(w*A, clip(w)*A) when A is positive,
    # and max(w*A, clip(w)*A) when A is negative
    # GSPO uses min() like PPO/GRPO
    is_positive_adv = seq_advantages > 0
    gspo_per_seq = torch.where(
        is_positive_adv,
        torch.max(loss_unclipped, loss_clipped),   # More conservative for positive adv
        torch.min(loss_unclipped, loss_clipped),   # Less conservative for negative adv
    )  # [B]

    # ---- Step 5: Identify unclipped samples for normalization ----
    # In GSPO, loss is normalized only over unclipped samples
    is_unclipped = (ratio >= 1.0 - epsilon_low) & (ratio <= 1.0 + epsilon_high)

    # Normalize: average per-sequence loss, weighted by sequence length
    n_unclipped = is_unclipped.sum().float()
    if n_unclipped > 0:
        # Normalize by number of unclipped sequences (per paper)
        gspo_loss = gspo_per_seq.sum() / n_unclipped
    else:
        # Fallback: use all sequences
        gspo_loss = gspo_per_seq.mean()

    # ---- Step 6: KL divergence penalty ----
    kl_loss = torch.tensor(0.0, device=current_per_token_logps.device)
    if kl_beta > 0 and ref_per_token_logps is not None:
        # Sequence-level KL: Σ_t KL(π_θ || π_ref)
        for i in range(B):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            kl_per_seq = (
                current_per_token_logps[start:end]
                - ref_per_token_logps[start:end]
            ).sum()
            kl_loss = kl_loss + kl_per_seq
        kl_loss = kl_beta * (kl_loss / B)

    # ---- Step 7: Per-token loss (for logging) ----
    # Distribute the sequence-level loss evenly across tokens
    total_tokens = seq_lengths.sum()
    if total_tokens > 0:
        per_token_loss = gspo_per_seq.sum() / total_tokens
    else:
        per_token_loss = gspo_loss

    total_loss = gspo_loss + kl_loss

    # Distributed reduction if needed
    if dp_group is not None:
        world_size = dp_group.size() if hasattr(dp_group, "size") else 1
        if world_size > 1:
            torch.distributed.all_reduce(total_loss, group=dp_group)
            total_loss = total_loss / world_size
            torch.distributed.all_reduce(per_token_loss, group=dp_group)
            per_token_loss = per_token_loss / world_size
            torch.distributed.all_reduce(kl_loss, group=dp_group)
            kl_loss = kl_loss / world_size

    return total_loss, per_token_loss, kl_loss


def compute_gspo_sequence_ratios(
    current_per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level importance ratios for monitoring.

    Args:
        current_per_token_logps: Current policy log-probs, shape [N_tokens].
        old_per_token_logps: Old policy log-probs, shape [N_tokens].
        cu_seqlens: Cumulative sequence lengths, shape [B+1].

    Returns:
        Tuple of (ratios, log_ratios) both shape [B].
    """
    B = cu_seqlens.shape[0] - 1
    ratios = torch.zeros(B, device=current_per_token_logps.device)
    log_ratios = torch.zeros(B, device=current_per_token_logps.device)

    for i in range(B):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_current = current_per_token_logps[start:end].sum()
        seq_old = old_per_token_logps[start:end].sum()
        log_ratios[i] = seq_current - seq_old
        ratios[i] = torch.exp(log_ratios[i])

    return ratios, log_ratios
