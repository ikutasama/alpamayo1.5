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

import os
import logging
from typing import Any, Dict, List

import cosmos_rl.utils.distributed as dist_util
import numpy as np
import torch
from cosmos_rl.dispatcher.replica import Rollout
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import compute_loss
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses
from cosmos_rl.utils.util import compute_mfu, is_master_rank
from torch.utils.tensorboard import SummaryWriter

from rl.base_trainer import AlpamayoGRPOTrainer

for _logger_name in ["cosmos", "cosmos_rl", "vllm", "transformers"]:
    _l = logging.getLogger(_logger_name)
    _l.setLevel(logging.WARNING)
    for _h in _l.handlers:
        _h.setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

_TB_WRITER = None


def _get_tb_writer(log_dir: str) -> SummaryWriter:
    global _TB_WRITER
    if _TB_WRITER is None:
        _TB_WRITER = SummaryWriter(log_dir=log_dir)
    return _TB_WRITER


def _get_advantage_routing_cfg(config: Any) -> dict[str, Any]:
    """Return optional token-level advantage routing config."""
    try:
        return getattr(config, "custom")["alpamayo"].get("advantage_routing", {})
    except (TypeError, KeyError, AttributeError):
        return {}


def _normalize_grouped(
    values: list[float],
    payloads: list[Any],
    fallback: list[float],
    n_generation: int = 8,
) -> list[float]:
    """Normalize component rewards within each prompt group, with safe fallback.

    Groups are formed by position: completions 0..n_generation-1 belong to
    prompt 0, n_generation..2*n_generation-1 belong to prompt 1, etc.
    This is robust because cosmos-rl always expands completions in order.
    """
    groups: dict[int, list[int]] = {}
    for idx in range(len(values)):
        group_key = idx // n_generation
        groups.setdefault(group_key, []).append(idx)

    out = list(fallback)
    eps = 1e-6
    for indices in groups.values():
        if len(indices) < 2:
            continue
        vals = np.array([values[i] for i in indices], dtype=np.float32)
        std = float(vals.std())
        if std <= eps:
            continue
        mean = float(vals.mean())
        for i in indices:
            out[i] = float((values[i] - mean) / std)
    return out


def _rollout_train_diagnostics(
    completions: list[str],
    rewards: list[float],
    reward_infos: list[Any],
) -> dict[str, float]:
    """Build raw reward and diversity diagnostics before advantage normalization."""
    report: dict[str, float] = {}
    if rewards:
        rewards_np = np.array([float(r) for r in rewards], dtype=np.float32)
        report["train/raw_reward_mean"] = float(rewards_np.mean())
        report["train/raw_reward_std"] = float(rewards_np.std())
        report["train/raw_reward_min"] = float(rewards_np.min())
        report["train/raw_reward_max"] = float(rewards_np.max())

    if completions:
        normalized = [" ".join(str(c).split()) for c in completions]
        report["train/unique_completion_ratio"] = len(set(normalized)) / max(1, len(normalized))
        cot_word_counts = []
        for text in completions:
            cot = str(text).split("<|cot_end|>", maxsplit=1)[0]
            cot_word_counts.append(len(cot.split()))
        report["train/cot_word_count_mean"] = float(np.mean(cot_word_counts))
        report["train/cot_word_count_min"] = float(np.min(cot_word_counts))

    metric_keys = [
        "scene_understanding",
        "grounded_fact_score",
        "grounded_fact_coverage",
        "grounded_fact_contradictions",
        "coc_quality",
        "raa_score",
        "raa_active_constraints",
        "traj_L2",
        "format_score",
        "consistency_penalty",
        # HCC-v2 grounded metrics
        "decision_alignment",
        "object_recall",
        "grounded_coc_reward",
        "hallucination_score",
        "spatial_accuracy",
        "threat_score",
        "num_high_threat_objects",
        "closest_obstacle_distance",
        "coc_unique_ratio",
        "diversity_score",
        "gt_decision_alignment",
        "correct_object_mentions",
        "false_object_mentions",
        "traj_quality",
        "cot_word_count",
    ]
    infos = [ri for ri in reward_infos if isinstance(ri, dict) and ri]
    for key in metric_keys:
        vals = [float(ri[key]) for ri in infos if key in ri]
        if vals:
            report[f"train/reward_{key}_mean"] = float(np.mean(vals))
            report[f"train/reward_{key}_std"] = float(np.std(vals))
    return report


@TrainerRegistry.register(trainer_type="reasoning_vla_grpo")
class ReasoningVLAGRPOTrainer(AlpamayoGRPOTrainer):
    """GRPO trainer for reasoning VLA models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tb_writer = None

    def _normalize_advantages(self, advantages_t: torch.Tensor, step: int, payloads: list[Any] = None) -> torch.Tensor:
        cfg = self._get_advantage_normalization_cfg()
        if not cfg.get("enable", True):
            return advantages_t

        nan_mask = torch.isnan(advantages_t) | torch.isinf(advantages_t)
        if nan_mask.any():
            n_nan = nan_mask.sum().item()
            logger.warning(
                f"[AdvNorm] step={step}: {n_nan}/{advantages_t.numel()} advantages are NaN/inf, replacing with 0"
            )
            advantages_t = advantages_t.clone()
            advantages_t[nan_mask] = 0.0

        vp_cfg = self._get_variance_protection_cfg()

        # === RANK-BASED ADVANTAGES (per-prompt group) ===
        # When enable_rank_advantages=True, use rank ordering instead of z-score.
        # This preserves relative ordering even when reward variance collapses,
        # and prevents advantage ≈ 0 that kills gradient updates.
        # Ranks are computed WITHIN each prompt group (same prompt → n_generation completions),
        # not globally, to match GRPO's per-group advantage semantics.
        if vp_cfg.get("enable_rank_advantages", False):
            rank_scale = vp_cfg.get("rank_scale", 1.0)
            n = advantages_t.numel()

            # Build prompt groups by position (most robust method).
            # cosmos-rl expands completions in order:
            # prompt0: completions[0..n_gen-1], prompt1: completions[n_gen..2*n_gen-1], ...
            n_gen = getattr(self.config.rollout, 'n_generation', 8) \
                if hasattr(self, 'config') else 8
            groups: dict[int, list[int]] = {}
            for idx in range(n):
                group_key = idx // n_gen
                groups.setdefault(group_key, []).append(idx)

            # Debug: log n_gen and group structure (every 50 steps)
            if step % 50 < 4:
                group_sizes = [len(v) for v in groups.values()]
                logger.warning(
                    f"[VarProtect] step={step}: DEBUG n_gen={n_gen}, n={n}, "
                    f"num_groups={len(groups)}, group_sizes={group_sizes}"
                )

            result = torch.zeros_like(advantages_t, dtype=torch.float32)
            for group_key, indices in groups.items():
                group_n = len(indices)
                if group_n < 2:
                    # Single sample group — advantage stays 0 (no relative ordering)
                    for idx in indices:
                        result[idx] = 0.0
                    continue
                # Extract group advantages
                group_vals = advantages_t[indices]
                # Sort and assign fractional ranks within group
                sorted_indices_local = torch.argsort(group_vals)
                ranks_local = torch.zeros(group_n, dtype=torch.float32, device=advantages_t.device)
                ranks_local[sorted_indices_local] = torch.arange(group_n, dtype=torch.float32, device=advantages_t.device)
                # Normalize ranks to [-1, 1] centered at 0
                normalized_ranks = (ranks_local - (group_n - 1) / 2.0) / ((group_n - 1) / 2.0 + 1e-8)
                # Scale by rank_scale
                for i, idx in enumerate(indices):
                    result[idx] = normalized_ranks[i] * rank_scale

            advantages_t = result

            logger.warning(
                f"[VarProtect] step={step}: rank-based advantages "
                f"(n={n}, groups={len(groups)}, rank_scale={rank_scale}, "
                f"adv_mean={advantages_t.mean().item():.4f}, "
                f"adv_std={advantages_t.std().item():.4f})"
            )

            # Still skip if all advantages are identical (no useful signal)
            raw_std = advantages_t.std().item()
            skip_std = vp_cfg.get("skip_update_std", 0.005)
            if raw_std < skip_std and n > 1:
                logger.warning(
                    f"[VarProtect] step={step}: advantages too uniform "
                    f"(std={raw_std:.6f}), zeroing"
                )
                advantages_t = torch.zeros_like(advantages_t)
        else:
            # Legacy z-score normalization (only used when rank_advantages=False)
            min_std = vp_cfg.get("min_group_std", 0.05)
            amplify_factor = vp_cfg.get("amplify_factor", 3.0)
            raw_std = advantages_t.std().item()
            if raw_std < min_std and raw_std > 1e-8:
                ratio = min(min_std / raw_std, amplify_factor)
                advantages_t = advantages_t * ratio
                if step % 5 == 0:
                    logger.warning(
                        f"[VarProtect] step={step}: amplified advantages by {ratio:.2f}x "
                        f"(raw_std={raw_std:.6f} < min_std={min_std})"
                    )

            # Check post-amplification variance, not pre-amplification raw_std
            post_amp_std = advantages_t.std().item()
            skip_std2 = vp_cfg.get("skip_update_std", 0.005)
            if post_amp_std < skip_std2:
                logger.warning(
                    f"[VarProtect] step={step}: post-amp variance too low "
                    f"(post-amp std={post_amp_std:.6f} < {skip_std2:.6f}), zeroing"
                )
                advantages_t = torch.zeros_like(advantages_t)

            eps = cfg.get("eps", 1e-8)
            clip_value = cfg.get("clip_value", 3.0)
            use_running_stats = cfg.get("use_running_stats", False)

            mean = advantages_t.mean().item()
            std = max(advantages_t.std().item(), eps)

            if use_running_stats:
                prev_mean = getattr(self, "_adv_mean", mean)
                prev_std = getattr(self, "_adv_std", std)
                alpha = cfg.get("ema_alpha", 0.99)
                mean = alpha * prev_mean + (1 - alpha) * mean
                std = alpha * prev_std + (1 - alpha) * std
                self._adv_mean = mean
                self._adv_std = std

            advantages_t = (advantages_t - mean) / std
            if clip_value > 0:
                advantages_t = torch.clamp(advantages_t, -clip_value, clip_value)

            if step % 10 == 0:
                logger.warning(
                    f"[AdvNorm] step={step}: raw mean={mean:.4f}, std={std:.4f} "
                    f"-> norm mean={advantages_t.mean().item():.4f}, "
                    f"std={advantages_t.std().item():.4f}"
                )

        return advantages_t

    def _get_variance_protection_cfg(self) -> dict[str, Any]:
        """Extract variance protection config from TOML [custom.alpamayo.variance_protection]."""
        try:
            return getattr(self.config, "custom", {}).get("alpamayo", {}).get(
                "variance_protection", {}
            )
        except (TypeError, AttributeError):
            return {}

    def _get_advantage_normalization_cfg(self) -> dict[str, Any]:
        try:
            return getattr(self.config, "custom", {}).get("alpamayo", {}).get(
                "advantage_normalization", {}
            )
        except (TypeError, AttributeError):
            return {}

    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Customized Reasoning VLA GRPO Trainer. Run one GRPO policy optimization step from a batch of rollouts.

        Args:
            rollouts: Completed rollouts (prompt, completion, advantage, masks, etc.).
            current_step: Global training step index (logging and checkpoint naming).
            total_steps: Planned total steps for the job.
            remain_samples_num: Samples remaining in the dataset pass (checkpoint policy).
            inter_policy_nccl: Communicator for cross-replica grad / loss all-reduces.
            is_master_replica: Whether this replica performs logging and checkpoint I/O.
            do_save_checkpoint: Reserved for the Cosmos trainer API (checkpointing uses
                ``_save_checkpoint`` and config inside this implementation).

        Returns:
            ``report_data``: metrics dict for logging (e.g. ``train_step``, ``train/loss_*``,
            timing, MFU) on the master rank; empty dict elsewhere.
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        logger.info("[Policy] Prepare training data.")
        self.metrics = {
            "entropy": 0.0,
            "effective_entropy": 0.0,
        }
        payloads_list = [rollout.prompt for rollout in rollouts]
        completions_list = [rollout.completion for rollout in rollouts]
        advantages_list = [rollout.advantage for rollout in rollouts]
        rewards_list_for_diag = [getattr(rollout, "reward", rollout.advantage) for rollout in rollouts]
        reward_infos_for_diag = [getattr(rollout, "reward_info", {}) for rollout in rollouts]
        has_reward_info_for_diag = any(
            isinstance(ri, dict) and ri for ri in reward_infos_for_diag
        )
        rollout_diag = _rollout_train_diagnostics(
            completions_list,
            rewards_list_for_diag,
            reward_infos_for_diag,
        )

        pos_coef_global = self.config.train.train_policy.positive_nll_coef
        if pos_coef_global is not None and pos_coef_global > 0.0:
            rewards_list = [rollout.reward for rollout in rollouts]
            self._positive_flags_t = torch.tensor(
                [1 if r > 0 else 0 for r in rewards_list],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            self._positive_flags_t = None
        n_ignore_prefix_tokens_list = [rollout.n_ignore_prefix_tokens for rollout in rollouts]
        processed_samples: List[Any] = [
            self.data_packer.get_policy_input(
                payloads_list[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(payloads_list))
        ]
        if not has_reward_info_for_diag:
            rollout_diag.update(
                _rollout_train_diagnostics(
                    completions_list,
                    rewards_list_for_diag,
                    [
                        s.get("reward_components", {})
                        if isinstance(s, dict)
                        else {}
                        for s in processed_samples
                    ],
                )
            )

        advantages_t = torch.tensor(advantages_list).to(self.device)
        advantages_t = self._normalize_advantages(advantages_t, current_step, payloads_list)
        component_advantages_t = self._build_component_advantages(
            payloads_list,
            processed_samples,
            advantages_list,
        )

        batch_size = len(rollouts)
        mini_batch_size = min(self.mini_batch, batch_size) if self.mini_batch > 0 else batch_size
        assert batch_size % mini_batch_size == 0, (
            "Batch size should be divided evenly by mini_batch"
        )
        num_mini_batch = batch_size // mini_batch_size

        self.old_per_token_logps = [None for _ in range(num_mini_batch)]
        self.ref_per_token_logps = [None for _ in range(num_mini_batch)]

        acc_n_tokens = 0

        need_compute_ref, kl_beta = self._swap_model_state_dict()

        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        grad_norm_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0
        is_computing_refs = [True, False] if need_compute_ref else [False]
        for is_computing_ref in is_computing_refs:
            if is_computing_ref:
                self.model.eval()
            else:
                if need_compute_ref:
                    need_compute_ref = False
                    self._swap_model_state_dict()
                self.model.train()

            with torch.set_grad_enabled(not is_computing_ref):
                for i_mu in range(1 if is_computing_ref else self.mu_iterations):
                    local_mini_step = 0
                    with torch.cuda.stream(self.train_stream):
                        for i in range(0, batch_size, mini_batch_size):
                            end = min(i + mini_batch_size, batch_size)

                            minibatched_processed_samples = processed_samples[i:end]

                            computed_max_len = (
                                self.config.policy.model_max_length
                                if self.parallel_dims.pp_enabled
                                else self.data_packer.policy_compute_max_len(
                                    minibatched_processed_samples
                                )
                            )

                            computed_max_len = (
                                (computed_max_len + self.seq_len_multiple - 1)
                                // self.seq_len_multiple
                                * self.seq_len_multiple
                            )
                            minibatched_scalar_advantages = (
                                advantages_t[i:end]
                                .unsqueeze(1)
                                .expand(-1, computed_max_len)
                                .to(self.device)
                            )

                            user_mini_batch: Dict[str, Any] = self.data_packer.policy_collate_fn(
                                minibatched_processed_samples,
                                computed_max_len=computed_max_len,
                            )

                            if (
                                self.parallel_dims.dp_shard_coord[1]
                                == self.parallel_dims.world_size
                            ):
                                user_mini_batch["interested_tokens"] = user_mini_batch[
                                    "logprob_masks"
                                ]

                            for k, v in list(user_mini_batch.items()):
                                if isinstance(v, torch.Tensor) and v.device != self.device:
                                    user_mini_batch[k] = v.to(self.device)

                            position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(
                                **user_mini_batch
                            )
                            acc_n_tokens += np.prod(input_ids.shape)
                            user_mini_batch["position_ids"] = position_ids
                            padding_mask = user_mini_batch.get("padding_mask", None)

                            input_ids_before_cp = user_mini_batch["input_ids"]
                            position_ids_before_cp = user_mini_batch["position_ids"]
                            padding_mask_before_cp = padding_mask

                            if self.parallel_dims.cp_enabled:
                                [input_ids, position_ids, padding_mask] = slice_inputs_for_ulysses(
                                    [input_ids, position_ids, padding_mask],
                                    self.parallel_dims.mesh["cp"],
                                )
                                user_mini_batch["position_ids"] = position_ids
                                user_mini_batch["input_ids"] = input_ids
                                if padding_mask is not None:
                                    user_mini_batch["padding_mask"] = padding_mask

                            if self.parallel_dims.pp_enabled:
                                raise NotImplementedError(
                                    "Pipeline Parallel is not supported for Reasoning VLA"
                                )
                            else:
                                model_out = self.model(**user_mini_batch)

                                if self.parallel_dims.cp_enabled:
                                    user_mini_batch["position_ids"] = position_ids_before_cp
                                    user_mini_batch["input_ids"] = input_ids_before_cp
                                    if padding_mask_before_cp is not None:
                                        user_mini_batch["padding_mask"] = padding_mask_before_cp

                                raw_logits = (
                                    model_out.logits if hasattr(model_out, "logits") else model_out
                                )
                                if self.config.train.train_policy.temperature > 1e-6:
                                    raw_logits = (
                                        raw_logits / self.config.train.train_policy.temperature
                                    )

                                current_per_token_logprobs, cu_seqlens, metrics = (
                                    self.compute_logprobs(
                                        user_mini_batch,
                                        logits=raw_logits,
                                        is_full_logits=True
                                        if getattr(raw_logits, "ndim", 0) == 3
                                        else False,
                                    )
                                )
                                logprob_masks = user_mini_batch["logprob_masks"]
                                current_advantages = self._build_token_advantages(
                                    user_mini_batch,
                                    minibatched_scalar_advantages,
                                    component_advantages_t,
                                    start=i,
                                    end=end,
                                    computed_max_len=computed_max_len,
                                )

                                if is_computing_ref:
                                    assert i_mu == 0, "Only first iteration should compute ref"
                                    self.ref_per_token_logps[local_mini_step] = (
                                        current_per_token_logprobs.detach()
                                    )
                                    local_mini_step += 1
                                    continue
                                else:
                                    if self.old_per_token_logps[local_mini_step] is None:
                                        assert i_mu == 0, (
                                            "Only first iteration should append "
                                            "`old_per_token_logps`"
                                        )
                                        self.old_per_token_logps[local_mini_step] = (
                                            current_per_token_logprobs.detach()
                                        )
                                    else:
                                        assert i_mu > 0, (
                                            "Only inner iteration should reuse "
                                            "`old_per_token_logps`"
                                        )
                                    loss, per_token_loss, kl_loss = compute_loss(
                                        current_per_token_logprobs,
                                        self.old_per_token_logps[local_mini_step],
                                        self.ref_per_token_logps[local_mini_step],
                                        current_advantages,
                                        cu_seqlens,
                                        self.config,
                                        logprob_masks,
                                        dp_group=self.parallel_dims.mesh["dp"].get_group()
                                        if self.parallel_dims.dp_enabled
                                        else None,
                                        ddp_comm=inter_policy_nccl,
                                    )

                                    if pos_coef_global is not None and pos_coef_global > 0.0:
                                        pos_flag_batch = self._positive_flags_t[i:end]
                                        pos_mask = pos_flag_batch.unsqueeze(1).expand_as(
                                            logprob_masks
                                        )
                                        pos_token_mask = pos_mask & logprob_masks
                                        if pos_token_mask.any():
                                            flat_mask = pos_token_mask[logprob_masks]
                                            l_nll = -current_per_token_logprobs[flat_mask].mean()
                                            loss = loss + pos_coef_global * l_nll

                                    loss = loss / num_mini_batch
                                    per_token_loss = per_token_loss / num_mini_batch
                                    kl_loss = kl_loss / num_mini_batch

                                    loss.backward()
                                    loss_sum += per_token_loss.item()
                                    kl_loss_sum += kl_loss.item()
                                    loss_count += 1
                                    for key in metrics:
                                        self.metrics[key] += metrics[key]

                                    # --- Diffusion Expert RL: advantage-weighted flow matching loss ---
                                    # After GRPO token-level loss backward, compute a separate
                                    # diffusion expert loss weighted by the per-sample advantage.
                                    # Gradient flows: advantage*MSE -> action_out_proj -> expert
                                    # transformer -> KV cache (detached) -> no VLM gradient.
                                    diffusion_rl_cfg = self._get_diffusion_rl_cfg()
                                    if diffusion_rl_cfg.get("enable", False):
                                        self._compute_and_backward_diffusion_rl_loss(
                                            minibatched_processed_samples=minibatched_processed_samples,
                                            minibatched_scalar_advantages=minibatched_scalar_advantages,
                                            user_mini_batch=user_mini_batch,
                                            computed_max_len=computed_max_len,
                                            diffusion_rl_cfg=diffusion_rl_cfg,
                                            current_step=current_step,
                                            num_mini_batch=num_mini_batch,
                                        )

                            self.mini_step += 1
                            local_mini_step += 1

                            if (
                                local_mini_step
                                % int(os.environ.get("COSMOS_GRPO_STEP_INTERVAL", "10"))
                                == 0
                            ) and local_mini_step > 1:
                                all_reduced = True
                                grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
                            else:
                                all_reduced = False
                        if not is_computing_ref and not all_reduced:
                            grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
        self.old_per_token_logps = []
        self.ref_per_token_logps = []
        end_event.record()

        self.lr_schedulers.step()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss = global_max_loss = loss.item()
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0
                report_data["train/iteration_time"] = iter_time
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                report_data["train/learning_rate"] = self.lr_schedulers.get_last_lr()[0]
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss
                report_data["train/grad_norm"] = grad_norm_sum.item()
                report_data["train/local_loss"] = loss.item()
                report_data.update(rollout_diag)
                report_data["train/advantage_mean"] = advantages_t.mean().item()
                report_data["train/advantage_std"] = advantages_t.std().item() if advantages_t.numel() > 1 else 0.0
                for key, tensor in component_advantages_t.items():
                    report_data[f"train/advantage_{key}_mean"] = tensor.mean().item()
                    report_data[f"train/advantage_{key}_std"] = tensor.std().item() if tensor.numel() > 1 else 0.0
                # Diffusion RL loss metrics
                if hasattr(self, '_diffusion_rl_loss_sum') and self._diffusion_rl_loss_count > 0:
                    report_data["train/diffusion_rl_loss"] = self._diffusion_rl_loss_sum / self._diffusion_rl_loss_count
                    # Reset for next step
                    self._diffusion_rl_loss_sum = 0.0
                    self._diffusion_rl_loss_count = 0
                raw_reward = report_data.get("train/raw_reward_mean", float("nan"))
                reward_std = report_data.get("train/raw_reward_std", float("nan"))
                logger.warning(
                    f"[Step {current_step}] loss={loss.item():.6f}, "
                    f"raw_reward={raw_reward:.4f}, reward_std={reward_std:.4f}, "
                    f"adv={advantages_t.mean().item():.4f}, gn={grad_norm_sum.item():.4f}"
                )

                if self.config.logging.report_mfu:
                    mfu = compute_mfu(
                        model=self.model,
                        n_tokens=acc_n_tokens,
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v
                if len(self.metrics) > 0:
                    for k, v in self.metrics.items():
                        report_data[f"train/{k}"] = (
                            v.item() if isinstance(v, torch.Tensor) else v
                        ) / loss_count

                tb_dir = os.path.join(os.environ.get("LOG_DIR", "/root/temp_log"), "tensorboard")
                writer = _get_tb_writer(tb_dir)
                for key, val in report_data.items():
                    if isinstance(val, (int, float)):
                        writer.add_scalar(key, val, current_step)
                writer.flush()

        self._save_checkpoint(current_step, total_steps, remain_samples_num, is_master_replica)
        return report_data

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            return loss.mean()

        return fake_compute_loss

    def _build_component_advantages(
        self,
        payloads: list[Any],
        processed_samples: list[Any],
        fallback_advantages: list[float],
    ) -> dict[str, torch.Tensor]:
        """Build grouped component advantages for optional token routing."""
        routing_cfg = _get_advantage_routing_cfg(self.config)
        if not bool(routing_cfg.get("enable", False)):
            return {}
        if not all(isinstance(s, dict) and "reward_components" in s for s in processed_samples):
            logger.warning(
                "[AdvantageRouting] Enabled but reward_components are missing; "
                "falling back to scalar GRPO advantages."
            )
            return {}

        components = [s["reward_components"] for s in processed_samples]
        fallback = [float(v) for v in fallback_advantages]
        n_gen = getattr(self.config.rollout, 'n_generation', 8)

        def value(name: str, default: float = 0.0) -> list[float]:
            return [float(c.get(name, default)) for c in components]

        def value_any(names: tuple[str, ...], default: float = 0.0) -> list[float]:
            vals = []
            for comp in components:
                val = default
                for name in names:
                    if name in comp:
                        val = comp[name]
                        break
                vals.append(float(val))
            return vals

        consistency = value_any(("consistency_reward", "raa_score"))
        risk = value("risk_reward")
        coc_values = [a + b for a, b in zip(value_any(("coc_reward", "coc_quality")), consistency)]
        traj_values = [
            a + b + c
            for a, b, c in zip(value_any(("traj_reward", "reward")), consistency, risk)
        ]
        format_values = value_any(("coc_format_score", "coc_factual", "scene_understanding"))

        adv = {
            "coc": _normalize_grouped(coc_values, payloads, fallback, n_generation=n_gen),
            "traj": _normalize_grouped(traj_values, payloads, fallback, n_generation=n_gen),
            "format": _normalize_grouped(format_values, payloads, fallback, n_generation=n_gen),
        }
        return {
            key: torch.tensor(vals, device=self.device, dtype=torch.float32)
            for key, vals in adv.items()
        }

    def _build_token_advantages(
        self,
        user_mini_batch: dict[str, Any],
        scalar_advantages: torch.Tensor,
        component_advantages: dict[str, torch.Tensor],
        *,
        start: int,
        end: int,
        computed_max_len: int,
    ) -> torch.Tensor:
        """Route component advantages to CoC/traj/format tokens when enabled."""
        logprob_masks = user_mini_batch["logprob_masks"]
        base = logprob_masks * scalar_advantages
        routing_cfg = _get_advantage_routing_cfg(self.config)
        if not bool(routing_cfg.get("enable", False)) or not component_advantages:
            return base

        coc_mask = user_mini_batch.get("coc_logprob_masks")
        traj_mask = user_mini_batch.get("traj_logprob_masks")
        fmt_mask = user_mini_batch.get("format_logprob_masks")
        if coc_mask is None or traj_mask is None or fmt_mask is None:
            logger.warning(
                "[AdvantageRouting] Token masks are missing; falling back to scalar advantages."
            )
            return base

        routed = torch.zeros_like(base)
        coc_weight = float(routing_cfg.get("coc_weight", 1.0))
        traj_weight = float(routing_cfg.get("traj_weight", 1.0))
        format_weight = float(routing_cfg.get("format_weight", 0.2))
        fallback_weight = float(routing_cfg.get("fallback_weight", 1.0))

        coc_adv = component_advantages["coc"][start:end].unsqueeze(1).expand(-1, computed_max_len)
        traj_adv = component_advantages["traj"][start:end].unsqueeze(1).expand(-1, computed_max_len)
        fmt_adv = (
            component_advantages["format"][start:end]
            .unsqueeze(1)
            .expand(-1, computed_max_len)
        )

        routed += coc_mask * (coc_weight * coc_adv)
        routed += traj_mask * (traj_weight * traj_adv)
        routed += fmt_mask * (format_weight * fmt_adv)

        routed_mask = (coc_mask | traj_mask | fmt_mask) & logprob_masks
        fallback_mask = logprob_masks & ~routed_mask
        routed += fallback_mask * (fallback_weight * scalar_advantages)
        return routed

    def _compute_and_backward_diffusion_rl_loss(
        self,
        minibatched_processed_samples: list[Any],
        minibatched_scalar_advantages: torch.Tensor,
        user_mini_batch: dict[str, Any],
        computed_max_len: int,
        diffusion_rl_cfg: dict[str, Any],
        current_step: int,
        num_mini_batch: int,
    ) -> None:
        """Compute advantage-weighted diffusion expert loss and backward it.

        After the GRPO token-level loss.backward() has been called, we do a
        second forward pass through the model with compute_diffusion_loss=True
        to get the flow matching MSE loss from the diffusion expert. This loss
        is then weighted by the per-sample advantage and backward'd separately.

        Gradient flow:
            advantage * diffusion_MSE -> action_out_proj -> expert transformer
            -> KV cache keys/values (DETACHED) -> NO VLM gradient

        VLM gets its own independent gradient via GRPO token-level loss.
        Expert gets gradient via advantage-weighted flow matching loss.
        The KV cache acts as the bridge: it carries VLM's contextual understanding
        to the expert, but gradient stops at the detached boundary.

        Args:
            minibatched_processed_samples: Raw data dicts for the current mini-batch.
            minibatched_scalar_advantages: [mini_batch_size, computed_max_len] expanded advantages.
            user_mini_batch: Collated batch dict (input_ids, position_ids, etc.).
            computed_max_len: Padded sequence length for this mini-batch.
            diffusion_rl_cfg: Config dict from TOML [custom.alpamayo.diffusion_rl].
            current_step: Global training step (for logging).
            num_mini_batch: Number of mini-batches (for loss scaling).
        """
        loss_weight = float(diffusion_rl_cfg.get("loss_weight", 0.1))
        advantage_mode = diffusion_rl_cfg.get("advantage_mode", "weighted_regression")
        advantage_clip = float(diffusion_rl_cfg.get("advantage_clip", 2.0))
        min_advantage_threshold = float(diffusion_rl_cfg.get("min_advantage_threshold", -1.0))

        # 1. Extract trajectory tensors from processed samples
        # Dataset tensors are [1, 1, T, 3] for xyz and [1, 1, T, 3, 3] for rot (rotation matrix).
        # We squeeze the B=1 and n_traj_group=1 dimensions from each sample,
        # then stack the per-sample 2D/3D tensors to create proper batch tensors.
        ego_history_xyz_list = []
        ego_history_rot_list = []
        ego_future_xyz_list = []
        ego_future_rot_list = []

        for sample in minibatched_processed_samples:
            if isinstance(sample, dict):
                h_xyz = sample.get("ego_history_xyz")
                h_rot = sample.get("ego_history_rot")
                f_xyz = sample.get("ego_future_xyz")
                f_rot = sample.get("ego_future_rot")

                if h_xyz is not None and f_xyz is not None:
                    # Dataset provides xyz as [1, 1, T, 3] (4D) or [T, 3] (2D)
                    # and rot as [1, 1, T, 3, 3] (5D, rotation matrix) or [T, 4] (2D, quaternion)
                    # We squeeze leading dimensions to get core [T, 3] / [T, 3, 3] / [T, 4]
                    # then stack will add the batch dimension correctly.
                    if h_xyz.dim() >= 4:
                        # Squeeze leading B and n_traj_group dims: [1, 1, T, 3] → [T, 3]
                        h_xyz = h_xyz.squeeze(0).squeeze(0)
                    if h_rot is not None and h_rot.dim() >= 5:
                        # Squeeze leading dims: [1, 1, T, 3, 3] → [T, 3, 3]
                        h_rot = h_rot.squeeze(0).squeeze(0)
                    elif h_rot is not None and h_rot.dim() == 4:
                        # [1, T, 3, 3] → [T, 3, 3]
                        h_rot = h_rot.squeeze(0)
                    elif h_rot is not None and h_rot.dim() == 3:
                        # Could be [T, 3, 3] (rotation matrix) or [T, 4] (quaternion)
                        # Keep as is — traj_to_action handles both
                        pass
                    if f_xyz.dim() >= 4:
                        f_xyz = f_xyz.squeeze(0).squeeze(0)
                    if f_rot is not None and f_rot.dim() >= 5:
                        f_rot = f_rot.squeeze(0).squeeze(0)
                    elif f_rot is not None and f_rot.dim() == 4:
                        f_rot = f_rot.squeeze(0)
                    elif f_rot is None:
                        # ego_future_rot missing: create unit rotation matrix [T, 3, 3]
                        T_steps = f_xyz.shape[0] if f_xyz.dim() == 2 else 1
                        f_rot = torch.zeros(T_steps, 3, 3,
                                            dtype=f_xyz.dtype, device=f_xyz.device)
                        f_rot[:, 0, 0] = 1.0  # Identity rotation matrix
                        f_rot[:, 1, 1] = 1.0
                        f_rot[:, 2, 2] = 1.0
                    elif f_rot.dim() == 3:
                        # [T, 3, 3] or [T, 4] — keep as is
                        pass

                    ego_history_xyz_list.append(h_xyz)
                    ego_history_rot_list.append(h_rot)
                    ego_future_xyz_list.append(f_xyz)
                    ego_future_rot_list.append(f_rot)
                else:
                    logger.warning(
                        f"[DiffusionRL] step={current_step}: sample missing "
                        f"trajectory data. Skipping diffusion loss."
                    )
                    return

        if not ego_future_xyz_list:
            logger.warning(
                f"[DiffusionRL] step={current_step}: No trajectory data found "
                f"in mini-batch. Skipping diffusion loss."
            )
            return

        # 2. Stack per-sample tensors into batch tensors
        # After squeeze, each tensor is [T, 3] for xyz, [T, 3, 3] for rot (rotation matrix).
        # torch.stack adds batch dim at dim=0: → [B, T, 3] for xyz, [B, T, 3, 3] for rot.
        # The model forward expects [B, n_traj_group, n_traj, 3] for xyz,
        # so we add n_traj_group=1 and n_traj=1 via unsqueeze.
        try:
            ego_history_xyz = torch.stack(ego_history_xyz_list).to(self.device)
            ego_history_rot = torch.stack(ego_history_rot_list).to(self.device)
            ego_future_xyz = torch.stack(ego_future_xyz_list).to(self.device)
            ego_future_rot = torch.stack(ego_future_rot_list).to(self.device)
        except Exception as e:
            logger.warning(
                f"[DiffusionRL] step={current_step}: Failed to stack trajectory "
                f"tensors: {e}. Skipping diffusion loss."
            )
            return

        # Add n_traj_group and n_traj dims for model forward compatibility
        # xyz: [B, T, 3] → [B, n_traj_group=1, n_traj=1, T, 3] is 5D...
        # BUT fuse_traj_tokens expects [B, n_traj, T, 3] (4D) — so add one dim.
        # rot: rotation matrix [B, T, 3, 3] → [B, n_traj, T, 3, 3] (5D)
        # After fuse_traj_tokens flattens [B, n_traj] → [B*n_traj],
        # _process_traj_future_training gets 4D [B*n_traj, T, 3] xyz
        # and 5D [B*n_traj, T, 3, 3] rot, which traj_to_action expects.
        ego_history_xyz = ego_history_xyz.unsqueeze(1)  # [B, T, 3] → [B, 1, T, 3]
        ego_history_rot = ego_history_rot.unsqueeze(1)  # [B, T, D, D] → [B, 1, T, D, D]
        ego_future_xyz = ego_future_xyz.unsqueeze(1)    # [B, T, 3] → [B, 1, T, 3]
        ego_future_rot = ego_future_rot.unsqueeze(1)    # [B, T, D, D] → [B, 1, T, D, D]

        # 3. Reconstruct the tokenized_data dict for a second model forward
        # We need a fresh forward pass with use_cache=True to get past_key_values
        # for the diffusion expert path.
        # CRITICAL: The model's forward() pops "input_ids" from tokenized_data
        # (line 289). The first GRPO forward already consumed it, so we must
        # provide a fresh copy. We reconstruct tokenized_data by copying all
        # relevant keys from user_mini_batch (which holds the original collated
        # tensors that haven't been popped yet — the pop happened inside the
        # model's forward, and the caller's user_mini_batch dict still has them).
        diffusion_tokenized_data = {}
        for k, v in user_mini_batch.items():
            if k in ("input_ids", "position_ids", "attention_mask",
                      "pixel_values", "image_grid_thw",
                      "labels_mask", "logprob_masks",
                      "coc_logprob_masks", "traj_logprob_masks",
                      "format_logprob_masks",
                      "interested_tokens", "padding_mask"):
                # Clone tensors so the pop() in model.forward() doesn't mutate
                # the original user_mini_batch (which is still needed by GRPO).
                if isinstance(v, torch.Tensor):
                    diffusion_tokenized_data[k] = v.clone()
                else:
                    diffusion_tokenized_data[k] = v

        # 4. Compute per-sample scalar advantages
        # minibatched_scalar_advantages is [mini_batch_size, computed_max_len]
        # where each row has the same value expanded across all tokens.
        # Since all positions in a row have identical advantage (it's just
        # scalar_advantage.unsqueeze(1).expand(-1, computed_max_len)),
        # we extract the original scalar by taking any single position.
        # However, due to _build_token_advantages routing, the expanded values
        # may differ across positions (coc/traj/format routing). For the
        # diffusion expert we want the OVERALL per-sample advantage, not
        # the per-token routed version. So we use the UN-normalized scalar
        # advantages that were computed before token routing.
        # Unfortunately, after routing we only have the routed version in
        # minibatched_scalar_advantages. The best approximation is to take
        # the mean across tokens for each sample.
        scalar_adv = minibatched_scalar_advantages.mean(dim=1)  # [mini_batch_size]

        # 5. Compute advantage-weighted diffusion loss
        # Call model forward with compute_diffusion_loss=True
        model_out = self.model(
            tokenized_data=diffusion_tokenized_data,
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            ego_future_xyz=ego_future_xyz,
            ego_future_rot=ego_future_rot,
            compute_diffusion_loss=True,
        )

        raw_diffusion_loss = model_out.diffusion_loss
        if raw_diffusion_loss is None:
            logger.warning(
                f"[DiffusionRL] step={current_step}: Model returned None "
                f"diffusion_loss. Skipping."
            )
            return

        # 6. Weight the diffusion loss by advantage
        # IMPORTANT: For diffusion expert, the flow matching MSE loss is always
        # positive. Multiplying by negative advantage produces negative loss,
        # which flips gradient direction — pushing the expert AWAY from the GT
        # trajectory. This is dangerous in continuous action space: "away from GT"
        # ≈ encouraging random/noisy actions, potentially causing model collapse.
        # Therefore, we handle negative advantages carefully:
        # - weighted_regression: only apply loss for POSITIVE advantages;
        #   negative advantages are zeroed (don't reinforce bad trajectories,
        #   but don't push toward randomness either).
        # - absolute_regression: treat all advantages as positive weights;
        #   the expert always learns toward GT, regardless of action quality.
        # - sign_regression: explicitly only use positive advantages.
        if advantage_mode == "weighted_regression":
            # Standard advantage-weighted regression, but SAFE for continuous space:
            # Positive advantage → reinforce GT trajectory (expert learns it)
            # Negative/zero advantage → no gradient from this sample
            # This prevents the "push away from GT = encourage noise" problem.
            clipped_adv = torch.clamp(scalar_adv, 0, advantage_clip)  # clamp to non-negative
            # Only apply loss where advantage is positive
            positive_mask = scalar_adv > 0
            n_positive = int(positive_mask.sum().item())
            if n_positive == 0:
                logger.warning(
                    f"[DiffusionRL] step={current_step}: No positive advantages. "
                    f"Skipping diffusion loss (no good trajectories to reinforce)."
                )
                return
            # Weight by mean of positive advantages only
            weighted_diffusion_loss = raw_diffusion_loss * clipped_adv[positive_mask].mean() * loss_weight

        elif advantage_mode == "absolute_regression":
            # Absolute regression: use |advantage| as weight regardless of sign.
            # All samples contribute equally to learning toward GT trajectory.
            # This is a conservative approach — the expert always improves,
            # but doesn't distinguish between good and bad actions.
            abs_adv = torch.abs(scalar_adv)
            abs_adv = torch.clamp(abs_adv, 0, advantage_clip)
            weighted_diffusion_loss = raw_diffusion_loss * abs_adv.mean() * loss_weight

        elif advantage_mode == "sign_regression":
            # Sign regression: only positive advantages reinforce the expert.
            # Negative advantages are completely ignored (zero gradient).
            # This is identical to weighted_regression with the same safety guard,
            # but more explicit about the intent.
            positive_mask = scalar_adv > 0
            n_positive = int(positive_mask.sum().item())
            if n_positive == 0:
                logger.warning(
                    f"[DiffusionRL] step={current_step}: No positive advantages. "
                    f"Skipping diffusion loss."
                )
                return
            signed_adv = torch.clamp(scalar_adv[positive_mask], 0, advantage_clip)
            weighted_diffusion_loss = raw_diffusion_loss * signed_adv.mean() * loss_weight

        else:
            logger.warning(
                f"[DiffusionRL] Unknown advantage_mode: {advantage_mode}. "
                f"Falling back to weighted_regression."
            )
            clipped_adv = torch.clamp(scalar_adv, 0, advantage_clip)
            positive_mask = scalar_adv > 0
            if positive_mask.sum().item() == 0:
                return
            weighted_diffusion_loss = raw_diffusion_loss * clipped_adv[positive_mask].mean() * loss_weight

        # 7. Backward the weighted diffusion loss
        weighted_diffusion_loss = weighted_diffusion_loss / num_mini_batch
        weighted_diffusion_loss.backward()

        # 8. Log diagnostics
        if current_step % 10 == 0:
            logger.warning(
                f"[DiffusionRL] step={current_step}: "
                f"raw_diff_loss={raw_diffusion_loss.item():.6f}, "
                f"weighted_diff_loss={weighted_diffusion_loss.item():.6f}, "
                f"adv_mean={scalar_adv.mean().item():.4f}, "
                f"adv_std={scalar_adv.std().item():.4f}, "
                f"mode={advantage_mode}, weight={loss_weight}"
            )

        # 9. Add to metrics for tensorboard logging
        diff_loss_val = weighted_diffusion_loss.item()
        if not hasattr(self, '_diffusion_rl_loss_sum'):
            self._diffusion_rl_loss_sum = 0.0
            self._diffusion_rl_loss_count = 0
        self._diffusion_rl_loss_sum += diff_loss_val
        self._diffusion_rl_loss_count += 1

    def _get_diffusion_rl_cfg(self) -> dict[str, Any]:
        """Extract diffusion RL config from TOML [custom.alpamayo.diffusion_rl].

        Config keys:
            enable (bool): Whether to enable diffusion expert RL training.
            loss_weight (float): Weight coefficient for the diffusion RL loss.
            advantage_mode (str): How to weight diffusion loss by advantage.
                "weighted_regression": adv * MSE (default)
                "absolute_regression": |adv| * MSE
                "sign_regression": only positive adv * MSE
            advantage_clip (float): Clip advantage magnitude.
            min_advantage_threshold (float): Skip samples below this threshold.

        Returns:
            Dict with config keys. Empty dict if config not found (disabled).
        """
        try:
            return getattr(self.config, "custom", {}).get("alpamayo", {}).get(
                "diffusion_rl", {}
            )
        except (TypeError, AttributeError):
            return {}

    
