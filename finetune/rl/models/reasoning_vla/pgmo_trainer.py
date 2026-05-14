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

"""Pareto-Guided Multi-Objective GRPO (PGMO-GRPO) Trainer.

Extends ReasoningVLAGRPOTrainer with:
  1. Multi-dimensional advantage decomposition
  2. Pareto-optimal sample identification and weighting
  3. Adaptive KL divergence scheduling
  4. Contrastive reasoning-action learning (CRAL)

All PGMO features are opt-in via TOML config [custom.alpamayo.pgmo].
When disabled, behaves identically to the base ReasoningVLAGRPOTrainer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

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

_TB_WRITER = None


def _get_tb_writer(log_dir: str) -> SummaryWriter:
    global _TB_WRITER
    if _TB_WRITER is None:
        _TB_WRITER = SummaryWriter(log_dir=log_dir)
    return _TB_WRITER


def _get_pgmo_cfg(config: object) -> dict[str, Any]:
    """Extract PGMO configuration from TOML [custom.alpamayo.pgmo]."""
    try:
        pgmo = getattr(config, "custom")["alpamayo"].get("pgmo", {})
    except (TypeError, KeyError, AttributeError):
        pgmo = {}
    return {
        "enable_pareto_weighting": bool(pgmo.get("enable_pareto_weighting", False)),
        "enable_adaptive_kl": bool(pgmo.get("enable_adaptive_kl", False)),
        "enable_contrastive": bool(pgmo.get("enable_contrastive", False)),
        "adaptive_kl_lambda": float(pgmo.get("adaptive_kl_lambda", 2.0)),
        "contrastive_weight": float(pgmo.get("contrastive_weight", 0.05)),
        "contrastive_temperature": float(pgmo.get("contrastive_temperature", 0.07)),
        "pareto_base_weight": float(pgmo.get("pareto_base_weight", 1.5)),
        "objective_weights": pgmo.get(
            "objective_weights", [0.25, 0.25, 0.25, 0.25]
        ),
    }


@TrainerRegistry.register(trainer_type="reasoning_vla_pgmo_grpo")
class PGMOTrainer(AlpamayoGRPOTrainer):
    """Pareto-Guided Multi-Objective GRPO Trainer for Reasoning VLA.

    Extends the base GRPO trainer with PGMO innovations:
    - Multi-dimensional advantage (CoC quality, RAA, trajectory, comfort)
    - Pareto-optimal sample weighting
    - Adaptive KL divergence scheduling
    - Contrastive reasoning-action learning
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tb_writer = None
        self._adaptive_kl_scheduler = None
        self._pgmo_cfg = None

    def _init_pgmo(self) -> None:
        """Lazy-initialize PGMO components from config."""
        if self._pgmo_cfg is not None:
            return
        self._pgmo_cfg = _get_pgmo_cfg(self.config)

        # Initialize adaptive KL scheduler if enabled
        if self._pgmo_cfg["enable_adaptive_kl"]:
            from rl.utils.adaptive_kl import AdaptiveKLScheduler

            self._adaptive_kl_scheduler = AdaptiveKLScheduler(
                base_beta=getattr(
                    self.config.train.train_policy, "kl_beta", 0.04
                ),
                lambda_decay=self._pgmo_cfg["adaptive_kl_lambda"],
            )

    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run one PGMO-GRPO policy optimization step.

        Extends the base GRPO step with Pareto weighting, adaptive KL,
        and contrastive loss.
        """
        self._init_pgmo()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        logger.info("[PGMO] Prepare training data.")
        self.metrics = {
            "entropy": 0.0,
            "effective_entropy": 0.0,
        }

        # ---- Extract rollout data ----
        payloads_list = [rollout.prompt for rollout in rollouts]
        completions_list = [rollout.completion for rollout in rollouts]
        advantages_list = [rollout.advantage for rollout in rollouts]

        # ---- Multi-objective advantage computation (PGMO) ----
        rewards_list = [rollout.reward for rollout in rollouts]
        reward_infos = [
            getattr(rollout, "reward_info", {}) for rollout in rollouts
        ]

        if (
            self._pgmo_cfg["enable_pareto_weighting"]
            and all(isinstance(ri, dict) and ri for ri in reward_infos)
        ):
            # Build multi-objective matrix from reward info dicts
            from rl.rewards.multi_objective import (
                compute_multi_objective_advantages,
                compute_pareto_weights,
                compute_weighted_scalar_advantage,
                decompose_reward_dict,
            )

            # Decompose each sample's reward into objective vector
            obj_list = [
                decompose_reward_dict(ri) for ri in reward_infos
            ]
            objectives = torch.stack(obj_list).to(self.device)  # [N, M]

            # Compute per-dimension advantages
            multi_advantages = compute_multi_objective_advantages(
                objectives, normalize=True
            )

            # Compute Pareto weights
            pareto_w = compute_pareto_weights(
                objectives,
                base_weight=self._pgmo_cfg["pareto_base_weight"],
            ).to(self.device)

            # Combine into scalar advantage
            obj_w = torch.tensor(
                self._pgmo_cfg["objective_weights"],
                device=self.device,
                dtype=torch.float32,
            )
            if obj_w.shape[0] != multi_advantages.shape[1]:
                obj_w = torch.ones(
                    multi_advantages.shape[1],
                    device=self.device,
                ) / multi_advantages.shape[1]

            advantages_t = compute_weighted_scalar_advantage(
                multi_advantages, obj_w, pareto_weights=pareto_w
            )

            # Log Pareto stats
            n_pareto = (
                pareto_w > 1.0
            ).sum().item() if pareto_w is not None else 0
            logger.info(
                f"[PGMO] Pareto-optimal: {n_pareto}/{len(rollouts)} "
                f"({100 * n_pareto / len(rollouts):.1f}%)"
            )

            # Adaptive KL update
            if self._pgmo_cfg["enable_adaptive_kl"] and self._adaptive_kl_scheduler:
                raa_mean = float(objectives[:, 1].mean().item())  # RAA is index 1
                new_beta = self._adaptive_kl_scheduler.update(raa_mean)
                # Override kl_beta in config for this step
                old_beta = getattr(
                    self.config.train.train_policy, "kl_beta", 0.04
                )
                try:
                    self.config.train.train_policy.kl_beta = new_beta
                except (AttributeError, TypeError):
                    pass
                logger.debug(
                    f"[PGMO] Adaptive KL: beta {old_beta:.4f} → {new_beta:.4f} "
                    f"(raa={raa_mean:.3f})"
                )
        else:
            # Standard scalar advantage
            advantages_t = torch.tensor(advantages_list).to(self.device)

        # ---- Positive NLL setup (unchanged) ----
        pos_coef_global = self.config.train.train_policy.positive_nll_coef
        if pos_coef_global is not None and pos_coef_global > 0.0:
            self._positive_flags_t = torch.tensor(
                [1 if r > 0 else 0 for r in rewards_list],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            self._positive_flags_t = None

        # ---- Process samples ----
        n_ignore_prefix_tokens_list = [
            rollout.n_ignore_prefix_tokens for rollout in rollouts
        ]
        processed_samples: List[Any] = [
            self.data_packer.get_policy_input(
                payloads_list[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(payloads_list))
        ]

        # ---- Training loop ----
        batch_size = len(rollouts)
        mini_batch_size = (
            min(self.mini_batch, batch_size)
            if self.mini_batch > 0
            else batch_size
        )
        assert batch_size % mini_batch_size == 0
        num_mini_batch = batch_size // mini_batch_size

        self.old_per_token_logps = [None for _ in range(num_mini_batch)]
        self.ref_per_token_logps = [None for _ in range(num_mini_batch)]

        acc_n_tokens = 0
        need_compute_ref, kl_beta = self._swap_model_state_dict()

        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        contrastive_loss_sum = torch.tensor(0.0, device=self.device)
        grad_norm_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0

        # ---- Contrastive loss setup ----
        do_contrastive = (
            self._pgmo_cfg["enable_contrastive"]
            and batch_size >= 4
            and not need_compute_ref
        )

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
                for i_mu in range(
                    1 if is_computing_ref else self.mu_iterations
                ):
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
                            minibatched_advantages = (
                                advantages_t[i:end]
                                .unsqueeze(1)
                                .expand(-1, computed_max_len)
                                .to(self.device)
                            )

                            user_mini_batch: Dict[str, Any] = (
                                self.data_packer.policy_collate_fn(
                                    minibatched_processed_samples,
                                    computed_max_len=computed_max_len,
                                )
                            )

                            if (
                                self.parallel_dims.dp_shard_coord[1]
                                == self.parallel_dims.world_size
                            ):
                                user_mini_batch["interested_tokens"] = (
                                    user_mini_batch["logprob_masks"]
                                )

                            for k, v in list(user_mini_batch.items()):
                                if (
                                    isinstance(v, torch.Tensor)
                                    and v.device != self.device
                                ):
                                    user_mini_batch[k] = v.to(self.device)

                            position_ids, input_ids, pos_seq_dim = (
                                self.model.get_position_ids(**user_mini_batch)
                            )
                            acc_n_tokens += np.prod(input_ids.shape)
                            user_mini_batch["position_ids"] = position_ids
                            padding_mask = user_mini_batch.get("padding_mask", None)

                            input_ids_before_cp = user_mini_batch["input_ids"]
                            position_ids_before_cp = user_mini_batch["position_ids"]
                            padding_mask_before_cp = padding_mask

                            if self.parallel_dims.cp_enabled:
                                [
                                    input_ids,
                                    position_ids,
                                    padding_mask,
                                ] = slice_inputs_for_ulysses(
                                    [input_ids, position_ids, padding_mask],
                                    self.parallel_dims.mesh["cp"],
                                )
                                user_mini_batch["position_ids"] = position_ids
                                user_mini_batch["input_ids"] = input_ids
                                if padding_mask is not None:
                                    user_mini_batch["padding_mask"] = padding_mask

                            if self.parallel_dims.pp_enabled:
                                raise NotImplementedError(
                                    "Pipeline Parallel not supported for PGMO"
                                )

                            model_out = self.model(**user_mini_batch)

                            if self.parallel_dims.cp_enabled:
                                user_mini_batch["position_ids"] = position_ids_before_cp
                                user_mini_batch["input_ids"] = input_ids_before_cp
                                if padding_mask_before_cp is not None:
                                    user_mini_batch["padding_mask"] = padding_mask_before_cp

                            raw_logits = (
                                model_out.logits
                                if hasattr(model_out, "logits")
                                else model_out
                            )
                            if self.config.train.train_policy.temperature > 1e-6:
                                raw_logits = (
                                    raw_logits
                                    / self.config.train.train_policy.temperature
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
                            current_advantages = (
                                logprob_masks * minibatched_advantages
                            )

                            if is_computing_ref:
                                assert i_mu == 0
                                self.ref_per_token_logps[local_mini_step] = (
                                    current_per_token_logprobs.detach()
                                )
                                local_mini_step += 1
                                continue
                            else:
                                if self.old_per_token_logps[local_mini_step] is None:
                                    assert i_mu == 0
                                    self.old_per_token_logps[local_mini_step] = (
                                        current_per_token_logprobs.detach()
                                    )
                                else:
                                    assert i_mu > 0

                                # ---- Standard GRPO loss ----
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

                                # ---- Contrastive loss (PGMO) ----
                                contrastive_loss_val = torch.tensor(
                                    0.0, device=self.device
                                )
                                if do_contrastive:
                                    from rl.utils.contrastive_loss import (
                                        compute_contrastive_loss_simple,
                                    )

                                    # Pool hidden states
                                    if hasattr(model_out, "hidden_states"):
                                        hidden = model_out.hidden_states[-1]
                                    elif hasattr(self.model, "vlm") and hasattr(
                                        self.model.vlm, "language_model"
                                    ):
                                        # Get last hidden from VLM forward
                                        hidden = raw_logits
                                    else:
                                        hidden = None

                                    if hidden is not None and hidden.dim() >= 2:
                                        pooled = hidden.mean(dim=1)  # [B, D]
                                        # Use rewards from the current mini-batch
                                        batch_rewards = torch.tensor(
                                            rewards_list[i:end],
                                            device=self.device,
                                        )
                                        contrastive_loss_val = (
                                            compute_contrastive_loss_simple(
                                                pooled,
                                                batch_rewards,
                                                temperature=self._pgmo_cfg[
                                                    "contrastive_temperature"
                                                ],
                                                weight=self._pgmo_cfg[
                                                    "contrastive_weight"
                                                ],
                                            )
                                        )
                                        contrastive_loss_sum += (
                                            contrastive_loss_val.item()
                                        )

                                # ---- Positive NLL (unchanged) ----
                                if (
                                    pos_coef_global is not None
                                    and pos_coef_global > 0.0
                                ):
                                    pos_flag_batch = self._positive_flags_t[i:end]
                                    pos_mask = pos_flag_batch.unsqueeze(1).expand_as(
                                        logprob_masks
                                    )
                                    pos_token_mask = pos_mask & logprob_masks
                                    if pos_token_mask.any():
                                        flat_mask = pos_token_mask[logprob_masks]
                                        l_nll = -current_per_token_logprobs[
                                            flat_mask
                                        ].mean()
                                        loss = loss + pos_coef_global * l_nll

                                # Combine with contrastive loss
                                total_loss_step = (
                                    loss + contrastive_loss_val
                                ) / num_mini_batch
                                per_token_loss = per_token_loss / num_mini_batch
                                kl_loss = kl_loss / num_mini_batch

                                total_loss_step.backward()
                                loss_sum += per_token_loss.item()
                                kl_loss_sum += kl_loss.item()
                                loss_count += 1
                                for key in metrics:
                                    self.metrics[key] += metrics[key]

                            self.mini_step += 1
                            local_mini_step += 1

                            if (
                                local_mini_step
                                % int(
                                    os.environ.get(
                                        "COSMOS_GRPO_STEP_INTERVAL", "10"
                                    )
                                )
                                == 0
                            ) and local_mini_step > 1:
                                all_reduced = True
                                grad_norm_sum += self.all_reduce_states(
                                    inter_policy_nccl
                                )
                            else:
                                all_reduced = False

                        if not is_computing_ref and not all_reduced:
                            grad_norm_sum += self.all_reduce_states(
                                inter_policy_nccl
                            )

        self.old_per_token_logps = []
        self.ref_per_token_logps = []
        end_event.record()

        self.lr_schedulers.step()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        contrastive_loss_avg = (
            (contrastive_loss_sum / loss_count)
            if loss_count > 0
            else contrastive_loss_sum
        )

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss = global_max_loss = loss.item()
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0
                report_data["train/iteration_time"] = iter_time
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                report_data["train/learning_rate"] = (
                    self.lr_schedulers.get_last_lr()[0]
                )
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss
                report_data["train/grad_norm"] = grad_norm_sum.item()
                report_data["train/local_loss"] = loss.item()
                report_data["train/reward_mean"] = advantages_t.mean().item()
                report_data["train/reward_std"] = advantages_t.std().item()
                if self._pgmo_cfg["enable_contrastive"]:
                    report_data["train/contrastive_loss"] = contrastive_loss_avg

                print(
                    f"[Step {current_step}] loss={loss.item():.6f}, "
                    f"reward={advantages_t.mean().item():.4f}, "
                    f"gn={grad_norm_sum.item():.4f}"
                    + (
                        f", c_loss={contrastive_loss_avg:.6f}"
                        if self._pgmo_cfg["enable_contrastive"]
                        else ""
                    )
                )

                # Adaptive KL stats
                if self._adaptive_kl_scheduler:
                    kl_stats = self._adaptive_kl_scheduler.get_stats()
                    for k, v in kl_stats.items():
                        report_data[f"train/{k}"] = v

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

                tb_dir = os.path.join(
                    os.environ.get("LOG_DIR", "/root/temp_log"), "tensorboard"
                )
                writer = _get_tb_writer(tb_dir)
                for key, val in report_data.items():
                    if isinstance(val, (int, float)):
                        writer.add_scalar(key, val, current_step)
                writer.flush()

        self._save_checkpoint(
            current_step, total_steps, remain_samples_num, is_master_replica
        )
        return report_data

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            return loss.mean()

        return fake_compute_loss
