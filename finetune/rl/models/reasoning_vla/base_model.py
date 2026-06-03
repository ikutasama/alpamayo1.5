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


"""Wrapped Reasoning VLA model implementation for RL training."""

from dataclasses import dataclass
from typing import Any

import einops
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.utils import ModelOutput

from alpamayo1_5.common import logging
from alpamayo1_5.models.base_model import ReasoningVLA, IGNORE_INDEX
from alpamayo1_5.models.token_utils import extract_text_tokens, extract_traj_tokens
from rl.models.reasoning_vla.config import RLWrapperReasoningVLAConfig

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


@dataclass
class ReasoningVLAOutput(ModelOutput):
    """Output of the ReasoningVLA model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    diffusion_loss: torch.FloatTensor | None = None


class RLWrapperReasoningVLA(ReasoningVLA):
    """RL Wrapped ReasoningVLA model."""

    config_class: type[RLWrapperReasoningVLAConfig] = RLWrapperReasoningVLAConfig

    def __init__(
        self,
        config: RLWrapperReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        """Initialize the model."""
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count)

        # Initialize diffusion-expert sub-modules (same as Alpamayo1_5)
        # These are needed when compute_diffusion_loss=True in RL training.
        # Config fields (action_space_cfg etc.) are stored as plain dicts from
        # checkpoint config.json; convert to OmegaConf for hydra instantiate.
        import copy
        import hydra.utils as hyu
        from omegaconf import OmegaConf
        from alpamayo1_5.action_space import ActionSpace
        from alpamayo1_5.diffusion.base import BaseDiffusion

        # Expert transformer (same architecture as VLM text backbone, no embed_tokens)
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if getattr(config, "expert_cfg", None) is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        # The diffusion expert does not support FlashAttention 2.
        if getattr(expert_config, "_attn_implementation", "flash_attention_2") == "flash_attention_2":
            expert_config._attn_implementation = "sdpa"
        self.expert = AutoModel.from_config(expert_config)
        del self.expert.embed_tokens

        # Action space (traj → action vector transformation)
        as_cfg = config.action_space_cfg
        if not OmegaConf.is_config(as_cfg):
            as_cfg = OmegaConf.create(as_cfg)
        self.action_space: ActionSpace = hyu.instantiate(as_cfg)

        # Diffusion module (flow matching)
        diff_cfg = config.diffusion_cfg
        if not OmegaConf.is_config(diff_cfg):
            diff_cfg = OmegaConf.create(diff_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            diff_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        # Action projection layers (noisy_x/t → expert embeds, expert hidden → action)
        inp_cfg = config.action_in_proj_cfg
        if not OmegaConf.is_config(inp_cfg):
            inp_cfg = OmegaConf.create(inp_cfg)
        self.action_in_proj = hyu.instantiate(
            inp_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )

        outp_cfg = config.action_out_proj_cfg
        if not OmegaConf.is_config(outp_cfg):
            outp_cfg = OmegaConf.create(outp_cfg)
        self.action_out_proj = hyu.instantiate(
            outp_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if getattr(config, "keep_same_dtype", True):
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments for gradient checkpointing.
        """
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def freeze_base_model_except_embeddings(self) -> None:
        """Only train the embeddings for new tokens."""
        for param in self.parameters():
            param.requires_grad = False

        self.vlm.language_model.embed_tokens.weight.requires_grad = True

        def reset_grad(grad: torch.Tensor) -> torch.Tensor:
            grad[: self.original_vocab_size] = 0
            return grad

        self.vlm.language_model.embed_tokens.weight.register_hook(reset_grad)

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for the next token prediction.

        Args:
            outputs: ModelOutput containing logits of shape (B, L, V)
            labels: [B, L]
            labels_mask: [B, L], indicates which tokens in the sequence are valid for loss
                computation
            token_mask: [V], indicates which tokens ids are valid for logits computation

        Returns:
            torch.Tensor: (,) loss value
        """
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        # Shift labels to the left by 1 position (predict next token)
        shift_labels = labels[..., 1:]
        # The logits should also be trimmed to match the shifted labels
        # NOTE: we clone the logits to avoid in-place operations if token_mask is present that will
        # modify the original
        shift_logits = outputs.logits[..., :-1, :].clone()

        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        loss = torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
            ),
            nan=0.0,
        )
        return loss

    def _process_traj_future_training(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        """Process the trajectory future data for diffusion training.

        Converts ego trajectory data to action space and constructs flow matching
        training data (noisy_x, timesteps, etc.).

        Args:
            traj_data: Dict with ego_history_xyz/rot and ego_future_xyz/rot tensors.

        Returns:
            Dict with keys 'x', 'noisy_x', 'timesteps', 'noise', 'is_drop_guidance'
            suitable for self.diffusion.compute_loss_from_pred.
        """
        ego_history_xyz = traj_data["ego_history_xyz"]
        ego_history_rot = traj_data["ego_history_rot"]
        ego_future_xyz = traj_data["ego_future_xyz"]
        ego_future_rot = traj_data["ego_future_rot"]
        action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz,
            traj_history_rot=ego_history_rot,
            traj_future_xyz=ego_future_xyz,
            traj_future_rot=ego_future_rot,
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        training_data: dict[str, Any] = self.diffusion.construct_training_data(action)
        return training_data

    def _compute_diffusion_expert_loss(
        self,
        vlm_outputs: Any,
        input_ids: torch.Tensor,
        traj_data: dict[str, Any],
        tokenized_data: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the diffusion expert (flow matching) loss.

        Runs the expert transformer on the VLM's KV cache + projected action
        embeddings, and returns the flow matching MSE loss. The KV cache is
        detached so gradients only flow through the expert/action_proj modules,
        not back into the VLM.

        This mirrors the SFT forward path in TrainableAlpamayo1_5 but is
        adapted for the RL setting where the VLM forward has already been done
        (we reuse the same VLM outputs from the GRPO token-level step).

        Args:
            vlm_outputs: VLM forward outputs (must have past_key_values).
            input_ids: The full input_ids sequence (including fused traj tokens).
            traj_data: Dict with ego_history_xyz/rot and ego_future_xyz/rot.
            tokenized_data: Remaining tokenized kwargs (pixel_values, etc.).

        Returns:
            Scalar flow matching MSE loss tensor.
        """
        batch_size = input_ids.shape[0]

        # 1. Construct flow matching training data from ground-truth future trajectory
        future_traj_data = self._process_traj_future_training(traj_data)

        # 2. Project noisy action + timestep into expert token embeddings
        action_embeds = self.action_in_proj(
            future_traj_data["noisy_x"], future_traj_data["timesteps"]
        )
        expert_embeds = action_embeds

        # 3. Locate <traj_future_start> position in input_ids
        future_start_token_id = self.config.traj_token_ids["future_start"]
        future_start_positions = (input_ids == future_start_token_id).nonzero(as_tuple=False)
        if future_start_positions.numel() == 0:
            logger.warning("[DiffusionRL] No <traj_future_start> found in input_ids; "
                           "skipping diffusion expert loss.")
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        # Take the LAST occurrence (in case history traj also has the token)
        last_traj_future_start_idx = future_start_positions[-1, 1] + 1

        # 4. Get KV cache and crop to <traj_future_start> position
        kv_cache = vlm_outputs.past_key_values
        # NOTE: We do NOT clone the KV cache. Instead we crop in-place and then
        # detach keys/values. After the expert forward, we restore the KV cache
        # to its original length so it doesn't affect subsequent operations.
        # This matches the SFT pattern in TrainableAlpamayo1_5.forward().
        original_kv_seq_len = kv_cache.get_seq_length()
        kv_cache.crop(last_traj_future_start_idx)

        # 5. Detach KV cache keys/values so gradient does NOT flow back to VLM
        #    VLM gets its own independent gradient via GRPO token-level loss.
        #    Expert gets gradient via advantage-weighted diffusion loss.
        for layer in kv_cache.layers:
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()

        # 6. Compute position IDs for the expert (Qwen2.5-VL 3-component RoPE)
        #    Same as SFT: position_ids = arange(n_tokens) + rope_deltas + kv_seq_len
        position_ids = torch.arange(expert_embeds.shape[1], device=expert_embeds.device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        # rope_deltas is set by Qwen2.5-VL during forward when use_cache=True
        rope_deltas = getattr(vlm_outputs, "rope_deltas", None)
        if rope_deltas is not None:
            delta = rope_deltas + kv_cache.get_seq_length()
            position_ids += delta.to(position_ids.device)
        else:
            # Fallback: no rope_deltas (rare, but happens if VLM model doesn't expose it)
            # Use the cropped KV cache length as offset
            logger.warning("[DiffusionRL] No rope_deltas in VLM outputs; using "
                           "KV cache seq_len as position offset.")
            position_ids += kv_cache.get_seq_length()

        # 7. Expert forward pass
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=expert_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )

        # 8. Project expert output back to action space and compute flow matching loss
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1]:]
        pred = self.action_out_proj(diffusion_out)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())
        diffusion_loss = self.diffusion.compute_loss_from_pred(
            training_data=future_traj_data,
            pred=pred,
        )

        # 9. Restore KV cache to original length (undo crop for safety)
        #    The KV cache is shared with the VLM outputs; restoring prevents
        #    side-effects on any downstream code that reads vlm_outputs.
        kv_cache.crop(original_kv_seq_len)

        return diffusion_loss

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        compute_diffusion_loss: bool = False,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Forward pass of the model.

        Args:
            tokenized_data: Tokenized input data dict.
            ego_history_xyz: History trajectory xyz [B, n_traj_group, T, 3].
            ego_history_rot: History trajectory rotation [B, n_traj_group, T, 3, 3] (rotation matrix).
            ego_future_xyz: Future trajectory xyz [B, n_traj_group, T, 3].
            ego_future_rot: Future trajectory rotation [B, n_traj_group, T, 3, 3] (rotation matrix).
            labels_mask: Mask for which tokens to include in loss.
            compute_diffusion_loss: If True, also compute diffusion expert loss
                (advantage weighting is done by the trainer, not here).

        Returns:
            ReasoningVLAOutput with VLM loss, logits, and optionally diffusion_loss.
        """
        # 1. tokenize trajectory and fuse into input_ids
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. get labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

# 3. vlm forward pass
        # Normal path: GRPO forward with gradient checkpointing, no KV cache needed.
        # Diffusion path: Need past_key_values for the expert. CRITICAL issue:
        #   torch.utils.checkpoint.checkpoint() DROPS non-tensor outputs like
        #   past_key_values regardless of no_grad mode. So we must temporarily
        #   disable gradient checkpointing on the VLM, then run forward in
        #   no_grad (VLM gradients come from GRPO, not from this forward).
        #   This is memory-safe: no_grad means no intermediate activations stored.
        vlm_kwargs = dict(tokenized_data)
        if compute_diffusion_loss:
            vlm_kwargs["use_cache"] = True
            # Temporarily disable gradient checkpointing — it drops past_key_values
            checkpointing_was_enabled = (
                hasattr(self.vlm, "is_gradient_checkpointing")
                and self.vlm.is_gradient_checkpointing
            )
            if checkpointing_was_enabled:
                self.vlm.gradient_checkpointing_disable()
            with torch.no_grad():
                outputs = self.vlm(input_ids=input_ids, labels=None, **vlm_kwargs)
            # Re-enable gradient checkpointing immediately
            if checkpointing_was_enabled:
                self.vlm.gradient_checkpointing_enable()
        else:
            try:
                outputs = self.vlm(input_ids=input_ids, labels=labels, **vlm_kwargs)
            except ValueError as e:
                import os
                rank = os.environ.get("RANK", "?")
                img_tok_id = None
                if hasattr(self.vlm, "config") and hasattr(self.vlm.config, "image_token_id"):
                    img_tok_id = self.vlm.config.image_token_id
                    img_tok_count = (input_ids == img_tok_id).sum().item()
                else:
                    img_tok_count = "N/A"
                pv = tokenized_data.get("pixel_values")
                gt = tokenized_data.get("image_grid_thw")
                pv_shape = pv.shape if pv is not None and hasattr(pv, "shape") else type(pv)
                gt_val = gt.tolist() if gt is not None and hasattr(gt, "tolist") else type(gt)
                print(f"[DEBUG] Crash at rank={rank}: input_ids.shape={input_ids.shape}, image_pad_count={img_tok_count}, pixel_values={pv_shape}, image_grid_thw={gt_val}, keys={list(tokenized_data.keys())}")
                raise

        # Compute VLM loss only for GRPO path (when not computing diffusion loss).
        # For diffusion path, VLM loss is meaningless (no_grad logits = no gradient),
        # and GRPO already handles VLM gradients via its own forward.
        if compute_diffusion_loss:
            outputs.loss = torch.tensor(0.0, device=input_ids.device)
        else:
            losses = {}
            # Identify trajectory tokens (tokens between traj_future and next special token)
            traj_mask = (
                (
                    (labels >= self.future_token_start_idx)
                    & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
                )
                | (labels == self.special_token_ids["traj_future_start"])
                | (labels == self.special_token_ids["traj_future_end"])
            )
            losses["future_traj"] = self._compute_next_token_loss(
                outputs, labels, traj_mask
            ) * self.config.loss_weights.get("future_traj", 1.0)
            labels[traj_mask] = IGNORE_INDEX

            # Include all other tokens in the loss
            losses["others"] = self._compute_next_token_loss(
                outputs, labels, labels != IGNORE_INDEX
            ) * self.config.loss_weights.get("others", 1.0)

            # Replace the original loss
            outputs.loss = sum(losses.values())

        # 4. Optionally compute diffusion expert loss
        diffusion_loss = None
        if compute_diffusion_loss and ego_future_xyz is not None and ego_future_rot is not None:
            # VLM outputs must have past_key_values for diffusion expert path
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                pkv_layers = len(outputs.past_key_values)
                pkv_first = outputs.past_key_values[0]
                # DynamicCache has .key attr; legacy tuple format is (key, value)
                if hasattr(pkv_first, "key"):
                    pkv_seq = pkv_first.key.shape[2]
                else:
                    pkv_seq = pkv_first[0].shape[2]  # tuple: (key_tensor, value_tensor)
                logger.warning(
                    f"[DiffusionRL] VLM past_key_values available: "
                    f"layers={pkv_layers}, seq_len={pkv_seq}"
                )
                diffusion_loss = self._compute_diffusion_expert_loss(
                    vlm_outputs=outputs,
                    input_ids=input_ids,
                    traj_data=traj_data,
                    tokenized_data=tokenized_data,
                )
            else:
                logger.warning(
                    "[DiffusionRL] VLM outputs lack past_key_values; "
                    "cannot compute diffusion expert loss. Need use_cache=True."
                )

        return ReasoningVLAOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            diffusion_loss=diffusion_loss,
        )

    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[Any, Any, Any, dict[str, list[str]]]
    ):
        """Sample trajectories from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        assert max_generation_length >= self.config.tokens_per_future_traj
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = n_samples_total
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generated = self.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        # remove input ids from the generated sequences
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        # extract trajectory tokens from generated sequences
        traj_token_ids = extract_traj_tokens(
            generated_tokens,
            self.special_token_ids,
            self.config.tokens_per_future_traj,
            self.future_token_start_idx,
            self.traj_tokenizer.vocab_size,
        )

        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=einops.repeat(
                ego_history_xyz[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            hist_rot=einops.repeat(
                ego_history_rot[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            tokens=traj_token_ids,
        )
        pred_xyz = einops.rearrange(
            pred_xyz,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        pred_rot = einops.rearrange(
            pred_rot,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        logger.warning(
            "logprob computation is not implemented; returning zeros. "
            "Do not use these values for ranking or importance weighting."
        )
        logprob = torch.zeros_like(pred_xyz[..., 0])

        # return additional information
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, generated_tokens)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, logprob, extra
        return pred_xyz, pred_rot, logprob


# Register the model with Auto classes
AutoConfig.register("alpamayo_reasoning_vla", RLWrapperReasoningVLAConfig)
AutoModel.register(RLWrapperReasoningVLAConfig, RLWrapperReasoningVLA)