# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for Causal Flow-GRPO reward helpers and token masks."""

from __future__ import annotations

import torch

from alpamayo1_5.models.base_model import SPECIAL_TOKENS
from rl.models.reasoning_vla.data_packer import build_completion_token_masks
from rl.models.reasoning_vla.trainer import _normalize_grouped
from rl.rewards.coc_reward import compute_coc_reward, extract_coc_sections
from rl.rewards.consistency_reward import compute_reasoning_action_consistency
from rl.rewards.hcc_reward import _compute_grounded_fact_score
from rl.rewards.raa_reward import compute_raa_score
from rl.rewards.risk_reward import compute_risk_reward


def test_coc_reward_parses_completion() -> None:
    text = (
        f"{SPECIAL_TOKENS['cot_start']}Slow down because a pedestrian is near the crosswalk."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<i1><i2>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    sections = extract_coc_sections(text)
    assert sections["has_cot_end"] == 1.0
    assert sections["has_traj_start"] == 1.0
    assert "pedestrian" in str(sections["reasoning"]).lower()

    reward = compute_coc_reward(text, {"cot": "slow down for pedestrian"})
    assert 0.0 <= reward.reward <= 1.0
    assert reward.metrics["risk_keyword_score"] == 1.0
    assert reward.metrics["action_keyword_score"] == 1.0


def test_completion_token_masks_are_disjoint_enough() -> None:
    ids = torch.tensor([1, 2, 3, 10, 11, 12, 13, 14, 15, 16])
    masks = build_completion_token_masks(
        ids,
        prompt_len=3,
        special_token_ids={
            "cot_end": 12,
            "traj_future_start": 13,
            "traj_future_end": 16,
        },
    )
    assert masks["logprob_masks"].sum().item() == 7
    assert masks["coc_logprob_masks"].sum().item() == 3
    assert masks["traj_logprob_masks"].sum().item() == 4
    assert masks["format_logprob_masks"].sum().item() == 3
    assert not masks["coc_logprob_masks"][:3].any()
    assert not masks["traj_logprob_masks"][:3].any()


def test_consistency_and_risk_rewards_accept_tensor_inputs() -> None:
    text = f"Slow down and keep straight.{SPECIAL_TOKENS['cot_end']}"
    predicted = torch.zeros((1, 64, 3), dtype=torch.float32)
    predicted[0, :, 0] = torch.linspace(0.0, 4.0, 64)

    consistency = compute_reasoning_action_consistency(text, predicted)
    risk = compute_risk_reward(predicted)

    assert 0.0 <= consistency.reward <= 1.0
    assert risk.reward == 0.0
    assert risk.metrics["valid"] == 1.0


def test_grouped_advantage_normalization_uses_prompt_groups() -> None:
    values = [1.0, 3.0, 10.0]
    fallback = [0.1, 0.2, 0.3]
    payloads = [
        {"split": "train", "idx": "7"},
        {"split": "train", "idx": "7"},
        {"split": "train", "idx": "8"},
    ]
    normalized = _normalize_grouped(values, payloads, fallback)
    assert normalized[0] < 0.0
    assert normalized[1] > 0.0
    assert normalized[2] == fallback[2]


def test_raa_does_not_reward_empty_intent() -> None:
    predicted = torch.zeros((64, 3), dtype=torch.float32)
    predicted[:, 0] = torch.linspace(0.0, 8.0, 64)
    rot = torch.eye(3).repeat(64, 1, 1)

    scores = compute_raa_score("The scene requires careful driving.", predicted, rot)

    assert scores["raa_active_constraints"] == 0.0
    assert scores["raa_score"] < 0.3


def test_grounded_fact_score_prefers_matching_action_facts() -> None:
    gt = torch.zeros((64, 3), dtype=torch.float32)
    gt[:, 0] = torch.linspace(0.0, 8.0, 64)
    rot = torch.eye(3).repeat(64, 1, 1)

    good, good_info = _compute_grounded_fact_score(
        "I observe the road ahead and will maintain a steady speed while keeping lane for safety.",
        {},
        gt,
        rot,
    )
    bad, bad_info = _compute_grounded_fact_score(
        "I will turn left and brake hard.",
        {},
        gt,
        rot,
    )

    assert good > bad
    assert good_info["grounded_fact_coverage"] > bad_info["grounded_fact_coverage"]
