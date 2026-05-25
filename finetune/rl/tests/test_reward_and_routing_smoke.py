# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for grounded reward helpers and token masks (v2)."""

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
from rl.rewards.obstacle_grounding_reward import (
    compute_obstacle_grounding_reward,
    _find_nearest_obstacle_mentions,
    OBSTACLE_TYPE_KEYWORDS,
)
from rl.rewards.decision_consistency_reward import (
    compute_decision_consistency_reward,
    _extract_gt_decision,
    _extract_cot_decision,
    _compute_decision_match,
)


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


# ---------------------------------------------------------------------------
# NEW v2 tests: obstacle grounding reward
# ---------------------------------------------------------------------------

def test_obstacle_grounding_reward_correct_pedestrian() -> None:
    """COT correctly mentions pedestrian when GT obstacle is a pedestrian."""
    text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "I see a pedestrian near the crosswalk ahead. This is a high threat hazard, "
        "I should slow down and yield to let them cross safely."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    reference = {
        "obstacle_info": {
            "types": ["pedestrian", "vehicle"],
            "positions": [[2.5, 1.0], [8.0, -0.5]],
            "distances": [2.5, 8.0],
            "is_high_threat": [True, False],
            "directions_from_ego": ["ahead", "ahead"],
        },
    }
    result = compute_obstacle_grounding_reward(text, reference)

    # Should score well: correctly identifies pedestrian as threat
    assert result.reward > 0.3
    assert result.metrics["obstacle_type_match"] > 0.3
    assert result.metrics["obstacle_grounding_score"] > 0.3


def test_obstacle_grounding_reward_wrong_type() -> None:
    """COT mentions vehicle when GT obstacle is actually a pedestrian."""
    text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "I observe a vehicle ahead in the lane. It is a risk so I will slow down."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    reference = {
        "obstacle_info": {
            "types": ["pedestrian"],
            "positions": [[2.5, 1.0]],
            "distances": [2.5],
            "is_high_threat": [True],
            "directions_from_ego": ["ahead"],
        },
    }
    result = compute_obstacle_grounding_reward(text, reference)

    # "vehicle" gets partial match (0.5) since "vehicle" is a generic fallback
    # But it shouldn't score as well as saying "pedestrian"
    assert result.metrics["obstacle_type_match"] < 0.8


def test_obstacle_grounding_reward_no_obstacles_in_scene() -> None:
    """No GT obstacles: COT should not hallucinate threats."""
    # Good COT: no obstacle mentions
    good_text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "The road ahead is clear. I will maintain my current speed and stay in lane."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    reference_no_obstacles = {"obstacle_info": None}

    good_result = compute_obstacle_grounding_reward(good_text, reference_no_obstacles)
    # Should get a neutral/baseline score for not hallucinating
    assert good_result.reward >= 0.0

    # Bad COT: hallucinating obstacles
    bad_text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "I see a vehicle and pedestrian hazard ahead. I must brake and yield."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    bad_result = compute_obstacle_grounding_reward(bad_text, reference_no_obstacles)
    # Should get a penalty for hallucinating
    assert bad_result.reward < good_result.reward


def test_obstacle_grounding_reward_empty_cot() -> None:
    """Empty COT should get zero obstacle grounding score."""
    text = f"{SPECIAL_TOKENS['cot_end']}{SPECIAL_TOKENS['traj_future_start']}<traj>{SPECIAL_TOKENS['traj_future_end']}"
    reference = {"obstacle_info": None}
    result = compute_obstacle_grounding_reward(text, reference)
    assert result.reward == 0.0


# ---------------------------------------------------------------------------
# NEW v2 tests: decision consistency reward
# ---------------------------------------------------------------------------

def test_decision_consistency_reward_yield_scene() -> None:
    """GT trajectory shows yielding (very slow), COT says yield → high score."""
    text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "I should yield and wait for the pedestrian to cross before proceeding."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    # GT: slow movement (yielding behavior)
    gt_xyz = torch.zeros((64, 3), dtype=torch.float32)
    gt_xyz[:, 0] = torch.linspace(0.0, 0.8, 64)  # very slow forward
    gt_rot = torch.eye(3).repeat(64, 1, 1)

    reference = {
        "ego_future_xyz": gt_xyz.unsqueeze(0).unsqueeze(0),
        "ego_future_rot": gt_rot.unsqueeze(0).unsqueeze(0),
    }

    result = compute_decision_consistency_reward(text, reference)
    # COT says yield, GT shows yield → should be positive
    assert result.reward > 0.3
    assert result.metrics["cot_gt_match"] > 0.2


def test_decision_consistency_reward_wrong_decision() -> None:
    """GT shows yielding but COT says maintain/accelerate → low score."""
    text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "I will maintain my speed and continue forward through the intersection."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    # GT: slow movement (yielding)
    gt_xyz = torch.zeros((64, 3), dtype=torch.float32)
    gt_xyz[:, 0] = torch.linspace(0.0, 0.5, 64)
    gt_rot = torch.eye(3).repeat(64, 1, 1)

    reference = {
        "ego_future_xyz": gt_xyz.unsqueeze(0).unsqueeze(0),
        "ego_future_rot": gt_rot.unsqueeze(0).unsqueeze(0),
    }

    result = compute_decision_consistency_reward(text, reference)
    # COT says maintain/accelerate, GT shows yield → should be low
    assert result.reward < 0.4


def test_decision_consistency_reward_stop_scene() -> None:
    """GT shows stopping, COT says stop → high score."""
    text = (
        f"{SPECIAL_TOKENS['cot_start']}"
        "I should stop here and wait for the red light."
        f"{SPECIAL_TOKENS['cot_end']}"
        f"{SPECIAL_TOKENS['traj_future_start']}<traj_data>"
        f"{SPECIAL_TOKENS['traj_future_end']}"
    )
    # GT: nearly stopped
    gt_xyz = torch.zeros((64, 3), dtype=torch.float32)
    gt_xyz[:, 0] = torch.linspace(0.0, 0.2, 64)  # almost no movement
    gt_rot = torch.eye(3).repeat(64, 1, 1)

    reference = {
        "ego_future_xyz": gt_xyz.unsqueeze(0).unsqueeze(0),
        "ego_future_rot": gt_rot.unsqueeze(0).unsqueeze(0),
    }

    result = compute_decision_consistency_reward(text, reference)
    assert result.reward > 0.3
    assert result.metrics["cot_gt_match"] > 0.3


def test_decision_consistency_reward_empty_cot() -> None:
    """Empty COT should get low baseline score."""
    text = f"{SPECIAL_TOKENS['cot_end']}{SPECIAL_TOKENS['traj_future_start']}<traj>{SPECIAL_TOKENS['traj_future_end']}"
    gt_xyz = torch.zeros((64, 3), dtype=torch.float32)
    gt_xyz[:, 0] = torch.linspace(0.0, 4.0, 64)
    gt_rot = torch.eye(3).repeat(64, 1, 1)

    reference = {
        "ego_future_xyz": gt_xyz.unsqueeze(0).unsqueeze(0),
        "ego_future_rot": gt_rot.unsqueeze(0).unsqueeze(0),
    }

    result = compute_decision_consistency_reward(text, reference)
    # Empty COT → low baseline (0.1)
    assert result.reward == 0.1


def test_gt_decision_extract_stop() -> None:
    """Test that GT decision extraction correctly identifies stop behavior."""
    gt_xyz = torch.zeros((64, 3), dtype=torch.float32)
    gt_xyz[:, 0] = torch.linspace(0.0, 0.15, 64)  # almost no movement
    gt_rot = torch.eye(3).repeat(64, 1, 1)

    decisions = _extract_gt_decision(gt_xyz, gt_rot)
    assert decisions.get("stop", 0.0) > 0.5


def test_gt_decision_extract_maintain() -> None:
    """Test that GT decision extraction correctly identifies maintain behavior."""
    gt_xyz = torch.zeros((64, 3), dtype=torch.float32)
    gt_xyz[:, 0] = torch.linspace(0.0, 8.0, 64)  # consistent forward speed
    gt_rot = torch.eye(3).repeat(64, 1, 1)

    decisions = _extract_gt_decision(gt_xyz, gt_rot)
    assert decisions.get("maintain", 0.0) > 0.5


def test_cot_decision_extract() -> None:
    """Test COT decision extraction from text."""
    # Yield decision
    decisions = _extract_cot_decision("I should yield and wait for the pedestrian to cross.")
    assert "yield" in decisions
    assert decisions["yield"] > 0.0

    # Maintain decision
    decisions = _extract_cot_decision("I will maintain my current speed and stay in lane.")
    assert "maintain" in decisions

    # Stop decision
    decisions = _extract_cot_decision("I must stop here and wait.")
    assert "stop" in decisions


def test_decision_match_same() -> None:
    """Test decision match score when source and target agree."""
    source = {"yield": 0.9}
    target = {"yield": 1.0}
    match = _compute_decision_match(source, target)
    assert match > 0.5


def test_decision_match_opposing() -> None:
    """Test decision match score when source contradicts target."""
    source = {"accelerate": 0.9}
    target = {"stop": 1.0}
    match = _compute_decision_match(source, target)
    assert match < 0.3


# ---------------------------------------------------------------------------
# NEW v2 tests: obstacle mention matching
# ---------------------------------------------------------------------------

def test_obstacle_mention_matching_with_gt_obstacles() -> None:
    """Test that obstacle mention matching works with GT obstacles."""
    obstacles = [
        {"type": "pedestrian", "distance": 3.0, "direction": "ahead", "is_high_threat": True},
        {"type": "vehicle", "distance": 10.0, "direction": "ahead", "is_high_threat": False},
    ]

    # Good COT: mentions pedestrian (closest, high-threat)
    good_scores = _find_nearest_obstacle_mentions(
        "A pedestrian ahead near the crosswalk. This is a high threat.",
        obstacles,
    )
    assert good_scores["obstacle_type_match"] > 0.3

    # Bad COT: only mentions vehicle (farther, lower threat)
    bad_scores = _find_nearest_obstacle_mentions(
        "A vehicle ahead in the lane.",
        obstacles,
    )
    # Should have lower type match because it missed the closest obstacle (pedestrian)
    assert good_scores["obstacle_type_match"] >= bad_scores["obstacle_type_match"]