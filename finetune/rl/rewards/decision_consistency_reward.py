# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision-trajectory consistency reward for Alpamayo RL.

This reward checks whether the COT's described ego decision matches the
actual GT trajectory behavior AND whether the COT decision matches the
predicted trajectory behavior. Unlike the old RAA reward which used regex
keyword matching (causing template-hacking), this uses a binary/differential
approach:

  1. Extract the GT decision from ground-truth trajectory (stop/go/nudge/
     yield/maintain/turn-left/turn-right).
  2. Extract the COT's described decision from its text.
  3. Check: COT decision == GT decision? → reward
  4. Check: COT decision == predicted trajectory behavior? → reward

This creates variance because different rollouts produce different decisions
with different accuracy. Template phrases like "maintain lane and follow
traffic" will score 0 when the GT says "stop and yield".
"""

from __future__ import annotations

import re
from typing import Any

import torch

from rl.rewards.coc_reward import extract_coc_text
from rl.rewards.reward_types import RewardComponents


# ---------------------------------------------------------------------------
# Decision taxonomy
# ---------------------------------------------------------------------------
DECISION_PATTERNS: dict[str, list[str]] = {
    "stop": [
        r"\bstop\b(?! at| for|ping)", r"\bhalt\b", r"\bwait\b",
        r"\bstand still\b", r"\bremain stationary\b",
    ],
    "yield": [
        r"\byield\b", r"\bgive way\b", r"\blet\s+\w+\s+(pass|go|cross|proceed)\b",
        r"\bwait for\b", r"\bhold\b(?!\s+steady)",
    ],
    "slow_down": [
        r"\bslow down\b", r"\bdecelerate\b", r"\bbrake\b",
        r"\breduce speed\b", r"\breduce\s+(my|our|the)\s+speed\b",
    ],
    "nudge": [
        r"\bnudge\b", r"\bcreep\b", r"\bslow(?:ly)?\s+(?:move|advance|proceed|go|forward)\b",
        r"\bcarefully\s+proceed\b", r"\bcautiously\s+advance\b",
        r"\bmove slowly\b", r"\binch forward\b",
    ],
    "maintain": [
        r"\bmaintain\b(?!.*distance)", r"\bkeep\s+(?:going|moving|speed|lane)\b",
        r"\bcontinue\b(?!\s+to\s+slow|to\s+stop|to\s+yield)",
        r"\bcruise\b", r"\bfollow\b(?!.*distance)", r"\bsteady\b",
        r"\bconstant speed\b",
    ],
    "accelerate": [
        r"\baccelerate\b", r"\bspeed up\b", r"\bincrease speed\b",
        r"\bgo faster\b", r"\bresume\b(?!.*cautious)",
    ],
    "turn_left": [
        r"\bturn left\b", r"\bleft turn\b", r"\bsteer left\b",
        r"\bchange\s+lane\s+to\s+the\s+left\b", r"\bmove left\b",
        r"\bveer left\b",
    ],
    "turn_right": [
        r"\bturn right\b", r"\bright turn\b", r"\bsteer right\b",
        r"\bchange\s+lane\s+to\s+the\s+right\b", r"\bmove right\b",
        r"\bveer right\b",
    ],
    "lane_change": [
        r"\bchange lane\b", r"\bswitch lane\b", r"\bmerge\b",
        r"\bshift\s+to\b", r"\bmove\s+to\b(?!\s+the\s+(left|right))",
    ],
}


def _extract_gt_decision(
    gt_future_xyz: torch.Tensor | None,
    gt_future_rot: torch.Tensor | None,
) -> dict[str, float]:
    """Extract the actual ego decision from GT trajectory.

    Returns a dict of decision confidence scores, with the primary decision
    having a high value (>0.5) and others near 0.
    """
    if gt_future_xyz is None:
        return {"maintain": 1.0}

    xyz = gt_future_xyz
    while xyz.ndim > 2:
        xyz = xyz[0]

    if xyz.shape[0] < 2:
        return {"stop": 1.0}

    dx = xyz[1:, 0] - xyz[:-1, 0]
    dy = xyz[1:, 1] - xyz[:-1, 1]
    speeds = torch.sqrt(dx**2 + dy**2)

    avg_speed = float(speeds.mean().item())
    speed_start = float(speeds[:5].mean().item()) if speeds.numel() >= 5 else avg_speed
    speed_end = float(speeds[-5:].mean().item()) if speeds.numel() >= 5 else avg_speed

    displacement = float(torch.linalg.norm(xyz[-1, :2] - xyz[0, :2]).item())
    lateral = float((xyz[-1, 1] - xyz[0, 1]).item())

    speed_change = (speed_end - speed_start) / max(speed_start, 0.1)

    # Heading change for turns
    heading_delta = 0.0
    if gt_future_rot is not None:
        rot = gt_future_rot
        while rot.ndim > 3:
            rot = rot[0]
        if rot.shape[0] >= 2:
            heading = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
            dh = heading[-1] - heading[0]
            heading_delta = float(torch.atan2(torch.sin(dh), torch.cos(dh)).item())

    decisions: dict[str, float] = {}

    # STOP: avg speed very low, displacement minimal
    if avg_speed < 0.25 and displacement < 1.5:
        decisions["stop"] = 1.0
    elif avg_speed < 0.3:
        decisions["stop"] = 0.5

    # YIELD: very low speed but slight movement (yielding then proceeding)
    if 0.2 < avg_speed < 0.6 and displacement < 3.0:
        decisions["yield"] = max(0.0, 1.0 - avg_speed / 0.6)

    # NUDGE: slow but moving forward
    if 0.15 < avg_speed < 0.5 and displacement > 1.0 and speed_change > -0.05:
        decisions["nudge"] = 0.7

    # SLOW DOWN: speed decreasing significantly
    if speed_change < -0.15 and avg_speed > 0.3:
        decisions["slow_down"] = float(min(1.0, abs(speed_change) / 0.3))

    # MAINTAIN: constant speed, straight
    if avg_speed >= 0.8 and abs(speed_change) < 0.15 and abs(lateral) < 0.8:
        decisions["maintain"] = 1.0
    elif avg_speed >= 0.5 and abs(speed_change) < 0.2:
        decisions["maintain"] = 0.5

    # ACCELERATE: speed increasing
    if speed_change > 0.15:
        decisions["accelerate"] = float(min(1.0, speed_change / 0.3))

    # TURN LEFT
    if lateral < -0.6 or heading_delta > 0.18:
        decisions["turn_left"] = float(min(1.0, max(abs(lateral), abs(heading_delta)) / 1.0))

    # TURN RIGHT
    if lateral > 0.6 or heading_delta < -0.18:
        decisions["turn_right"] = float(min(1.0, max(abs(lateral), abs(heading_delta)) / 1.0))

    # If no decision stands out, default to maintain
    if not decisions or max(decisions.values()) < 0.3:
        decisions["maintain"] = 1.0

    return decisions


def _extract_cot_decision(coc_text: str) -> dict[str, float]:
    """Extract the COT's described ego decision from its text.

    Returns dict of decision confidence scores based on keyword matching.
    """
    if not coc_text or len(coc_text.strip()) < 10:
        return {}

    text_l = coc_text.lower()
    decisions: dict[str, float] = {}

    for decision, patterns in DECISION_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, text_l, re.IGNORECASE))
        if hits > 0:
            decisions[decision] = min(0.7 + 0.15 * (hits - 1), 1.0)

    return decisions


def _extract_predicted_decision(
    predicted_fut_xyz: torch.Tensor | None,
    predicted_fut_rot: torch.Tensor | None,
) -> dict[str, float]:
    """Extract the actual decision from predicted trajectory.

    Uses the same logic as _extract_gt_decision but on predicted trajectory.
    """
    return _extract_gt_decision(predicted_fut_xyz, predicted_fut_rot)


def _compute_decision_match(
    source_decisions: dict[str, float],
    target_decisions: dict[str, float],
) -> float:
    """Compute how well source decisions match target decisions.

    Returns a score in [0, 1]:
    - 1.0 if source and target agree on the primary decision
    - 0.0 if source mentions a decision that contradicts target
    - Penalty for source mentioning wrong decisions
    """
    if not source_decisions or not target_decisions:
        return 0.0

    # Find primary decision in target (GT)
    target_primary = max(target_decisions, key=target_decisions.get)
    target_primary_score = target_decisions[target_primary]

    # Opposition pairs: decisions that contradict each other
    oppositions = [
        ("stop", "accelerate"),
        ("stop", "maintain"),
        ("yield", "accelerate"),
        ("slow_down", "accelerate"),
        ("nudge", "stop"),
        ("turn_left", "turn_right"),
        ("maintain", "turn_left"),
        ("maintain", "turn_right"),
    ]

    def are_opposing(a: str, b: str) -> bool:
        for pair in oppositions:
            if (a == pair[0] and b == pair[1]) or (a == pair[1] and b == pair[0]):
                return True
        return False

    # Score source decision match to target
    match_score = 0.0

    # Bonus: source mentions the correct primary decision
    if target_primary in source_decisions:
        match_score += source_decisions[target_primary] * target_primary_score

    # Penalty: source mentions decisions opposing the target primary
    for src_decision, src_score in source_decisions.items():
        if are_opposing(src_decision, target_primary):
            match_score -= src_score * 0.5

    return max(0.0, min(1.0, match_score))


def compute_decision_consistency_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    predicted_fut_xyz: torch.Tensor | None = None,
    predicted_fut_rot: torch.Tensor | None = None,
) -> RewardComponents:
    """Compute decision-trajectory consistency reward.

    Checks three things:
    1. COT decision matches GT trajectory decision (most important)
    2. COT decision matches predicted trajectory decision
    3. Predicted trajectory decision matches GT trajectory decision

    This creates clear variance within GRPO groups because:
    - A rollout that correctly says "yield" when GT is yield → score ~0.8
    - A rollout that says "maintain" when GT is yield → score ~0.1
    - Template phrase "maintain lane" always scores ~0 when GT is stop/yield

    Args:
        to_be_evaluated: Full rollout completion string.
        reference: Reference data dict containing GT trajectory.
        predicted_fut_xyz: Predicted future trajectory (optional).
        predicted_fut_rot: Predicted future trajectory rotations (optional).

    Returns:
        RewardComponents with reward in [0, 1] and detailed metrics.
    """
    coc_text = extract_coc_text(to_be_evaluated)

    # Get GT decision
    gt_future_xyz = reference.get("ego_future_xyz", None)
    gt_future_rot = reference.get("ego_future_rot", None)

    if isinstance(gt_future_xyz, torch.Tensor):
        while gt_future_xyz.ndim > 2:
            gt_future_xyz = gt_future_xyz[0]
    if isinstance(gt_future_rot, torch.Tensor):
        while gt_future_rot.ndim > 3:
            gt_future_rot = gt_future_rot[0]

    gt_decision = _extract_gt_decision(gt_future_xyz, gt_future_rot)
    cot_decision = _extract_cot_decision(coc_text)

    # 1. COT vs GT decision match (most important)
    cot_gt_match = _compute_decision_match(cot_decision, gt_decision)

    # 2. COT vs predicted trajectory decision match
    pred_decision = _extract_predicted_decision(predicted_fut_xyz, predicted_fut_rot)
    cot_pred_match = _compute_decision_match(cot_decision, pred_decision)

    # 3. Predicted vs GT decision match
    pred_gt_match = _compute_decision_match(pred_decision, gt_decision)

    # If COT is empty/generic, give low baseline score (not zero to avoid
    # complete collapse, but low enough to differentiate from good reasoning)
    if not cot_decision:
        reward = 0.1  # Very low reward for no decision commitment
    else:
        # Weighted aggregation: COT-GT match is most important
        reward = (
            0.60 * cot_gt_match
            + 0.20 * cot_pred_match
            + 0.20 * pred_gt_match
        )

    # Get primary GT decision name for logging
    gt_primary = max(gt_decision, key=gt_decision.get) if gt_decision else "unknown"
    cot_primary = max(cot_decision, key=cot_decision.get) if cot_decision else "none"

    return RewardComponents(
        reward=float(reward),
        metrics={
            "decision_consistency_score": float(reward),
            "cot_gt_match": float(cot_gt_match),
            "cot_pred_match": float(cot_pred_match),
            "pred_gt_match": float(pred_gt_match),
            "gt_primary_decision": float(hash(gt_primary) % 100),  # encoded for logging
            "cot_primary_decision": float(hash(cot_primary) % 100),
            "gt_is_stopped": float(gt_decision.get("stop", 0.0)),
            "gt_is_yield": float(gt_decision.get("yield", 0.0)),
            "gt_is_nudge": float(gt_decision.get("nudge", 0.0)),
            "gt_is_maintain": float(gt_decision.get("maintain", 0.0)),
            "gt_is_slow_down": float(gt_decision.get("slow_down", 0.0)),
            "gt_is_accelerate": float(gt_decision.get("accelerate", 0.0)),
            "gt_is_turn_left": float(gt_decision.get("turn_left", 0.0)),
            "gt_is_turn_right": float(gt_decision.get("turn_right", 0.0)),
            "cot_has_decision": float(len(cot_decision) > 0),
            "cot_num_decisions": float(len(cot_decision)),
        },
    )