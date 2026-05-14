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

"""Reasoning-Action Alignment (RAA) reward for HCC-RM framework.

Measures how well the generated trajectory follows the intent expressed
in the Chain-of-Causation reasoning text. This is the key innovation
for bridging the CoC-trajectory gap.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from rl.rewards.coc_reward import extract_coc_text


# ---------------------------------------------------------------------------
# Constraint extraction patterns
# ---------------------------------------------------------------------------

DECELERATE_PATTERNS: list[str] = [
    r"\b(slow down|decelerate|brake|reduce speed|stop|halt)\b",
    r"\b(yield|give way|wait)\b",
]

ACCELERATE_PATTERNS: list[str] = [
    r"\b(speed up|accelerate|increase speed|proceed|go|advance)\b",
    r"\b(resume|continue|move forward)\b",
]

TURN_LEFT_PATTERNS: list[str] = [
    r"\b(turn left|left turn|steer left|veer left|change to left)\b",
]

TURN_RIGHT_PATTERNS: list[str] = [
    r"\b(turn right|right turn|steer right|veer right|change to right)\b",
]

LANE_CHANGE_PATTERNS: list[str] = [
    r"\b(change lane|switch lane|merge|move to|shift to)\b",
]

MAINTAIN_PATTERNS: list[str] = [
    r"\b(maintain|keep|stay|hold|remain|continue)\b",
    r"\b(follow|track|cruise)\b",
]

KEEP_DISTANCE_PATTERNS: list[str] = [
    r"\b(keep distance|maintain distance|safe distance|following distance|stay behind)\b",
    r"\b(gap|clearance|buffer|room)\b",
]


def extract_behavioral_constraints(coc_text: str) -> dict[str, float]:
    """Extract behavioral constraints from CoC text.

    Parses the reasoning text to identify intended driving behaviors
    and returns a dict of constraint strengths.

    Args:
        coc_text: The extracted CoC reasoning text.

    Returns:
        Dict mapping constraint names to confidence scores in [0, 1].
        Keys include: ``decelerate``, ``accelerate``, ``turn_left``,
        ``turn_right``, ``lane_change``, ``maintain``, ``keep_distance``,
        ``has_navigation``, ``has_obstacle_avoidance``.
    """
    constraints: dict[str, float] = {}

    # Longitudinal actions
    constraints["decelerate"] = _match_confidence(coc_text, DECELERATE_PATTERNS)
    constraints["accelerate"] = _match_confidence(coc_text, ACCELERATE_PATTERNS)

    # Lateral actions
    constraints["turn_left"] = _match_confidence(coc_text, TURN_LEFT_PATTERNS)
    constraints["turn_right"] = _match_confidence(coc_text, TURN_RIGHT_PATTERNS)
    constraints["lane_change"] = _match_confidence(coc_text, LANE_CHANGE_PATTERNS)

    # Steady-state
    constraints["maintain"] = _match_confidence(coc_text, MAINTAIN_PATTERNS)
    constraints["keep_distance"] = _match_confidence(coc_text, KEEP_DISTANCE_PATTERNS)

    # Additional context
    constraints["has_obstacle_avoidance"] = _match_confidence(
        coc_text, [
            r"\b(obstacle|object|barrier|block|hazard|pedestrian|cyclist|vehicle ahead)\b",
        ]
    )
    constraints["has_navigation"] = _match_confidence(
        coc_text, [
            r"\b(navigation|route|destination|waypoint|map|direction|heading)\b",
        ]
    )

    return constraints


def _match_confidence(text: str, patterns: list[str]) -> float:
    """Compute confidence that the text expresses a given behavior.

    Returns 0.0 if no pattern matches, scales up to 1.0 with multiple matches.
    """
    if not text:
        return 0.0
    hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    # Sigmoid-like scaling: 1 hit → 0.7, 2 hits → 0.9, 3+ → 1.0
    if hits == 0:
        return 0.0
    return min(0.7 + 0.15 * (hits - 1), 1.0)


def infer_trajectory_behavior(
    predicted_xyz: torch.Tensor,
    predicted_rot: torch.Tensor,
) -> dict[str, float]:
    """Infer actual driving behavior from predicted trajectory.

    Args:
        predicted_xyz: Predicted future positions, shape [T, 3] or [B, T, 3].
        predicted_rot: Predicted future rotations, shape [T, 3, 3] or [B, T, 3, 3].

    Returns:
        Dict mapping behavior names to confidence scores in [0, 1].
    """
    # Ensure 2D shape [T, 3]
    if predicted_xyz.dim() == 3:
        predicted_xyz = predicted_xyz[0]
    if predicted_rot.dim() == 4:
        predicted_rot = predicted_rot[0]

    # Extract heading from rotation matrix
    heading = torch.atan2(predicted_rot[..., 1, 0], predicted_rot[..., 0, 0])

    # Compute velocities
    dx = predicted_xyz[1:, 0] - predicted_xyz[:-1, 0]
    dy = predicted_xyz[1:, 1] - predicted_xyz[:-1, 1]
    speeds = torch.sqrt(dx**2 + dy**2)

    # Heading changes
    dh = heading[1:] - heading[:-1]
    # Wrap to [-pi, pi]
    dh = torch.atan2(torch.sin(dh), torch.cos(dh))

    behaviors: dict[str, float] = {}

    # Deceleration: speed decreasing over trajectory
    speed_start = speeds[:5].mean()    # First 0.5s average
    speed_end = speeds[-5:].mean()     # Last 0.5s average
    if speed_start > 0.1:
        speed_change = (speed_end - speed_start) / speed_start
        behaviors["decelerate"] = float(torch.sigmoid(-speed_change * 10.0).item())
        behaviors["accelerate"] = float(torch.sigmoid(speed_change * 10.0).item())
    else:
        behaviors["decelerate"] = 0.0
        behaviors["accelerate"] = 0.0

    # Turning: cumulative heading change
    total_dh = dh.sum().item()
    behaviors["turn_left"] = float(torch.sigmoid(total_dh * 5.0).item())
    behaviors["turn_right"] = float(torch.sigmoid(-total_dh * 5.0).item())

    # Lane change: lateral displacement
    total_dy = (predicted_xyz[-1, 1] - predicted_xyz[0, 1]).item()
    if abs(total_dy) > 0.5:  # At least 0.5m lateral movement
        behaviors["lane_change"] = min(abs(total_dy) / 3.5, 1.0)
    else:
        behaviors["lane_change"] = 0.0

    # Maintain: low speed variance and low heading variance
    if len(speeds) > 1:
        speed_std = speeds.std().item()
        behaviors["maintain"] = float(1.0 - min(speed_std / 2.0, 1.0))

        if behaviors["decelerate"] < 0.3 and behaviors["accelerate"] < 0.3:
            behaviors["maintain"] = max(behaviors["maintain"], 0.6)
    else:
        behaviors["maintain"] = 0.5

    # Keep distance: stable speed with small changes
    behaviors["keep_distance"] = behaviors.get("maintain", 0.5) * (
        1.0 - max(behaviors.get("decelerate", 0), behaviors.get("accelerate", 0))
    )

    return behaviors


def compute_raa_score(
    coc_text: str,
    predicted_xyz: torch.Tensor,
    predicted_rot: torch.Tensor,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute Reasoning-Action Alignment score.

    This is the core of the RAA reward: it measures how well the trajectory
    follows the intent expressed in the reasoning text.

    Args:
        coc_text: The Chain-of-Causation reasoning text.
        predicted_xyz: Predicted future XYZ positions.
        predicted_rot: Predicted future rotation matrices.
        weights: Optional per-constraint weights. Defaults to balanced.

    Returns:
        Dict with per-constraint alignment scores and the aggregate ``raa_score``.
        Each alignment score is in [0, 1] where 1 means perfectly aligned.
    """
    if weights is None:
        weights = {
            "decelerate": 0.18,
            "accelerate": 0.12,
            "turn_left": 0.15,
            "turn_right": 0.15,
            "lane_change": 0.15,
            "maintain": 0.10,
            "keep_distance": 0.15,
        }

    # Extract intended behavior from CoC text
    constraints = extract_behavioral_constraints(coc_text)

    # Infer actual behavior from trajectory
    inferred = infer_trajectory_behavior(predicted_xyz, predicted_rot)

    # Compute per-constraint alignment
    alignments: dict[str, float] = {}
    for key in weights:
        intended = constraints.get(key, 0.0)
        actual = inferred.get(key, 0.0)
        # Alignment = 1 - |intended - actual|, but only for constraints
        # that were actually intended (intended > 0.3 threshold)
        if intended > 0.3:
            # For intended behaviors, we want actual to be high
            alignments[f"raa_{key}"] = 1.0 - abs(intended - max(actual, 0.0))
        else:
            # For non-intended behaviors, actual being low is fine
            # But if actual is high and intended is low, penalize
            if actual > 0.5 and intended < 0.2:
                alignments[f"raa_{key}"] = 1.0 - actual
            else:
                alignments[f"raa_{key}"] = 1.0

    # Weighted aggregate
    weighted_sum = 0.0
    total_weight = 0.0
    for key, w in weights.items():
        weighted_sum += w * alignments[f"raa_{key}"]
        total_weight += w
    aggregate = weighted_sum / total_weight if total_weight > 0 else 0.5

    alignments["raa_score"] = float(max(0.0, min(1.0, aggregate)))
    return alignments


def compute_raa_reward(
    to_be_evaluated: str,
    predicted_fut_xyz: torch.Tensor,
    predicted_fut_rot: torch.Tensor,
    *,
    weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Compute the full RAA reward from a rollout completion.

    Args:
        to_be_evaluated: The full rollout completion string.
        predicted_fut_xyz: Predicted future trajectory XYZ.
        predicted_fut_rot: Predicted future rotation matrices.
        weight: Weight multiplier for the final reward.

    Returns:
        Tuple of (raa_reward, info_dict).
    """
    coc_text = extract_coc_text(to_be_evaluated)

    if not coc_text or len(coc_text.strip()) < 20:
        return 0.0, {"raa_score": 0.0, "raa_empty_coc": 1.0}

    scores = compute_raa_score(coc_text, predicted_fut_xyz, predicted_fut_rot)
    info = {k: float(v) if isinstance(v, (float, int)) else v for k, v in scores.items()}
    return weight * scores["raa_score"], info
