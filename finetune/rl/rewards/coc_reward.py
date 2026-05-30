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

"""Chain-of-Causation (CoC) quality reward for Alpamayo RL training.

Part of the Hierarchical Causal-Consistent Reward Model (HCC-RM) framework.
Evaluates reasoning text quality across multiple dimensions:
  - Factual Accuracy: Does the CoC mention entities actually present in the scene?
  - Causal Coherence: Is the reasoning chain logically consistent (obs → analysis → decision → action)?
  - Safety Awareness: Does the CoC identify potential hazards and safety-critical elements?
  - Completeness: Does the CoC cover key driving dimensions?
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from alpamayo1_5.models.base_model import SPECIAL_TOKENS
from rl.rewards.reward_types import RewardComponents

# ---------------------------------------------------------------------------
# Scene entity patterns for factual accuracy checking
# ---------------------------------------------------------------------------
SCENE_ENTITY_PATTERNS: dict[str, list[str]] = {
    "vehicle": [
        r"\b(vehicle|car|truck|bus|motorcycle|bicycle|cyclist|pedestrian)\b",
        r"\b(ego vehicle|leading vehicle|oncoming|adjacent lane)\b",
    ],
    "road_structure": [
        r"\b(lane|intersection|crosswalk|traffic light|stop sign|yield sign)\b",
        r"\b(highway|ramp|roundabout|junction|merge|exit|road)\b",
    ],
    "traffic_condition": [
        r"\b(congestion|heavy traffic|light traffic|stopped|moving|speed)\b",
    ],
}

# ---------------------------------------------------------------------------
# Causal chain patterns for coherence checking
# ---------------------------------------------------------------------------
CAUSAL_CONNECTORS: list[str] = [
    r"\b(because|since|due to|as a result|therefore|thus|hence|consequently)\b",
    r"\b(so|leading to|causing|resulting in|indicating|suggesting)\b",
]

DECISION_KEYWORDS: list[str] = [
    r"\b(should|must|need to|will|plan to|decide to|choose to)\b",
    r"\b(slow down|speed up|maintain|change lane|turn|yield|stop|proceed|follow)\b",
    r"\b(keep distance|brake|accelerate|steer|merge|overtake)\b",
]

SAFETY_KEYWORDS: list[str] = [
    r"\b(safety|danger|hazard|risk|collision|crash|accident)\b",
    r"\b(cautious|careful|attention|vigilant|monitor|watch)\b",
    r"\b(safe distance|reaction time|blind spot|cut-in|hard brake)\b",
]

DRIVING_DIMENSIONS: dict[str, list[str]] = {
    "hazard_identification": [
        r"\b(hazard|danger|risk|threat|unsafe|potential)\b",
    ],
    "intent_prediction": [
        r"\b(may|might|could|likely|probably|intend|plan to|about to)\b",
    ],
    "action_justification": [
        r"\b(accelerate|decelerate|brake|steer|turn|lane change|maintain)\b",
    ],
    "safety_check": [
        r"\b(safe|clear|check|verify|ensure|confirm)\b",
    ],
    "spatial_reasoning": [
        r"\b(left|right|front|rear|ahead|behind|adjacent|beside)\b",
    ],
    "temporal_reasoning": [
        r"\b(now|currently|soon|within|after|before|during)\b",
    ],
}


def _strip_image_tokens(text: str) -> str:
    """Remove trajectory/image placeholder tokens (<i...>) and leaked special tokens from CoC text.

    The VLA model sometimes generates <i{N}> tokens (trajectory discrete tokens)
    in the CoC reasoning section, which corrupt both the text and downstream
    token count calculations. This strips them out to recover the actual
    reasoning content.
    """
    # Remove <i{number}> tokens (trajectory vocabulary placeholders)
    text = re.sub(r"<i\d+>", "", text)
    # Remove <|answer_end|> and any other leaked special tokens in the CoC section
    text = re.sub(r"<\|answer_end\|>", "", text)
    # Remove <|...|> special tokens that shouldn't appear in CoC text
    text = re.sub(r"<\|(?:cot_end|traj_future_start|traj_future_end|meta_action_start|meta_action_end|answer_start)\|>", "", text)
    # Clean up multiple spaces left by removal
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_coc_text(to_be_evaluated: str) -> str:
    """Extract the Chain-of-Causation reasoning text from a rollout completion.

    The CoC text is the portion before ``<|cot_end|>``.
    Image/trajectory tokens (<i...>) are stripped to prevent corruption.

    Args:
        to_be_evaluated: The full rollout completion string.

    Returns:
        The extracted CoC text (cleaned of image tokens), or empty string if not found.
    """
    coc = ""
    if "<|cot_end|>" in to_be_evaluated:
        coc = to_be_evaluated.split("<|cot_end|>")[0].strip()
    elif "<|traj_future_start|>" in to_be_evaluated:
        coc = to_be_evaluated.split("<|traj_future_start|>")[0].strip()
    else:
        coc = to_be_evaluated.strip()
    return _strip_image_tokens(coc)


def score_factual_accuracy(coc_text: str) -> float:
    """Score the factual accuracy of a CoC text based on entity mentions.

    Ensures the CoC references real-world driving entities rather than
    generating hallucinated or generic content. A higher score indicates
    specific, contextualized reasoning.

    Args:
        coc_text: The extracted CoC reasoning text.

    Returns:
        Score in [0, 1] where 1 means highly factual/specific.
    """
    if not coc_text:
        return 0.0
    category_scores = []
    for category, patterns in SCENE_ENTITY_PATTERNS.items():
        cat_hits = 0
        for pattern in patterns:
            if re.search(pattern, coc_text, re.IGNORECASE):
                cat_hits += 1
        category_scores.append(min(cat_hits / len(patterns), 1.0))

    factual_score = float(np.mean(category_scores))
    word_count = len(coc_text.strip().split())
    if word_count < 10:
        return 0.0
    if word_count < 20:
        return float(word_count - 10) / 10.0 * float(np.clip(factual_score, 0.0, 1.0))
    return float(np.clip(factual_score, 0.0, 1.0))


def score_causal_coherence_unchecked(coc_text: str) -> float:
    scores = []
    causal_hits = sum(
        1 for pattern in CAUSAL_CONNECTORS if re.search(pattern, coc_text, re.IGNORECASE)
    )
    scores.append(min(causal_hits / 3.0, 1.0))
    decision_hits = sum(
        1 for pattern in DECISION_KEYWORDS if re.search(pattern, coc_text, re.IGNORECASE)
    )
    scores.append(min(decision_hits / 2.0, 1.0))
    has_observation = any(
        re.search(p, coc_text, re.IGNORECASE) for p in [
            r"\b(see|observe|notice|detect|visible|appears|present)\b",
            r"\b(currently|now|situation|scenario|condition)\b",
        ]
    )
    has_action = any(
        re.search(p, coc_text, re.IGNORECASE) for p in [
            r"\b(should|will|need to|must|plan to|decide to)\b",
        ]
    )
    structure_score = (has_observation + has_action) / 2.0
    scores.append(structure_score)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def score_causal_coherence(coc_text: str) -> float:
    """Score the logical coherence of a CoC reasoning chain.

    Evaluates whether the CoC follows a proper causal structure:
    observation → analysis → risk assessment → decision → planned action.

    Args:
        coc_text: The extracted CoC reasoning text.

    Returns:
        Score in [0, 1] where 1 means logically coherent.
    """
    word_count = len(coc_text.strip().split())
    if word_count < 10:
        return 0.0
    if word_count < 20:
        return float(word_count - 10) / 10.0 * score_causal_coherence_unchecked(coc_text)
    return score_causal_coherence_unchecked(coc_text)


def score_safety_awareness(coc_text: str) -> float:
    """Score the safety awareness expressed in a CoC text.

    Evaluates whether the reasoning explicitly identifies hazards,
    proposes safety measures, and demonstrates risk-aware thinking.

    Args:
        coc_text: The extracted CoC reasoning text.

    Returns:
        Score in [0, 1] where 1 means highly safety-aware.
    """
    word_count = len(coc_text.strip().split())
    if word_count < 10:
        return 0.0
    safety_hits = sum(
        1 for pattern in SAFETY_KEYWORDS if re.search(pattern, coc_text, re.IGNORECASE)
    )
    base = float(np.clip(safety_hits / 4.0, 0.0, 1.0))
    if word_count < 20:
        return float(word_count - 10) / 10.0 * base
    return base


def score_completeness(coc_text: str) -> float:
    """Score the completeness of a CoC text across key driving dimensions.

    Evaluates coverage of: hazard identification, intent prediction,
    action justification, safety check, spatial reasoning, temporal reasoning.

    Args:
        coc_text: The extracted CoC reasoning text.

    Returns:
        Score in [0, 1] where 1 means all dimensions are covered.
    """
    word_count = len(coc_text.strip().split())
    if word_count < 10:
        return 0.0

    dimension_scores = []
    for dim_name, patterns in DRIVING_DIMENSIONS.items():
        dim_hits = any(
            re.search(pattern, coc_text, re.IGNORECASE) for pattern in patterns
        )
        dimension_scores.append(1.0 if dim_hits else 0.0)

    base = float(np.mean(dimension_scores))
    if word_count < 20:
        return float(word_count - 10) / 10.0 * base
    return base


def compute_coc_quality(
    to_be_evaluated: str,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute multi-dimensional CoC quality scores.

    This is the main entry point for evaluating Chain-of-Causation quality
    as part of the HCC-RM framework.

    Args:
        to_be_evaluated: The full rollout completion string.
        weights: Optional per-dimension weights. Defaults to equal weighting.

    Returns:
        Dict with individual dimension scores and aggregated CoC quality score.
        Keys: ``coc_factual``, ``coc_coherence``, ``coc_safety``,
        ``coc_completeness``, ``coc_quality`` (aggregated).
    """
    if weights is None:
        weights = {
            "factual": 0.30,
            "coherence": 0.35,
            "safety": 0.10,
            "completeness": 0.25,
        }

    coc_text = extract_coc_text(to_be_evaluated)

    import logging as _logging
    _logger = _logging.getLogger("cosmos")
    _logger.info(f"[CoC-Debug] coc_text={coc_text[:300]!r}")

    scores = {
        "coc_factual": score_factual_accuracy(coc_text),
        "coc_coherence": score_causal_coherence(coc_text),
        "coc_safety": score_safety_awareness(coc_text),
        "coc_completeness": score_completeness(coc_text),
    }

    # Weighted aggregation
    aggregated = (
        weights.get("factual", 0.25) * scores["coc_factual"]
        + weights.get("coherence", 0.30) * scores["coc_coherence"]
        + weights.get("safety", 0.25) * scores["coc_safety"]
        + weights.get("completeness", 0.20) * scores["coc_completeness"]
    )
    scores["coc_quality"] = float(np.clip(aggregated, 0.0, 1.0))

    return scores


_LIGHTWEIGHT_RISK_WORDS = (
    "risk",
    "hazard",
    "obstacle",
    "vehicle",
    "car",
    "pedestrian",
    "crosswalk",
    "intersection",
    "lane",
    "merge",
    "cut in",
    "cut-in",
    "traffic",
    "light",
    "stop",
    "yield",
    "curb",
    "collision",
)
_LIGHTWEIGHT_ACTION_WORDS = (
    "keep",
    "maintain",
    "slow",
    "decelerate",
    "accelerate",
    "brake",
    "stop",
    "turn",
    "left",
    "right",
    "straight",
    "follow",
    "yield",
    "avoid",
)


def _between(text: str, start_token: str | None, end_token: str | None) -> str:
    """Extract text between two optional markers."""
    out = text
    if start_token and start_token in out:
        out = out.split(start_token, maxsplit=1)[-1]
    if end_token and end_token in out:
        out = out.split(end_token, maxsplit=1)[0]
    return out.strip()


def _word_overlap(a: str, b: str) -> float:
    """Small lexical overlap score in [0, 1] for optional reference CoC text."""
    toks_a = set(re.findall(r"[\w\u4e00-\u9fff]+", a.lower()))
    toks_b = set(re.findall(r"[\w\u4e00-\u9fff]+", b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(1, len(toks_a | toks_b))


def extract_coc_sections(to_be_evaluated: str) -> dict[str, str | float]:
    """Extract CoC and trajectory text plus format indicators from a rollout completion."""
    cot_start = SPECIAL_TOKENS["cot_start"]
    cot_end = SPECIAL_TOKENS["cot_end"]
    traj_start = SPECIAL_TOKENS["traj_future_start"]
    traj_end = SPECIAL_TOKENS["traj_future_end"]

    has_cot_end = cot_end in to_be_evaluated
    has_traj_start = traj_start in to_be_evaluated
    has_traj_end = traj_end in to_be_evaluated
    ordered = (
        (not has_cot_end or not has_traj_start)
        or to_be_evaluated.find(cot_end) <= to_be_evaluated.find(traj_start)
    )

    reasoning = _between(
        to_be_evaluated,
        cot_start if cot_start in to_be_evaluated else None,
        cot_end,
    )
    # Strip trajectory/image placeholder tokens from CoC reasoning text
    reasoning = _strip_image_tokens(reasoning)
    traj_text = _between(to_be_evaluated, traj_start, traj_end)
    format_score = (
        float(has_cot_end) * 0.4
        + float(has_traj_start) * 0.3
        + float(has_traj_end) * 0.2
    )
    format_score += float(ordered) * 0.1
    # Penalize image token pollution: if the original text contained <i...> tokens,
    # the CoC is malformed and should score lower
    has_image_token_pollution = bool(re.search(r"<i\d+>", _between(
        to_be_evaluated,
        cot_start if cot_start in to_be_evaluated else None,
        cot_end,
    )))
    if has_image_token_pollution:
        format_score = min(format_score, 0.2)  # cap at 0.2 max for polluted CoC

    return {
        "reasoning": reasoning,
        "traj_text": traj_text,
        "has_cot_end": float(has_cot_end),
        "has_traj_start": float(has_traj_start),
        "has_traj_end": float(has_traj_end),
        "format_score": min(1.0, format_score),
        "has_image_token_pollution": float(has_image_token_pollution),
    }


def compute_coc_reward(
    to_be_evaluated: str,
    reference: dict[str, Any] | None = None,
    *,
    min_chars: int = 24,
) -> RewardComponents:
    """Score whether the generated CoC is structured and action-relevant.

    This lightweight score is kept for reward ablations and token-level
    advantage routing. The richer HCC-RM path continues to use
    ``compute_coc_quality`` above.
    """
    sections = extract_coc_sections(to_be_evaluated)
    reasoning = str(sections["reasoning"])
    reasoning_l = reasoning.lower()

    length_score = min(1.0, len(reasoning.strip()) / max(1, min_chars))
    # Use regex word boundary matching to avoid false positives
    # (e.g. "light" matching "lightly", "stop" matching "stopped")
    risk_pattern = re.compile(r'|'.join(r'\b' + re.escape(w) + r'\b' for w in _LIGHTWEIGHT_RISK_WORDS))
    action_pattern = re.compile(r'|'.join(r'\b' + re.escape(w) + r'\b' for w in _LIGHTWEIGHT_ACTION_WORDS))
    risk_score = float(bool(risk_pattern.search(reasoning_l)))
    action_score = float(bool(action_pattern.search(reasoning_l)))
    format_score = float(sections["format_score"])

    ref_score = 0.0
    if reference and isinstance(reference.get("cot", ""), str):
        ref_score = _word_overlap(reasoning, reference["cot"])

    reward = (
        0.35 * format_score
        + 0.20 * length_score
        + 0.20 * risk_score
        + 0.20 * action_score
        + 0.05 * ref_score
    )

    return RewardComponents(
        reward=float(reward),
        metrics={
            "format_score": float(format_score),
            "length_score": float(length_score),
            "risk_keyword_score": float(risk_score),
            "action_keyword_score": float(action_score),
            "reference_overlap": float(ref_score),
            "has_cot_end": float(sections["has_cot_end"]),
            "has_traj_start": float(sections["has_traj_start"]),
            "has_traj_end": float(sections["has_traj_end"]),
        },
    )
