#!/usr/bin/env python3
"""Smoke test for the improved grounded CoC reward and variance protection.

Run with: python -m finetune.rl.tests.test_grounded_reward
Or: python finetune/rl/tests/test_grounded_reward.py
"""

from __future__ import annotations

import sys
import os
import math

# Ensure project root is on sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)
if os.path.join(_root, "finetune") not in sys.path:
    sys.path.insert(0, os.path.join(_root, "finetune"))

import numpy as np
import torch


def test_gt_decision_extraction():
    """Test that GT trajectory decisions are correctly classified."""
    from rl.rewards.grounded_coc_reward import extract_gt_decision

    # Case 1: Deceleration trajectory
    t = torch.linspace(0, 1, 64)
    xyz_decel = torch.stack([t * 10, torch.zeros(64), torch.zeros(64)], dim=-1)
    # Speed decreases from ~10 to ~0
    xyz_decel[:, 0] = 10 * t - 5 * t**2  # Quadratic: fast start, slow end
    decel = extract_gt_decision(xyz_decel)
    assert decel["slow_down"] > 0.3 or decel["stop"] > 0.3, f"Decel trajectory not detected: {decel}"
    print(f"  ✓ Deceleration: slow_down={decel['slow_down']:.2f}, stop={decel['stop']:.2f}")

    # Case 2: Constant speed straight
    xyz_const = torch.stack([t * 5, torch.zeros(64), torch.zeros(64)], dim=-1)
    const = extract_gt_decision(xyz_const)
    assert const["maintain"] > 0.3 or const["accelerate"] > 0.1, f"Constant speed not detected: {const}"
    print(f"  ✓ Constant speed: maintain={const['maintain']:.2f}, accelerate={const['accelerate']:.2f}")

    # Case 3: Stop trajectory (positions barely change)
    xyz_stop = torch.stack([torch.ones(64) * 0.1, torch.zeros(64), torch.zeros(64)], dim=-1)
    stop = extract_gt_decision(xyz_stop)
    assert stop["stop"] > 0.3, f"Stop trajectory not detected: {stop}"
    print(f"  ✓ Stop: stop={stop['stop']:.2f}")

    # Case 4: Left nudge
    xyz_left = torch.stack([t * 5, -t * 2, torch.zeros(64)], dim=-1)
    left = extract_gt_decision(xyz_left)
    assert left["nudge_left"] > 0.1, f"Left nudge not detected: {left}"
    print(f"  ✓ Left nudge: nudge_left={left['nudge_left']:.2f}")

    print("test_gt_decision_extraction: PASSED\n")


def test_coc_decision_extraction():
    """Test CoC text decision extraction."""
    from rl.rewards.grounded_coc_reward import extract_coc_decisions

    # Test 1: Clear deceleration intent
    text1 = "The vehicle ahead is slowing down, so I should brake and reduce speed to maintain safe distance."
    dec1 = extract_coc_decisions(text1)
    assert dec1["slow_down"] > 0.5, f"Slow down not detected: {dec1}"
    print(f"  ✓ Slow down text: {dec1['slow_down']:.2f}")

    # Test 2: Yield intent
    text2 = "A pedestrian is crossing, I need to yield and wait for them to pass."
    dec2 = extract_coc_decisions(text2)
    assert dec2["yield"] > 0.5, f"Yield not detected: {dec2}"
    print(f"  ✓ Yield text: {dec2['yield']:.2f}")

    # Test 3: Maintain speed
    text3 = "The road is clear ahead, I will continue at the current speed."
    dec3 = extract_coc_decisions(text3)
    assert dec3["maintain"] > 0.3 or dec3["accelerate"] > 0.3, f"Maintain not detected: {dec3}"
    print(f"  ✓ Maintain text: maintain={dec3['maintain']:.2f}, accelerate={dec3['accelerate']:.2f}")

    print("test_coc_decision_extraction: PASSED\n")


def test_object_mentions():
    """Test object type mention extraction."""
    from rl.rewards.grounded_coc_reward import extract_coc_object_mentions

    text = "I see a truck in the left lane and a pedestrian near the crosswalk."
    mentions = extract_coc_object_mentions(text)
    assert mentions["vehicle"], f"Vehicle not detected: {mentions}"
    assert mentions["pedestrian"], f"Pedestrian not detected: {mentions}"
    print(f"  ✓ Object mentions: {mentions}")

    text2 = "The road ahead is clear."
    mentions2 = extract_coc_object_mentions(text2)
    assert not any(mentions2.values()), f"False positive: {mentions2}"
    print(f"  ✓ No objects: {mentions2}")

    print("test_object_mentions: PASSED\n")


def test_grounded_coc_reward():
    """Test the full grounded CoC reward computation."""
    from rl.rewards.grounded_coc_reward import compute_grounded_coc_reward

    # Scenario: vehicle ahead, CoC correctly identifies it and says slow down
    scene_facts = {
        "has_vehicle_nearby": True,
        "has_pedestrian_nearby": False,
        "has_cyclist_nearby": False,
        "closest_object_type": "vehicle",
        "closest_distance": 15.0,
        "object_on_left": False,
        "object_on_right": False,
        "object_ahead": True,
        "approaching_object": False,
        "num_obstacles": 2,
        "high_threat_objects": [
            {"track_id": 1, "object_type": "vehicle", "distance": 15.0,
             "position": [15.0, 0.0, 0.0], "radial_velocity": -1.0, "is_approaching": True}
        ],
    }

    # GT trajectory: deceleration
    t = torch.linspace(0, 1, 64)
    gt_xyz = torch.stack([10 * t - 5 * t**2, torch.zeros(64), torch.zeros(64)], dim=-1)

    # Good CoC: correctly identifies vehicle and says slow down
    good_coc = "I observe a vehicle ahead in the same lane. It appears to be slowing down. I should reduce speed and maintain safe following distance to avoid a collision."
    good_reward = compute_grounded_coc_reward(good_coc, scene_facts, gt_xyz)
    print(f"  Good CoC reward: {good_reward.reward:.3f}")
    print(f"  Metrics: object_recall={good_reward.metrics.get('object_recall', 0):.2f}, "
          f"decision_alignment={good_reward.metrics.get('decision_alignment', 0):.2f}")

    # Bad CoC: hallucinates pedestrian, says accelerate
    bad_coc = "I see a pedestrian crossing from the left. I will accelerate to pass quickly."
    bad_reward = compute_grounded_coc_reward(bad_coc, scene_facts, gt_xyz)
    print(f"  Bad CoC reward: {bad_reward.reward:.3f}")
    print(f"  Metrics: hallucination_score={bad_reward.metrics.get('hallucination_score', 0):.2f}, "
          f"false_mentions={bad_reward.metrics.get('false_object_mentions', 0):.0f}")

    # Template CoC: generic keywords, no real content
    template_coc = "Because of the current situation, I should be careful and maintain safe distance. The vehicle may stop therefore I should be vigilant."
    template_reward = compute_grounded_coc_reward(template_coc, scene_facts, gt_xyz)
    print(f"  Template CoC reward: {template_reward.reward:.3f}")

    # Good reward should be higher than bad reward
    assert good_reward.reward > bad_reward.reward, \
        f"Good CoC ({good_reward.reward:.3f}) should score higher than bad ({bad_reward.reward:.3f})"

    print("test_grounded_coc_reward: PASSED\n")


def test_variance_protection():
    """Test rank-based advantages and variance amplification."""
    from rl.rewards.variance_protection import (
        compute_rank_advantages,
        amplify_low_variance_advantages,
        compute_completion_diversity,
        should_skip_update,
    )

    # Test 1: Rank advantages preserve ordering even with tiny differences
    rewards = [0.501, 0.502, 0.500, 0.503, 0.499, 0.501]
    groups = [0, 0, 0, 0, 0, 0]
    rank_adv = compute_rank_advantages(rewards, groups)
    print(f"  Raw rewards: {[f'{r:.4f}' for r in rewards]}")
    print(f"  Rank advantages: {[f'{a:.3f}' for a in rank_adv]}")
    # Check that the highest reward gets highest advantage
    assert rank_adv[3] > rank_adv[4], "Rank ordering violated"
    print(f"  ✓ Rank ordering preserved")

    # Test 2: Amplification
    adv = torch.tensor([0.001, -0.001, 0.002, -0.002, 0.0, 0.001])
    raw_r = torch.tensor([0.501, 0.499, 0.502, 0.498, 0.500, 0.501])
    amplified = amplify_low_variance_advantages(adv, raw_r, min_std=0.05, amplify_factor=3.0)
    assert amplified.abs().max() > adv.abs().max(), "Amplification did not increase magnitude"
    print(f"  ✓ Amplification: {adv.abs().max():.4f} -> {amplified.abs().max():.4f}")

    # Test 3: Diversity
    completions = [
        "I see a car ahead and slow down.",
        "I see a car ahead and slow down.",  # Duplicate
        "The truck in front is braking, I yield.",  # Different
    ]
    diversity = compute_completion_diversity(completions, [0, 0, 0])
    print(f"  Diversity: {[f'{d:.2f}' for d in diversity]}")
    assert diversity[0] < diversity[2] or abs(diversity[0] - diversity[2]) < 0.3, \
        "Duplicate should have lower diversity"
    print(f"  ✓ Diversity computed")

    # Test 4: Skip update
    rewards_low = [0.5, 0.5, 0.5, 0.5]
    skip, diag = should_skip_update(rewards_low, [0, 0, 0, 0], min_group_std=0.01)
    print(f"  Skip on low variance: {skip}, diag: {diag}")
    assert skip, "Should skip when variance is zero"
    print(f"  ✓ Skip update on zero variance")

    print("test_variance_protection: PASSED\n")


def test_obstacle_scene_facts():
    """Test obstacle data parsing and scene fact extraction."""
    from rl.rewards.obstacle_reward import extract_scene_facts_from_obstacles

    # Mock obstacle data
    obstacle_data = {
        "obstacles": [
            {
                "track_id": 1,
                "object_type": "vehicle",
                "positions": np.array([[15.0, 0.5, 0.0], [14.0, 0.5, 0.0]]),
                "velocities": np.array([[-2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
                "heading": np.array([3.14, 3.14]),
                "bbox_lwh": np.array([4.5, 2.0, 1.5]),
                "distances": np.array([15.0, 14.0]),
                "timestamps_rel": np.array([0.0, 0.1]),
            },
            {
                "track_id": 2,
                "object_type": "pedestrian",
                "positions": np.array([[20.0, 3.0, 0.0], [19.5, 2.8, 0.0]]),
                "velocities": np.array([[-0.5, -0.3, 0.0], [-0.5, -0.3, 0.0]]),
                "heading": np.array([3.5, 3.5]),
                "bbox_lwh": np.array([0.5, 0.5, 1.7]),
                "distances": np.array([20.2, 19.7]),
                "timestamps_rel": np.array([0.0, 0.1]),
            },
        ],
        "closest_obstacle": {
            "track_id": 1,
            "object_type": "vehicle",
            "distance": 15.0,
            "position": [15.0, 0.5, 0.0],
            "velocity": [-2.0, 0.0, 0.0],
            "heading": 3.14,
        },
        "obstacle_summary": "1 vehicle, 1 pedestrian",
    }

    facts = extract_scene_facts_from_obstacles(obstacle_data)
    assert facts["has_vehicle_nearby"], "Vehicle not detected"
    assert facts["has_pedestrian_nearby"], "Pedestrian not detected"
    assert facts["closest_object_type"] == "vehicle", f"Wrong closest: {facts['closest_object_type']}"
    assert facts["object_on_left"], "Object should be on left (positive y)"
    print(f"  Scene facts: {facts}")
    print(f"  ✓ Obstacle scene facts extracted correctly")

    # Test empty obstacle data
    empty_facts = extract_scene_facts_from_obstacles(None)
    assert not empty_facts["has_vehicle_nearby"], "False positive on empty data"
    print(f"  ✓ Empty data handled correctly")

    print("test_obstacle_scene_facts: PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Grounded Reward & Variance Protection Tests")
    print("=" * 60 + "\n")

    tests = [
        test_gt_decision_extraction,
        test_coc_decision_extraction,
        test_object_mentions,
        test_grounded_coc_reward,
        test_variance_protection,
        test_obstacle_scene_facts,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            print(f"--- {test_fn.__name__} ---")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
