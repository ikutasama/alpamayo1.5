#!/bin/bash
# Alpamayo 1.5 RL v2 Grounded Reward — Training Metrics Extraction Script
# Designed for the HCC-RM v2 reward system (obstacle-grounding + decision-consistency).
# Usage: bash extract_v2_metrics.sh [LOG_DIR]
#   If no LOG_DIR given, auto-detects the latest log directory under /root/temp_log*
# Output: prints report to stdout AND saves to /tmp/training_v2_report.txt
#
# Key metrics to watch (what I need to judge training health):
#   1. reward trend (is it increasing? stable? collapsing?)
#   2. reward_std (is variance healthy? or collapsed to ~0?)
#   3. obstacle_grounding_score (are obstacle mentions grounded to GT?)
#   4. decision_consistency (are COT decisions matching GT decisions?)
#   5. obstacle data availability (is obstacle_info being loaded?)
#   6. COT diversity (unique_completion_ratio — are rollouts diverse or templated?)
#   7. gt_decision distribution (what fraction of samples are stop/yield/maintain?)
#   8. advantage_std (gradient signal strength)

# ---- Auto-detect log directory ----
if [ -n "$1" ]; then
    LOG_BASE="$1"
else
    # Find the latest log directory
    LOG_BASE=$(ls -td /root/temp_log* 2>/dev/null | head -1)
fi

if [ -z "$LOG_BASE" ] || [ ! -d "$LOG_BASE" ]; then
    echo "ERROR: Cannot find log directory. Tried: /root/temp_log*"
    echo "Usage: bash extract_v2_metrics.sh <path_to_log_dir>"
    exit 1
fi

# Find the latest subdirectory (timestamped folder)
LOG_DIR=$(ls -td "$LOG_BASE"/logs_* 2>/dev/null | head -1)
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$LOG_BASE"
fi

ROLL_LOG=$(find "$LOG_DIR" -name "rollout_0.log" -o -name "rollout_*.log" 2>/dev/null | head -1)
POLICY_LOG=$(find "$LOG_DIR" -name "policy_0.log" -o -name "policy_*.log" 2>/dev/null | head -1)
CTRL_LOG=$(find "$LOG_DIR" -name "controller.log" 2>/dev/null | head -1)
TB_PATH=$(find "$LOG_BASE" -type d -name "tensorboard" 2>/dev/null | head -1)

# Fallback: try broader search
if [ -z "$POLICY_LOG" ]; then
    POLICY_LOG=$(find "$LOG_BASE" -name "policy_0.log" 2>/dev/null | head -1)
fi
if [ -z "$ROLL_LOG" ]; then
    ROLL_LOG=$(find "$LOG_BASE" -name "rollout_0.log" 2>/dev/null | head -1)
fi
if [ -z "$CTRL_LOG" ]; then
    CTRL_LOG=$(find "$LOG_BASE" -name "controller.log" 2>/dev/null | head -1)
fi
if [ -z "$TB_PATH" ]; then
    TB_PATH=$(find "$LOG_BASE" -type d -name "tensorboard" 2>/dev/null | head -1)
fi

R="/tmp/training_v2_report.txt"
> "$R"

echo "======== Alpamayo RL v2 Grounded Reward — Training Report ========" | tee -a "$R"
echo "Time: $(date)" | tee -a "$R"
echo "Log dir: $LOG_DIR" | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 0. File Sizes ===" | tee -a "$R"
for f in "$POLICY_LOG" "$ROLL_LOG" "$CTRL_LOG"; do
  if [ -f "$f" ]; then
    echo "  $(wc -l < "$f") lines  $(du -sh "$f" | cut -f1)  $f" | tee -a "$R"
  else
    echo "  MISSING  $f" | tee -a "$R"
  fi
done
if [ -n "$TB_PATH" ] && [ -d "$TB_PATH" ]; then
  echo "  TB: $(ls "$TB_PATH"/events* 2>/dev/null | wc -l) event files" | tee -a "$R"
else
  echo "  TB: NOT FOUND" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 1. Controller Log (last 30 lines) ===" | tee -a "$R"
if [ -f "$CTRL_LOG" ]; then
  tail -30 "$CTRL_LOG" | tee -a "$R"
else
  echo "  No controller log found" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 2. [Step] lines — core training progress ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep "\[Step " "$POLICY_LOG" | tail -40 | tee -a "$R"
  total_steps=$(grep -c '\[Step ' "$POLICY_LOG" 2>/dev/null || echo 0)
  echo "  ($total_steps total training steps logged)" | tee -a "$R"
else
  echo "  No policy log found" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 3. Reward Trend (first → last) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  first_r=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'raw_reward=[0-9.\-]+')
  last_r=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'raw_reward=[0-9.\-]+')
  # Also extract loss and grad_norm for first and last
  first_l=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'loss=[0-9.\-]+')
  last_l=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'loss=[0-9.\-]+')
  first_g=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'gn=[0-9.\-]+')
  last_g=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'gn=[0-9.\-]+')
  first_a=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'adv=[0-9.\-]+')
  last_a=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'adv=[0-9.\-]+')
  echo "  first step: reward=$first_r  loss=$first_l  grad=$first_g  adv=$first_a" | tee -a "$R"
  echo "  last  step: reward=$last_r  loss=$last_l  grad=$last_g  adv=$last_a" | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 4. HCC-v2 Reward Breakdown ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  # v2 uses [HCC-v2] tag
  grep -E "\[HCC-v2\]|\[HCC-RM\]|\[HCC-Reward\]" "$POLICY_LOG" | tail -20 | tee -a "$R"
  echo "  ($(grep -c '\[HCC-v2\]' "$POLICY_LOG" 2>/dev/null || echo 0) HCC-v2 entries)" | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 5. Reward Variance (CRITICAL — measures if template-collapse is fixed) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  # Extract raw_reward_std from [Step] lines
  grep "\[Step " "$POLICY_LOG" | tail -20 | grep -oP 'raw_reward=[0-9.\-]+' | grep -oP '[0-9.\-]+' | python3 -c "
import sys
vals = [float(v.strip()) for v in sys.stdin if v.strip()]
if vals:
    import numpy as np
    a = np.array(vals)
    print(f'  last 20 steps reward: mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}')
    print(f'  variance health: GOOD (std>0.05) if std>{a.std():.4f}')
else:
    print('  No reward values extracted')
" | tee -a "$R"

  # Also check advantage_std (gradient signal)
  grep "\[Step " "$POLICY_LOG" | tail -10 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 6. Obstacle Grounding Status ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  # Check if obstacle data is being loaded
  obs_load_fail=$(grep -c "Failed to load obstacle" "$POLICY_LOG" 2>/dev/null || echo 0)
  obs_success=$(grep -c "obstacle_info" "$POLICY_LOG" 2>/dev/null || echo 0)
  obs_source=$(grep "scene_source=" "$POLICY_LOG" 2>/dev/null | tail -5 | tee -a "$R")
  echo "  obstacle load failures: $obs_load_fail" | tee -a "$R"
  echo "  obstacle info references: $obs_success" | tee -a "$R"
  # Check obstacle_grounding_score from HCC-v2 logs
  grep "obstacle" "$POLICY_LOG" | tail -10 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 7. Decision Distribution (what GT decisions look like) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  # Count how many steps have each GT decision type
  for dec in stop yield nudge maintain slow_down accelerate turn_left turn_right; do
    count=$(grep -c "gt_is_${dec}" "$POLICY_LOG" 2>/dev/null || echo 0)
    echo "  gt_is_${dec}: $count mentions" | tee -a "$R"
  done
  # Extract decision consistency scores from HCC-v2
  grep "s2(decision)" "$POLICY_LOG" | tail -10 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 8. COT Diversity (anti-template indicator) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep "unique_completion_ratio" "$POLICY_LOG" | tail -10 | tee -a "$R"
  grep "cot_word_count" "$POLICY_LOG" | tail -10 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 9. CoC Debug Output ===" | tee -a "$R"
for f in "$POLICY_LOG" "$ROLL_LOG"; do
  if [ -f "$f" ]; then
    grep "\[CoC-Debug" "$f" | tail -5 | tee -a "$R"
  fi
done
echo "" | tee -a "$R"

echo "=== 10. Advantage Normalization ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep -E "\[AdvNorm" "$POLICY_LOG" | tail -20 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 11. Errors (excluding known noise) ===" | tee -a "$R"
for f in "$POLICY_LOG" "$ROLL_LOG" "$CTRL_LOG"; do
  if [ -f "$f" ]; then
    grep -iE "error|traceback|exception|failed|killed|oom|cuda" "$f" 2>/dev/null \
      | grep -v "UserWarning\|dim_slice_info\|Failed to load obstacle" \
      | tail -20 | tee -a "$R"
  fi
done
echo "" | tee -a "$R"

echo "=== 12. GPU Status ===" | tee -a "$R"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>/dev/null | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 13. Rollout Log (last 30 lines) ===" | tee -a "$R"
if [ -f "$ROLL_LOG" ]; then
  tail -30 "$ROLL_LOG" | tee -a "$R"
else
  echo "  No rollout log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 14. Policy Log (last 30 lines) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  tail -30 "$POLICY_LOG" | tee -a "$R"
else
  echo "  No policy log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 15. TB Scalars (v2 reward components) ===" | tee -a "$R"
if [ -n "$TB_PATH" ] && [ -d "$TB_PATH" ]; then
  python3 -c "
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator('$TB_PATH', size_guidance={'scalars': 0})
    ea.Reload()
    tags = sorted(ea.Tags().get('scalars', []))
    print(f'  {len(tags)} scalar tags found')

    # v2 priority keys — the most important metrics
    v2_keys = [
        'train/raw_reward_mean', 'train/raw_reward_std',
        'train/raw_reward_min', 'train/raw_reward_max',
        'train/loss_avg', 'train/grad_norm',
        'train/advantage_mean', 'train/advantage_std',
        # v2 grounded reward components
        'train/reward_scene_understanding_mean',
        'train/reward_obstacle_grounding_score_mean',
        'train/reward_obstacle_type_match_mean',
        'train/reward_obstacle_direction_match_mean',
        'train/reward_decision_consistency_mean',
        'train/reward_decision_consistency_score_mean',
        'train/reward_cot_gt_match_mean',
        'train/reward_traj_L2_mean',
        'train/reward_format_score_mean',
        'train/reward_consistency_penalty_mean',
        'train/reward_num_gt_obstacles_mean',
        'train/reward_obstacle_hallucination_penalty_mean',
        # GT decision distribution
        'train/reward_gt_is_stopped_mean',
        'train/reward_gt_is_yield_mean',
        'train/reward_gt_is_nudge_mean',
        'train/reward_gt_is_maintain_mean',
        'train/reward_gt_is_slow_down_mean',
        'train/reward_gt_is_accelerate_mean',
        # COT diversity
        'train/unique_completion_ratio',
        'train/cot_word_count_mean',
        'train/reward_cot_has_decision_mean',
        'train/reward_cot_num_decisions_mean',
    ]

    for t in v2_keys:
        if t in tags:
            evts = ea.Scalars(t)[-15:]
            print(f'  {t}:')
            for e in evts:
                print(f'    step={e.step} val={e.value:.4f}')
        else:
            print(f'  {t}: NOT FOUND')

    # Show other tags not in v2_keys
    other = [t for t in tags if t not in v2_keys]
    if other:
        print(f'  Other tags ({len(other)}):')
        for t in other[:30]:
            evts = ea.Scalars(t)[-3:]
            last_val = evts[-1].value if evts else 'N/A'
            print(f'    {t}: last={last_val}')
except Exception as ex:
    print(f'  TB dump failed: {ex}')
" 2>&1 | tee -a "$R"
else
  echo "  No tensorboard data found" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 16. Quick Health Check ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  total_steps=$(grep -c "\[Step " "$POLICY_LOG" 2>/dev/null || echo "0")
  echo "  Total training steps: $total_steps" | tee -a "$R"

  if [ "$total_steps" -gt 0 ]; then
    first_v=$(grep "\[Step " "$POLICY_LOG" | head -1 | grep -oP 'raw_reward=[0-9.\-]+' | grep -oP '[0-9.\-]+$')
    last_v=$(grep "\[Step " "$POLICY_LOG" | tail -1 | grep -oP 'raw_reward=[0-9.\-]+' | grep -oP '[0-9.\-]+$')

    if [ -n "$first_v" ] && [ -n "$last_v" ]; then
      echo "  Reward: $first_v -> $last_v" | tee -a "$R"
      improved=$(python3 -c "
f=float('$first_v'); l=float('$last_v')
if l > f + 0.01: print('INCREASING (good)')
elif abs(l-f) < 0.01: print('FLAT (check variance)')
else: print('DECREASING (concern)')
" 2>/dev/null || echo "UNKNOWN")
      echo "  Reward trend: $improved" | tee -a "$R"
    fi

    # Check if v2 reward is actually being used
    v2_count=$(grep -c "\[HCC-v2\]" "$POLICY_LOG" 2>/dev/null || echo 0)
    v1_count=$(grep -c "\[HCC-RM\]" "$POLICY_LOG" 2>/dev/null || echo 0)
    echo "  HCC-v2 entries: $v2_count  |  HCC-v1 entries: $v1_count" | tee -a "$R"
    if [ "$v2_count" -gt 0 ]; then
      echo "  Reward version: v2 (grounded) — CORRECT" | tee -a "$R"
    elif [ "$v1_count" -gt 0 ]; then
      echo "  Reward version: v1 (regex) — WARNING: v2 not active!" | tee -a "$R"
    else
      echo "  Reward version: unknown — check HCC config" | tee -a "$R"
    fi
  else
    echo "  No [Step] output yet — still in rollout/warmup phase" | tee -a "$R"
  fi
else
  echo "  No policy log to analyze" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "======== END OF v2 REPORT ========" | tee -a "$R"
echo ""
echo "Report saved to: $R ($(wc -l < "$R") lines)"
echo "Send the contents of $R to me for analysis."