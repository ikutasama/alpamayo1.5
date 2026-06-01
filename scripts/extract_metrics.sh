#!/bin/bash
# Alpamayo 1.5 RL Training Metrics Extraction Script
# Updated for diffusion_rl_0601 branch: adds diffusion expert RL monitoring
# Usage: bash extract_metrics.sh [LOG_DIR]
#   If no LOG_DIR given, auto-detects latest logs under /root/temp_log*
# Output: prints report to stdout AND saves to /root/temp_log/training_report.txt

# ---- Auto-detect log directory ----
if [ -n "$1" ]; then
    LOG_BASE="$1"
else
    LOG_BASE=$(ls -td /root/temp_log* 2>/dev/null | head -1)
fi

if [ -z "$LOG_BASE" ] || [ ! -d "$LOG_BASE" ]; then
    echo "ERROR: Cannot find log directory. Tried: /root/temp_log*"
    echo "Usage: bash extract_metrics.sh <path_to_log_dir>"
    exit 1
fi

LOG_DIR=$(ls -td "$LOG_BASE"/logs_* 2>/dev/null | head -1)
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$LOG_BASE"
fi

ROLL_LOG=$(find "$LOG_DIR" -name "rollout_0.log" 2>/dev/null | head -1)
POLICY_LOG=$(find "$LOG_DIR" -name "policy_0.log" 2>/dev/null | head -1)
CTRL_LOG=$(find "$LOG_DIR" -name "controller.log" 2>/dev/null | head -1)
TB_PATH=$(find "$LOG_BASE" -type d -name "tensorboard" 2>/dev/null | head -1)

# Fallback searches
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

R="/root/temp_log/training_report.txt"
> "$R"

echo "======== Alpamayo RL Training Report (diffusion_rl_0601) ========" | tee -a "$R"
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
  tb_files=$(ls "$TB_PATH"/events* 2>/dev/null | wc -l)
  tb_latest=$(ls -t "$TB_PATH"/events.out.tfevents.* 2>/dev/null | head -1)
  echo "  TB: $tb_files event files, latest: $tb_latest" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 1. Controller Log (last 30 lines) ===" | tee -a "$R"
if [ -f "$CTRL_LOG" ]; then
  tail -30 "$CTRL_LOG" | tee -a "$R"
else
  echo "  No controller log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 2. [Step] lines — core training progress ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep "\[Step " "$POLICY_LOG" | tail -40 | tee -a "$R"
  total_steps=$(grep -c '\[Step ' "$POLICY_LOG" 2>/dev/null || echo 0)
  echo "  ($total_steps total training steps)" | tee -a "$R"
else
  echo "  No policy log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 3. Reward Trend ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  first_r=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'raw_reward=[0-9.\-]+')
  last_r=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'raw_reward=[0-9.\-]+')
  first_l=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'loss=[0-9.\-]+')
  last_l=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'loss=[0-9.\-]+')
  first_g=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'gn=[0-9.\-]+')
  last_g=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'gn=[0-9.\-]+')
  first_a=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'adv=[0-9.\-]+')
  last_a=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'adv=[0-9.\-]+')
  echo "  first: reward=$first_r  loss=$first_l  grad=$first_g  adv=$first_a" | tee -a "$R"
  echo "  last:  reward=$last_r  loss=$last_l  grad=$last_g  adv=$last_a" | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 4. Diffusion RL Loss (NEW) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  diff_entries=$(grep "\[DiffusionRL\]" "$POLICY_LOG" 2>/dev/null | tail -20)
  if [ -n "$diff_entries" ]; then
    echo "  Active! Last 20 [DiffusionRL] entries:" | tee -a "$R"
    echo "$diff_entries" | tee -a "$R"
    diff_count=$(grep -c '\[DiffusionRL\]' "$POLICY_LOG" 2>/dev/null || echo 0)
    echo "  ($diff_count total DiffusionRL entries)" | tee -a "$R"
    # Extract loss trend
    first_diff=$(grep "\[DiffusionRL\]" "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'raw_diff_loss=[0-9.\-]+')
    last_diff=$(grep "\[DiffusionRL\]" "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'raw_diff_loss=[0-9.\-]+')
    first_wdiff=$(grep "\[DiffusionRL\]" "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'weighted_diff_loss=[0-9.\-]+')
    last_wdiff=$(grep "\[DiffusionRL\]" "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'weighted_diff_loss=[0-9.\-]+')
    echo "  raw_diff_loss:    $first_diff -> $last_diff" | tee -a "$R"
    echo "  weighted_diff_loss: $first_wdiff -> $last_wdiff" | tee -a "$R"
  else
    echo "  INACTIVE — no [DiffusionRL] entries found" | tee -a "$R"
    echo "  Check TOML: [custom.alpamayo.diffusion_rl] enable=true?" | tee -a "$R"
  fi
  # Check for no-token warning
  no_token_warn=$(grep -c "No <traj_future_start> found" "$POLICY_LOG" 2>/dev/null || echo 0)
  no_adv_warn=$(grep -c "No positive advantages" "$POLICY_LOG" 2>/dev/null || echo 0)
  no_traj_warn=$(grep -c "No trajectory data found" "$POLICY_LOG" 2>/dev/null || echo 0)
  stack_fail=$(grep -c "Failed to stack trajectory" "$POLICY_LOG" 2>/dev/null || echo 0)
  echo "  Warnings: no_token=$no_token_warn  no_adv=$no_adv_warn  no_traj=$no_traj_warn  stack_fail=$stack_fail" | tee -a "$R"
else
  echo "  No policy log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 5. HCC Reward ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep -E "\[HCC-v2\]|\[HCC-RM\]|\[HCC-Reward\]" "$POLICY_LOG" | tail -20 | tee -a "$R"
  v2_count=$(grep -c '\[HCC-v2\]' "$POLICY_LOG" 2>/dev/null || echo 0)
  echo "  ($v2_count HCC-v2 entries)" | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 6. Reward Variance (template collapse check) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep "\[Step " "$POLICY_LOG" | tail -20 | grep -oP 'raw_reward=[0-9.\-]+' | grep -oP '[0-9.\-]+' | python3 -c "
import sys
vals = [float(v.strip()) for v in sys.stdin if v.strip()]
if vals:
    import numpy as np
    a = np.array(vals)
    print(f'  last 20 steps: mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}')
    status = 'HEALTHY (std>0.05)' if a.std() > 0.05 else 'COLLAPSED (std<0.05)'
    print(f'  variance: {status}')
else:
    print('  No values extracted')
" | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 7. CoC Debug ===" | tee -a "$R"
for f in "$POLICY_LOG" "$ROLL_LOG"; do
  if [ -f "$f" ]; then
    grep "\[CoC-Debug" "$f" | tail -10 | tee -a "$R"
  fi
done
echo "" | tee -a "$R"

echo "=== 8. VarProtect / AdvNorm ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep -E "\[VarProtect\]|\[AdvNorm" "$POLICY_LOG" | tail -20 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 9. Obstacle status ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  obs_count=$(grep -c "Failed to load obstacle" "$POLICY_LOG" 2>/dev/null || echo 0)
  echo "  obstacle load failures: $obs_count" | tee -a "$R"
  grep -i "obstacle" "$POLICY_LOG" "$ROLL_LOG" 2>/dev/null | head -5 | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 10. Errors (excluding known noise) ===" | tee -a "$R"
for f in "$POLICY_LOG" "$ROLL_LOG" "$CTRL_LOG"; do
  if [ -f "$f" ]; then
    grep -iE "error|traceback|exception|failed|warning|non-finite|killed|oom|cuda" "$f" 2>/dev/null \
      | grep -v "UserWarning\|dim_slice_info\|obstacle.offline\|Failed to load obstacle\|\[DiffusionRL\]\|\[HCC\]\|\[Step\]\|\[CoC-Debug\]\|\[VarProtect\]\|\[AdvNorm\]" \
      | tail -20 | tee -a "$R"
  fi
done
echo "" | tee -a "$R"

echo "=== 11. GPU Status ===" | tee -a "$R"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>/dev/null | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 12. TB Scalars (with diffusion RL) ===" | tee -a "$R"
if [ -n "$TB_PATH" ] && [ -d "$TB_PATH" ]; then
  python3 -c "
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # Use the full directory so all event files are read
    ea = EventAccumulator('$TB_PATH', size_guidance={'scalars': 0})
    ea.Reload()
    tags = sorted(ea.Tags().get('scalars', []))
    print(f'  {len(tags)} scalar tags found')

    keys = [
        # Core RL metrics
        'train/raw_reward_mean', 'train/raw_reward_std',
        'train/raw_reward_min', 'train/raw_reward_max',
        'train/loss_avg', 'train/grad_norm',
        'train/advantage_mean', 'train/advantage_std',
        # Reward components
        'train/reward_traj_L2_mean',
        'train/reward_scene_understanding_mean',
        'train/reward_decision_alignment_mean',
        'train/reward_coc_quality_mean',
        'train/reward_format_score_mean',
        # HCC-v2 grounded metrics
        'train/reward_grounded_coc_reward_mean',
        'train/reward_hallucination_score_mean',
        'train/reward_spatial_accuracy_mean',
        'train/reward_threat_score_mean',
        'train/reward_coc_unique_ratio_mean',
        'train/reward_diversity_score_mean',
        'train/reward_gt_decision_alignment_mean',
        # Diversity
        'train/unique_completion_ratio',
        'train/cot_word_count_mean',
        # NEW: Diffusion RL loss
        'train/diffusion_rl_loss',
    ]
    for t in keys:
        if t in tags:
            evts = ea.Scalars(t)[-15:]
            print(f'  {t}:')
            for e in evts:
                print(f'    step={e.step} val={e.value:.4f}')
        else:
            print(f'  {t}: NOT FOUND')

    other = [t for t in tags if t not in keys]
    if other:
        print(f'  Other ({len(other)}):')
        for t in other[:30]:
            evts = ea.Scalars(t)[-3:]
            last_val = evts[-1].value if evts else 'N/A'
            print(f'    {t}: last={last_val}')
except Exception as ex:
    print(f'  TB dump failed: {ex}')
" 2>&1 | tee -a "$R"
else
  echo "  No tensorboard data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 13. Quick Health Check ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  total_steps=$(grep -c '\[Step ' "$POLICY_LOG" 2>/dev/null || echo 0)
  total_steps=$(echo "$total_steps" | tr -d '[:space:]')
  echo "  Total steps: $total_steps" | tee -a "$R"

  if [ "$total_steps" -gt 0 ]; then
    first_v=$(grep "\[Step " "$POLICY_LOG" | head -1 | grep -oP 'raw_reward=[0-9.\-]+' | grep -oP '[0-9.\-]+$')
    last_v=$(grep "\[Step " "$POLICY_LOG" | tail -1 | grep -oP 'raw_reward=[0-9.\-]+' | grep -oP '[0-9.\-]+$')
    if [ -n "$first_v" ] && [ -n "$last_v" ]; then
      echo "  Reward: $first_v -> $last_v" | tee -a "$R"
      improved=$(python3 -c "
f=float('$first_v'); l=float('$last_v')
if l > f + 0.01: print('INCREASING')
elif abs(l-f) < 0.01: print('FLAT')
else: print('DECREASING')
" 2>/dev/null || echo "UNKNOWN")
      echo "  Reward trend: $improved" | tee -a "$R"
    fi

    # Diffusion RL status
    diff_count=$(grep -c '\[DiffusionRL\]' "$POLICY_LOG" 2>/dev/null || echo 0)
    if [ "$diff_count" -gt 0 ]; then
      echo "  Diffusion RL: ACTIVE ($diff_count entries)" | tee -a "$R"
      last_diff_raw=$(grep "\[DiffusionRL\]" "$POLICY_LOG" | tail -1 | grep -oP 'raw_diff_loss=[0-9.\-]+' | grep -oP '[0-9.\-]+$')
      echo "  Last raw_diff_loss: $last_diff_raw" | tee -a "$R"
    else
      echo "  Diffusion RL: INACTIVE" | tee -a "$R"
    fi

    # HCC version
    v2_count=$(grep -c '\[HCC-v2\]' "$POLICY_LOG" 2>/dev/null || echo 0)
    if [ "$v2_count" -gt 0 ]; then
      echo "  HCC version: v2 (grounded)" | tee -a "$R"
    else
      echo "  HCC version: v1 or unknown" | tee -a "$R"
    fi
  else
    echo "  No [Step] output — still in rollout/warmup" | tee -a "$R"
  fi
else
  echo "  No policy log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 14. Diffusion RL Loss Trend (last 30 steps) ===" | tee -a "$R"
if [ -f "$POLICY_LOG" ]; then
  grep "\[DiffusionRL\]" "$POLICY_LOG" 2>/dev/null | grep -oP 'raw_diff_loss=[0-9.\-]+' | grep -oP '[0-9.\-]+' | python3 -c "
import sys
vals = [float(v.strip()) for v in sys.stdin if v.strip()]
if vals:
    import numpy as np
    a = np.array(vals)
    print(f'  n={len(vals)} values')
    print(f'  mean={a.mean():.6f} std={a.std():.6f} min={a.min():.6f} max={a.max():.6f}')
    # Trend: compare first half vs second half
    if len(vals) >= 4:
        half = len(vals)//2
        first_half = a[:half].mean()
        second_half = a[half:].mean()
        pct = (second_half - first_half) / max(abs(first_half), 1e-6) * 100
        trend = 'INCREASING (bad — loss growing)' if pct > 5 else 'DECREASING (good — expert learning)' if pct < -5 else 'STABLE'
        print(f'  trend: {trend} ({pct:+.1f}%)')
        print(f'  first_half_mean={first_half:.6f}  second_half_mean={second_half:.6f}')
else:
    print('  No diffusion RL loss values found')
" | tee -a "$R"
else
  echo "  No data" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 15. Rollout Log (last 30 lines) ===" | tee -a "$R"
if [ -f "$ROLL_LOG" ]; then
  tail -30 "$ROLL_LOG" | tee -a "$R"
else
  echo "  No rollout log" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "======== END OF REPORT ========" | tee -a "$R"
echo ""
echo "Report saved to: $R ($(wc -l < "$R") lines)"