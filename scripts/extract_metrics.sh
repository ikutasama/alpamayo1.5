#!/bin/bash
# Alpamayo 1.5 RL Training Metrics Extraction Script
# Usage: bash extract_metrics.sh
# Output: prints report to stdout AND saves to /root/temp_log/training_report.txt

ROLL_LOG="/root/temp_log_0425/logs_latest/rollout_0.log"
POLICY_LOG="/root/temp_log_0425/logs_latest/policy_0.log"
CTRL_LOG="/root/temp_log_0425/logs_latest/controller.log"
TB_PATH=$(ls -t /root/temp_log/tensorboard/events.out.tfevents.* 2>/dev/null | head -1)
R="/root/temp_log/training_report.txt"
> "$R"

echo "======== Alpamayo RL Training Report ========" | tee -a "$R"
echo "Time: $(date)" | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 0. File Sizes ===" | tee -a "$R"
for f in "$POLICY_LOG" "$ROLL_LOG" "$CTRL_LOG"; do
  if [ -f "$f" ]; then
    echo "  $(wc -l < "$f") lines  $f" | tee -a "$R"
  else
    echo "  MISSING  $f" | tee -a "$R"
  fi
done
if [ -n "$TB_PATH" ]; then
  echo "  TB: $TB_PATH" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "=== 1. Controller Log (full) ===" | tee -a "$R"
cat "$CTRL_LOG" 2>/dev/null | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 2. Policy Log (last 50 lines) ===" | tee -a "$R"
tail -50 "$POLICY_LOG" 2>/dev/null | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 3. Rollout Log (last 50 lines) ===" | tee -a "$R"
tail -50 "$ROLL_LOG" 2>/dev/null | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 4. [Step] lines ===" | tee -a "$R"
grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -30 | tee -a "$R"
echo "  ($(grep -c '\[Step ' "$POLICY_LOG" 2>/dev/null || echo 0) total steps)" | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 5. HCC Reward ===" | tee -a "$R"
grep -E "\[HCC-v2\]|\[HCC-RM\]|\[HCC-Reward\]" "$POLICY_LOG" 2>/dev/null | tail -20 | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 6. CoC Debug ===" | tee -a "$R"
grep "\[CoC-Debug" "$POLICY_LOG" "$ROLL_LOG" 2>/dev/null | tail -15 | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 7. VarProtect / AdvNorm ===" | tee -a "$R"
grep -E "\[VarProtect\]|\[AdvNorm" "$POLICY_LOG" 2>/dev/null | tail -20 | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 8. Errors (excluding known warnings) ===" | tee -a "$R"
grep -iE "error|traceback|exception|failed|warning|non-finite|killed|oom|cuda" \
  "$POLICY_LOG" "$ROLL_LOG" "$CTRL_LOG" 2>/dev/null \
  | grep -v "UserWarning\|dim_slice_info\|obstacle.offline\|Failed to load obstacle" \
  | tail -30 | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 9. Obstacle status ===" | tee -a "$R"
obs_count=$(grep -c "Failed to load obstacle" "$POLICY_LOG" 2>/dev/null || echo 0)
echo "  obstacle load failures: $obs_count" | tee -a "$R"
grep -i "obstacle" "$POLICY_LOG" "$ROLL_LOG" 2>/dev/null | head -5 | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 10. Reward trend ===" | tee -a "$R"
first_r=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | head -1 | grep -oP 'raw_reward=[0-9.\-]+')
last_r=$(grep "\[Step " "$POLICY_LOG" 2>/dev/null | tail -1 | grep -oP 'raw_reward=[0-9.\-]+')
echo "  first: $first_r" | tee -a "$R"
echo "  last:  $last_r" | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 11. TB Scalars ===" | tee -a "$R"
python3 -c "
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator('$TB_PATH', size_guidance={'scalars': 0})
    ea.Reload()
    tags = sorted(ea.Tags().get('scalars', []))
    print(f'  {len(tags)} scalar tags')
    keys = ['train/raw_reward_mean','train/raw_reward_std','train/loss_avg',
            'train/grad_norm','train/advantage_mean','train/advantage_std',
            'train/reward_traj_L2_mean','train/reward_scene_understanding_mean',
            'train/reward_decision_alignment_mean','train/reward_coc_quality_mean',
            'train/cot_word_count_mean','train/unique_completion_ratio',
            'train/reward_object_recall_mean','train/reward_format_score_mean',
            'train/reward_grounded_coc_reward_mean','train/reward_hallucination_score_mean',
            'train/reward_spatial_accuracy_mean','train/reward_threat_score_mean',
            'train/reward_coc_unique_ratio_mean','train/reward_diversity_score_mean',
            'train/reward_gt_decision_alignment_mean']
    for t in keys:
        if t in tags:
            evts = ea.Scalars(t)[-10:]
            print(f'  {t}:')
            for e in evts:
                print(f'    step={e.step} val={e.value:.4f}')
    other = [t for t in tags if t not in keys]
    if other:
        print(f'  Other ({len(other)}): {other[:30]}')
except Exception as ex:
    print(f'  TB dump failed: {ex}')
" 2>&1 | tee -a "$R"
echo "" | tee -a "$R"

echo "=== 12. Quick Health Check ===" | tee -a "$R"
total_steps=$(grep -c "\[Step " "$POLICY_LOG" 2>/dev/null || echo "0")
echo "  Total training steps: $total_steps" | tee -a "$R"
if [ "$total_steps" -gt 0 ] && [ -n "$first_r" ] && [ -n "$last_r" ]; then
  first_v=$(echo "$first_r" | grep -oP '[0-9.\-]+$')
  last_v=$(echo "$last_r" | grep -oP '[0-9.\-]+$')
  echo "  Reward: $first_v -> $last_v" | tee -a "$R"
  improved=$(python3 -c "print('YES' if float('$last_v') > float('$first_v') + 0.01 else 'FLAT' if abs(float('$last_v')-float('$first_v'))<0.01 else 'DOWN')" 2>/dev/null)
  echo "  Reward trend: $improved" | tee -a "$R"
else
  echo "  No [Step] output yet — still in rollout/warmup phase" | tee -a "$R"
fi
echo "" | tee -a "$R"

echo "======== END OF REPORT ========" | tee -a "$R"
echo ""
echo "Report saved to: $R ($(wc -l < "$R") lines)"
