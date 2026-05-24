# Grounded CoT-Action RL Reward Update

This note documents the May 2026 update to the Alpamayo 1.5 RL reward and diagnostics.

## Goal

The reward now optimizes for grounded reasoning-action consistency rather than CoT length. A good rollout should:

- mention facts that are checkable from the current sample,
- state an action intent when relevant,
- generate a trajectory that agrees with that stated intent,
- keep improving trajectory quality through ADE and comfort.

Long CoT is not rewarded by itself. Short CoT is not automatically clamped to `-1`; it simply receives low grounded fact and action-alignment scores.

## Reward Changes

Main files:

- `finetune/rl/rewards/hcc_reward.py`
- `finetune/rl/rewards/raa_reward.py`

The HCC reward now includes these components:

- `scene_understanding`: blended generic factual score and grounded fact score.
- `grounded_fact_score`: verifies whether CoT mentions expected ego-motion facts from the sample, such as maintain/decelerate/accelerate/turn/straight, plus scene and safety grounding.
- `grounded_fact_contradictions`: counts unsupported action claims, for example saying "turn left" when the GT future indicates right/straight.
- `raa_score`: alignment between action intent parsed from CoT and the decoded generated trajectory.
- `format_score`: presence and ordering of `<|cot_end|>`, `<|traj_future_start|>`, and `<|traj_future_end|>`.
- `traj_L2` and `comfort_reward`: trajectory quality signals retained from the previous implementation.

Important behavior change:

- The previous short-CoT hard gate was removed.
- The old positive length bonus is no longer part of the final reward.
- Empty/generic CoT no longer gets a high RAA score just because it avoids committing to an action.

## Diagnostics Added

Rollout logs now emit `rollout_generation_diagnostics`:

- `unique_completion_ratio_mean`
- `unique_completion_ratio_min`
- `cot_word_count_mean`
- `cot_word_count_min`
- `format_ok_rate`

Trainer logs now emit raw reward diagnostics before advantage normalization:

- `train/raw_reward_mean`
- `train/raw_reward_std`
- `train/raw_reward_min`
- `train/raw_reward_max`
- `train/unique_completion_ratio`
- `train/cot_word_count_mean`
- `train/reward_grounded_fact_score_mean`
- `train/reward_grounded_fact_coverage_mean`
- `train/reward_grounded_fact_contradictions_mean`
- `train/reward_raa_score_mean`
- `train/reward_raa_active_constraints_mean`
- `train/reward_traj_L2_mean`
- `train/advantage_mean`
- `train/advantage_std`

Use raw reward metrics to judge reward health. `advantage_mean/std` are normalized training signals and should not be interpreted as raw reward growth.

## Config Changes

`finetune/rl/toml/alpamayo_rvla_rl_local_test.toml` now defaults to:

- `trainer_type = "reasoning_vla_pgmo_grpo"`
- `kl_beta = 0.01`
- rollout `temperature = 0.8`
- rollout `top_p = 0.95`
- `repetition_penalty = 1.05`
- `max_new_tokens = 384`
- `allowed_outdated_steps = 10`
- token-level advantage routing enabled, with higher CoT weight.

The entry script also auto-selects the PGMO trainer when `[custom.alpamayo.pgmo].enable = true`, unless `COSMOS_TRAINER_TYPE` is explicitly set.

## Recommended First Server Run

Use the same command:

```bash
cosmos-rl \
  --config finetune/rl/toml/alpamayo_rvla_rl_local_test.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
```

Watch these first:

- `rollout_generation_diagnostics.unique_completion_ratio_mean`: should be well above zero; if near `1/n_generation`, the model is still templating.
- `train/raw_reward_std`: should not collapse to zero within a prompt group.
- `train/reward_grounded_fact_score_mean`: should rise if CoT becomes more sample-specific.
- `train/reward_raa_active_constraints_mean`: should rise if the model states actionable intent.
- `train/reward_raa_score_mean` and `train/reward_traj_L2_mean`: should improve together for the intended paper claim.

## Follow-Up Ablations

If rollouts are still too templated:

- raise rollout `temperature` to `0.9` or `1.0`,
- lower `top_p` to `0.9`,
- increase `n_generation` if memory allows,
- add a small SFT warmup set with high-quality grounded CoT before RL.

If trajectory quality drops while CoT improves:

- increase `traj_l2_weight`,
- lower `coc_quality_weight`,
- raise `kl_beta` to `0.02`.

