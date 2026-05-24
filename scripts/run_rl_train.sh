#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Alpamayo 1.5 RL Training Launch Script
# Usage: bash scripts/run_rl_train.sh [TOML_CONFIG]
#
# Prerequisites:
#   1. source switch_env a1_5_venv
#   2. cd /data/mnt_m181/z59900495/workspace/alpamayo1.5
#   3. Ensure PAI dataset exists at ALPAMAYO_PAI_LOCAL_DIR

set -e

# =============================================================================
# Environment Variables
# =============================================================================
export ALPAMAYO_WORKSPACE=/data/mnt_m181/z59900495/workspace/alpamayo1.5
export ALPAMAYO_MODEL_DIR=/data/mnt_m181/z59900495/workspace/alpamayo1.5/temp_model
export ALPAMAYO_PAI_LOCAL_DIR=/data/mnt_m181/z59900495/workspace/DownloadTool-master/pai_dataset
export ALPAMAYO_LOG_DIR=/root/temp_log_0425
export WANDB_MODE=offline
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export TMPDIR=/data/mnt_m181/z59900495/workspace/alpamayo1.5/torchelastic
export COSMOS_ROLLOUT_SKIP_ON_ERROR=1
export COSMOS_ROLLOUT_MAX_RETRIES=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# =============================================================================
# Config: default TOML, can override via argument
# =============================================================================
TOML_CONFIG="${1:-finetune/rl/toml/alpamayo_rvla_rl_local_test.toml}"

# =============================================================================
# Launch
# =============================================================================
echo "============================================"
echo "Alpamayo 1.5 RL Training"
echo "============================================"
echo "Workspace:  $ALPAMAYO_WORKSPACE"
echo "Model Dir:  $ALPAMAYO_MODEL_DIR"
echo "PAI Data:   $ALPAMAYO_PAI_LOCAL_DIR"
echo "Log Dir:    $ALPAMAYO_LOG_DIR"
echo "Config:     $TOML_CONFIG"
echo "============================================"

cosmos-rl \
    --config "$TOML_CONFIG" \
    --policy 1 \
    --rollout 1 \
    --log-dir "$ALPAMAYO_LOG_DIR" \
    finetune/rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py