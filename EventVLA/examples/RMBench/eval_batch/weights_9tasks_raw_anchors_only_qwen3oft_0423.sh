#!/usr/bin/env bash
# 9-task EventVLA RMBench batch evaluation config for the legacy starVLA Qwen3OFT
# raw_anchors_only checkpoint.
#
# Usage:
#   bash examples/RMBench/eval_batch/run_batch_eval.sh \
#     examples/RMBench/eval_batch/weights_9tasks_raw_anchors_only_qwen3oft_0423.sh

TASK_CONFIG="demo_clean"
SEED=0
HOST="127.0.0.1"
UNNORM_KEY="new_embodiment"
ACTION_MODE="abs"
INSTRUCTION_TYPE="unseen"
USE_BF16=1

SERVER_READY_TIMEOUT=600
SERVER_START_MAX_RETRIES=2
SERVER_RETRY_SLEEP=20
SERVER_START_STAGGER_SECONDS=8
SERVER_STARTUP_STALL_TIMEOUT=300
AUTO_KILL_PORT_OCCUPY=1

EVENTVLA_PYTHON="/path/to/eventvla/env/bin/python"
ROBOTWIN_MEM_PYTHON="/path/to/RoboTwin-Mem/env/bin/python"
ROBOTWIN_MEM_ROOT="/path/to/RoboTwin-Mem"

POLICY_NAME="model2robotwin_mem_interface"

TASKS=(
  "battery_try"
  "blocks_ranking_try"
  "cover_blocks"
  "observe_and_pickup"
  "press_button"
  "put_back_block"
  "rearrange_blocks"
  "swap_T"
  "swap_blocks"
)

GPU_SLOTS=(4 4 5 5 6 6 7 7 7)
PORT_SLOTS=(5802 5803 5804 5805 5806 5807 5808 5809 5810)

RAW_ANCHORS_ONLY_CKPT="/path/to/checkpoints/EventVLA/RMBench/final_model/pytorch_model.pt"

WEIGHT_TASK1="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK2="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK3="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK4="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK5="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK6="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK7="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK8="$RAW_ANCHORS_ONLY_CKPT"
WEIGHT_TASK9="$RAW_ANCHORS_ONLY_CKPT"

CKPT_SETTING_TASK1="raw_anchors_only"
CKPT_SETTING_TASK2="raw_anchors_only"
CKPT_SETTING_TASK3="raw_anchors_only"
CKPT_SETTING_TASK4="raw_anchors_only"
CKPT_SETTING_TASK5="raw_anchors_only"
CKPT_SETTING_TASK6="raw_anchors_only"
CKPT_SETTING_TASK7="raw_anchors_only"
CKPT_SETTING_TASK8="raw_anchors_only"
CKPT_SETTING_TASK9="raw_anchors_only"
