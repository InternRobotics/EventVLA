#!/usr/bin/env bash
# 9-task EventVLA RMBench batch evaluation config for the legacy starVLA QwenOFT
# pure_image_keyframe_memory checkpoint.
#
# Usage:
#   bash examples/RMBench/eval_batch/run_batch_eval.sh \
#     examples/RMBench/eval_batch/weights_9tasks_pure_image_keyframe_memory_teacher_qwenoft.sh

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

GPU_SLOTS=(0 0 1 1 2 2 3 3 3)
PORT_SLOTS=(5902 5903 5904 5905 5906 5907 5908 5909 5910)

RMBENCH_9TASK_CKPT="/path/to/checkpoints/EventVLA/RMBench/final_model/pytorch_model.pt"

WEIGHT_TASK1="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK2="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK3="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK4="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK5="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK6="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK7="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK8="$RMBENCH_9TASK_CKPT"
WEIGHT_TASK9="$RMBENCH_9TASK_CKPT"

CKPT_SETTING_TASK1="pure_image_keyframe_memory"
CKPT_SETTING_TASK2="pure_image_keyframe_memory"
CKPT_SETTING_TASK3="pure_image_keyframe_memory"
CKPT_SETTING_TASK4="pure_image_keyframe_memory"
CKPT_SETTING_TASK5="pure_image_keyframe_memory"
CKPT_SETTING_TASK6="pure_image_keyframe_memory"
CKPT_SETTING_TASK7="pure_image_keyframe_memory"
CKPT_SETTING_TASK8="pure_image_keyframe_memory"
CKPT_SETTING_TASK9="pure_image_keyframe_memory"
