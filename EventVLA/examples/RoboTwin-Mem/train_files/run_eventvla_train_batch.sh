#!/bin/bash
#SBATCH -J robotwin_mem_eventvla
#SBATCH -p ebench_t
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm-%j-%x.out
#SBATCH -e slurm-%j-%x.err

set -euo pipefail

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT=$(cd -- "${REPO_ROOT}" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
else
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
fi
cd "${REPO_ROOT}"

Framework_name=${FRAMEWORK_NAME:-EventVLA}
freeze_module_list=${FREEZE_MODULE_LIST:-""}
base_vlm=${BASE_VLM:-/mnt/inspurfs/efm_t/yangganlin/models/download_models/Qwen3-VL-4B-Instruct}
config_yaml=${CONFIG_YAML:-./examples/RoboTwin-Mem/train_files/eventvla_robotwin_mem.yaml}
deepspeed_config=${DEEPSPEED_CONFIG:-eventvla/config/deepseeds/deepspeed_zero2.yaml}
train_entry=${TRAIN_ENTRY:-eventvla/training/train_eventvla.py}
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints}
data_root_dir=${1:-${ROBOTWIN_MEM_DATA_ROOT:-/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata}}
data_mix=${DATA_MIX:-robotwin_mem8}
memory_ablation_mode=${MEMORY_ABLATION_MODE:-pure_image_keyframe_memory}
resolved_profile=${memory_ablation_mode}
keyframe_train_memory_source=${KEYFRAME_TRAIN_MEMORY_SOURCE:-teacher_to_predict}
keyframe_train_memory_schedule=${KEYFRAME_TRAIN_MEMORY_SCHEDULE:-teacher_to_predict}
keyframe_schedule_teacher_prob_start=${KEYFRAME_SCHEDULE_TEACHER_PROB_START:-1.0}
keyframe_schedule_teacher_prob_end=${KEYFRAME_SCHEDULE_TEACHER_PROB_END:-0.0}

per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}
max_train_steps=${MAX_TRAIN_STEPS:-100000}
save_interval=${SAVE_INTERVAL:-10000}
keep_recent_checkpoints=${KEEP_RECENT_CHECKPOINTS:-2}
eval_interval=${EVAL_INTERVAL:-1000}
logging_frequency=${LOGGING_FREQUENCY:-100}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}
max_keyframe_images=${MAX_KEYFRAME_IMAGES:-5}
wandb_project=${WANDB_PROJECT:-null}
wandb_entity=${WANDB_ENTITY:-null}
memory_debug=${MEMORY_DEBUG:-true}
memory_debug_interval=${MEMORY_DEBUG_INTERVAL:-1}
memory_debug_first_steps=${MEMORY_DEBUG_FIRST_STEPS:-1}
run_date=$(date +%Y%m%d)
run_id=${RUN_ID:-${run_date}_${data_mix}_${memory_ablation_mode}_eventvla}

num_nodes=${SLURM_NNODES:-${NUM_NODES:-2}}
gpus_per_node=${GPUS_PER_NODE:-8}
total_gpus=$((gpus_per_node * num_nodes))
master_addr=${MASTER_ADDR:-$(if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1; else hostname; fi)}
master_port=${MASTER_PORT:-$((20000 + RANDOM % 10000))}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "[train] run_id=${run_id} data_mix=${data_mix}"
echo "[train] data_root_dir=${data_root_dir}"
echo "[train] config_yaml=${config_yaml}"
echo "[train] batch=${per_device_batch_size} steps=${max_train_steps} keyframes=${max_keyframe_images} debug=${memory_debug}"
echo "[dist] nodes=${num_nodes} gpus_per_node=${gpus_per_node} total_gpus=${total_gpus} master=${master_addr}:${master_port}"

export REPO_ROOT
export Framework_name freeze_module_list base_vlm config_yaml deepspeed_config train_entry
export run_root_dir data_root_dir data_mix run_id
export memory_ablation_mode resolved_profile
export keyframe_train_memory_source keyframe_train_memory_schedule
export keyframe_schedule_teacher_prob_start keyframe_schedule_teacher_prob_end
export per_device_batch_size max_train_steps save_interval keep_recent_checkpoints
export eval_interval logging_frequency gradient_accumulation_steps max_keyframe_images
export wandb_project wandb_entity
export memory_debug memory_debug_interval memory_debug_first_steps
export num_nodes gpus_per_node total_gpus master_addr master_port

srun_args=()
if [[ -n "${SLURM_JOBID:-}" ]]; then
  srun_args+=(--jobid "${SLURM_JOBID}")
fi

srun "${srun_args[@]}" bash -c '
set -euo pipefail
cd -- "$REPO_ROOT"
echo "[rank ${SLURM_PROCID:-0}] host=$(hostname)"

wandb_entity_args=()
if [[ -n "$wandb_entity" && "$wandb_entity" != "null" ]]; then
  wandb_entity_args=(--wandb_entity "$wandb_entity")
fi

accelerate launch \
  --config_file "$deepspeed_config" \
  --main_process_ip "$master_addr" \
  --main_process_port "$master_port" \
  --machine_rank "${SLURM_PROCID:-0}" \
  --num_machines "$num_nodes" \
  --num_processes "$total_gpus" \
  "$train_entry" \
  --config_yaml "$config_yaml" \
  --framework.name "$Framework_name" \
  --framework.memory_ablation_mode "$memory_ablation_mode" \
  --framework.qwenvl.base_vlm "$base_vlm" \
  --framework.memory_buffer.qwen_memory_injection.keyframe_image_position after_anchor_images_before_action \
  --framework.memory_buffer.qwen_memory_injection.max_keyframe_images "$max_keyframe_images" \
  --framework.memory_buffer.qwen_memory_injection.use_image_role_text true \
  --framework.memory_buffer.keyframe_loss_weight 1.0 \
  --framework.memory_buffer.keyframe_positive_weight 7.0 \
  --framework.memory_buffer.keyframe_threshold 0.5 \
  --framework.memory_buffer.keyframe_predict_mode chunk_future \
  --framework.memory_buffer.event_future_min_offset 1 \
  --framework.memory_buffer.event_commit_threshold 0.55 \
  --framework.memory_buffer.enable_delayed_chunk_event_commit true \
  --framework.memory_buffer.keyframe_train_memory_source "$keyframe_train_memory_source" \
  --framework.memory_buffer.keyframe_eval_memory_source predict \
  --framework.memory_buffer.keyframe_train_memory_schedule "$keyframe_train_memory_schedule" \
  --framework.memory_buffer.keyframe_schedule_warmup_steps 10000 \
  --framework.memory_buffer.keyframe_schedule_transition_steps 30000 \
  --framework.memory_buffer.keyframe_schedule_teacher_prob_start "$keyframe_schedule_teacher_prob_start" \
  --framework.memory_buffer.keyframe_schedule_teacher_prob_end "$keyframe_schedule_teacher_prob_end" \
  --framework.memory_buffer.keyframe_schedule_mix_granularity sample \
  --framework.memory_buffer.debug "$memory_debug" \
  --framework.memory_buffer.debug_interval "$memory_debug_interval" \
  --framework.memory_buffer.debug_first_steps "$memory_debug_first_steps" \
  --datasets.vla_data.use_sequential_episode_sampler true \
  --datasets.vla_data.sampling_interval 50 \
  --datasets.vla_data.chunk_keyframe_target_dilation 8 \
  --datasets.vla_data.chunk_keyframe_target_kernel raised_cosine \
  --datasets.vla_data.event_future_min_offset 1 \
  --datasets.vla_data.teacher_event_threshold 0.55 \
  --datasets.vla_data.keyframe_image_memory.max_keyframes "$max_keyframe_images" \
  --datasets.vla_data.keyframe_image_memory.include_current_keyframe true \
  --datasets.vla_data.keyframe_image_memory.order chronological \
  --datasets.vla_data.keyframe_image_memory.selection latest \
  --datasets.vla_data.keyframe_image_memory.view_mode include_names \
  --datasets.vla_data.keyframe_image_memory.include_names "[cam_high,head,main]" \
  --datasets.vla_data.keyframe_image_memory.exclude_name_patterns "[wrist]" \
  --datasets.vla_data.keyframe_image_memory.strict_single_view true \
  --datasets.vla_data.per_device_batch_size "$per_device_batch_size" \
  --datasets.vla_data.data_root_dir "$data_root_dir" \
  --datasets.vla_data.data_mix "$data_mix" \
  --trainer.freeze_modules "$freeze_module_list" \
  --trainer.max_train_steps "$max_train_steps" \
  --trainer.learning_rate.keyframe_head 1.0e-04 \
  --trainer.save_interval "$save_interval" \
  --trainer.keep_recent_checkpoints "$keep_recent_checkpoints" \
  --trainer.logging_frequency "$logging_frequency" \
  --trainer.eval_interval "$eval_interval" \
  --run_root_dir "$run_root_dir" \
  --run_id "$run_id" \
  --wandb_project "$wandb_project" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation_steps" \
  "${wandb_entity_args[@]}"
'
