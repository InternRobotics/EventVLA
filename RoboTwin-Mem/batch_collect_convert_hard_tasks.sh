#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
GO1_DIR="${REPO_ROOT}/policy/GO1"

DEFAULT_MEM_TASKS="cover_blocks_hard,pick_the_unhidden_block,find_seal_and_seal_stamp,pick_objects_in_order,put_back_block_hard,press_button_keyframe,rearrange_blocks_hard,reproduce_route"

TASK_CONFIG="${1:-demo_clean}"
EXPERT_DATA_NUM="${2:-50}"
GPU_ID="${3:-1}"
REPO_ID_PREFIX="${4:-}"
HF_LEROBOT_HOME_DIR="${5:-/tmp/robotwin_mem_lerobotdata}"
TASKS_CSV="${6:-${DEFAULT_MEM_TASKS}}"

ROBOTWIN_MEM_ENV="${ROBOTWIN_MEM_ENV:-/shared/smartbot/yangganlin/anaconda3/envs/RMBench}"
LEROBOT_ENV="${LEROBOT_ENV:-/shared/smartbot/yangganlin/anaconda3/envs/lerobot_data}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/tmp/robotwin_mem_data}"
LOCAL_GO1_PROCESSED_DIR="${LOCAL_GO1_PROCESSED_DIR:-/tmp/robotwin_mem_go1_processed_data}"
FFMPEG_BIN="${ROBOTWIN_MEM_ENV}/bin/ffmpeg"

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No tasks configured. Please provide TASKS_CSV, e.g. pick_the_unhidden_block,cover_blocks_hard" >&2
  exit 1
fi

ROBOTWIN_MEM_PYTHON="${ROBOTWIN_MEM_ENV}/bin/python"
LEROBOT_PYTHON="${LEROBOT_ENV}/bin/python"

if [[ ! -x "${ROBOTWIN_MEM_PYTHON}" ]]; then
  echo "RoboTwin-Mem python not found: ${ROBOTWIN_MEM_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${LEROBOT_PYTHON}" ]]; then
  echo "LeRobot python not found: ${LEROBOT_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${FFMPEG_BIN}" ]]; then
  echo "ffmpeg not found in RMBench env: ${FFMPEG_BIN}" >&2
  exit 1
fi

export PATH="${ROBOTWIN_MEM_ENV}/bin:${LEROBOT_ENV}/bin:${PATH}"

setup_local_data_dir() {
  local repo_data_dir="${REPO_ROOT}/data"
  local backup_dir

  mkdir -p "${LOCAL_DATA_DIR}"

  if [[ -d "${repo_data_dir}" ]]; then
    echo "Migrating existing data to local NVMe: ${LOCAL_DATA_DIR}"
    cp -a "${repo_data_dir}/." "${LOCAL_DATA_DIR}/"
    backup_dir="${REPO_ROOT}/data_oss_backup_$(date +%Y%m%d_%H%M%S)"
    mv "${repo_data_dir}" "${backup_dir}"
    echo "Original OSS data directory backed up at: ${backup_dir}"
    return
  fi

  if [[ -e "${repo_data_dir}" && ! -L "${repo_data_dir}" ]]; then
    echo "Cannot replace non-directory data path: ${repo_data_dir}" >&2
    exit 1
  fi
}

setup_local_data_dir
export ROBOTWIN_MEM_DATA_DIR="${LOCAL_DATA_DIR}"
export ROBOTWIN_MEM_GO1_PROCESSED_DIR="${LOCAL_GO1_PROCESSED_DIR}"
mkdir -p "${HF_LEROBOT_HOME_DIR}"
mkdir -p "${ROBOTWIN_MEM_GO1_PROCESSED_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Data dir: ${ROBOTWIN_MEM_DATA_DIR}"
echo "GO1 processed data dir: ${ROBOTWIN_MEM_GO1_PROCESSED_DIR}"
echo "Task config: ${TASK_CONFIG}"
echo "Expert data num: ${EXPERT_DATA_NUM}"
echo "GPU id: ${GPU_ID}"
echo "HF_LEROBOT_HOME: ${HF_LEROBOT_HOME_DIR}"
echo "ffmpeg: $(command -v ffmpeg)"
echo "Tasks: ${TASKS_CSV}"
echo

collect_raw_data() {
  local task_name="$1"
  echo "=== [1/3] Collect raw data for ${task_name} ==="
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONWARNINGS=ignore::UserWarning \
    "${ROBOTWIN_MEM_PYTHON}" script/collect_data.py "${task_name}" "${TASK_CONFIG}"
    rm -rf "${ROBOTWIN_MEM_DATA_DIR}/${task_name}/${TASK_CONFIG}/.cache"
  )
}

convert_to_hdf5() {
  local task_name="$1"
  echo "=== [2/3] Convert raw data to HDF5 for ${task_name} ==="
  (
    cd "${GO1_DIR}"
    "${ROBOTWIN_MEM_PYTHON}" scripts/process_data.py "${task_name}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}"
  )
}

convert_to_lerobot() {
  local task_name="$1"
  local repo_id
  if [[ -n "${REPO_ID_PREFIX}" ]]; then
    repo_id="${REPO_ID_PREFIX}_${task_name}"
  else
    repo_id="${task_name}"
  fi

  echo "=== [3/3] Convert HDF5 to LeRobot for ${task_name} ==="
  echo "repo_id: ${repo_id}"
  (
    cd "${GO1_DIR}"
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME_DIR}" \
    "${LEROBOT_PYTHON}" scripts/convert_aloha_data_to_lerobot_robotwin.py \
      --raw_dir "${ROBOTWIN_MEM_GO1_PROCESSED_DIR}/${task_name}-${TASK_CONFIG}-${EXPERT_DATA_NUM}" \
      --repo_id "${repo_id}"
  )
}

for task_name in "${TASKS[@]}"; do
  task_name="$(echo "${task_name}" | xargs)"
  if [[ -z "${task_name}" ]]; then
    continue
  fi

  collect_raw_data "${task_name}"
  convert_to_hdf5 "${task_name}"
  convert_to_lerobot "${task_name}"
  echo "=== Finished ${task_name} ==="
  echo
done

echo "All RoboTwin-Mem task data pipelines finished."
