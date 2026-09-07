# EventVLA

**EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies**

> News: EventVLA has been accepted to the Conference on Robot Learning (CoRL) 2026.

## Quick Links

- 🌐 **Project Page**: [EventVLA Project Page](https://ganlin-yang.github.io/EventVLA.github.io/)
- 🤗 **Models**: [Hugging Face - EventVLA](https://huggingface.co/ganlinyang/EventVLA/tree/main)
- 📦 **Datasets**: [Hugging Face - RoboTwin-MeM](https://huggingface.co/datasets/ganlinyang/RoboTwin-MeM)
- 📄 **Paper**: [arXiv 2606.20092](https://arxiv.org/pdf/2606.20092)

EventVLA is a vision-language-action framework for long-horizon robotic manipulation. It introduces event-driven visual evidence memory: the policy detects task-relevant events, stores key visual evidence as raw keyframe images, and reuses this memory during action prediction.

This repository is released together with **RoboTwin-Mem**, a memory-dependent manipulation benchmark built on RoboTwin 2.0.


## 📆 To Do List
- [x] RoboTwin-MeM benchmark training and evaluation.
- [x] EventVLA training and evaluation code in simulation (RMBench and RoboTwin-MeM).
- [ ] Real-world inference and evaluation code.
- [ ] Real-world fine-tuned model.


<p align="center">
  <img src="assets/Overview.png" width="95%" alt="EventVLA Overview">
</p>
<p align="center">
  <img src="assets/RoboTwin-MeM.png" width="95%" alt="RoboTwin-MeM Benchmark">
</p>


## Released Resources

### Model Checkpoints

Model repository:

https://huggingface.co/ganlinyang/EventVLA/tree/main

| Checkpoint   | Usage                                               | Recommended file                            |
| ------------ | --------------------------------------------------- | ------------------------------------------- |
| RoboTwin-MeM | EventVLA evaluation on the eight RoboTwin-MeM tasks | `RoboTwin-MeM/final_model/pytorch_model.pt` |
| RMBench      | Evaluation on the RMBench benchmark                 | `RMBench/final_model/pytorch_model.pt`      |

Each release also contains:

```text
checkpoints/steps_100000_pytorch_model.pt
final_model/pytorch_model.pt
config.yaml
dataset_statistics.json
summary.jsonl
```

Use the RoboTwin-MeM checkpoint with the RoboTwin-MeM evaluation scripts. The RMBench checkpoint should be used with the corresponding RMBench environment and evaluation configuration.

### RoboTwin-MeM Dataset

Dataset repository:

https://huggingface.co/datasets/ganlinyang/RoboTwin-MeM

The dataset contains:

```text
RoboTwin-MeM/
├── hdf5/          # HDF5 trajectories
└── lerobot_2.1/   # LeRobot 2.1 training data
```

Both formats contain the following eight tasks:

```text
cover_blocks_hard
find_seal_and_seal_stamp
pick_objects_in_order
pick_the_unhidden_block
press_button_keyframe
put_back_block_hard
rearrange_blocks_hard
reproduce_route
```

## Installation

We recommend using two conda environments: one for RoboTwin-Mem simulation, and one for EventVLA model training / inference.

### RoboTwin-MeM Environment

```bash
conda create -n RoboTwin-MeM python=3.10 -y
conda activate RoboTwin-MeM

cd /path/to/RoboTwin-MeM
pip install -r script/requirements.txt
bash script/_install.sh
```

Download the simulation assets:

```bash
cd /path/to/RoboTwin-MeM
bash script/_download_assets.sh
```

### EventVLA Environment

```bash
conda create -n eventvla python=3.10 -y
conda activate eventvla

cd /path/to/EventVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```


Download the released resources from Hugging Face:

EventVLA checkpoints: https://huggingface.co/ganlinyang/EventVLA

RoboTwin-MeM dataset: https://huggingface.co/datasets/ganlinyang/RoboTwin-MeM

We recommend organizing them as follows:

```text
workspace/
├── checkpoints/
│   └── EventVLA/
│       ├── RoboTwin-MeM/
│       │   ├── final_model/pytorch_model.pt
│       │   ├── config.yaml
│       │   └── dataset_statistics.json
│       └── RMBench/
│           ├── final_model/pytorch_model.pt
│           ├── config.yaml
│           └── dataset_statistics.json
└── data/
    └── RoboTwin-MeM/
        ├── lerobot_2.1/
        └── hdf5/
```

Use the RoboTwin-MeM checkpoint for RoboTwin-MeM evaluation and the RMBench checkpoint for RMBench evaluation.

## Path Configuration

Before training or evaluation, replace the example paths with paths on your machine.

For training, set the base Qwen-VL / Qwen3-VL model path and the LeRobot data root:

```bash
BASE_VLM=/path/to/Qwen3-VL-4B-Instruct
ROBOTWIN_MEM_DATA_ROOT=/path/to/RoboTwin-MeM/lerobot_2.1
```

These values are passed to:

```text
framework.qwenvl.base_vlm
datasets.vla_data.data_root_dir
```

For evaluation, set the policy checkpoint, simulator checkout, and Python environments:

```bash
POLICY_CKPT_PATH=/path/to/checkpoints/EventVLA/RMBench/final_model/pytorch_model.pt
ROBOTWIN_MEM_ROOT=/path/to/RoboTwin-Mem
ROBOTWIN_MEM_PYTHON=/path/to/RoboTwin-Mem/env/bin/python
EVENTVLA_PYTHON=/path/to/eventvla/env/bin/python
```

Before running inference from a released checkpoint, also check the checkpoint directory's `config.yaml`. If it was produced on another machine, update `framework.qwenvl.base_vlm` to your local Qwen-VL / Qwen3-VL model path:

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/Qwen3-VL-4B-Instruct
```

## Evaluation

### Single-Task Evaluation

Start an EventVLA policy server:

```bash
cd /path/to/EventVLA

EVENTVLA_PYTHON=/path/to/eventvla/env/bin/python \
bash examples/RoboTwin-Mem/eval_files/run_policy_server.sh \
  /path/to/checkpoint.pt \
  0 \
  5840
```

Run RoboTwin-Mem evaluation:

```bash
cd /path/to/EventVLA

ROBOTWIN_MEM_ROOT=/path/to/RoboTwin-Mem \
ROBOTWIN_MEM_PYTHON=/path/to/RoboTwin-Mem/env/bin/python \
POLICY_CKPT_PATH=/path/to/checkpoint.pt \
HOST=127.0.0.1 \
PORT=5840 \
UNNORM_KEY=robotwin_mem \
bash examples/RoboTwin-Mem/eval_files/eval.sh \
  cover_blocks_hard \
  demo_clean \
  pure_image_keyframe_memory \
  0 \
  0
```

The arguments are:

```text
bash eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]
```

### Batch Evaluation

Run the default 8-task batch evaluation:

```bash
cd /path/to/EventVLA

bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh
```

Before running, edit the weight config file and replace local checkpoint / Python paths with your own paths:

```bash
EVENTVLA_PYTHON="/path/to/eventvla/env/bin/python"
ROBOTWIN_MEM_PYTHON="/path/to/RoboTwin-Mem/env/bin/python"
ROBOTWIN_MEM_ROOT="/path/to/RoboTwin-Mem"

WEIGHT_TASK1="/path/to/checkpoint.pt"
CKPT_SETTING_TASK1="pure_image_keyframe_memory"
```

### RMBench Evaluation

Start an EventVLA policy server:

```bash
cd /path/to/EventVLA

EVENTVLA_PYTHON=/path/to/eventvla/env/bin/python \
bash examples/RMBench/eval_files/run_policy_server.sh \
  /path/to/checkpoints/EventVLA/RMBench/final_model/pytorch_model.pt \
  0 \
  5840
```

Run a single RMBench task:

```bash
cd /path/to/EventVLA

ROBOTWIN_MEM_ROOT=/path/to/RoboTwin-Mem \
ROBOTWIN_MEM_PYTHON=/path/to/RoboTwin-Mem/env/bin/python \
POLICY_CKPT_PATH=/path/to/checkpoints/EventVLA/RMBench/final_model/pytorch_model.pt \
HOST=127.0.0.1 \
PORT=5840 \
UNNORM_KEY=new_embodiment \
ACTION_MODE=abs \
MEMORY_ABLATION_MODE=raw_anchors_only \
bash examples/RMBench/eval_files/eval.sh \
  battery_try \
  demo_clean \
  raw_anchors_only \
  0 \
  0
```

The arguments are:

```text
bash eval.sh <task_name> <task_config> [ckpt_setting] [seed] [gpu_id]
```

Run the 9-task RMBench batch evaluation:

```bash
cd /path/to/EventVLA

bash examples/RMBench/eval_batch/run_batch_eval.sh \
  examples/RMBench/eval_batch/weights_9tasks_raw_anchors_only_qwen3oft_0423.sh
```

Before running, edit the weight config file and replace the example checkpoint / Python paths with your own paths:

```bash
EVENTVLA_PYTHON="/path/to/eventvla/env/bin/python"
ROBOTWIN_MEM_PYTHON="/path/to/RoboTwin-Mem/env/bin/python"
ROBOTWIN_MEM_ROOT="/path/to/RoboTwin-Mem"
RAW_ANCHORS_ONLY_CKPT="/path/to/checkpoints/EventVLA/RMBench/final_model/pytorch_model.pt"
```

RMBench tasks:

```text
battery_try
blocks_ranking_try
cover_blocks
observe_and_pickup
press_button
put_back_block
rearrange_blocks
swap_T
swap_blocks
```

### Evaluation Outputs

Batch evaluation summary:

```text
examples/RoboTwin-Mem/eval_batch/logs/batch_eval/<RUN_TAG>/summary.csv
```

Per-task logs:

```text
examples/RoboTwin-Mem/eval_batch/logs/batch_eval/<RUN_TAG>/round_<ROUND_ID>/
```

RoboTwin-Mem result files:

```text
RoboTwin-Mem/eval_result/<task>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
├── _result.txt
└── eval_log.txt
```

## Training

Training scripts are under:

```text
examples/RoboTwin-Mem/train_files/
```


### Single-Node Training

```bash
cd "${EVENTVLA_ROOT}"

BASE_VLM=/path/to/Qwen3-VL-4B-Instruct \
ROBOTWIN_MEM_DATA_ROOT=/path/to/RoboTwin-MeM/lerobot_2.1 \
bash examples/RoboTwin-Mem/train_files/run_eventvla_train.sh
```

The data root can also be passed as the first argument:

```bash
bash examples/RoboTwin-Mem/train_files/run_eventvla_train.sh \
  /path/to/RoboTwin-MeM/lerobot_2.1
```

### SLURM Training

```bash
cd "${EVENTVLA_ROOT}"

BASE_VLM=/path/to/Qwen3-VL-4B-Instruct \
ROBOTWIN_MEM_DATA_ROOT=/path/to/RoboTwin-MeM/lerobot_2.1 \
sbatch examples/RoboTwin-Mem/train_files/run_eventvla_train_batch.sh
```

Alternatively:

```bash
sbatch examples/RoboTwin-Mem/train_files/run_eventvla_train_batch.sh \
  /path/to/RoboTwin-MeM/lerobot_2.1
```

The released RoboTwin-MeM configuration uses:

```text
memory_ablation_mode: pure_image_keyframe_memory
keyframe_image_memory.enabled: true
keyframe_train_memory_source: teacher_to_predict
keyframe_eval_memory_source: predict
max_keyframes: 4
temporal image anchors: first frame, t-30, t-15, current frame
action horizon: 50
maximum training steps: 100000
```

Training checkpoints are written under:

```text
EventVLA/results/Checkpoints/<run_id>/checkpoints/
```

Typical checkpoint names are:

```text
steps_<global_step>_pytorch_model.pt
steps_<global_step>_model.safetensors
```

## Citation

If you find this project useful, please cite:

```bibtex
@article{yang2026eventvla,
  title={EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies},
  author={Yang, Ganlin and Tu, Zhangzheng and Yang, Yuqiang and Mao, Sitong and Dong, Junyi and Chen, Tianxing and Peng, Jiaqi and Xiong, Jing and Cao, Jiafei and Dai, Jifeng and others},
  journal={arXiv preprint arXiv:2606.20092},
  year={2026}
}
```

This project is built upon the following open-source projects:

* [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin)
* [RMBench](https://github.com/robotwin-Platform/RMBench)
* [StarVLA](https://github.com/starVLA/starVLA)

Please also cite the corresponding papers when using EventVLA, RoboTwin-MeM, or the above projects in your research.

## Acknowledgement

EventVLA and RoboTwin-MeM are developed based on [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin), [RMBench](https://github.com/robotwin-Platform/RMBench), and [StarVLA](https://github.com/starVLA/starVLA). We thank the authors for releasing their code and simulation environments.

We also acknowledge Qwen-VL / Qwen3-VL, LeRobot, SAPIEN, MPLib, and CuRobo.


## License

This repository is released under the MIT license.
