# EventVLA RoboTwin-Mem

This example contains the RoboTwin-Mem training and evaluation entrypoints for
EventVLA.

## Training

The default training config uses `pure_image_keyframe_memory`:

```bash
bash examples/RoboTwin-Mem/train_files/run_eventvla_train.sh
```

`SequentialEpisodeBatchSampler` is the only supported VLA training sampler. It
keeps each episode in a stable batch slot across consecutive batches so
slot-indexed runtime memory stays aligned with the episode stream.

To train/evaluate the visual-anchors-only setting, set:

```bash
MEMORY_ABLATION_MODE=raw_anchors_only
```

## Single Eval

Start a policy server:

```bash
bash examples/RoboTwin-Mem/eval_files/run_policy_server.sh <ckpt_path> <gpu_id> <port>
```

Run RoboTwin-Mem eval:

```bash
bash examples/RoboTwin-Mem/eval_files/eval.sh <task_name> <ckpt_path> <gpu_id> <port>
```

Supported memory modes are `pure_image_keyframe_memory` and
`raw_anchors_only`. Legacy starVLA `QwenOFT` checkpoint configs in those two
non-token modes are normalized to the public `EventVLA` framework path.

## Batch Eval

Run the default 8-task pure-image checkpoint config:

```bash
bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh
```

Run the raw-anchors-only checkpoint config:

```bash
bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_raw_anchors_only_qwen3oft_0423.sh
```

Dry-run without launching servers or simulation:

```bash
DRY_RUN=1 bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh
```

Batch eval writes a summary to:

```text
examples/RoboTwin-Mem/eval_batch/logs/batch_eval/<RUN_TAG>/summary.csv
```
