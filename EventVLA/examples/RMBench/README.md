# EventVLA RMBench

This example mirrors `examples/RoboTwin-Mem` and runs against the same
RoboTwin-Mem checkout:

```text
/path/to/RoboTwin-Mem
```

The RMBench task list is:

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

The RMBench setting uses `raw_anchors_only`.

## Training

Run training with the RMBench example config:

```bash
bash examples/RMBench/train_files/run_eventvla_train.sh
```

`SequentialEpisodeBatchSampler` is the only supported VLA training sampler. It
keeps each episode in a stable batch slot across consecutive batches so
slot-indexed runtime memory stays aligned with the episode stream.

To force the RMBench raw-anchors-only setting, set:

```bash
MEMORY_ABLATION_MODE=raw_anchors_only
```

## RMBench Inference

Start a policy server:

```bash
bash examples/RMBench/eval_files/run_policy_server.sh <ckpt_path> <gpu_id> <port>
```

Run a single RMBench task:

```bash
MEMORY_ABLATION_MODE=raw_anchors_only \
ROBOTWIN_MEM_ROOT=/path/to/RoboTwin-Mem \
bash examples/RMBench/eval_files/eval.sh <task_name> demo_clean raw_anchors_only <seed> <gpu_id>
```

Supported RMBench inference mode: `raw_anchors_only`.

## Batch Eval

Run the 9-task RMBench raw-anchors-only checkpoint config:

```bash
bash examples/RMBench/eval_batch/run_batch_eval.sh \
  examples/RMBench/eval_batch/weights_9tasks_raw_anchors_only_qwen3oft_0423.sh
```

`run_batch_eval.sh` uses the raw-anchors-only config above by default, so this is
equivalent:

```bash
bash examples/RMBench/eval_batch/run_batch_eval.sh
```

Dry-run without launching servers or simulation:

```bash
DRY_RUN=1 bash examples/RMBench/eval_batch/run_batch_eval.sh \
  examples/RMBench/eval_batch/weights_9tasks_raw_anchors_only_qwen3oft_0423.sh
```

Batch eval writes a summary to:

```text
examples/RMBench/eval_batch/logs/batch_eval/<RUN_TAG>/summary.csv
```
