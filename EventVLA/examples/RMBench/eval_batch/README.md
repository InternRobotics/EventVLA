# EventVLA RMBench Batch Eval

RMBench batch eval uses the same RoboTwin-Mem evaluation checkout:

```text
/path/to/RoboTwin-Mem
```

Run the 9-task legacy Qwen3OFT raw-anchors-only checkpoint:

```bash
cd /path/to/EventVLA
bash examples/RMBench/eval_batch/run_batch_eval.sh \
  examples/RMBench/eval_batch/weights_9tasks_raw_anchors_only_qwen3oft_0423.sh
```

`run_batch_eval.sh` defaults to that raw-anchors-only config:

```bash
cd /path/to/EventVLA
bash examples/RMBench/eval_batch/run_batch_eval.sh
```

Dry-run the wiring without launching policy servers or simulation:

```bash
DRY_RUN=1 bash examples/RMBench/eval_batch/run_batch_eval.sh \
  examples/RMBench/eval_batch/weights_9tasks_raw_anchors_only_qwen3oft_0423.sh
```

Outputs are written under:

```text
examples/RMBench/eval_batch/logs/batch_eval/<RUN_TAG>/summary.csv
/path/to/RoboTwin-Mem/eval_result/<task>/<policy>/<task_config>/<ckpt_setting>/<timestamp>/
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
