# YOLOv3-tiny Pipeline Baseline

- Generated: 2026-05-12 16:41:56
- Board: root@192.168.1.185
- Port: 9090
- Modes: native_pipeline
- Compile only: False

## Results

| mode | status | fps | mean ms | notes |
|---|---|---:|---:|---|
| rpc_baseline | ok | 3.724 | 268.504 |  |
| native_pipeline | ok | 2.962 | 1254.550 |  |

## Correctness

- Status: passed
- Notes: n/a

Native modes currently validate a single all-VTA stage through the native runner with raw multi-output summaries. A true YOLO `cpu/vta/cpu` Relay stage split is still a separate next step.
