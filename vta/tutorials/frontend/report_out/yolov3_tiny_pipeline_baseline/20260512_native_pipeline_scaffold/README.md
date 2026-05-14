# YOLOv3-tiny Pipeline Baseline

- Generated: 2026-05-12 15:58:12
- Board: n/a
- Port: 9090
- Modes: native_serial,native_pipeline
- Compile only: True

## Results

| mode | status | fps | mean ms | notes |
|---|---|---:|---:|---|
| native_serial | not_implemented | 0.000 | 0.000 | YOLOv3-tiny native stage pipeline needs a detection graph stage splitter. The existing ResNet18 stage splitter cannot be reused directly because it depends on ResNet unit metadata and MXNet block boundaries. |
| native_pipeline | not_implemented | 0.000 | 0.000 | YOLOv3-tiny native stage pipeline needs a detection graph stage splitter. The existing ResNet18 stage splitter cannot be reused directly because it depends on ResNet unit metadata and MXNet block boundaries. |

## Correctness

- Status: no_comparisons
- Notes: RPC baseline is missing or failed.; native_serial is not implemented yet.; native_pipeline is not implemented yet.

Native serial/pipeline modes are intentionally explicit `not_implemented` until a YOLO stage splitter is added.
