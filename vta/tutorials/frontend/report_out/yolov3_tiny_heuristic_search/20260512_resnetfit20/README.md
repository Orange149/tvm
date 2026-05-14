# YOLOv3-tiny ResNet-Fit20 Heuristic Search

## Status

- ResNet18 heuristic fitting completed from 200 native measurements.
- YOLO candidate pool expanded to 30 ranked candidates to obtain 20 buildable true `cpu/vta/cpu` packages.
- Buildability: 20 buildable, 10 build failed.
- Board serial smoke: 20/20 native packages ran successfully, but 0/20 passed raw-output correctness against RPC all-VTA.
- Pipeline measurement was intentionally skipped because no true split passed the serial correctness gate.

## Heuristic Fit

- Training samples: 200 ResNet18 native measured candidates.
- Target: `measured_cycle_ms = 1000 / pipeline_throughput_fps`.
- Fit MAE: `7.047 ms`.
- Fit RMSE: `9.905 ms`.
- Predicted Top20 recall@50: `0.700`.
- Original static Top20 recall@50: `0.150`.
- Parameter file: `../heuristic_params_resnet18_v23_200.json`.

## YOLO Candidate Results

- Candidate ranking file: `yolo_heuristic_candidates.csv`.
- Buildability file: `buildability.csv`.
- Measurement file: `summary.json` / `summary.csv`.
- Build failures were concentrated in:
  - `*_to_logits`: VTA stage includes YOLO head/logit path that is too complex for this first split.
  - `data_to_*`: VTA stage starts from raw input and fails in the current explicit-pack path.
- Buildable candidates were coarse true `cpu/vta/cpu` variants ending at `trunk12`, `shared13`, `small_pre18`, or `dual_pre18_14`.

## Correctness Finding

All 20 buildable candidates ran in native serial mode, but all failed raw-output correctness.

The failures are numeric, not structural:

- Output count matches.
- Shapes and dtypes match.
- YOLO metadata outputs match.
- Tensor outputs differ substantially; representative max mean delta is about `0.52`.

This means the first true split implementation is functionally wired correctly, but independent per-stage quantization is not numerically equivalent to the RPC all-VTA graph. The split VTA stage currently quantizes only the isolated subgraph, while the RPC baseline quantizes and packs the full graph as one unit.

## Next Fix

Before pipeline ranking, fix quantization equivalence:

1. Build the true split from the already quantized full YOLO graph instead of quantizing each VTA stage independently.
2. Or relax the correctness reference to a split-specific RPC/native serial baseline and use detection-level correctness, but keep raw gate as a diagnostic metric.
3. Re-test serial for the 20 buildable candidates; only candidates passing correctness should enter pipeline throughput ranking.

No pipeline throughput result from this run should be treated as a valid YOLO split ranking.
