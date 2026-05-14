# YOLOv3-tiny full-graph quantized stage split build

This run changes the YOLO stage split flow to cut stages from the already quantized full YOLO Relay graph, instead of quantizing each VTA stage independently.

## What changed

- `search_yolov3_tiny_stage_splits.py` now has `--split-quantization-mode full_graph|per_stage`.
- The default is `full_graph`.
- In `full_graph` mode the script:
  - imports YOLOv3-tiny once,
  - quantizes the complete Relay module once,
  - collects stage split anchors from that quantized graph,
  - builds CPU/VTA/CPU stage packages from the already quantized subgraphs,
  - skips per-stage VTA quantization.
- Package cache reuse now includes `split_quantization_mode`, so old per-stage-quantized packages are not reused accidentally.

## Command

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
TEST_DATA_ROOT_PATH=/tmp/tvm_test_data \
MPLCONFIGDIR=/tmp/mpl \
PYTHONUNBUFFERED=1 \
VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/search_yolov3_tiny_stage_splits.py \
  --mode build \
  --build-only \
  --candidate-count 30 \
  --measure-count 20 \
  --split-quantization-mode full_graph \
  --output-root vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/20260512_resnetfit20_fullq \
  --heuristic-params-json vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/heuristic_params_resnet18_v23_200.json \
  --weights-path /tmp/tvm_test_data/darknet/yolov3-tiny.weights
```

## Result

- Board measurement was not run.
- Buildability rows: 30.
- Buildable packages: 20.
- Build failed packages: 10.
- Buildable package manifests record `"split_quantization_mode": "full_graph"`.

Most remaining build failures are candidates ending directly at both YOLO logits, where Relay shape inference rejects a reshape after full-graph quantization. The 20 buildable candidates use coarser tail boundaries such as `shared13`, `trunk12`, `small_pre18`, and `dual_pre18_14`.

## Next step

When board measurement is desired, run serial smoke on one buildable full-graph-quantized package first and compare raw outputs against the RPC all-VTA baseline. If serial raw correctness passes, continue with the 20-candidate pipeline measurement.
