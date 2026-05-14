# YOLOv3-tiny Baseline Status

## Completed

- Added standalone script: `vta/tutorials/frontend/test_yolov3_tiny_pipeline_baseline.py`
- `deploy_detection.py` was not modified.
- `python -m py_compile` passed for the new script.
- YOLOv3-tiny weights were downloaded to:
  `/tmp/tvm_test_data/darknet/yolov3-tiny.weights`
- RPC server was started on `192.168.1.185:9090` using the HPC runtime.
- RPC all-VTA compile-only passed:
  `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_rpc_compile_only`
- RPC all-VTA smoke passed:
  `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_rpc_all_vta_smoke`
- RPC all-VTA formal baseline passed:
  `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_rpc_all_vta_runs20`
- Native runner raw-output support was added to:
  `vta/apps/native_deploy/vta_stage_pipeline_runner.cc`
- Native serial all-VTA stage smoke passed:
  `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_native_serial_smoke`
- Native pipeline all-VTA stage baseline passed:
  `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_native_pipeline_runs20`

## RPC all-VTA Baseline

- Status: `ok`
- Mean latency: `268.50351293333335 ms`
- Std: `0.0722622058311879 ms`
- FPS: `3.724346058177234`
- Raw outputs: `8`
- Detection count: `3`

Top detections:

| rank | class | score | box |
|---:|---|---:|---|
| 1 | person | 0.9619008302688599 | left=192 top=111 right=277 bottom=372 |
| 2 | dog | 0.9111211895942688 | left=90 top=240 right=184 bottom=369 |
| 3 | horse | 0.7099802494049072 | left=404 top=122 right=612 bottom=340 |

## Native Runner Baseline

Current native baseline uses one all-VTA stage (`stage_split=single_stage_all_vta`) through the generic native runner. This validates native packaging, board deployment, runner execution, multi-output YOLO raw tensor capture, and RPC raw-output correctness comparison. It is not yet the final coarse `cpu/vta/cpu` split.

| mode | runs | throughput fps | mean total latency ms | stage0 mean ms | correctness |
|---|---:|---:|---:|---:|---|
| native_serial | 2 | 2.854595595465187 | 350.312318 | 350.2586325 | passed |
| native_pipeline | 20 | 2.9619880760327932 | 1254.5500459999998 | 338.92856779999994 | passed |

The native pipeline `fps` is computed from completion timestamps, not queued per-frame latency. Raw correctness is passed against RPC all-VTA with matching output count, shape, dtype, and low numeric drift; max mean delta is `7.14412994318181e-08`.

## Next Required Work

Implement a true YOLOv3-tiny coarse stage splitter before heuristic split search:

- Extract YOLO conv/head coarse units from the Relay/Darknet graph.
- Define stage input/output schemas for multi-output YOLO heads.
- Build a fixed coarse `cpu/vta/cpu` native package.
- Run the true coarse split in native serial against the RPC all-VTA raw output signature.
- Only after true coarse serial passes, run true coarse native pipeline.
