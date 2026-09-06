# Paper Evidence and Open Items

## Working Title

Resource-Aware Stage Partitioning for Pipelined Neural Network Inference on
Embedded CPU--FPGA Systems

## Claims Backed by Existing Artifacts

### ResNet18

- 200 correctness-passing native pipeline candidates.
- Best: `islands_3_9__10_19`, 10.5323018542 FPS.
- Best stage times: 82.483 / 46.073 / 42.465 / 2.932 ms.
- Manual best: `three_stage_f`, 10.3209254680 FPS.
- Measured worst: `three_stage_b`, 4.5591653502 FPS.
- RPC all-VTA params-once total: 176.712 ms, approximately 5.659 FPS.
- Fitted heuristic on 200 samples:
  - MAE 7.047 ms.
  - RMSE 9.905 ms.
  - predicted Top-20 recall@50 0.70.
  - original static Top-20 recall@50 0.15.

Primary data:

- `vta/tutorials/frontend/report_out/native_stage_pipeline_searches/20260507_v23_warm100`
- `vta/tutorials/frontend/report_out/native_stage_pipeline_searches/20260512_theory_build100_board`
- `vta/tutorials/frontend/report_out/yolov3_tiny_heuristic_search/heuristic_params_resnet18_v23_200.json`

### YOLOv3-tiny

- 200 candidates generated in two 100-candidate searches.
- 169 correctness-passing pipeline measurements; 31 build failures.
- Best: `yolo_multi_pool2_logits`, 5.4402324518 FPS.
- Best stage times: 169.398 / 176.032 / 5.163 ms.
- Earlier coarse best: 5.1131424908 FPS.
- Measured worst across the two searches: 1.2555704574 FPS.
- RPC all-VTA: 3.7243460582 FPS, 268.5035 ms.
- Native single-stage all-VTA: 2.9619880760 FPS, stage mean 338.9286 ms.

Primary data:

- `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_rpc_all_vta_runs20`
- `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260514_resnet_style_cpu_vta_cpu_100`
- `vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260514_yolo_multisplit100`

### Cross-Model Analysis

- `vta/tutorials/frontend/report_out/cross_model_split_search_analysis_20260515.md`

## Claims Requiring Additional Experiments

1. Fair all-VTA speedup comparison:
   run the all-VTA reference through the same native executor and warm-cache
   protocol used for split candidates.
2. Cost-model ablation:
   compare FLOPs only, balance only, balance+boundary, and the full model.
3. Cross-model generalization:
   train on one model and report Top-K recall/rank correlation on another.
4. Hardware generalization:
   repeat on another VTA configuration or board if available.
5. Third model:
   SqueezeNet is the most practical next target because its Fire modules and
   concatenations differ from both ResNet and YOLO while remaining tractable.
6. Energy:
   record board power and report frames/J in addition to frames/s.

## Wording Constraints

- Do not claim that GraphExecutor multi-request support is the main novelty.
- Do not claim that multiple VTA stages execute concurrently; the stable
  runner serializes VTA stage access with a global mutex.
- Do not claim raw equality for per-stage-quantized YOLO candidates; use the
  recorded raw sanity and detection correctness policy.
- Do not compare the 6.5215 FPS packed-graph smoke directly as a CPU/VTA split
  result; it used a different packed execution path.
