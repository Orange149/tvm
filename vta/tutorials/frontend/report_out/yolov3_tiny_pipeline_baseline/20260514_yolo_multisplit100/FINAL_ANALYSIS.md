# YOLOv3-tiny Multi-Island Search Final Analysis

## Run Summary

- Board: `root@192.168.1.112`
- Search policy: Relay per-stage split, up to 3 VTA islands
- Candidate target: 100
- Buildable candidates: 81 / 100
- Build failed candidates: 19 / 100
- Serial smoke: 81 / 81 passed detection gate and raw sanity gate
- Pipeline runs: 81 / 81 completed successfully
- Pipeline correctness: 81 / 81 passed detection gate and raw sanity gate
- Remote cleanup: no candidate run directory remained under `/var/volatile/yolov3_tiny_pipeline_search`

Raw tensor exact matching is still not expected for these per-stage quantized splits. The ranking uses `detection_gate_passed=true` and `raw_sanity_passed=true`; all ranked pipeline rows satisfy both.

## Top Results

| Rank | Candidate | Pattern | FPS | Stage Times ms | Bottleneck |
|---:|---|---|---:|---|---|
| 1 | `yolo_multi_pool2_logits` | `cpu/vta/cpu` | 5.4402 | `[169.40, 176.03, 5.16]` | VTA stage |
| 2 | `yolo_multi_pool2_shared13__small_pre18_logits` | `cpu/vta/cpu/vta/cpu` | 5.2789 | `[182.87, 108.71, 5.02, 80.09, 7.45]` | CPU prefix |
| 3 | `yolo_multi_pool2_trunk12__small_pre18_logits` | `cpu/vta/cpu/vta/cpu` | 5.1381 | `[192.33, 103.90, 25.12, 82.41, 7.54]` | CPU prefix |
| 4 | `yolo_multi_pool2_dual_pre_logits` | `cpu/vta/cpu` | 4.9278 | `[203.31, 165.50, 87.95]` | CPU prefix |
| 5 | `yolo_multi_pool1_trunk12__small_pre18_dual_pre_logits` | `cpu/vta/cpu/vta/cpu` | 4.9060 | `[148.63, 163.66, 27.30, 56.79, 88.85]` | first VTA island |
| 6 | `yolo_multi_pool3_logits` | `cpu/vta/cpu` | 4.8877 | `[205.44, 152.06, 7.44]` | CPU prefix |
| 7 | `yolo_multi_pool0_pool1__pool3_small_pre18__dual_pre18_14_logits` | `cpu/vta/cpu/vta/cpu/vta/cpu` | 4.8520 | `[81.94, 73.85, 173.15, 94.08, 126.24, 84.04, 9.24]` | middle CPU stage |
| 8 | `yolo_multi_pool2_shared13__small_pre18_dual_pre_logits` | `cpu/vta/cpu/vta/cpu` | 4.7821 | `[208.29, 119.09, 2.55, 58.37, 88.93]` | CPU prefix |
| 9 | `yolo_multi_pool0_pool1__pool3_dual_pre_logits` | `cpu/vta/cpu/vta/cpu` | 4.7728 | `[79.34, 76.34, 149.34, 145.23, 85.88]` | middle CPU stage |
| 10 | `yolo_multi_pool1_pool2__pool3_dual_pre_logits` | `cpu/vta/cpu/vta/cpu` | 4.6340 | `[162.00, 68.35, 79.95, 147.96, 111.97]` | CPU prefix |

## Baseline Comparison

- RPC all-VTA baseline: 3.724 fps
- Previous packed smoke: 6.5215 fps
- Previous packed hetero best: 4.2922 fps
- Previous ResNet-style 3-stage YOLO candidate: 5.113 fps
- Current multi-island best: 5.4402 fps

The new best improves over the prior ResNet-style 3-stage candidate and packed hetero best, but it still does not beat the packed smoke result. The packed smoke path likely benefits from lower per-stage boundary overhead and a more favorable pre-existing packed graph execution path.

## Granularity Findings

| Island Count | Stage Count | N | Best FPS | Median FPS | Mean FPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 28 | 5.4402 | 2.7280 | 3.0663 |
| 2 | 5 | 35 | 5.2789 | 4.4055 | 4.2023 |
| 3 | 7 | 18 | 4.8520 | 4.2981 | 4.3146 |

One-island candidates have the single best result, but their quality is highly sensitive to the boundary. Two- and three-island candidates have better median throughput, because the search avoids very bad coarse cuts, but the best fine-grained candidates do not exceed the best coarse split.

The key reason is visible in the stage times. The best one-island split is well balanced between CPU prefix and VTA body: `169.40 ms` vs `176.03 ms`, with a tiny CPU tail. Many finer splits reduce one VTA island but create either a large CPU middle stage or extra boundary transfers. With the VTA global mutex, multiple VTA islands do not run concurrently on VTA; their VTA occupancy adds up, while CPU/VTA boundary cost increases.

## Practical Heuristic

For YOLOv3-tiny on this board, the next heuristic should favor:

1. Start VTA after `pool2`.
   Earlier starts make CPU prefix smaller but increase boundary pressure and can hurt balance. Later starts leave too much CPU prefix.
2. Keep the main trunk through logits in one VTA island when possible.
   `pool2 -> logits` is currently the best balance point.
3. Allow a second VTA island only when it separates a real conv-heavy head and does not introduce a large CPU middle stage.
   Good examples are `pool2 -> shared13` plus `small_pre18 -> logits`.
4. Avoid 3-island splits unless they are used for diversity/control.
   They can be correct and stable, but the best observed result is lower than the best 1- and 2-island schemes.
5. Penalize large route/pool4 boundaries and middle CPU stages.
   The 19 build failures are concentrated around `pool4_route` and `shared13` route-style boundaries, and successful fine splits often lose throughput when the middle CPU stage becomes the bottleneck.

## Recommended Next Step

Use `yolo_multi_pool2_logits` as the current best YOLOv3-tiny native pipeline baseline. For a second pass, search locally around `pool2` start boundaries and compare:

- `pool2 -> logits`
- `pool2 -> dual_pre_logits`
- `pool2 -> shared13` plus a head VTA island
- `pool2 -> trunk12` plus a head VTA island

The search should not expand to arbitrary op-level cuts until there is evidence that boundary overhead can be reduced or VTA stage submission can overlap more effectively.
