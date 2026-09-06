# Cross-Model Split Search Analysis

Date: 2026-05-15

This note summarizes the measured split-search results for ResNet18 and YOLOv3-tiny, then extracts rules that can be reused as a heuristic search policy.

## Data Scope

### ResNet18

The ResNet18 dataset contains 200 measured native pipeline candidates:

| Dataset | Pipeline OK | Best FPS | Median FPS | Mean FPS |
|---|---:|---:|---:|---:|
| `20260507_v23_warm100` | 100 | 10.5323 | 8.1576 | 8.1268 |
| `20260512_theory_build100_board` | 100 | 10.2003 | 8.0106 | 8.0553 |
| Combined | 200 | 10.5323 | 8.0894 | 8.0911 |

The global best is:

```text
islands_3_9__10_19
throughput = 10.5323 fps
pattern = cpu / vta / vta / cpu
effective pattern ~= cpu / vta-block / cpu
```

The two VTA stages are adjacent, so this is physically a 4-stage graph but resource-wise behaves like a large VTA block between CPU prefix and CPU tail.

### YOLOv3-tiny

The YOLOv3-tiny dataset contains two 100-candidate runs. 200 candidates were generated; 169 became measured, correctness-passing pipeline rows because build failures were filtered before board ranking.

| Dataset | Generated | Pipeline OK | Build Failed | Best FPS | Median FPS | Mean FPS |
|---|---:|---:|---:|---:|---:|---:|
| `20260514_resnet_style_cpu_vta_cpu_100` | 100 | 88 | 12 | 5.1131 | 2.4033 | 2.7630 |
| `20260514_yolo_multisplit100` | 100 | 81 | 19 | 5.4402 | 4.1412 | 3.8346 |
| Combined measured | 200 generated | 169 measured | 31 | 5.4402 | 3.1794 | 3.2766 |

The global best is:

```text
yolo_multi_pool2_logits
throughput = 5.4402 fps
pattern = cpu / vta / cpu
stage times = [169.40, 176.03, 5.16] ms
```

## Best Candidates

### ResNet18 Top Patterns

| Rank | Candidate | FPS | Stage Count | VTA Islands | Boundary Bytes |
|---:|---|---:|---:|---:|---:|
| 1 | `islands_3_9__10_19` | 10.5323 | 4 | 2 | 1.3046 MB |
| 2 | `islands_3_9__10_18` | 10.5243 | 4 | 2 | 1.4049 MB |
| 3 | `islands_3_8__10_19` | 10.2417 | 5 | 2 | 2.1074 MB |
| 4 | `islands_3_7__10_18` | 10.2003 | 5 | 2 | 1.8063 MB |
| 5 | `islands_3_7__10_19` | 9.8739 | 5 | 2 | 1.7060 MB |

The strongest ResNet18 family is:

```text
CPU: stem + layer1_block0
VTA island 1: layer1_block1 through layer2_block1 area
VTA island 2: layer3_block0 through layer4_block1 area
CPU: head
```

### YOLOv3-tiny Top Patterns

| Rank | Candidate | FPS | Pattern | Stage Times ms |
|---:|---|---:|---|---|
| 1 | `yolo_multi_pool2_logits` | 5.4402 | `cpu/vta/cpu` | `[169.40, 176.03, 5.16]` |
| 2 | `yolo_multi_pool2_shared13__small_pre18_logits` | 5.2789 | `cpu/vta/cpu/vta/cpu` | `[182.87, 108.71, 5.02, 80.09, 7.45]` |
| 3 | `yolo_multi_pool2_trunk12__small_pre18_logits` | 5.1381 | `cpu/vta/cpu/vta/cpu` | `[192.33, 103.90, 25.12, 82.41, 7.54]` |
| 4 | `yolo_pool2_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000` | 5.1131 | `cpu/vta/cpu` | `[196.00, 165.12, 137.61]` |
| 5 | `yolo_pool2_to_dual_pre_logits__rt_s04_s23_q2_poll1000_post1000` | 5.0391 | `cpu/vta/cpu` | `[200.92, 169.98, 109.21]` |

The strongest YOLOv3-tiny family is:

```text
CPU: input through pool2
VTA: pool2 through logits, or pool2 through shared/trunk plus a small head island
CPU: final detection/decode/NMS tail
```

## Granularity Patterns

### ResNet18

| VTA Island Count | N | Best FPS | Median FPS | Mean FPS |
|---:|---:|---:|---:|---:|
| 1 | 47 | 10.5323 | 7.4267 | 7.6649 |
| 2 | 140 | 10.2417 | 8.1709 | 8.2497 |
| 3 | 13 | 8.2245 | 7.9280 | 7.9241 |

The top two ResNet rows have two adjacent VTA islands and only one effective CPU/VTA/CPU resource block. True three-island splits have a lower ceiling.

### YOLOv3-tiny

| VTA Island Count | N | Best FPS | Median FPS | Mean FPS |
|---:|---:|---:|---:|---:|
| 1 | 116 | 5.4402 | 2.5537 | 2.8362 |
| 2 | 35 | 5.2789 | 4.4055 | 4.2023 |
| 3 | 18 | 4.8520 | 4.2981 | 4.3146 |

YOLO's 2- and 3-island candidates have better median throughput because the candidate policy avoids many bad coarse cuts. However, the best 1-island split still wins because it gives the cleanest stage balance with the least boundary overhead.

## Cross-Model Rules

### 1. Optimize stage balance, not VTA coverage

The best candidates are not the ones that put the most ops on VTA. They are the ones where:

```text
max_cpu_stage_ms ~= total_vta_occupied_ms
```

Good examples:

- ResNet18 best: CPU prefix is the bottleneck, while adjacent VTA stages are shorter but well utilized.
- YOLO best: CPU prefix `169.40 ms` and VTA body `176.03 ms` are nearly balanced, with a tiny CPU tail.

Bad examples:

- YOLO splits ending before logits leave a large CPU tail.
- YOLO splits starting too early overload VTA with high-resolution early layers.
- ResNet splits with three islands introduce extra boundaries without raising effective overlap.

### 2. Treat VTA as one serialized resource

The native runner uses a global VTA mutex, so multiple VTA stages cannot be scored as independent parallel stages. The correct core metric is:

```text
predicted_cycle_ms =
  max(max_cpu_stage_ms, total_vta_occupied_ms)
  + boundary_penalty_ms
  + dma_fragmentation_penalty_ms
  + sram_spill_penalty_ms
  + launch_sync_penalty_ms
```

This explains why fine-grained multi-island YOLO splits rarely beat `pool2 -> logits`: VTA work is serialized, while each extra island adds boundary and launch overhead.

### 3. CPU is usually best for high-resolution prefix and small semantic tail

Across both networks:

- High-resolution prefix often has large activation tensors and high pack/DMA overhead.
- Very small tails are not worth moving to VTA.
- VTA is strongest on the middle conv-heavy trunk.

Model-specific placements:

- ResNet18: keep `stem + layer1_block0` on CPU, move `layer1_block1` through late residual blocks to VTA, keep `head` on CPU.
- YOLOv3-tiny: keep input through `pool2` on CPU, move the main trunk/logits path to VTA, keep detection decode/NMS on CPU.

### 4. Boundary cost is architecture-dependent

Boundary bytes alone are not a universal predictor. ResNet18 can tolerate around 1-2 MB boundaries in the best family. YOLOv3-tiny has more fragile route/head boundaries:

- Best YOLO one-island boundary: about 3.63 MB.
- Good two-island YOLO boundary: about 5.45-5.97 MB.
- A strong three-island YOLO candidate can exceed 20 MB boundary traffic but then becomes bottlenecked by middle CPU stages.

For YOLO, route/concat/head boundaries should receive a stronger penalty than normal sequential boundaries.

### 5. More stages improve median only when candidate filtering removes bad coarse cuts

ResNet18 and YOLO both show that more stages do not guarantee a better peak.

- ResNet18 peak: effective CPU/VTA/CPU block.
- YOLO peak: CPU/VTA/CPU.
- YOLO 2- and 3-island median is better than 1-island median, but their peak is lower.

This means stage count is useful as a diversity axis, not as a direct objective.

## Updated Heuristic

Use the following ranking policy for future models:

```text
score_ms =
  max(max_cpu_stage_ms, total_vta_occupied_ms)
  + boundary_crossing_ms
  + dma_fragmentation_ms
  + sram_spill_risk_ms
  + cpu_tail_penalty_ms
  + small_vta_island_penalty_ms
  + branch_boundary_penalty_ms
  + launch_sync_penalty_ms
```

Where:

- `max_cpu_stage_ms`: maximum predicted CPU stage time.
- `total_vta_occupied_ms`: sum of all VTA stage set_input + run + get_output time under global VTA mutex.
- `boundary_crossing_ms`: bytes and tensor count penalty for CPU/VTA crossings.
- `cpu_tail_penalty_ms`: stronger when head/decode/tail dominates pipeline cycle.
- `small_vta_island_penalty_ms`: rejects or penalizes single small conv islands.
- `branch_boundary_penalty_ms`: high for YOLO route/concat/upsample/head boundaries.
- `launch_sync_penalty_ms`: grows with effective stage count and VTA island count.

Selection policy:

1. Generate coarse candidates first.
2. Keep 1-island and 2-island candidates as the main pool.
3. Include 3-island candidates only as diversity/control unless they reduce a measured CPU bottleneck.
4. Reject candidates where a CPU middle/tail stage is predicted to exceed VTA occupancy.
5. Prefer candidates whose largest CPU stage and total VTA occupancy differ by less than about 20-30%.
6. After every measured batch, fit residuals for:
   - CPU prefix,
   - CPU tail,
   - total VTA occupied time,
   - boundary type,
   - route/branch boundary risk.

## Model-Specific Next Steps

### ResNet18

Search locally around:

```text
3_7/8/9 + 10_18/19
1_6/7 + 8_13/14
```

Do not expand much into 3-island space unless the objective changes, because measured 3-island peak is lower.

### YOLOv3-tiny

Search locally around:

```text
pool2 -> logits
pool2 -> dual_pre_logits
pool2 -> shared13 + small_pre18 -> logits
pool2 -> trunk12 + small_pre18 -> logits
```

Avoid broad route-boundary expansion until the build failures around `pool4_route` and `shared13` are fixed. If those boundaries are fixed, they should still carry a high branch-boundary penalty until measured otherwise.

## Main Conclusion

The transferable rule is not "more fine-grained is better" or "always use CPU/VTA/CPU". The rule is:

```text
choose the fewest boundaries that make CPU time and serialized VTA occupancy balanced
```

For ResNet18, this naturally becomes two large adjacent/mid-late VTA blocks. For YOLOv3-tiny, it becomes a coarse `pool2 -> logits` VTA body with CPU prefix and CPU decode tail.
