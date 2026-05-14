# YOLOv3-tiny CPU/VTA/CPU 100-Candidate Search Analysis

## Scope

This run tested ResNet18-style per-stage YOLOv3-tiny CPU/VTA/CPU pipeline candidates. The search expanded coarse split points with runtime configurations for CPU thread allocation and queue/poll settings, then measured only candidates that passed serial detection-gate correctness.

## Run Summary

- Candidate target: 100 generated configs
- Buildable configs: 88
- Serial smoke passed: 88
- Pipeline measured OK: 88
- Pipeline failures/timeouts: 0
- Build failures: 12, all from the experimental `data_to_*` start-boundary families, so they did not enter serial or pipeline ranking.
- Correctness policy: detection_gate; raw tensor mismatch is recorded but not used as the sole hard gate
- Remote cleanup: `/var/volatile/yolov3_tiny_pipeline_search` empty after run

## Best Result

- Candidate: `yolo_pool2_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000`
- Throughput: 5.113142 fps
- Measured cycle: 195.574 ms
- Stage times: [195.9976552, 165.1157442, 137.6137912] ms
- Bottleneck stage: 0 (`stage0_cpu`)
- Runtime config: `s04_s22_q2_poll1000_post1000` = {'poll_sleep_ns': 1000, 'post_start_sleep_ns': 1000, 'queue_depth': 2, 'stage0_threads': 4, 'stage2_threads': 2}
- Detection gate passed: True
- Raw gate passed: False
- Boundary bytes estimate: 1038336 B
- Boundary tensors: ['small_pre26', 'big64']

## Top 10

| Rank | Candidate | FPS | Stage Times ms | Detection Gate |
|---:|---|---:|---|---|
| 1 | `yolo_pool2_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000` | 5.113142 | `[195.9976552, 165.1157442, 137.6137912]` | True |
| 2 | `yolo_pool2_to_dual_pre_logits__rt_s04_s23_q2_poll1000_post1000` | 5.039127 | `[200.91954375, 169.9750413, 109.2062189]` | True |
| 3 | `yolo_pool2_to_dual_pre_logits__rt_s04_s24_q2_poll1000_post1000` | 4.920562 | `[202.95903745, 178.3111616, 102.28362295]` | True |
| 4 | `yolo_pool2_to_dual_pre_logits__rt_s03_s24_q2_poll1000_post1000` | 4.917438 | `[203.78464730000002, 157.89904755, 78.94503485]` | True |
| 5 | `yolo_pool1_to_dual_pre_logits__rt_s03_s22_q2_poll1000_post1000` | 4.874753 | `[155.83812890000002, 208.4422088, 133.53268085]` | True |
| 6 | `yolo_pool2_to_dual_pre_logits__rt_s03_s23_q2_poll1000_post1000` | 4.693044 | `[212.819116, 162.87733635, 100.0384003]` | True |
| 7 | `yolo_pool1_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000` | 4.685594 | `[125.4550414, 217.2552154, 119.52733015]` | True |
| 8 | `yolo_pool1_to_dual_pre_logits__rt_s04_s23_q2_poll1000_post1000` | 4.557074 | `[134.34582235, 224.10779705, 116.7783353]` | True |
| 9 | `yolo_pool1_to_dual_pre_logits__rt_s03_s24_q2_poll1000_post1000` | 4.528239 | `[145.69066535, 220.1533375, 81.9588772]` | True |
| 10 | `yolo_pool1_to_dual_pre_logits__rt_s03_s23_q2_poll1000_post1000` | 4.515128 | `[151.81565565, 223.4425043, 104.5058967]` | True |

## Best Split Families

| Split family | Tested configs | Best FPS | Mean FPS | Best stage times ms |
|---|---:|---:|---:|---|
| `yolo_pool2_to_dual_pre_logits` | 6 | 5.113142 | 4.843047 | `[195.9976552, 165.1157442, 137.6137912]` |
| `yolo_pool1_to_dual_pre_logits` | 6 | 4.874753 | 4.573653 | `[155.83812890000002, 208.4422088, 133.53268085]` |
| `yolo_pool0_to_dual_pre_logits` | 6 | 4.201505 | 3.938214 | `[65.5835657, 240.8372963, 103.29346585]` |
| `yolo_pool3_to_dual_pre_logits` | 6 | 4.154513 | 3.945641 | `[239.11632905, 143.61456795, 139.647924]` |
| `yolo_pool0_to_dual_pre18_14` | 6 | 3.269188 | 3.137042 | `[52.040724, 259.9426739, 303.87054555]` |
| `yolo_pool1_to_dual_pre18_14` | 6 | 3.209512 | 2.933755 | `[159.0259202, 182.89024075, 316.8771801]` |
| `yolo_pool2_to_dual_pre18_14` | 6 | 2.818544 | 2.434808 | `[254.18259065, 124.5732818, 359.79163875]` |
| `yolo_pool3_to_dual_pre18_14` | 6 | 2.574374 | 2.191070 | `[307.17433005, 106.57037455, 393.12285245]` |
| `yolo_pool2_to_shared13` | 4 | 2.495036 | 2.285123 | `[247.34493735, 120.0274382, 407.5056414]` |
| `yolo_pool3_to_small_pre18` | 6 | 2.285970 | 1.937865 | `[312.90356385, 96.9066345, 438.53189605]` |
| `yolo_pool3_to_shared13` | 6 | 2.271433 | 1.933733 | `[316.08574219999997, 92.653369, 444.915399]` |
| `yolo_pool3_to_trunk12` | 6 | 2.252781 | 1.890212 | `[313.321984, 84.1896138, 451.7822861]` |

## Interpretation

1. The strongest family is `pool2_to_dual_pre_logits`. It keeps the early feature extractor on CPU, runs the main conv-heavy trunk on VTA, and leaves only the final YOLO logits/decode tail on CPU. Its best stage times `[195.998, 165.116, 137.614] ms` are the most balanced among high-throughput candidates.
2. `pool1_to_dual_pre_logits` is the next best family. It moves more early work into VTA, but the VTA stage becomes the bottleneck around 208-229 ms, so throughput falls below the best `pool2` split.
3. Cutting at `dual_pre18_14`, `shared13`, `trunk12`, or `small_pre18` makes the CPU tail too heavy or the VTA island too short. Those candidates are useful controls, but they do not improve throughput.
4. Runtime thread allocation matters. For the best split, `stage0_threads=4, stage2_threads=2` wins; using more threads on the tail does not help enough to compensate for stage0/VTA scheduling effects.
5. The raw tensor gate still fails for the best candidate, mostly on the large 26x26 logits tensor, but the detection gate passes with matching top detections. This matches the intended policy for per-stage quantized YOLO: raw drift is diagnostic, detection-level correctness is the ranking gate.
6. The `data_to_*` start-boundary candidates are not yet usable in this implementation. Their first base build failed, and repeated runtime variants were skipped through the build-failure cache. I fixed the cache diagnostic path after this run so future retries report the original base failure cleanly.

## Baseline Comparison

| Result | FPS | Notes |
|---|---:|---|
| RPC all-VTA baseline | 3.724 | reference correctness/performance baseline from prior runs |
| Packed graph smoke | 6.5215 | packed all-graph pipeline smoke, not the same CPU/VTA/CPU per-stage split path |
| Packed hetero best | 4.2922 | previous packed hetero result |
| This run best | 5.113142 | true CPU/VTA/CPU, detection-gate correct |

## Practical Heuristic

For YOLOv3-tiny on this board, the useful heuristic is: choose a coarse CPU/VTA/CPU split that keeps VTA occupied with the main conv trunk while keeping CPU stage0 and CPU tail below the VTA-stage time. The target is not maximal VTA coverage; it is balanced stage time after boundary pack/unpack and final head cost. In this run, that points to `pool2 -> dual_pre_logits`, then tune CPU threads with stage0 favored over tail threads.

## Artifacts

- `summary.json` and `summary.csv`: full measured results
- `buildability_summary.json` and `buildability.csv`: package/build outcome
- `safe_candidates.csv`: generated candidate list
- Per-candidate directories: logs, serial/pipeline JSONL, raw output dumps
