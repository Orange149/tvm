# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T15:14:46
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 256
- Buildability tested: 128
- Buildable candidates: 128
- Board test configs: 20
- File cache policy: prewarm
- File cache warmups: 20
- Max VTA islands: 3
- Measure top N: 20
- Refine top N: 0
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_3_9__10_19
- Candidate: islands_3_9__10_19
- Stage devices: cpu/vta/vta/cpu
- Threads: 3,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=True reason=cat_equivalent baseline_top1=282 native_top1=285
- Pipeline throughput: 10.532 fps
- Static throughput estimate: 9.126 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=0.702 vta=2.930
- CPU/VTA achieved GOPS: cpu=8.340 vta=36.210
- PS-PL bandwidth: 2.102 Gbit/s, DMA bytes=362567680.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.305 static=0.015

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_9__10_19 | ok | True | 10.532 | 4 | 3,1,1,4 | stage0_cpu | 2.102 | 38.281 | 1.305 | 1304576.000 |
| 2 | islands_3_9__10_18 | ok | True | 10.524 | 4 | 3,1,1,4 | stage0_cpu | 2.100 | 38.281 | 1.305 | 1404928.000 |
| 3 | islands_1_7__8_13 | ok | True | 9.719 | 4 | 3,1,1,4 | stage3_cpu | 1.300 | 38.281 | 1.198 | 1605632.000 |
| 4 | islands_1_7__8_14 | ok | True | 9.450 | 4 | 3,1,1,4 | stage3_cpu | 1.289 | 38.281 | 1.198 | 1404928.000 |
| 5 | islands_1_4__5_12__13_13 | ok | True | 8.776 | 5 | 3,1,1,1,4 | stage4_cpu | 1.303 | 38.281 | 1.198 | 2207744.000 |
| 6 | islands_1_5__8_13 | ok | True | 8.619 | 5 | 3,1,1,1,4 | stage4_cpu | 1.227 | 38.281 | 1.176 | 2809856.000 |
| 7 | islands_1_4__5_7__8_14 | ok | True | 8.564 | 5 | 3,1,1,1,4 | stage4_cpu | 1.291 | 38.281 | 1.198 | 2207744.000 |
| 8 | islands_1_4__5_12__13_14 | ok | True | 8.508 | 5 | 3,1,1,1,4 | stage4_cpu | 1.306 | 38.281 | 1.198 | 2007040.000 |
| 9 | islands_1_5__8_14 | ok | True | 8.484 | 5 | 3,1,1,1,4 | stage4_cpu | 1.229 | 38.281 | 1.176 | 2609152.000 |
| 10 | islands_1_4__5_7__8_13 | ok | True | 8.482 | 5 | 3,1,1,1,4 | stage4_cpu | 1.279 | 38.281 | 1.198 | 2408448.000 |
| 11 | islands_1_3__5_12__13_15 | ok | True | 8.273 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.602 | 38.281 | 1.228 | 3713024.000 |
| 12 | islands_1_3__5_11__13_15 | ok | True | 8.004 | 7 | 3,1,1,1,1,1,4 | stage1_vta | 1.554 | 38.281 | 1.228 | 4114432.000 |
| 13 | islands_1_3__5_11__13_17 | ok | True | 7.980 | 7 | 3,1,1,1,1,1,4 | stage1_vta | 1.570 | 38.281 | 1.244 | 3913728.000 |
| 14 | islands_1_3__5_11__13_16 | ok | True | 7.954 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.566 | 38.281 | 1.244 | 4014080.000 |
| 15 | islands_1_3__5_11__13_18 | ok | True | 7.863 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.966 | 38.281 | 1.273 | 4014080.000 |
| 16 | islands_1_3__5_10__13_18 | ok | True | 7.826 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.957 | 38.281 | 1.259 | 4214784.000 |
| 17 | islands_1_3__5_11__13_19 | ok | True | 7.749 | 7 | 3,1,1,1,1,1,4 | stage1_vta | 1.942 | 38.281 | 1.273 | 3913728.000 |
| 18 | islands_1_3__5_10__13_19 | ok | True | 7.715 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.939 | 38.281 | 1.259 | 4114432.000 |
| 19 | islands_3_9__10_14 | ok | True | 7.425 | 4 | 3,1,1,4 | stage0_cpu | 1.327 | 38.281 | 1.228 | 1404928.000 |
| 20 | islands_3_9__10_13 | ok | True | 7.395 | 4 | 3,1,1,4 | stage0_cpu | 1.356 | 38.281 | 1.228 | 1605632.000 |

Full metrics are in `summary.csv` and `summary.json`.
