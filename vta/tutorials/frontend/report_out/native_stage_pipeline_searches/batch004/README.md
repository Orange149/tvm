# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T15:41:53
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

- Scheme: islands_3_8__10_19
- Candidate: islands_3_8__10_19
- Stage devices: cpu/vta/cpu/vta/cpu
- Threads: 3,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=True reason=cat_equivalent baseline_top1=282 native_top1=285
- Pipeline throughput: 10.242 fps
- Static throughput estimate: 9.126 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=0.702 vta=2.930
- CPU/VTA achieved GOPS: cpu=7.245 vta=35.671
- PS-PL bandwidth: 2.073 Gbit/s, DMA bytes=362567680.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.305 static=0.015

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_8__10_19 | ok | True | 10.242 | 5 | 3,1,1,1,4 | stage0_cpu | 2.073 | 38.281 | 1.305 | 2107392.000 |
| 2 | islands_1_6__8_14 | ok | True | 9.494 | 5 | 3,1,1,1,4 | stage4_cpu | 1.329 | 38.281 | 1.198 | 2207744.000 |
| 3 | islands_3_8__10_18 | ok | True | 9.432 | 5 | 3,1,1,1,4 | stage0_cpu | 2.080 | 38.281 | 1.305 | 2207744.000 |
| 4 | islands_1_6__8_13 | ok | True | 9.164 | 5 | 3,1,1,1,4 | stage4_cpu | 1.329 | 38.281 | 1.198 | 2408448.000 |
| 5 | islands_3_9__10_16 | ok | True | 9.030 | 4 | 3,1,1,4 | stage0_cpu | 1.747 | 38.281 | 1.277 | 1404928.000 |
| 6 | islands_3_9__10_17 | ok | True | 8.859 | 4 | 3,1,1,4 | stage0_cpu | 1.725 | 38.281 | 1.277 | 1304576.000 |
| 7 | islands_1_2__3_7__8_14 | ok | True | 8.591 | 5 | 3,1,1,1,4 | stage4_cpu | 1.297 | 38.281 | 1.198 | 2207744.000 |
| 8 | islands_1_4__5_9__10_13 | ok | True | 8.512 | 5 | 3,1,1,1,4 | stage4_cpu | 1.286 | 38.281 | 1.198 | 2408448.000 |
| 9 | islands_1_2__3_7__8_13 | ok | True | 8.439 | 5 | 3,1,1,1,4 | stage4_cpu | 1.267 | 38.281 | 1.198 | 2408448.000 |
| 10 | islands_1_4__5_9__10_14 | ok | True | 8.277 | 5 | 3,1,1,1,4 | stage4_cpu | 1.249 | 38.281 | 1.198 | 2207744.000 |
| 11 | islands_3_8__10_14 | ok | True | 7.958 | 5 | 3,1,1,1,4 | stage0_cpu | 1.359 | 38.281 | 1.228 | 2207744.000 |
| 12 | islands_3_6__8_13 | ok | True | 7.614 | 5 | 3,1,1,1,4 | stage0_cpu | 1.290 | 38.281 | 1.228 | 2408448.000 |
| 13 | islands_3_9__10_12__13_14 | ok | True | 7.495 | 5 | 3,1,1,1,4 | stage0_cpu | 1.412 | 38.281 | 1.228 | 1605632.000 |
| 14 | islands_3_6__8_14 | ok | True | 7.488 | 5 | 3,1,1,1,4 | stage0_cpu | 1.376 | 38.281 | 1.228 | 2207744.000 |
| 15 | islands_3_7__8_13 | ok | True | 7.454 | 4 | 3,1,1,4 | stage0_cpu | 1.426 | 38.281 | 1.228 | 1605632.000 |
| 16 | islands_3_8__10_13 | ok | True | 7.360 | 5 | 3,1,1,1,4 | stage0_cpu | 1.366 | 38.281 | 1.228 | 2408448.000 |
| 17 | islands_3_7__8_14 | ok | True | 7.359 | 4 | 3,1,1,4 | stage0_cpu | 1.312 | 38.281 | 1.228 | 1404928.000 |
| 18 | islands_3_4__5_7__8_14 | ok | True | 7.299 | 5 | 3,1,1,1,4 | stage0_cpu | 1.448 | 38.281 | 1.228 | 2207744.000 |
| 19 | islands_3_9__10_12__13_13 | ok | True | 7.248 | 5 | 3,1,1,1,4 | stage0_cpu | 1.337 | 38.281 | 1.228 | 1806336.000 |
| 20 | islands_3_4__5_7__8_13 | ok | True | 7.163 | 5 | 3,1,1,1,4 | stage0_cpu | 1.336 | 38.281 | 1.228 | 2408448.000 |

Full metrics are in `summary.csv` and `summary.json`.
