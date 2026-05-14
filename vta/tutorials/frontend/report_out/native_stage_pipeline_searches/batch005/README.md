# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T15:57:22
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

- Scheme: islands_1_4__5_10__13_14
- Candidate: islands_1_4__5_10__13_14
- Stage devices: cpu/vta/vta/cpu/vta/cpu
- Threads: 3,1,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 8.995 fps
- Static throughput estimate: 9.708 fps
- Bottleneck: stage5_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.074 vta=2.557
- CPU/VTA achieved GOPS: cpu=7.862 vta=32.091
- PS-PL bandwidth: 1.335 Gbit/s, DMA bytes=217303040.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.176 static=0.016

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_1_4__5_10__13_14 | ok | True | 8.995 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.335 | 38.281 | 1.176 | 2609152.000 |
| 2 | islands_1_4__5_8__10_13 | ok | True | 8.782 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.337 | 38.281 | 1.198 | 3211264.000 |
| 3 | islands_1_3__5_7__8_13 | ok | True | 8.655 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.350 | 38.281 | 1.198 | 4014080.000 |
| 4 | islands_1_4__5_11__13_13 | ok | True | 8.625 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.312 | 38.281 | 1.198 | 2609152.000 |
| 5 | islands_1_4__5_10__13_13 | ok | True | 8.613 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.248 | 38.281 | 1.176 | 2809856.000 |
| 6 | islands_1_3__5_12__13_14 | ok | True | 8.558 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.337 | 38.281 | 1.198 | 3612672.000 |
| 7 | islands_1_4__5_6__8_14 | ok | True | 8.547 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.340 | 38.281 | 1.198 | 3010560.000 |
| 8 | islands_1_4__5_11__13_14 | ok | True | 8.465 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.299 | 38.281 | 1.198 | 2408448.000 |
| 9 | islands_1_4__5_8__10_14 | ok | True | 8.447 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.295 | 38.281 | 1.198 | 3010560.000 |
| 10 | islands_1_3__5_7__8_14 | ok | True | 8.409 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.317 | 38.281 | 1.198 | 3813376.000 |
| 11 | islands_1_4__5_6__8_13 | ok | True | 8.408 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.281 | 38.281 | 1.198 | 3211264.000 |
| 12 | islands_1_3__5_12__13_13 | ok | True | 8.171 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.263 | 38.281 | 1.198 | 3813376.000 |
| 13 | islands_1_3__5_9__10_14 | ok | True | 8.122 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.269 | 38.281 | 1.198 | 3813376.000 |
| 14 | islands_1_4__5_5__8_13 | ok | True | 8.037 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.262 | 38.281 | 1.176 | 3612672.000 |
| 15 | islands_3_9__10_15 | ok | True | 8.031 | 4 | 3,1,1,4 | stage0_cpu | 1.780 | 38.281 | 1.259 | 1505280.000 |
| 16 | islands_3_4__5_12__13_14 | ok | True | 7.955 | 5 | 3,1,1,1,4 | stage0_cpu | 1.374 | 38.281 | 1.228 | 2007040.000 |
| 17 | islands_3_4__5_12__13_13 | ok | True | 7.940 | 5 | 3,1,1,1,4 | stage0_cpu | 1.291 | 38.281 | 1.228 | 2207744.000 |
| 18 | islands_1_4__5_5__8_14 | ok | True | 7.609 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.235 | 38.281 | 1.176 | 3411968.000 |
| 19 | islands_3_5__8_14 | ok | True | 7.592 | 5 | 3,1,1,1,4 | stage0_cpu | 1.298 | 38.281 | 1.202 | 2609152.000 |
| 20 | islands_3_5__8_13 | ok | True | 7.110 | 5 | 3,1,1,1,4 | stage0_cpu | 1.338 | 38.281 | 1.202 | 2809856.000 |

Full metrics are in `summary.csv` and `summary.json`.
