# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T14:21:56
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 20
- Buildability tested: 20
- Buildable candidates: 20
- Board test configs: 20
- File cache policy: prewarm
- File cache warmups: 20
- Max VTA islands: 3
- Measure top N: 20
- Refine top N: 0
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_3_6__10_12__13_18
- Candidate: islands_3_6__10_12__13_18
- Stage devices: cpu/vta/cpu/vta/vta/cpu
- Threads: 3,1,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 9.868 fps
- Static throughput estimate: 6.046 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.165 vta=2.467
- CPU/VTA achieved GOPS: cpu=6.419 vta=34.377
- PS-PL bandwidth: 2.211 Gbit/s, DMA bytes=329512960.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.353 static=0.014

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_6__10_12__13_18 | ok | True | 9.868 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.211 | 38.281 | 1.353 | 2408448.000 |
| 2 | islands_3_5__10_19 | ok | True | 9.726 | 5 | 3,1,1,1,4 | stage0_cpu | 2.136 | 38.281 | 1.338 | 2508800.000 |
| 3 | islands_3_6__10_12__13_19 | ok | True | 9.631 | 6 | 3,1,1,1,1,4 | stage2_cpu | 2.228 | 38.281 | 1.353 | 2308096.000 |
| 4 | islands_3_5__10_18 | ok | True | 9.162 | 5 | 3,1,1,1,4 | stage0_cpu | 2.218 | 38.281 | 1.338 | 2609152.000 |
| 5 | islands_3_12__13_15 | ok | True | 9.084 | 4 | 3,1,1,4 | stage0_cpu | 1.752 | 38.281 | 1.259 | 1304576.000 |
| 6 | islands_3_7__10_12__13_18 | ok | True | 8.570 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.176 | 38.281 | 1.353 | 2007040.000 |
| 7 | islands_3_5__10_17 | ok | True | 8.103 | 5 | 3,1,1,1,4 | stage0_cpu | 1.694 | 38.281 | 1.308 | 2508800.000 |
| 8 | islands_3_10__15_17 | ok | True | 8.046 | 5 | 3,1,1,1,4 | stage2_cpu | 1.840 | 38.281 | 1.215 | 1705984.000 |
| 9 | islands_3_7__8_9__13_17 | ok | True | 8.032 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.623 | 38.281 | 1.251 | 1906688.000 |
| 10 | islands_3_5__10_16 | ok | True | 8.007 | 5 | 3,1,1,1,4 | stage0_cpu | 1.764 | 38.281 | 1.308 | 2609152.000 |
| 11 | islands_3_4__8_12__13_17 | ok | True | 7.929 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.727 | 38.281 | 1.294 | 2308096.000 |
| 12 | islands_3_5__10_15 | ok | True | 7.887 | 5 | 3,1,1,1,4 | stage0_cpu | 1.777 | 38.281 | 1.287 | 2709504.000 |
| 13 | islands_3_4__5_9__13_17 | ok | True | 7.884 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.776 | 38.281 | 1.251 | 2308096.000 |
| 14 | islands_3_4__8_17 | ok | True | 7.715 | 5 | 3,1,1,1,4 | stage0_cpu | 1.763 | 38.281 | 1.294 | 2107392.000 |
| 15 | islands_3_4__5_8__13_17 | ok | True | 7.699 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.627 | 38.281 | 1.251 | 2709504.000 |
| 16 | islands_3_4__8_12__13_16 | ok | True | 7.620 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.707 | 38.281 | 1.294 | 2408448.000 |
| 17 | islands_3_4__5_9__13_16 | ok | True | 7.589 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.740 | 38.281 | 1.251 | 2408448.000 |
| 18 | islands_3_4__5_8__13_16 | ok | True | 7.586 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.633 | 38.281 | 1.251 | 2809856.000 |
| 19 | islands_3_4__8_16 | ok | True | 7.572 | 5 | 3,1,1,1,4 | stage0_cpu | 1.629 | 38.281 | 1.294 | 2207744.000 |
| 20 | islands_3_7__8_9__13_16 | ok | True | 7.550 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.738 | 38.281 | 1.251 | 2007040.000 |

Full metrics are in `summary.csv` and `summary.json`.
