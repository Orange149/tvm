# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T14:35:56
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

- Scheme: islands_3_5__10_14__15_19
- Candidate: islands_3_5__10_14__15_19
- Stage devices: cpu/vta/cpu/vta/vta/cpu
- Threads: 3,1,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 9.405 fps
- Static throughput estimate: 5.883 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.177 vta=2.454
- CPU/VTA achieved GOPS: cpu=6.452 vta=33.284
- PS-PL bandwidth: 2.147 Gbit/s, DMA bytes=319488000.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.338 static=0.014

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_5__10_14__15_19 | ok | True | 9.405 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.147 | 38.281 | 1.338 | 2709504.000 |
| 2 | islands_3_5__10_14__15_18 | ok | True | 9.287 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.178 | 38.281 | 1.338 | 2809856.000 |
| 3 | islands_3_9__10_12__13_17 | ok | True | 9.072 | 5 | 3,1,1,1,4 | stage0_cpu | 1.721 | 38.281 | 1.277 | 1505280.000 |
| 4 | islands_3_6__8_15 | ok | True | 8.803 | 5 | 3,1,1,1,4 | stage0_cpu | 1.682 | 38.281 | 1.259 | 2308096.000 |
| 5 | islands_3_10__13_16 | ok | True | 8.689 | 5 | 3,1,1,1,4 | stage0_cpu | 1.650 | 38.281 | 1.258 | 1806336.000 |
| 6 | islands_3_10__13_17 | ok | True | 8.627 | 5 | 3,1,1,1,4 | stage0_cpu | 1.670 | 38.281 | 1.258 | 1705984.000 |
| 7 | islands_3_7__8_14__15_17 | ok | True | 8.621 | 5 | 3,1,1,1,4 | stage0_cpu | 1.728 | 38.281 | 1.277 | 1505280.000 |
| 8 | islands_3_9__10_12__13_16 | ok | True | 8.591 | 5 | 3,1,1,1,4 | stage0_cpu | 1.691 | 38.281 | 1.277 | 1605632.000 |
| 9 | islands_1_5__10_15 | ok | True | 8.559 | 5 | 3,1,1,1,4 | stage2_cpu | 1.563 | 38.281 | 1.243 | 2709504.000 |
| 10 | islands_3_14__15_17 | ok | True | 8.488 | 4 | 3,1,1,4 | stage0_cpu | 1.757 | 38.281 | 1.277 | 1103872.000 |
| 11 | islands_3_7__10_12__13_15 | ok | True | 8.193 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.755 | 38.281 | 1.308 | 2107392.000 |
| 12 | islands_3_9__10_14__15_17 | ok | True | 8.155 | 5 | 3,1,1,1,4 | stage0_cpu | 1.780 | 38.281 | 1.277 | 1505280.000 |
| 13 | islands_3_9__10_12__15_17 | ok | True | 8.088 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.641 | 38.281 | 1.239 | 1705984.000 |
| 14 | islands_3_7__8_12__13_15 | ok | True | 8.015 | 5 | 3,1,1,1,4 | stage0_cpu | 1.694 | 38.281 | 1.259 | 1705984.000 |
| 15 | islands_3_6__10_12__13_15 | ok | True | 7.771 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.832 | 38.281 | 1.308 | 2508800.000 |
| 16 | islands_5_14__15_17 | ok | True | 6.432 | 4 | 3,1,1,4 | stage0_cpu | 1.907 | 38.281 | 1.322 | 1103872.000 |
| 17 | islands_5_12__13_15 | ok | True | 6.395 | 4 | 3,1,1,4 | stage0_cpu | 1.915 | 38.281 | 1.302 | 1304576.000 |
| 18 | islands_5_7__8_17 | ok | True | 6.318 | 4 | 3,1,1,4 | stage0_cpu | 1.920 | 38.281 | 1.322 | 1304576.000 |
| 19 | islands_5_7__8_15 | ok | True | 6.300 | 4 | 3,1,1,4 | stage0_cpu | 1.916 | 38.281 | 1.302 | 1505280.000 |
| 20 | islands_5_7__8_16 | ok | True | 6.280 | 4 | 3,1,1,4 | stage0_cpu | 1.919 | 38.281 | 1.322 | 1404928.000 |

Full metrics are in `summary.csv` and `summary.json`.
