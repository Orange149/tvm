# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T14:07:41
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

- Scheme: islands_3_7__10_12__13_19
- Candidate: islands_3_7__10_12__13_19
- Stage devices: cpu/vta/cpu/vta/vta/cpu
- Threads: 3,1,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 8.779 fps
- Static throughput estimate: 6.049 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.164 vta=2.467
- CPU/VTA achieved GOPS: cpu=6.297 vta=34.844
- PS-PL bandwidth: 2.234 Gbit/s, DMA bytes=329512960.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.353 static=0.015

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_7__10_12__13_19 | ok | True | 8.779 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.234 | 38.281 | 1.353 | 1906688.000 |
| 2 | islands_3_4__8_15 | ok | True | 7.824 | 5 | 3,1,1,1,4 | stage0_cpu | 1.734 | 38.281 | 1.273 | 2308096.000 |
| 3 | islands_5_9__10_19 | ok | True | 7.688 | 4 | 3,1,1,4 | stage0_cpu | 2.276 | 38.281 | 1.347 | 1304576.000 |
| 4 | islands_5_18 | ok | True | 7.641 | 3 | 3,1,4 | stage0_cpu | 2.276 | 38.281 | 1.347 | 1003520.000 |
| 5 | islands_5_14__15_18 | ok | True | 7.533 | 4 | 3,1,1,4 | stage0_cpu | 2.281 | 38.281 | 1.347 | 1204224.000 |
| 6 | islands_5_14__15_19 | ok | True | 7.427 | 4 | 3,1,1,4 | stage0_cpu | 2.276 | 38.281 | 1.347 | 1103872.000 |
| 7 | islands_5_12__13_19 | ok | True | 7.378 | 4 | 3,1,1,4 | stage0_cpu | 2.277 | 38.281 | 1.347 | 1103872.000 |
| 8 | islands_5_12__13_18 | ok | True | 7.360 | 4 | 3,1,1,4 | stage0_cpu | 2.279 | 38.281 | 1.347 | 1204224.000 |
| 9 | islands_5_19 | ok | True | 7.295 | 3 | 3,1,4 | stage0_cpu | 2.276 | 38.281 | 1.347 | 903168.000 |
| 10 | islands_5_9__10_14__15_19 | ok | True | 7.259 | 5 | 3,1,1,1,4 | stage0_cpu | 2.278 | 38.281 | 1.347 | 1505280.000 |
| 11 | islands_5_9__10_18 | ok | True | 7.159 | 4 | 3,1,1,4 | stage0_cpu | 2.279 | 38.281 | 1.347 | 1404928.000 |
| 12 | islands_5_15 | ok | True | 6.674 | 3 | 3,1,4 | stage0_cpu | 1.917 | 38.281 | 1.302 | 1103872.000 |
| 13 | islands_5_9__10_14__15_18 | ok | True | 6.646 | 5 | 3,1,1,1,4 | stage0_cpu | 2.274 | 38.281 | 1.347 | 1605632.000 |
| 14 | islands_5_9__10_17 | ok | True | 6.541 | 4 | 3,1,1,4 | stage0_cpu | 1.920 | 38.281 | 1.322 | 1304576.000 |
| 15 | islands_5_17 | ok | True | 6.532 | 3 | 3,1,4 | stage0_cpu | 1.918 | 38.281 | 1.322 | 903168.000 |
| 16 | islands_5_9__10_16 | ok | True | 6.513 | 4 | 3,1,1,4 | stage0_cpu | 1.919 | 38.281 | 1.322 | 1404928.000 |
| 17 | islands_5_16 | ok | True | 6.509 | 3 | 3,1,4 | stage0_cpu | 1.919 | 38.281 | 1.322 | 1003520.000 |
| 18 | islands_5_9__10_15 | ok | True | 6.379 | 4 | 3,1,1,4 | stage0_cpu | 1.916 | 38.281 | 1.302 | 1505280.000 |
| 19 | islands_5_12__13_16 | ok | True | 6.365 | 4 | 3,1,1,4 | stage0_cpu | 1.921 | 38.281 | 1.322 | 1204224.000 |
| 20 | islands_5_12__13_17 | ok | True | 6.201 | 4 | 3,1,1,4 | stage0_cpu | 1.920 | 38.281 | 1.322 | 1103872.000 |

Full metrics are in `summary.csv` and `summary.json`.
