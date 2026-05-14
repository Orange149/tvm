# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T13:39:33
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

- Scheme: islands_3_7__10_18
- Candidate: islands_3_7__10_18
- Stage devices: cpu/vta/cpu/vta/cpu
- Threads: 3,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 10.200 fps
- Static throughput estimate: 6.049 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.164 vta=2.467
- CPU/VTA achieved GOPS: cpu=6.635 vta=35.279
- PS-PL bandwidth: 2.243 Gbit/s, DMA bytes=329512960.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.353 static=0.014

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_7__10_18 | ok | True | 10.200 | 5 | 3,1,1,1,4 | stage0_cpu | 2.243 | 38.281 | 1.353 | 1806336.000 |
| 2 | islands_3_7__10_19 | ok | True | 9.874 | 5 | 3,1,1,1,4 | stage0_cpu | 2.220 | 38.281 | 1.353 | 1705984.000 |
| 3 | islands_3_6__10_18 | ok | True | 9.821 | 5 | 3,1,1,1,4 | stage0_cpu | 2.195 | 38.281 | 1.353 | 2207744.000 |
| 4 | islands_3_6__10_19 | ok | True | 9.749 | 5 | 3,1,1,1,4 | stage0_cpu | 2.230 | 38.281 | 1.353 | 2107392.000 |
| 5 | islands_3_7__10_14__15_19 | ok | True | 9.176 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.244 | 38.281 | 1.353 | 1906688.000 |
| 6 | islands_3_6__10_14__15_19 | ok | True | 8.949 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.244 | 38.281 | 1.353 | 2308096.000 |
| 7 | islands_3_7__10_14__15_18 | ok | True | 8.934 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.178 | 38.281 | 1.353 | 2007040.000 |
| 8 | islands_3_6__10_14__15_18 | ok | True | 8.782 | 6 | 3,1,1,1,1,4 | stage0_cpu | 2.225 | 38.281 | 1.353 | 2408448.000 |
| 9 | islands_3_9__13_16 | ok | True | 8.228 | 5 | 3,1,1,1,4 | stage0_cpu | 1.642 | 38.281 | 1.251 | 1605632.000 |
| 10 | islands_3_9__13_17 | ok | True | 8.195 | 5 | 3,1,1,1,4 | stage0_cpu | 1.651 | 38.281 | 1.251 | 1505280.000 |
| 11 | islands_3_6__10_17 | ok | True | 8.171 | 5 | 3,1,1,1,4 | stage0_cpu | 1.899 | 38.281 | 1.328 | 2107392.000 |
| 12 | islands_3_7__10_15 | ok | True | 8.090 | 5 | 3,1,1,1,4 | stage0_cpu | 1.757 | 38.281 | 1.308 | 1906688.000 |
| 13 | islands_3_6__10_16 | ok | True | 8.015 | 5 | 3,1,1,1,4 | stage0_cpu | 1.765 | 38.281 | 1.328 | 2207744.000 |
| 14 | islands_3_8__13_17 | ok | True | 7.992 | 5 | 3,1,1,1,4 | stage0_cpu | 1.629 | 38.281 | 1.251 | 1906688.000 |
| 15 | islands_3_8__13_16 | ok | True | 7.921 | 5 | 3,1,1,1,4 | stage2_cpu | 1.633 | 38.281 | 1.251 | 2007040.000 |
| 16 | islands_3_8__13_15 | ok | True | 7.875 | 5 | 3,1,1,1,4 | stage0_cpu | 1.772 | 38.281 | 1.228 | 2107392.000 |
| 17 | islands_3_7__10_17 | ok | True | 7.859 | 5 | 3,1,1,1,4 | stage0_cpu | 1.827 | 38.281 | 1.328 | 1705984.000 |
| 18 | islands_3_6__10_15 | ok | True | 7.843 | 5 | 3,1,1,1,4 | stage0_cpu | 1.817 | 38.281 | 1.308 | 2308096.000 |
| 19 | islands_3_7__10_16 | ok | True | 7.807 | 5 | 3,1,1,1,4 | stage0_cpu | 1.809 | 38.281 | 1.328 | 1806336.000 |
| 20 | islands_3_9__13_15 | ok | True | 7.796 | 5 | 3,1,1,1,4 | stage0_cpu | 1.737 | 38.281 | 1.228 | 1705984.000 |

Full metrics are in `summary.csv` and `summary.json`.
