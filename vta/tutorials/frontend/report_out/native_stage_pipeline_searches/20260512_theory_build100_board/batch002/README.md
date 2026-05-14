# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T13:54:22
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

- Scheme: islands_3_16
- Candidate: islands_3_16
- Stage devices: cpu/vta/cpu
- Threads: 3,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=True reason=cat_equivalent baseline_top1=282 native_top1=285
- Pipeline throughput: 9.461 fps
- Static throughput estimate: 5.432 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.164 vta=2.467
- CPU/VTA achieved GOPS: cpu=8.312 vta=35.364
- PS-PL bandwidth: 1.781 Gbit/s, DMA bytes=265103360.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.277 static=0.016

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_16 | ok | True | 9.461 | 3 | 3,1,4 | stage0_cpu | 1.781 | 38.281 | 1.277 | 1003520.000 |
| 2 | islands_3_17 | ok | True | 9.206 | 3 | 3,1,4 | stage0_cpu | 1.715 | 38.281 | 1.277 | 903168.000 |
| 3 | islands_3_7__8_15 | ok | True | 9.135 | 4 | 3,1,1,4 | stage0_cpu | 1.768 | 38.281 | 1.259 | 1505280.000 |
| 4 | islands_3_12__13_17 | ok | True | 9.124 | 4 | 3,1,1,4 | stage0_cpu | 1.749 | 38.281 | 1.277 | 1103872.000 |
| 5 | islands_3_7__8_12__13_17 | ok | True | 8.788 | 5 | 3,1,1,1,4 | stage0_cpu | 1.740 | 38.281 | 1.277 | 1505280.000 |
| 6 | islands_3_15 | ok | True | 8.769 | 3 | 3,1,4 | stage0_cpu | 1.723 | 38.281 | 1.259 | 1103872.000 |
| 7 | islands_3_7__8_17 | ok | True | 8.707 | 4 | 3,1,1,4 | stage0_cpu | 1.747 | 38.281 | 1.277 | 1304576.000 |
| 8 | islands_3_7__8_16 | ok | True | 8.497 | 4 | 3,1,1,4 | stage0_cpu | 1.783 | 38.281 | 1.277 | 1404928.000 |
| 9 | islands_3_12__13_16 | ok | True | 8.378 | 4 | 3,1,1,4 | stage0_cpu | 1.775 | 38.281 | 1.277 | 1204224.000 |
| 10 | islands_3_12__15_17 | ok | True | 8.337 | 5 | 3,1,1,1,4 | stage0_cpu | 1.796 | 38.281 | 1.239 | 1304576.000 |
| 11 | islands_3_7__8_12__13_16 | ok | True | 8.213 | 5 | 3,1,1,1,4 | stage0_cpu | 1.761 | 38.281 | 1.277 | 1605632.000 |
| 12 | islands_3_7__10_12__13_17 | ok | True | 8.050 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.800 | 38.281 | 1.328 | 1906688.000 |
| 13 | islands_3_6__10_14__15_17 | ok | True | 8.017 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.742 | 38.281 | 1.328 | 2308096.000 |
| 14 | islands_3_7__8_12__15_17 | ok | True | 7.972 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.676 | 38.281 | 1.239 | 1705984.000 |
| 15 | islands_3_7__8_11__15_17 | ok | True | 7.903 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.719 | 38.281 | 1.239 | 1906688.000 |
| 16 | islands_3_7__10_12__13_16 | ok | True | 7.834 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.861 | 38.281 | 1.328 | 2007040.000 |
| 17 | islands_3_6__10_12__13_17 | ok | True | 7.818 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.814 | 38.281 | 1.328 | 2308096.000 |
| 18 | islands_3_6__10_12__13_16 | ok | True | 7.768 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.842 | 38.281 | 1.328 | 2408448.000 |
| 19 | islands_3_11__15_17 | ok | True | 7.673 | 5 | 3,1,1,1,4 | stage0_cpu | 1.869 | 38.281 | 1.239 | 1505280.000 |
| 20 | islands_3_7__10_14__15_17 | ok | True | 7.552 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.853 | 38.281 | 1.328 | 1906688.000 |

Full metrics are in `summary.csv` and `summary.json`.
