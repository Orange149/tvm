# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T13:22:31
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 20
- Buildability tested: 20
- Buildable candidates: 20
- Board test configs: 20
- File cache policy: prewarm
- File cache warmups: 0
- Max VTA islands: 3
- Measure top N: 20
- Refine top N: 0
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_3_6__10_12__13_16
- Candidate: islands_3_6__10_12__13_16
- Stage devices: cpu/vta/cpu/vta/vta/cpu
- Threads: 3,1,1,1,1,4
- Correctness gate: n/a
- Correctness policy: n/a relaxed=n/a reason=n/a baseline_top1=n/a native_top1=n/a
- Pipeline throughput: n/a fps
- Static throughput estimate: 6.046 fps
- Bottleneck: stage2_cpu (n/a)
- CPU/VTA GOPS est: cpu=1.627 vta=2.004
- CPU/VTA achieved GOPS: cpu=n/a vta=n/a
- PS-PL bandwidth: n/a Gbit/s, DMA bytes=4280576.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=n/a static=0.015

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_6__10_12__13_16 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2408448.000 |
| 2 | islands_3_6__10_12__13_16 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2408448.000 |
| 3 | islands_3_7__10_12__13_16 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 2007040.000 |
| 4 | islands_3_7__10_12__13_16 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 2007040.000 |
| 5 | islands_3_11__15_17 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 1505280.000 |
| 6 | islands_3_11__15_17 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 1505280.000 |
| 7 | islands_3_7__8_11__15_17 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage3_cpu | n/a | 38.281 | 0.016 | 1906688.000 |
| 8 | islands_3_7__8_11__15_17 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage3_cpu | n/a | 38.281 | 0.016 | 1906688.000 |
| 9 | islands_3_15 | buildable |  | n/a | 3 | 3,1,4 | stage1_vta | n/a | 38.281 | 0.016 | 1103872.000 |
| 10 | islands_3_15 | timeout |  | n/a | 3 | 3,1,4 | stage1_vta | n/a | 38.281 | 0.016 | 1103872.000 |
| 11 | islands_3_7__8_15 | buildable |  | n/a | 4 | 3,1,1,4 | stage2_vta | n/a | 38.281 | 0.016 | 1505280.000 |
| 12 | islands_3_7__8_15 | timeout |  | n/a | 4 | 3,1,1,4 | stage2_vta | n/a | 38.281 | 0.016 | 1505280.000 |
| 13 | islands_3_6__10_14__15_17 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 2308096.000 |
| 14 | islands_3_6__10_12__13_17 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 2308096.000 |
| 15 | islands_3_6__10_14__15_17 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 2308096.000 |
| 16 | islands_3_6__10_12__13_17 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 2308096.000 |
| 17 | islands_3_12__15_17 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 1304576.000 |
| 18 | islands_3_12__15_17 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 1304576.000 |
| 19 | islands_3_7__8_12__15_17 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage3_cpu | n/a | 38.281 | 0.016 | 1705984.000 |
| 20 | islands_3_7__8_12__15_17 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage3_cpu | n/a | 38.281 | 0.016 | 1705984.000 |

Full metrics are in `summary.csv` and `summary.json`.
