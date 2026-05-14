# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T13:21:19
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

- Scheme: islands_3_6__10_18
- Candidate: islands_3_6__10_18
- Stage devices: cpu/vta/cpu/vta/cpu
- Threads: 3,1,1,1,4
- Correctness gate: n/a
- Correctness policy: n/a relaxed=n/a reason=n/a baseline_top1=n/a native_top1=n/a
- Pipeline throughput: n/a fps
- Static throughput estimate: 6.046 fps
- Bottleneck: stage2_cpu (n/a)
- CPU/VTA GOPS est: cpu=1.165 vta=2.467
- CPU/VTA achieved GOPS: cpu=n/a vta=n/a
- PS-PL bandwidth: n/a Gbit/s, DMA bytes=5586688.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=n/a static=0.014

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_6__10_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2207744.000 |
| 2 | islands_3_6__10_18 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2207744.000 |
| 3 | islands_3_6__10_14__15_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2408448.000 |
| 4 | islands_3_6__10_14__15_18 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2408448.000 |
| 5 | islands_3_7__10_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 1806336.000 |
| 6 | islands_3_7__10_18 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 1806336.000 |
| 7 | islands_3_7__10_14__15_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2007040.000 |
| 8 | islands_3_7__10_14__15_18 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2007040.000 |
| 9 | islands_3_8__13_15 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2107392.000 |
| 10 | islands_3_8__13_15 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 2107392.000 |
| 11 | islands_3_9__13_15 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 1705984.000 |
| 12 | islands_3_9__13_15 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.014 | 1705984.000 |
| 13 | islands_3_6__10_19 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2107392.000 |
| 14 | islands_3_6__10_19 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2107392.000 |
| 15 | islands_3_6__10_14__15_19 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2308096.000 |
| 16 | islands_3_6__10_14__15_19 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2308096.000 |
| 17 | islands_3_6__10_15 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2308096.000 |
| 18 | islands_3_6__10_15 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 2308096.000 |
| 19 | islands_3_7__10_19 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 1705984.000 |
| 20 | islands_3_7__10_19 | timeout |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 1705984.000 |

Full metrics are in `summary.csv` and `summary.json`.
