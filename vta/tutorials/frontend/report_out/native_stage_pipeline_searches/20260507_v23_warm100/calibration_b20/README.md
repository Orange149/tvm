# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T13:41:41
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 256
- Buildability tested: 0
- Buildable candidates: 0
- Board test configs: 0
- File cache policy: prewarm
- File cache warmups: 0
- Max VTA islands: 3
- Measure top N: 64
- Refine top N: 8
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_18_18
- Candidate: islands_18_18
- Stage devices: cpu/vta/cpu
- Threads: 3,1,4
- Correctness gate: n/a
- Correctness policy: n/a relaxed=n/a reason=n/a baseline_top1=n/a native_top1=n/a
- Pipeline throughput: n/a fps
- Static throughput estimate: 2.020 fps
- Bottleneck: stage0_cpu (n/a)
- CPU/VTA GOPS est: cpu=3.169 vta=0.462
- CPU/VTA achieved GOPS: cpu=n/a vta=n/a
- PS-PL bandwidth: n/a Gbit/s, DMA bytes=1255936.000
- SRAM peak utilization: 7.037%
- DMA fragmentation: measured=n/a static=0.007

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_18_18 | static |  | n/a | 3 | 3,1,4 | stage0_cpu | n/a | 7.037 | 0.007 | 301056.000 |
| 2 | islands_15_15__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage0_cpu | n/a | 9.570 | 0.007 | 802816.000 |
| 3 | islands_15_15 | static |  | n/a | 3 | 3,1,4 | stage0_cpu | n/a | 9.570 | 0.008 | 501760.000 |
| 4 | islands_5_5__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.008 | 2809856.000 |
| 5 | islands_3_3__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.008 | 3211264.000 |
| 6 | islands_1_1__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.008 | 3211264.000 |
| 7 | islands_5_5__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 2308096.000 |
| 8 | islands_15_16__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage0_cpu | n/a | 9.570 | 0.009 | 702464.000 |
| 9 | islands_3_4__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 2408448.000 |
| 10 | islands_1_2__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 2408448.000 |
| 11 | islands_10_10__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage0_cpu | n/a | 19.141 | 0.009 | 1806336.000 |
| 12 | islands_3_3__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 2709504.000 |
| 13 | islands_1_1__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 2709504.000 |
| 14 | islands_15_15__18_19 | static |  | n/a | 5 | 3,1,1,1,4 | stage0_cpu | n/a | 9.570 | 0.009 | 702464.000 |
| 15 | islands_13_13__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage0_cpu | n/a | 9.570 | 0.009 | 1404928.000 |
| 16 | islands_3_4__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 1906688.000 |
| 17 | islands_1_2__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 1906688.000 |
| 18 | islands_5_6__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.009 | 2408448.000 |
| 19 | islands_10_10__18_18 | static |  | n/a | 5 | 3,1,1,1,4 | stage0_cpu | n/a | 19.141 | 0.009 | 1304576.000 |
| 20 | islands_8_8__15_15__18_18 | static |  | n/a | 7 | 3,1,1,1,1,1,4 | stage2_cpu | n/a | 19.141 | 0.009 | 2007040.000 |

Full metrics are in `summary.csv` and `summary.json`.
