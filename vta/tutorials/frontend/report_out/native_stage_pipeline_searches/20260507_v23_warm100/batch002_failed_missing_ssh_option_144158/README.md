# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T14:41:30
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 256
- Buildability tested: 128
- Buildable candidates: 128
- Board test configs: 1
- File cache policy: prewarm
- File cache warmups: 0
- Max VTA islands: 3
- Measure top N: 20
- Refine top N: 0
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_1_4__5_10__13_18
- Candidate: islands_1_4__5_10__13_18
- Stage devices: cpu/vta/vta/cpu/vta/cpu
- Threads: 3,1,1,1,1,4
- Correctness gate: n/a
- Correctness policy: n/a relaxed=n/a reason=n/a baseline_top1=n/a native_top1=n/a
- Pipeline throughput: n/a fps
- Static throughput estimate: 11.399 fps
- Bottleneck: stage2_vta (n/a)
- CPU/VTA GOPS est: cpu=0.252 vta=3.380
- CPU/VTA achieved GOPS: cpu=n/a vta=n/a
- PS-PL bandwidth: n/a Gbit/s, DMA bytes=7176064.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=n/a static=0.014

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_1_4__5_10__13_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2609152.000 |
| 2 | islands_1_3__5_10__13_18 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4214784.000 |
| 3 | islands_1_4__5_11__13_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2408448.000 |
| 4 | islands_1_4__5_10__13_15 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2709504.000 |
| 5 | islands_1_4__5_10__13_19 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2508800.000 |
| 6 | islands_1_4__5_10__13_19 | timeout |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2508800.000 |
| 7 | islands_1_3__5_11__13_18 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4014080.000 |
| 8 | islands_1_3__5_10__13_19 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4114432.000 |
| 9 | islands_1_7__8_12__13_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage1_vta | n/a | 38.281 | 0.014 | 1605632.000 |
| 10 | islands_1_4__5_12__13_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2007040.000 |
| 11 | islands_1_4__5_9__10_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 2207744.000 |
| 12 | islands_1_3__5_10__13_15 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4315136.000 |
| 13 | islands_1_4__5_11__13_19 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.015 | 2308096.000 |
| 14 | islands_1_4__5_11__13_15 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.015 | 2508800.000 |
| 15 | islands_3_8__10_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage0_cpu | n/a | 38.281 | 0.015 | 2207744.000 |
| 16 | islands_1_5__8_12__13_15 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage1_vta | n/a | 38.281 | 0.015 | 2910208.000 |
| 17 | islands_1_4__5_10__13_16 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.015 | 2609152.000 |
| 18 | islands_1_3__5_12__13_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.015 | 3612672.000 |
| 19 | islands_1_3__5_11__13_19 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.015 | 3913728.000 |
| 20 | islands_3_9__10_18 | buildable |  | n/a | 4 | 3,1,1,4 | stage0_cpu | n/a | 38.281 | 0.015 | 1404928.000 |

Full metrics are in `summary.csv` and `summary.json`.
