# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-06T19:54:11
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 256
- Buildability tested: 128
- Buildable candidates: 128
- Board test configs: 12
- Max VTA islands: 3
- Measure top N: 64
- Refine top N: 8
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: three_stage_f
- Candidate: three_stage_f
- Stage devices: cpu/vta/cpu
- Threads: 3,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 10.321 fps
- Static throughput estimate: 4.421 fps
- Bottleneck: stage1_vta (vta)
- CPU/VTA GOPS est: cpu=0.701 vta=2.930
- CPU/VTA achieved GOPS: cpu=7.336 vta=34.874
- PS-PL bandwidth: 1.677 Gbit/s, DMA bytes=303022080.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.244 static=0.016

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | three_stage_f | ok | True | 10.321 | 3 | 3,1,4 | stage1_vta | 1.677 | 38.281 | 1.244 | 903168.000 |
| 2 | three_stage_e | ok | True | 8.722 | 3 | 3,1,4 | stage0_cpu | 1.762 | 38.281 | 1.277 | 903168.000 |
| 3 | three_stage_a | ok | True | 5.882 | 3 | 3,1,4 | stage0_cpu | 1.513 | 38.281 | 1.271 | 1003520.000 |
| 4 | block_stage_c | ok | True | 4.742 | 3 | 3,1,4 | stage0_cpu | 1.492 | 38.281 | 1.115 | 1204224.000 |
| 5 | fine_conv_vta_a | ok | True | 4.727 | 5 | 3,1,1,1,4 | stage0_cpu | 1.299 | 38.281 | 1.115 | 2809856.000 |
| 6 | three_stage_b | ok | True | 4.647 | 3 | 3,1,4 | stage0_cpu | 1.532 | 19.141 | 1.438 | 602112.000 |
| 7 | islands_1_4__5_10__13_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2609152.000 |
| 8 | islands_1_3__5_10__13_18 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4214784.000 |
| 9 | islands_1_4__5_11__13_18 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2408448.000 |
| 10 | islands_1_4__5_10__13_15 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2709504.000 |
| 11 | islands_1_4__5_10__13_19 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2508800.000 |
| 12 | islands_1_3__5_11__13_18 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4014080.000 |
| 13 | islands_1_3__5_10__13_19 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4114432.000 |
| 14 | islands_1_7__8_12__13_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage1_vta | n/a | 38.281 | 0.014 | 1605632.000 |
| 15 | islands_1_4__5_12__13_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.014 | 2007040.000 |
| 16 | islands_1_4__5_9__10_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 2207744.000 |
| 17 | islands_1_3__5_10__13_15 | buildable |  | n/a | 7 | 3,1,1,1,1,1,4 | stage3_vta | n/a | 38.281 | 0.014 | 4315136.000 |
| 18 | islands_1_4__5_11__13_19 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.015 | 2308096.000 |
| 19 | islands_1_4__5_11__13_15 | buildable |  | n/a | 6 | 3,1,1,1,1,4 | stage2_vta | n/a | 38.281 | 0.015 | 2508800.000 |
| 20 | islands_3_8__10_18 | buildable |  | n/a | 5 | 3,1,1,1,4 | stage0_cpu | n/a | 38.281 | 0.015 | 2207744.000 |

Full metrics are in `summary.csv` and `summary.json`.
