# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T14:55:37
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 256
- Buildability tested: 128
- Buildable candidates: 128
- Board test configs: 20
- File cache policy: prewarm
- File cache warmups: 20
- Max VTA islands: 3
- Measure top N: 20
- Refine top N: 0
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_1_4__5_12__13_17
- Candidate: islands_1_4__5_12__13_17
- Stage devices: cpu/vta/vta/vta/cpu
- Threads: 3,1,1,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=False reason=exact_match baseline_top1=282 native_top1=282
- Pipeline throughput: 8.569 fps
- Static throughput estimate: 11.232 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=0.701 vta=2.930
- CPU/VTA achieved GOPS: cpu=7.675 vta=32.360
- PS-PL bandwidth: 1.618 Gbit/s, DMA bytes=303022080.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.244 static=0.016

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_1_4__5_12__13_17 | ok | True | 8.569 | 5 | 3,1,1,1,4 | stage0_cpu | 1.618 | 38.281 | 1.244 | 1906688.000 |
| 2 | islands_1_4__5_10__13_15 | ok | True | 8.456 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.556 | 38.281 | 1.210 | 2709504.000 |
| 3 | islands_1_4__5_11__13_17 | ok | True | 8.454 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.616 | 38.281 | 1.244 | 2308096.000 |
| 4 | islands_1_4__5_12__13_15 | ok | True | 8.448 | 5 | 3,1,1,1,4 | stage0_cpu | 1.566 | 38.281 | 1.228 | 2107392.000 |
| 5 | islands_1_4__5_12__13_16 | ok | True | 8.299 | 5 | 3,1,1,1,4 | stage0_cpu | 1.578 | 38.281 | 1.244 | 2007040.000 |
| 6 | islands_1_4__5_11__13_16 | ok | True | 8.256 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.581 | 38.281 | 1.244 | 2408448.000 |
| 7 | islands_1_4__5_10__13_19 | ok | True | 8.227 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.939 | 38.281 | 1.259 | 2508800.000 |
| 8 | islands_1_4__5_11__13_15 | ok | True | 8.226 | 6 | 3,1,1,1,1,4 | stage2_vta | 1.560 | 38.281 | 1.228 | 2508800.000 |
| 9 | islands_1_3__5_10__13_16 | ok | True | 8.224 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.608 | 38.281 | 1.227 | 4214784.000 |
| 10 | islands_1_4__5_10__13_17 | ok | True | 8.178 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.545 | 38.281 | 1.227 | 2508800.000 |
| 11 | islands_1_4__5_11__13_19 | ok | True | 8.175 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.967 | 38.281 | 1.273 | 2308096.000 |
| 12 | islands_1_3__5_10__13_15 | ok | True | 8.155 | 7 | 3,1,1,1,1,1,4 | stage3_vta | 1.561 | 38.281 | 1.210 | 4315136.000 |
| 13 | islands_1_4__5_11__13_18 | ok | True | 8.109 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.947 | 38.281 | 1.273 | 2408448.000 |
| 14 | islands_1_3__5_12__13_17 | ok | True | 8.108 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.641 | 38.281 | 1.244 | 3512320.000 |
| 15 | islands_1_3__5_10__13_17 | ok | True | 8.101 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.582 | 38.281 | 1.227 | 4114432.000 |
| 16 | islands_1_4__5_10__13_16 | ok | True | 8.098 | 6 | 3,1,1,1,1,4 | stage2_vta | 1.540 | 38.281 | 1.227 | 2609152.000 |
| 17 | islands_1_4__5_10__13_18 | ok | True | 7.971 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.913 | 38.281 | 1.259 | 2609152.000 |
| 18 | islands_1_3__5_12__13_16 | ok | True | 7.891 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.596 | 38.281 | 1.244 | 3612672.000 |
| 19 | islands_1_3__5_12__13_18 | ok | True | 7.777 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.935 | 38.281 | 1.273 | 3612672.000 |
| 20 | islands_1_3__5_12__13_19 | ok | True | 7.721 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.961 | 38.281 | 1.273 | 3512320.000 |

Full metrics are in `summary.csv` and `summary.json`.
