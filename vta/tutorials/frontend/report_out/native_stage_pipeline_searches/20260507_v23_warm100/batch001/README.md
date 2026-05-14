# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-07T14:00:57
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

- Scheme: three_stage_e
- Candidate: three_stage_e
- Stage devices: cpu/vta/cpu
- Threads: 3,1,4
- Correctness gate: True
- Correctness policy: cat_equivalent relaxed=True reason=cat_equivalent baseline_top1=282 native_top1=285
- Pipeline throughput: 9.311 fps
- Static throughput estimate: 5.430 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.164 vta=2.467
- CPU/VTA achieved GOPS: cpu=8.361 vta=35.488
- PS-PL bandwidth: 1.781 Gbit/s, DMA bytes=265103360.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.277 static=0.017

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | three_stage_e | ok | True | 9.311 | 3 | 3,1,4 | stage0_cpu | 1.781 | 38.281 | 1.277 | 903168.000 |
| 2 | islands_1_4__5_12__13_19 | ok | True | 8.324 | 5 | 3,1,1,1,4 | stage0_cpu | 1.971 | 38.281 | 1.273 | 1906688.000 |
| 3 | islands_1_4__5_9__10_16 | ok | True | 8.316 | 5 | 3,1,1,1,4 | stage0_cpu | 1.578 | 38.281 | 1.244 | 2207744.000 |
| 4 | islands_1_4__5_9__10_15 | ok | True | 8.315 | 5 | 3,1,1,1,4 | stage0_cpu | 1.559 | 38.281 | 1.228 | 2308096.000 |
| 5 | islands_1_4__5_12__13_18 | ok | True | 8.176 | 5 | 3,1,1,1,4 | stage0_cpu | 1.957 | 38.281 | 1.273 | 2007040.000 |
| 6 | islands_1_4__5_8__10_15 | ok | True | 8.161 | 6 | 3,1,1,1,1,4 | stage5_cpu | 1.547 | 38.281 | 1.228 | 3110912.000 |
| 7 | three_stage_f | ok | True | 8.139 | 3 | 3,1,4 | stage1_vta | 1.612 | 38.281 | 1.244 | 903168.000 |
| 8 | islands_1_4__5_9__10_17 | ok | True | 8.072 | 5 | 3,1,1,1,4 | stage0_cpu | 1.519 | 38.281 | 1.244 | 2107392.000 |
| 9 | islands_1_4__5_8__10_16 | ok | True | 7.991 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.575 | 38.281 | 1.244 | 3010560.000 |
| 10 | islands_1_3__5_9__10_17 | ok | True | 7.977 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.571 | 38.281 | 1.244 | 3713024.000 |
| 11 | islands_1_4__5_8__10_17 | ok | True | 7.964 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.540 | 38.281 | 1.244 | 2910208.000 |
| 12 | islands_1_3__5_8__10_16 | ok | True | 7.928 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.627 | 38.281 | 1.244 | 4616192.000 |
| 13 | islands_1_3__5_9__10_16 | ok | True | 7.922 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.582 | 38.281 | 1.244 | 3813376.000 |
| 14 | islands_1_3__5_9__10_15 | ok | True | 7.804 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.528 | 38.281 | 1.228 | 3913728.000 |
| 15 | islands_1_3__5_8__10_15 | ok | True | 7.769 | 7 | 3,1,1,1,1,1,4 | stage1_vta | 1.545 | 38.281 | 1.228 | 4716544.000 |
| 16 | islands_1_3__5_8__10_17 | ok | True | 7.744 | 7 | 3,1,1,1,1,1,4 | stage1_vta | 1.576 | 38.281 | 1.244 | 4515840.000 |
| 17 | three_stage_a | ok | True | 5.946 | 3 | 3,1,4 | stage0_cpu | 1.514 | 38.281 | 1.271 | 1003520.000 |
| 18 | block_stage_c | ok | True | 4.746 | 3 | 3,1,4 | stage0_cpu | 1.492 | 38.281 | 1.115 | 1204224.000 |
| 19 | fine_conv_vta_a | ok | True | 4.684 | 5 | 3,1,1,1,4 | stage0_cpu | 1.492 | 38.281 | 1.115 | 2809856.000 |
| 20 | three_stage_b | ok | True | 4.559 | 3 | 3,1,4 | stage0_cpu | 1.535 | 19.141 | 1.438 | 602112.000 |

Full metrics are in `summary.csv` and `summary.json`.
