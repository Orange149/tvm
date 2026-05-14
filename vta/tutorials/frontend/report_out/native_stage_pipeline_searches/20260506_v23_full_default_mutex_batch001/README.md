# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-06T20:37:50
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 256
- Buildability tested: 128
- Buildable candidates: 128
- Board test configs: 20
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
- Pipeline throughput: 9.075 fps
- Static throughput estimate: 5.430 fps
- Bottleneck: stage0_cpu (cpu)
- CPU/VTA GOPS est: cpu=1.164 vta=2.467
- CPU/VTA achieved GOPS: cpu=7.862 vta=33.300
- PS-PL bandwidth: 1.669 Gbit/s, DMA bytes=265103360.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=1.277 static=0.017

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | three_stage_e | ok | True | 9.075 | 3 | 3,1,4 | stage0_cpu | 1.669 | 38.281 | 1.277 | 903168.000 |
| 2 | islands_1_4__5_9__10_15 | ok | True | 8.424 | 5 | 3,1,1,1,4 | stage0_cpu | 1.606 | 38.281 | 1.228 | 2308096.000 |
| 3 | islands_1_4__5_9__10_17 | ok | True | 8.368 | 5 | 3,1,1,1,4 | stage0_cpu | 1.558 | 38.281 | 1.244 | 2107392.000 |
| 4 | islands_1_4__5_9__10_16 | ok | True | 8.339 | 5 | 3,1,1,1,4 | stage0_cpu | 1.608 | 38.281 | 1.244 | 2207744.000 |
| 5 | islands_1_4__5_12__13_19 | ok | True | 8.334 | 5 | 3,1,1,1,4 | stage0_cpu | 1.976 | 38.281 | 1.273 | 1906688.000 |
| 6 | islands_1_4__5_8__10_16 | ok | True | 8.223 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.599 | 38.281 | 1.244 | 3010560.000 |
| 7 | islands_1_4__5_8__10_17 | ok | True | 8.123 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.563 | 38.281 | 1.244 | 2910208.000 |
| 8 | islands_1_4__5_12__13_18 | ok | True | 8.110 | 5 | 3,1,1,1,4 | stage0_cpu | 1.949 | 38.281 | 1.273 | 2007040.000 |
| 9 | islands_1_4__5_8__10_15 | ok | True | 8.108 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.552 | 38.281 | 1.228 | 3110912.000 |
| 10 | islands_1_3__5_9__10_16 | ok | True | 7.962 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.574 | 38.281 | 1.244 | 3813376.000 |
| 11 | three_stage_f | ok | True | 7.952 | 3 | 3,1,4 | stage1_vta | 1.646 | 38.281 | 1.244 | 903168.000 |
| 12 | islands_1_3__5_9__10_15 | ok | True | 7.893 | 6 | 3,1,1,1,1,4 | stage1_vta | 1.553 | 38.281 | 1.228 | 3913728.000 |
| 13 | islands_1_3__5_8__10_17 | ok | True | 7.834 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.584 | 38.281 | 1.244 | 4515840.000 |
| 14 | islands_1_3__5_8__10_15 | ok | True | 7.816 | 7 | 3,1,1,1,1,1,4 | stage1_vta | 1.544 | 38.281 | 1.228 | 4716544.000 |
| 15 | islands_1_3__5_8__10_16 | ok | True | 7.808 | 7 | 3,1,1,1,1,1,4 | stage0_cpu | 1.549 | 38.281 | 1.244 | 4616192.000 |
| 16 | islands_1_3__5_9__10_17 | ok | True | 7.712 | 6 | 3,1,1,1,1,4 | stage0_cpu | 1.567 | 38.281 | 1.244 | 3713024.000 |
| 17 | three_stage_a | ok | True | 5.783 | 3 | 3,1,4 | stage0_cpu | 1.513 | 38.281 | 1.271 | 1003520.000 |
| 18 | block_stage_c | ok | True | 4.738 | 3 | 3,1,4 | stage0_cpu | 1.493 | 38.281 | 1.115 | 1204224.000 |
| 19 | fine_conv_vta_a | ok | True | 4.623 | 5 | 3,1,1,1,4 | stage0_cpu | 1.246 | 38.281 | 1.115 | 2809856.000 |
| 20 | three_stage_b | ok | True | 4.561 | 3 | 3,1,4 | stage0_cpu | 1.538 | 19.141 | 1.438 | 602112.000 |

Full metrics are in `summary.csv` and `summary.json`.
