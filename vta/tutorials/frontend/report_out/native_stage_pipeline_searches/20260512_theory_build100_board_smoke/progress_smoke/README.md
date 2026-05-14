# ResNet18 N-stage Stage Split Search

- Generated: 2026-05-12T13:18:59
- Raw island candidates: 13457
- Native candidates: 7374
- Static shortlist: 2
- Buildability tested: 0
- Buildable candidates: 2
- Board test configs: 2
- File cache policy: prewarm
- File cache warmups: 0
- Max VTA islands: 3
- Measure top N: 2
- Refine top N: 0
- Primary objective: measured pipeline throughput fps

## Best

- Scheme: islands_3_9__13_16
- Candidate: islands_3_9__13_16
- Stage devices: cpu/vta/cpu/vta/cpu
- Threads: 3,1,1,1,4
- Correctness gate: n/a
- Correctness policy: n/a relaxed=n/a reason=n/a baseline_top1=n/a native_top1=n/a
- Pipeline throughput: n/a fps
- Static throughput estimate: 7.777 fps
- Bottleneck: stage2_cpu (n/a)
- CPU/VTA GOPS est: cpu=1.524 vta=2.107
- CPU/VTA achieved GOPS: cpu=n/a vta=n/a
- PS-PL bandwidth: n/a Gbit/s, DMA bytes=4325120.000
- SRAM peak utilization: 38.281%
- DMA fragmentation: measured=n/a static=0.015

## Ranking

| rank | scheme | status | gate | fps | stages | threads | bottleneck | ps-pl Gbit/s | SRAM peak % | DMA frag | boundary bytes |
|---:|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 1 | islands_3_9__13_16 | dry_run |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.015 | 1605632.000 |
| 2 | islands_3_9__13_17 | dry_run |  | n/a | 5 | 3,1,1,1,4 | stage2_cpu | n/a | 38.281 | 0.016 | 1505280.000 |

Full metrics are in `summary.csv` and `summary.json`.
