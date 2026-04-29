# Native Stage Pipeline Run: three_stage_f

- Date: 2026-04-29
- Board: root@192.168.1.133
- Scheme: three_stage_f
- Runs: 20
- Stage runtime threads: stage0=3, stage1=1, stage2=1
- Serial/pipeline top1 match: True
- Pipeline top1 values: [282]

## Summary, skip first 3 frames

- Pipeline throughput: 8.801 fps
- Pipeline stage0 interval: 90.461 ms
- Pipeline avg stage0: 36.809 ms
- Pipeline avg stage1: 98.789 ms
- Pipeline avg stage2: 69.471 ms

## Interpretation

stage0 drops as expected, but stage1 becomes the bottleneck and total pipeline throughput is worse than three_stage_e. The top1 also changes from the prior three_stage_e default-input result, so this split requires RPC/all_vta correctness validation before it can be considered valid.
