# Native Stage Pipeline Run: 3/1/2

- Date: 2026-04-29
- Board: root@192.168.1.133
- Scheme: three_stage_e
- Runs: 20
- Input: default input.bin repeated
- Stage runtime threads: stage0=3, stage1=1, stage2=2
- Queue depth: 2
- Top1 serial/pipeline match: True

## Summary, skip first 3 frames

- Serial throughput: 5.309 fps
- Pipeline throughput: 8.230 fps
- Pipeline stage0 submit interval: 115.339 ms
- Pipeline avg stage0: 115.002 ms
- Pipeline avg stage1: 73.134 ms
- Pipeline avg stage2: 61.355 ms

## Interpretation

This run keeps correctness, but it is slower than 3/1/1. Giving stage2 two runtime threads reduces stage2 time, but stage0 becomes much slower and becomes the dominant bottleneck.

See `run_command.sh` for the exact command.
