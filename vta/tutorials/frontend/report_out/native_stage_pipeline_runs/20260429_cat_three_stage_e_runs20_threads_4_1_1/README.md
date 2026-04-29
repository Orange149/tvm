# Native Stage Pipeline Run: 4/1/1

- Date: 2026-04-29
- Board: root@192.168.1.133
- Scheme: three_stage_e
- Runs: 20
- Input: default input.bin repeated
- Stage runtime threads: stage0=4, stage1=1, stage2=1
- Queue depth: 2
- Top1 serial/pipeline match: True

## Summary, skip first 3 frames

- Serial throughput: 5.210 fps
- Pipeline throughput: 8.689 fps
- Pipeline stage0 submit interval: 90.785 ms
- Pipeline avg stage0: 85.898 ms
- Pipeline avg stage1: 74.337 ms
- Pipeline avg stage2: 102.131 ms

## Interpretation

This run proves functional overlap and correctness, but 4 CPU threads on stage0 starve stage2. Pipeline stage2 expands far beyond serial stage2 and becomes the tail bottleneck, so this configuration is worse than 3/1/1.

See `run_command.sh` for the exact command.
