# Native Stage Pipeline Run: 3/1/1 Poll 1ms

- Date: 2026-04-29
- Board: root@192.168.1.133
- Scheme: three_stage_e
- Runs: 20
- Input: default input.bin repeated
- Stage runtime threads: stage0=3, stage1=1, stage2=1
- AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000000
- AXU5EVB_DRIVER_POLL_SLEEP_NS=1000000
- Top1 serial/pipeline match: True

## Summary, skip first 3 frames

- Serial throughput: 4.326 fps
- Pipeline throughput: 9.022 fps
- Pipeline stage0 submit interval: 103.415 ms
- Pipeline avg stage0: 102.609 ms
- Pipeline avg stage1: 72.619 ms
- Pipeline avg stage2: 72.042 ms

## Interpretation

Increasing VTA poll sleep to 1ms did not improve throughput. Stage0 becomes slower than the original 3/1/1 run while stage1 remains around the same order, so CPU contention is not primarily caused by tight VTA polling.
