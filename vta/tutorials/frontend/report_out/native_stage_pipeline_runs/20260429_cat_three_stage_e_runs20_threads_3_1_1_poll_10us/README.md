# Native Stage Pipeline Run: 3/1/1 Poll 10us

- Date: 2026-04-29
- Board: root@192.168.1.133
- Runs: 20
- Stage runtime threads: stage0=3, stage1=1, stage2=1
- AXU5EVB_DRIVER_POST_START_SLEEP_NS=10000
- AXU5EVB_DRIVER_POLL_SLEEP_NS=10000
- Top1 serial/pipeline match: True

## Summary, skip first 3 frames

- Serial throughput: 4.600 fps
- Pipeline throughput: 10.158 fps
- Pipeline stage0 submit interval: 89.758 ms
- Pipeline avg stage0: 90.252 ms
- Pipeline avg stage1: 74.877 ms
- Pipeline avg stage2: 80.900 ms

## Interpretation

10us poll sleep keeps correctness, but it is slower than the original 1us poll 3/1/1 run. Stage2 shows early backlog and the overall span throughput falls below the 1us baseline.
