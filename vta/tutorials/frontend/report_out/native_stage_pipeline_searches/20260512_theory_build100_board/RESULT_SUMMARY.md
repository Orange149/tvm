# Theory-Guided Build100 Board Results

- Board: `root@192.168.1.185`
- Candidate set: `theory_build100_buildable_ids.txt`
- Batches: `batch001` to `batch005`, 20 measured configs each
- File cache policy: `prewarm`
- Correctness policy: `cat_equivalent`
- Attempted candidates: 100
- Unique candidate ids: 100
- Successful rows: 100
- Failed rows: 0
- Remote cleanup: only `/var/volatile/vta_stage_pipeline_search/_file_cache` remains

## Best

- Scheme: `islands_3_7__10_18`
- Batch: `batch001`
- Stage devices: `cpu/vta/cpu/vta/cpu`
- Threads: `3,1,1,1,4`
- Pipeline throughput: `10.200349406848721 fps`
- Serial throughput: `4.430743130620745 fps`
- Correctness gate: `true`
- Correctness relaxed: `false`
- Baseline/native top1: `282 / 282`
- Bottleneck: `stage0_cpu`
- PS-PL bandwidth: `2.24320612687742 Gbit/s`
- PS-PL DMA bytes: `329512960`
- SRAM peak utilization: `38.28125%`
- DMA fragmentation score: `1.3525179856115108`

Measured stage times for the best candidate:

| stage | device | ms | run_ms | achieved GOPS |
|---|---|---:|---:|---:|
| stage0_cpu | cpu | 88.986 | 87.699 | 7.989 |
| stage1_vta | vta | 32.464 | 28.677 | 28.688 |
| stage2_cpu | cpu | 85.598 | 84.806 | 5.455 |
| stage3_vta | vta | 43.200 | 41.254 | 39.861 |
| stage4_cpu | cpu | 3.143 | 2.985 | 0.368 |

## Top 10

| rank | batch | scheme | fps | devices | threads | gate | relaxed | bottleneck |
|---:|---|---|---:|---|---|---|---|---|
| 1 | batch001 | `islands_3_7__10_18` | 10.200 | cpu/vta/cpu/vta/cpu | 3,1,1,1,4 | true | false | stage0_cpu |
| 2 | batch001 | `islands_3_7__10_19` | 9.874 | cpu/vta/cpu/vta/cpu | 3,1,1,1,4 | true | false | stage0_cpu |
| 3 | batch004 | `islands_3_6__10_12__13_18` | 9.868 | cpu/vta/cpu/vta/vta/cpu | 3,1,1,1,1,4 | true | false | stage0_cpu |
| 4 | batch001 | `islands_3_6__10_18` | 9.821 | cpu/vta/cpu/vta/cpu | 3,1,1,1,4 | true | false | stage0_cpu |
| 5 | batch001 | `islands_3_6__10_19` | 9.749 | cpu/vta/cpu/vta/cpu | 3,1,1,1,4 | true | false | stage0_cpu |
| 6 | batch004 | `islands_3_5__10_19` | 9.726 | cpu/vta/cpu/vta/cpu | 3,1,1,1,4 | true | false | stage0_cpu |
| 7 | batch004 | `islands_3_6__10_12__13_19` | 9.631 | cpu/vta/cpu/vta/vta/cpu | 3,1,1,1,1,4 | true | false | stage2_cpu |
| 8 | batch002 | `islands_3_16` | 9.461 | cpu/vta/cpu | 3,1,4 | true | true | stage0_cpu |
| 9 | batch005 | `islands_3_5__10_14__15_19` | 9.405 | cpu/vta/cpu/vta/vta/cpu | 3,1,1,1,1,4 | true | false | stage0_cpu |
| 10 | batch005 | `islands_3_5__10_14__15_18` | 9.287 | cpu/vta/cpu/vta/vta/cpu | 3,1,1,1,1,4 | true | false | stage0_cpu |

## Batch Summary

| batch | board configs | ok | failed | best | best fps |
|---|---:|---:|---:|---|---:|
| batch001 | 20 | 20 | 0 | `islands_3_7__10_18` | 10.200 |
| batch002 | 20 | 20 | 0 | `islands_3_16` | 9.461 |
| batch003 | 20 | 20 | 0 | `islands_3_7__10_12__13_19` | 8.779 |
| batch004 | 20 | 20 | 0 | `islands_3_6__10_12__13_18` | 9.868 |
| batch005 | 20 | 20 | 0 | `islands_3_5__10_14__15_19` | 9.405 |

## Comparisons

- Warm100 best, `islands_3_9__10_19`: `10.5323 fps`
- Previous manual pipeline best, `three_stage_e`: `10.2049 fps`
- RPC all_vta params-once baseline: about `5.66 fps`
- Theory-guided Build100 best is about tied with the previous manual `three_stage_e`, but below the earlier warm100 best.
- Theory-guided Build100 best is about `1.80x` the RPC all_vta end-to-end baseline.

## Artifacts

- Aggregate summary: `theory_board100_aggregate_summary.json`
- Per-batch summaries: `batch001/summary.json` ... `batch005/summary.json`
- Per-batch progress: `batch001/progress.txt` ... `batch005/progress.txt`
- Remote cleanup record: `remote_after.txt`
