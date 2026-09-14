# Experiment D: cross-boot frozen-topology replay

This intermediate report covers the first two independent board boots. Boot 2 contains
two 22-frame runs and one 62-frame run; these are repeated runs on one boot, not three
independent boots. The subsequently completed third-boot Latin-square replay is reported
separately in `EXPERIMENT_D_THIRD_BOOT_LATIN.md`.
All runs discard the first two completions for throughput.

| topology | boot 1, 22 frames | boot 2 run A, 22 | boot 2 run B, 22 | boot 2, 62 | 62-frame interval P95 |
|---|---:|---:|---:|---:|---:|
| A | 10.872 FPS | 10.872 FPS | 10.892 FPS | 10.955 FPS | 105.848 ms |
| B | 10.310 FPS | 10.220 FPS | 10.697 FPS | 10.020 FPS | 136.753 ms |
| C | 10.994 FPS | 11.559 FPS | 10.431 FPS | 10.818 FPS | 113.255 ms |
| D | 10.501 FPS | 10.648 FPS | 10.582 FPS | 10.840 FPS | 108.952 ms |

## Findings

- Every topology has one stable output hash across serial/pipeline replay and across boots.
- LOAD/STORE calls and payload are exactly invariant in every run. The memory mechanism
  conclusions (same-DMA A/C and the extra-island demand of D) therefore reproduce.
- Fine FPS ordering does not reproduce: boot 1 orders C>A>D>B, while the primary boot-2
  62-frame run orders A>D>C>B. Cycle-rank Spearman is 0.400.
- The two short runs on boot 2 expose substantial B/C variation. The 62-frame B run still
  has a 136.753 ms completion-interval P95. This points to CPU-stage/runtime jitter, not
  variable VTA DMA.

## Decision

Use these data to validate invariant memory features and coarse pruning rules, but do not
fit a DDR-time coefficient or claim a significant sub-FPS topology ordering from these two
boots. See `EXPERIMENT_D_THIRD_BOOT_LATIN.md` for the completed 4x4 Latin-square replay.
