# Experiment D: third-boot Latin-square replay

Boot `ba873689-d01b-47df-beaf-f536e8508824` uses four 62-frame rounds. The order
is `A-B-C-D / B-C-D-A / C-D-A-B / D-A-B-C`, so each frozen topology occupies
each execution position once. The first two completions are discarded per trial.

| topology | four FPS trials | median FPS | min--max FPS | median interval P95 |
|---|---|---:|---:|---:|
| A | 10.870, 10.928, 10.567, 10.921 | 10.896 | 10.567--10.928 | 107.361 ms |
| B | 10.404, 10.078, 9.973, 10.103 | 10.091 | 9.973--10.404 | 131.550 ms |
| C | 10.851, 10.383, 10.821, 10.691 | 10.756 | 10.383--10.851 | 106.789 ms |
| D | 10.498, 10.480, 10.463, 10.588 | 10.489 | 10.463--10.588 | 108.046 ms |

## Round ordering and paired directions

- Round 1: `A>C>D>B`.
- Round 2: `A>D>C>B`.
- Round 3: `C>A>D>B`.
- Round 4: `A>C>D>B`.
- `A_vs_C`: left is faster in 3/4 rounds; median left-minus-right is +0.125 FPS.
- `D_vs_B`: left is faster in 4/4 rounds; median left-minus-right is +0.444 FPS.

Aggregate boot-3 median ordering is `A>C>D>B`. Cross-boot cycle-rank Spearman is boot1/boot2 `0.400`, boot1/boot3 `0.800`, and boot2/boot3 `0.800`.

## Reproducibility decision

- All topology-specific output hashes remain stable across all three boots.
- Per-frame LOAD/STORE calls and payload remain exactly invariant across all three boots.
- The four within-boot FPS orderings are not identical; the fine ordering is therefore
  not a robust target for a scalar DDR-time regression.
- Retain static tile-derived DMA and graph boundary bytes as mechanism-grounded search
  features and safe dominance/pruning signals. Do not add a fitted DMA time on top of
  measured VTA service time, because that would double-count transfers already inside it.

Raw archive SHA-256: `65d80cc7cc6d989b82f38631a4eac495d285bdd49daae9ff1992b510636614aa`.
