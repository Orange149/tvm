# P7R30 E01 tile-ci discriminating contract

- Frozen after observing config 951/957 failures and before executing config 877.
- Config 877 preserves `tile_h=3, tile_w=21, tile_co=5, oc_nthread=2` and two
  post-cthread output-channel groups from config 957, but changes `tile_ci=12`
  to `tile_ci=1`.
- Purpose: test whether the remaining FSim-to-FPGA correctness gap is tied to
  the reduction-tile depth rather than DMA orientation.
- Correctness only: three exact-output seeds, no latency or FPS.

