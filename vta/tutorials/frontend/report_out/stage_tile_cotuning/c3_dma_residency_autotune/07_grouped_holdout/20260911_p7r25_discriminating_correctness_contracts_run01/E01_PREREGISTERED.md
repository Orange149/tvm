# P7R25 E01 discriminating correctness contract

- Frozen before executing E01 config 957 on the real FPGA.
- Purpose: distinguish the long-height 2-D DMA geometry of failing config 951
  from a general E01 or two-context failure.
- Config 957 preserves the same channel/reduction tiles and two post-cthread
  output-channel groups, but rotates the spatial tile from `21x3` to `3x21`.
- Three exact-output seeds; no latency or FPS measurement.

