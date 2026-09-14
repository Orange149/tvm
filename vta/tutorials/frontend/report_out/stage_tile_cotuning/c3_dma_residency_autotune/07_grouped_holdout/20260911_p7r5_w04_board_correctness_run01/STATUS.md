# P7R5 W04 board correctness

- Status: `stopped_on_wrong_same_tile_original`
- Protected original config463: 3/3 seeds exact.
- Same-tile original config330: 0/3 seeds exact; mismatch counts 334, 336, and 334.
- Input-stationary config330: not executed.
- Config455/461 pairs: not executed.
- Timing/FPS: not collected.
- Board after stop: same boot, FPGA operating, u-dma-buf unchanged, no new storage error after the frozen dmesg cutoff.

This is a real-FPGA legality failure that FSim did not expose. Config330 and its transformed
candidate are permanently excluded from timing. Remaining geometry probes require a new adaptive
development contract; this failed batch cannot simply resume past the stop point.
