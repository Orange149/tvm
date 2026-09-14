# P7c9 W04 tile-width board probe

Frozen before board execution.

- Width-contract SHA-256: `ea0f9b3ec0a5b751e286f07e4934102c84c186f269bf2408d7be500c393bf20f`.
- Runner SHA-256: `1571d1e3b9baa166f5f8eca451cc660ea0159c54ef3636848f4422df6be49609`.
- Fixed dimensions: W04, `input_stationary`, `tile_h=14`, `tile_ci=1`, `tile_co=4`, virtual threads 1.
- Widths 1 and 2 are new board observations. Width 7 reuses its prior failure. Width 14 is rejected without dispatch because its 3,136-vector accumulator region exceeds the frozen 2,048-vector FPGA capacity.
- Each new candidate receives three exact seed checks followed by a known-correct original-config139 seed-0 sentinel.
- No replacement and no timing label.
