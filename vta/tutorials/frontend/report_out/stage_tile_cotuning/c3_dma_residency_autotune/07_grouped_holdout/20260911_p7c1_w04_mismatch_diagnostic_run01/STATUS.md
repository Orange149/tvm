# Status

Confirmed a target-specific, board-only correctness failure.

- The same-config `original` sentinel passed before and after the target (6/6 seed checks).
- The `input_stationary` target failed all 9 target executions (three repeats times three seeds), with 2,197--2,432 mismatched output elements per execution.
- Target output hashes and first mismatch positions changed across repeats, while the sentinels stayed exact.
- Board boot, FPGA state, u-dma-buf size, RPC tmpfs location, SD mount, free-space reserve, and storage-error watermark remained valid.
- No timing label was collected.

Interpretation: this is consistent with a real-FPGA scratchpad-address/dependency hazard that local FSim did not expose. Exact cause still requires lowered-TIR/address-span analysis.
