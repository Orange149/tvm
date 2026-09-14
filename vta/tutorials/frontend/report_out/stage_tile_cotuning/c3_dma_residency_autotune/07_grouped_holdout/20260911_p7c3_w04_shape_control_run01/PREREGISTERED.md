# P7c3 W04 shape control

Frozen before execution.

- Diagnostic runner SHA-256: `5ec21a27639f287f72304fb1957c434938fe015e75045fce52e59c2b429a5067`.
- Target: W04 `input_stationary` ConfigSpace index 142 (`tile_h=7`, `tile_w=14`), which passed its first formal board check and has the same 1,568-vector working-set count as failed index 139 (`tile_h=14`, `tile_w=7`).
- Control: W04 `original` ConfigSpace index 139 before and after the target.
- Purpose: distinguish a raw scratchpad-capacity failure from an orientation/address-pattern or synchronization failure.
- Three repeats times three seeds; correctness only; no timing label.
