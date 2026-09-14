# HANDOFF

P3e passes every preregistered local proof gate. The positive result has two coupled
parts: the TE probe exposes weight reuse, while the compiler represents the explicit
full drain with the registered TIR Op and treats it as a dependency-segment boundary.
Schedule-only P3d remains a valid historical negative result because the compiler did
not have either capability at that point.

Evidence:

- W00: weight bytes `258048 -> 36864` (`-85.71%`), 2 residency drains plus the existing
  final drain, no `1 <-> 3`, three seeds correct.
- W02: weight bytes `589824 -> 147456` (`-75.00%`), 4 residency drains plus final drain,
  no `1 <-> 3`, three seeds correct.
- W09: weight bytes unchanged at `131072`, but calls `32 -> 2`; 2 residency drains plus
  final drain, no `1 <-> 3`, three seeds correct.
- Pure TIR and full VTA regression: `50 passed` in the final combined command.

The full drain trades fewer weight requests for a synchronization boundary. No latency,
FPS, or board-side benefit is claimed. Mode 4 must remain a probe until a later phase
explicitly admits it to candidate identity and measures it on hardware. Any broader
compiler adoption should first test nested conditionals/loops beyond the minimal segment
cases here.

No runtime, driver, RTL, or ISA file was modified. The existing runtime CHECK remains.
