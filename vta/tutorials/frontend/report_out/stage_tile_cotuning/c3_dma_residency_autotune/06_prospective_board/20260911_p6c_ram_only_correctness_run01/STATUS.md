# C3-P6c board correctness result

Status: **passed**.

- 9/9 frozen candidates executed on the AXU5EVB under boot ID `a68a7983-719f-47bf-94d7-41f974c5342c`.
- 27/27 candidate/seed checks matched the independent NumPy reference exactly.
- This includes all three mode-4 weight-barrier candidates, so their semantics are now verified on real VTA hardware rather than only FSim/cross-compilation.
- Per-seed runtime counters confirm that the generated mechanisms changed physical runtime requests. For example, W00 input-stationary reduced input LOAD bytes from 487,424 to 243,712, while W00 weight-barrier reduced weight LOAD bytes from 258,048 to 36,864.
- No latency was collected by P6c.
- The SD ext4 partition remained read-only and no storage error occurred after dmesg timestamp 467.

This passes the P6d correctness precondition. It does not establish a speedup, generic search benefit, stage/FPS benefit, or SD-card repair.
