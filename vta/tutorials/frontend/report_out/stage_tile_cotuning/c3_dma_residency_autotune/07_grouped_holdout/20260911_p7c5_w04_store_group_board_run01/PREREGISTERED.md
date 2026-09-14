# P7c5 W04 output-store grouping board probe

Frozen before board execution.

- Probe-contract SHA-256: `04d214515077254fb61c2b88428684975a5392571b57779208e349d7fd091198`.
- Runner SHA-256: `7a6431262b79c6e670f9fe83501f78beeaf2df555f2cd8fff5494ab08697f70c`.
- Hardware: exact `vta_hpc.bit` SHA-256 `7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6`, confirmed in current boot logs and on-board firmware.
- Fixed dimensions: W04, `input_stationary`, `tile_h=14`, `tile_w=7`, `tile_ci=1`, virtual threads 1, and 1,568 accumulator vectors (below the 2,048-vector capacity).
- Changed dimension: `tile_co` in 1, 2, 4, 8, 16. Config139/tile_co4 reuses its already-recorded failure and is not rerun; the other four are new observations.
- Every new candidate receives three exact seed checks. A known-correct original-config139 seed-0 sentinel runs immediately afterward; sentinel failure stops the probe.
- All candidate pass/fail results are retained, with no replacement and no timing label.
