# P7c7 W04 tile-height board probe

Frozen before board execution.

- Height-contract SHA-256: `636e437846d6aeb5357577972ee2bc0b629ecfcf2893080876bdbb30b895949f`.
- Runner SHA-256: `8d74856259c6ae3c9ae476788062cfb89774336f260b04e73baeb66fb08a69a0`.
- Fixed dimensions: W04, `input_stationary`, `tile_w=7`, `tile_ci=1`, `tile_co=4`, virtual threads 1.
- Planned heights were 1, 2, 7, and 14. Heights 2 and 7 were statically rejected because VTA's DMA injection could not express their local-ACC store as a legal 2-D transfer. Height 14 reuses its prior board failure and is not rerun. Therefore only the previously unobserved height-1 candidate is dispatched.
- The new candidate receives three exact seed checks, followed by the known-correct original-config139 seed-0 sentinel.
- No replacement and no timing label.
