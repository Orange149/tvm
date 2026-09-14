# C3-P6c RAM-only board correctness preregistration

Frozen before observing P6c board output on 2026-09-11.

- Input contract: the immutable P6b mechanism manifest and dispatch plan from `20260911_p6b_offline_mechanism_canary_run01`.
- Board boot ID: `a68a7983-719f-47bf-94d7-41f974c5342c`.
- Storage constraint: `/dev/mmcblk1p2` must remain read-only; runtime, bitstream, kernel module, uploaded binaries, and RPC working files must remain on RAM-backed filesystems.
- Candidates: all 9 frozen candidates across W00, W02, and W09.
- Correctness: exact equality against the independent NumPy oracle for seeds 0, 20250901, and 20260910.
- Stop rule: stop the batch at the first candidate failure, boot-ID change, RPC loss, FPGA state change, u-dma-buf size change, SD becoming writable, or new storage error after dmesg timestamp 467.
- Timing: forbidden in P6c. This run can qualify candidates for later P6d timing but cannot support a performance claim.
