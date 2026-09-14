# P7R20 E02 prospective timing contract

Frozen before any E02 latency label.

- E02 was predeclared on 2026-09-10 and had no prior latency measurement.
- Correctness ledger: `eb886e1a8f3da32f28b3bef4b69df0ad942608a1f1d94ce8bc3678e983ff391e`;
  all 15 real-FPGA seed checks passed.
- Board contract: `04573afd7317d5db7f3050654ae77ecfd0de1d0d33e0079a89a9b149157228b3`.
- Timing runner: `7fd82e53b38f125cc86ac57a4825642c0a24d1e4a08642ec44e973273947f2b4`.
- Request-shape point: config1163, `(tile_h,tile_w,tile_ci,tile_co)=(6,3,1,6)`.
- Contiguous control: config1167, `(6,6,1,6)`.
- Frozen prediction: the request-shape point has greater same-tile residency speedup than the
  contiguous control. Absolute latency ordering is secondary because tile work differs.
- Seven deterministic randomized complete blocks; three warmups per module; five inferences per
  reported sample; order seed 20260911.
- Stop on mismatch, board fingerprint/storage change, or 300-second timeout. Never reboot/power off.
- Scope: prospective unseen-geometry operator test; no stage or FPS claim.
