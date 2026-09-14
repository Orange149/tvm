# P7R358: power-loss restart connectivity check

Status: BOARD_UNREACHABLE_BEFORE_SSH; no candidate dispatched.

User reported power loss and board restart. Read-only host diagnostics found:

- Host eth1 UP, address 192.168.1.100/24.
- Route to 192.168.1.247 uses eth1 with source 192.168.1.100.
- SSH port 22: `No route to host` (exit 255).
- Neighbor entry for 192.168.1.247: FAILED.
- Three ping probes: zero replies, Destination Host Unreachable.

No current-boot board ID, SD filesystem health, FPGA state, or runtime state
could be read. This is a connectivity observation, not evidence of SD failure.
No board files or hardware state were modified.

Resume from P7R354 Y08 conv12 immutable candidate contract, P7R355 budget
protocol, P7R356 completed local qualification, and P7R357 frozen search orders.
P7R357 has six eligible candidates and a one-candidate Pareto front; it contains
no FPGA correctness or performance labels. Do not regenerate the candidate pool
or claim prebuild-versus-lazy discrimination from a singleton front.

Next gate: restore target connectivity; inspect current boot/storage state;
restore hash-qualified runtime and bitstream in RAM; run health canary before
new Y08 board measurements. Prior-boot health qualification is not current-boot
qualification.
