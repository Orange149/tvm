# P7R4 board preflight attempt

- Status: `stopped_before_candidate_build_or_execution`
- Cause: the runner required RPC cwd to start with `/tmp`, while the valid tmpfs-backed RPC
  cwd is `/var/volatile/vta_c3_ram/runtime`.
- Board labels collected: 0
- Candidate selection/order changed: no
- Recovery: corrected only the preflight path predicate; retry uses a new output directory.
