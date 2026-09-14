# P7R8 W04 paired timing contract

Frozen before any P7R timing label on 2026-09-11.

- Eligibility evidence: P7R7 real FPGA correctness, 18/18 exact seed checks.
- Correctness ledger SHA-256: `7cf17eb6ee6bf34861ac11eca45d27f186a7d1d421a7095e11be1679d86cc98d`.
- Recovery contract SHA-256: `9c6173c8fc2b6f48a6f74937d2fc4daeb2c9da332baddbb98d642c084a0b58c7`.
- Timing runner SHA-256: `7172af99b75246be9b4802ddeae957866bb9cd656745bd95b053c5aba2697381`.
- Candidates: original/input-stationary config455 and config461; protected original config463.
- Warmup: 3 direct executions per module.
- Measurement: 7 deterministic randomized complete blocks, 5 inferences per reported sample,
  one sample per middle candidate per block, incumbent before and after every block.
- Random order seed: 20260911.
- Primary endpoint: median same-tile relative speedup for 455 and 461 separately.
- Protected endpoint: each residency median relative to TopHub config463 median.
- Scope: one boot, development workload; no unseen-confirmation or full-network FPS claim.
- Stop on wrong output, changed board fingerprint, new SD error, or 300-second outer timeout;
  never reboot or power off.
