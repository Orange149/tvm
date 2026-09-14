# Handoff

The only dispatchable artifacts in this run are:

- `dispatches/W00_B7_budget4/dispatch_manifest.json`
- `dispatches/W02_B7_budget4/dispatch_manifest.json`
- `dispatches/W09_B7_budget4/dispatch_manifest.json`

Do not edit or overwrite them. A later board phase must consume the selected contract by SHA-256 and write a separate execution record containing the actual board fingerprint, frozen timing parameters, per-seed board correctness, failure category, and latency observations.

Before any board action:

1. Verify the contract hash from `artifact_hashes.json`.
2. Require the actual target, VTA config SHA-256, and bitstream SHA-256 to match the expected fingerprint; also record the current boot identifier.
3. Freeze the currently empty timing fields in a separate preregistered execution protocol before observing results.
4. Preserve the candidate order and charge every attempted candidate, including compile/correctness/timeout/RPC/environment failures, to the budget.
5. Treat the three-seed P4e FSim result only as local admission evidence. Re-run the contract's three correctness seeds on the board.
6. Do not infer mode benefit, latency, regret, or G6 from P6a.

The preparation tool contains no SSH/RPC path and refuses to overwrite an existing `dispatch_manifest.json`.
