# Handoff

Use `shortlist.json` as the immutable input to a later board dispatcher. Select the requested workload, strategy, and budget, and dispatch only the listed `candidate_ids`; do not recompute their order after seeing any board outcome.

Review before dispatch:

1. Verify `input_manifest.json` hashes against the frozen P4b inputs and planner source.
2. Verify the requested prefix retains `protected_original_incumbent` and contains no ID listed by `filtered_failures.json`.
3. Count budget by unique candidate ID per workload and strategy. Budget 16 can saturate below 16 when fewer static lower-successes exist.
4. Preserve compile, correctness, timeout, RPC, environment, and latency outcomes as later labels; they were not consumed here.
5. Treat B7's mode diversity as exploration coverage only. It does not establish benefit from weight or hybrid mode.
6. Evaluate B4--B7 only after prospective board execution under the separately preregistered protocol. Do not infer G5 or performance from this run.

No existing schedule, runtime, driver, generator, or hardware file was changed. The only implementation additions are the standalone planner and its unit test.
