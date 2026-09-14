# C3 P6a offline dispatch contract status

Status: **offline_dispatch_contract_frozen**. This run generated contracts only; it used no SSH, RPC, board execution, or latency measurement.

Three immutable B7/budget-4 dispatch manifests are frozen for W00, W02, and W09. Each contains four ordered, unique candidate IDs with the protected original incumbent first, complete ConfigEntity payloads, workload/mode/template/schedule identity, three correctness seeds (`0`, `20250901`, `20260910`), an unfilled timing protocol, and a pending actual board fingerprint. Every manifest explicitly records `board_executed=false` and `G6_passed=false`.

The input validator reproduces the complete P5b shortlist and feature-set hash, verifies P5b/P4b file hashes, joins all 250 candidate identities, and requires the unified P4e qualification evidence. All 12 selected entries have `local_status=fsim_passed` for all three required seeds. P4e accounts for all 250 candidates: 165 three-seed FSim passes and 85 lower failures, with no unqualified lower-success candidate.

The B7 four-mode composition is exploratory coverage only. It is not evidence that weight-stationary or hybrid mode improves performance. P4e FSim qualification is local correctness evidence, not board correctness or performance evidence. G6 was not evaluated.
