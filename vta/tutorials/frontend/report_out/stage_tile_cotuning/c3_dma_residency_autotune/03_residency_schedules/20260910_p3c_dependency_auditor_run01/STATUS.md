# C3-P3c dependency auditor run01

- Status: `completed_local_tooling`
- Result: `P3C_DEPENDENCY_AUDITOR_PASS`
- Scope: lowered-TIR audit and local tests only.
- Production changes: none. The schedule, transform, build module, and runtime were not modified.
- Board access: none; no SSH, RPC board connection, or performance claim.

## Tool result

`audit_vta_coproc_dependencies.py` accepts a `tvm.IRModule`/`tir.PrimFunc` directly and also provides JSON-in/JSON-out CLI operation for modules serialized by `tvm.ir.save_json`. It reports each dependency callsite and each per-function directed edge, with push/pop counts and the following checks:

- allowed: load(1) ↔ compute(2), compute(2) ↔ store(3);
- forbidden: load(1) ↔ store(3);
- unsupported stage/self edges rejected;
- nonconstant or malformed endpoints rejected;
- push/pop callsites balanced per function and directed edge.

The formal report contract is `audit_vta_coproc_dependencies.schema.jsonschema`, schema ID `vta_coproc_dependency_audit_v1`.

## Verification

- Pytest: 10 passed, 15 warnings, exit 0.
- Artificial legal CLI case: exit 0, `valid=true`.
- Artificial forbidden 3→1 CLI case with `--fail-on-invalid`: exit 2, `valid=false`, error `forbidden_direct_load_store`.
- Current W00 safe modes (`original`, `input_stationary`, `weight_stationary`, `paper_inspired_hybrid`): 4/4 valid. Each contains 12 dependency callsites (6 push, 6 pop), four supported directed edges, balanced counts, and no 1↔3 edge.

## Interpretation limit

Balance is a static callsite check. Calls nested in a loop are counted once, and the auditor is explicitly not path-sensitive. It is an early compile-time rejection gate for the known VTA topology failure, not a replacement for build, FSim, or runtime validation.
