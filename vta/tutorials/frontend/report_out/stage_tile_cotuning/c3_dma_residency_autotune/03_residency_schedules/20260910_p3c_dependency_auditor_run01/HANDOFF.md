# C3-P3c handoff

The dependency auditor is ready to serve as a pre-build rejection gate:

```python
report = audit_vta_coproc_dependencies(lowered_module)
if not report["valid"]:
    reject_candidate(report["errors"])
```

Recommended placement is after VTA lowering has inserted dependency intrinsics and before accepting a candidate for FSim/board measurement. Persist the full report with the candidate identity.

Important limits:

- This gate recognizes the concrete runtime topology rule that exposed the first outer-weight failure.
- Balance is per-function/per-directed-edge static callsite balance, not symbolic path balance.
- Passing the audit does not prove DMA compactness, SRAM capacity, instruction capacity, numerical correctness, or performance.
- Keep FSim and later board validation mandatory.

The W00 four-mode evidence is in `w00_safe_modes_audit.json`. Artificial CLI accept/reject cases demonstrate exit-code behavior. No existing production file was edited.
