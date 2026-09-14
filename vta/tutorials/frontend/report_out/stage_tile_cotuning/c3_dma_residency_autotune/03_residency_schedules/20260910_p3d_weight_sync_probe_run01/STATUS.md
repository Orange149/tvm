# C3-P3d weight residency sync probe run 01

- Status: `completed_negative`
- Decision: **NO-GO for schedule-only explicit drain**
- Probe mode: `4 / weight_stationary_sync_probe`
- Production modes 0--3: preserved
- Original registered schedule: preserved
- Board/SSH: not used
- FSim performance timing: not collected

Mode 4 restores `ckernel` at the outer output-channel tile and requests one drain after
the complete spatial loop of each weight-residency tile. The TE schedule is constructible,
but current `InjectCoProcSync` rejects the explicit pragma while constructing
`vta.coproc_sync`; therefore executable TIR is not produced and three-seed FSim is not
reachable.

A counterfactual diagnostic that strips only the broken explicit-sync pragma confirms the
reason not to call this a partial success: W00/W02 show 85.71%/75% fewer weight bytes, but
generic `CoProcSync` still inserts balanced yet forbidden `STORE(3) -> LOAD(1)` pairs. W09
shows no weight-byte reduction. The schedule-only gate fails both executable lowering and
dependency legality.

Regression tests:

- Dedicated negative probe tests: 10/10 passed.
- Existing residency suite under local FSim: 28/28 passed, including frozen-original TIR
  and all existing modes 0--3.

