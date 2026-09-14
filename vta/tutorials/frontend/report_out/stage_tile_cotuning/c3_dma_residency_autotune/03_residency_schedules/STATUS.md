# P3-local residency schedule status

- Date: 2026-09-10
- Scope: local lowering and FSim only; no SSH, board, runtime, driver, or hardware changes.
- Decision: **G3 NO-GO**.  The isolated input-stationary mechanism is locally valid, but the complete weight/hybrid residency objective is not achieved.
- Paper boundary: mode 3 is named `paper_inspired_hybrid`.  It is not an exact reproduction because the full 2026 paper method is unavailable.

## What is established

The registered `conv2d_packed.vta` API still selects the original schedule.  Its three frozen incumbent TIR hashes are unchanged:

- W00: `597666fdc6b1f366edd557e357ada30872bfddf1ea67eec47dec4e9c07be4bc1`
- W02: `1067810187559ffb7943426d76991d9cbd5791a2058643579198397c6924b047`
- W09: `483ed2916b9aef8d68fbf510ae740ce952d93a92a21b2b8676dd9ad68ccbdd08`

The pre/post-refactor ten-workload original DMA snapshots were byte-identical (SHA-256 `ff154ced1154c54fdd168597a1ab23433854c06652004538b821bd96fb48a3a5`).  The independent experimental template exposes modes 0--3 without adding a knob to the original ConfigSpace, and rejects non-original virtual-thread settings greater than one.

On conservative representatives, input-stationary reduces input bytes by 50% (W00/W09) and 75% (W02), with unchanged weight and store bytes.  All 12 mode/workload combinations lower and match the reference in FSim.  These are compiler-static DMA counts, not measured AXI transactions or latency.

## What is not established

The first outer-cache weight/hybrid implementation reduced static weight LOAD bytes, but FSim rejected it with an unsupported `STORE -> LOAD` dependency.  Keeping both DMA producers inside the VTA task removes that error, but weight-stationary produces no DMA reduction on W00/W02/W09.  The safe hybrid only retains an input-traffic reduction; it does not reduce weight traffic.  Therefore there is no board performance claim and no completed hybrid innovation claim.

## Verification

- Non-FSim: `16 passed, 12 deselected` (2.62 s), exit 0.
- FSim: `12 passed, 16 deselected` (3.64 s), exit 0, guarded by a 30 s timeout.
- Initial unsafe FSim: original 3/3 and input 3/3 passed; weight 3/3 failed; hybrid W00/W02 failed and W09 was interrupted.  First error is preserved in `failed_outer_cache_fsim.log`.

Detailed counts and hashes are in `p3_local_results.json`.  This directory is a negative/partial P3 result and must not advance G3.
