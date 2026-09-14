# P3 local residency run 01

- Status: `partial_completed_input_only`
- Gate: `G3_LOCAL_MECHANISM`
- Gate result: only the `input_stationary` sub-mechanism passed; the `weight_stationary` and `paper_inspired_hybrid` targets did not pass.
- Scope: local lower/static-DMA/FSim only. No SSH, board, runtime, driver, or hardware changes.
- Academic label: `paper_inspired_hybrid`, not an exact reproduction of the 2026 paper.

The final safe implementation preserved the original registered schedule and isolated four experimental modes. On W00/W02/W09, all modes lower and match the reference in FSim. Input-stationary reduced static input DMA bytes by 50%, 75%, and 50%, respectively. Static counts are not measured AXI transactions or board performance.

The first outer-weight-cache implementation reduced static weight LOADs but failed FSim with an unsupported `STORE -> LOAD` dependency. The safe recovery kept both DMA producers inside the VTA task; this restored correctness but eliminated weight-DMA reduction. The safe hybrid retains only input-traffic reduction, so the full hybrid target remains unproved.

Final verification accepted by the root agent:

- non-FSim: 16 passed, 12 deselected, exit 0;
- FSim: 12 passed, 16 deselected, exit 0.

Conclusion: preserve this as a partial positive result plus a negative boundary result. Do not advance the complete C3 mechanism through G3 on this run.
