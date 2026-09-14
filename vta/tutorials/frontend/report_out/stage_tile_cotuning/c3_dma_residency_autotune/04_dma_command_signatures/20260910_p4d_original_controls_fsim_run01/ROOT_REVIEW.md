# Root review: invalid environment run

This run is rejected as an environment/preflight failure, not as candidate evidence.
The parent process was launched without `VTA_HW_PATH=/tmp/vta-hw-fsim`, so all 45
workers observed the board configuration rather than `TARGET=sim` and stopped before
build or execution.  Consequently, `0/45 passed` must **not** be interpreted as a
compile or correctness failure.  No seed was executed and no candidate was evaluated.

The corrected rerun is `../20260910_p4d_original_controls_fsim_run02/`.  Run 01 is
retained to preserve the failure and recovery trail.
