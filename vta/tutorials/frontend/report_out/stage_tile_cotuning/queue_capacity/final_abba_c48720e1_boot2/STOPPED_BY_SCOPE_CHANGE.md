# Boot 2 stopped after scope change

Date: 2026-09-10.

The user stopped the remaining queue-capacity performance campaign to prioritize the
AutoTVM research direction. The process was interrupted during
`block1_B_PF_1_F`; that incomplete run is not present in `runs_completed` and must
not be reconstructed or used.

The preserved `summary.json` contains 33 complete runs and 8 complete ABBA blocks
from boot `c48720e1-b116-4cfb-b6c4-a360b0161245`. Its `status: running` reflects
the interrupted preregistered script, not an active process. No local controller or
board-side `vta_stage_pipeline_runner` remained after interruption.

This directory is exploratory incomplete evidence only. Do not combine it with
boot 1 for the preregistered three-boot confidence interval and do not report the
three-boot non-inferiority criterion as passed.
