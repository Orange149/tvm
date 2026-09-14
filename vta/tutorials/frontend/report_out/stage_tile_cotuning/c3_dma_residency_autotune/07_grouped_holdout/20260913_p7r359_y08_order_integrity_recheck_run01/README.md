# P7R359: offline order integrity recheck

Added checks for the implemented tie rule and Pareto axes, manifest coverage of
all consumed inputs, duplicate/missing candidate identities, candidate-level
board exposure, static success preceding FSim acceptance, and nonnegative
integer DMA metrics. Nine synthetic regression tests pass.

Executed the strengthened freezer against immutable P7R355/P7R356 inputs.
All five search orders and all six program metric records are exactly equal
to P7R357. P7R357 remains the original pre-board freeze; it was not overwritten.
P7R359 is an integrity recheck, not a new independent holdout or performance run.

Remaining boundary: this check does not independently establish the ancestry
of the v2 candidate set against the original v1 candidate commitment, nor does
it certify FPGA correctness. Artifact hashes check integrity, not scientific
validity. No performance labels were collected.

Board SSH was retried and still returned `No route to host`; current-boot
health qualification and Y08 measurements remain pending.
