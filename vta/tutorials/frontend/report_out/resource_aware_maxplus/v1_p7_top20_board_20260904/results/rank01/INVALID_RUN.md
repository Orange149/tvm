# Invalid Run

This directory is excluded from every P7 result summary. The reuse deployment command omitted
`--stage-runtime-num-threads`, so the generated runner used `4,1,4` instead of the candidate's
required `4,1,1` configuration.

The valid rank-1 result is in `../rank01_valid/`. The deployment tool now inherits the package
thread map when the option is omitted, and every accepted P7 result rechecks the board-returned
manifest against the ranked candidate.
