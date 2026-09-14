# P7Q2 W04 qualified timing

Frozen before board timing.

- Qualified-contract SHA-256: `dbed44f6260f6acade9dd723875f86f53b843402603447c4a927b7fb618f2246`.
- Timing-runner SHA-256: `8bfe549490c8ddd22eec9d1b526be9dfccf222faa3e159b13ce385b8981562d5`.
- Scope: the 18 W04 candidates that passed the frozen correctness gate.
- The one FPGA-invalid W04 candidate stays in gross search orders and consumes a dispatch if reached, but receives no timing label.
- Three warmups per timed candidate, then five frozen randomized complete blocks with number=1.
- Primary candidate label: median of five samples. Incumbent before/after block sentinels are retained separately and excluded from candidate medians.
- One data/weight/output set is reused for the complete workload.
