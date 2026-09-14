# C3-P4f AXU5EVB cross-compile qualification

- Status: `completed`
- P4e FSim-passed candidates: 165
- Passed build/export qualification: 165
- Failed: 0
- Fresh isolated cross-compiles: 155
- P2 run02 incumbent citations: 10
- Candidate binaries retained: 0
- SSH/RPC/board: not used
- Performance: not measured; this proves build/export deployability only

Verification notes:

- Post-P3e `libtvm.so` was guarded at SHA-256 `3feea276219c2ace9e33f4d31c664d7c0bc971a3efb88becfc1b0fe2fd6e76a9`; all six production/library guard hashes were unchanged from run start to end.
- Pre-batch fresh smoke candidate `017ba6fa...fd9` passed identity, TIR, VTA build, and AXU export; the formal batch then rebuilt it normally.
- Unit tests: 4 passed, exit 0.
- Across the 155 fresh records, median VTA build/export time was 0.0689/0.0909 seconds. These are compilation costs, not inference latency.
