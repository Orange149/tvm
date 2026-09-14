# Full-pool command-capacity generality analysis

- Coverage: 197 identities, W00--W09, five modes; reduced-capacity pass 197/197
- Pool capacity: 24 KiB instruction + 12 KiB UOP = 36 KiB
- Per-identity total capacity: min 8 KiB, median 8 KiB, p90 12 KiB, max 28 KiB
- Leave-one-workload-out fixed-capacity fit: 9/10
- Rule: derive capacity from the exact compiled allowlist; never assume 36 KiB for unseen identities
