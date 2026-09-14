# P7R158 bitstream-isolated command-capacity diagnostic

- Status: `passed_with_capacity_induced_persistent_correctness_anomaly`
- A1 and A3 after exact bitstream reload: 16 KiB + 4 KiB, same Y00 control correct
- A2 without reload after B: `wrong_answer`
- B control: 12 KiB instruction + 4 KiB UOP with 4 KiB offset compensation, `wrong_answer`
- B rejection: exact 13,488-byte Y00 fallback rejected before device run
- Deployment decision: retain 16 KiB + 4 KiB for this exact allowlist
- Default RPC restored; W05 health passed before and after
