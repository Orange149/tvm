import run_vta_p7r126_y01_recovery as recovery


def test_recovery_is_only_interrupted_final_identity():
    _, failure, rows, candidate = recovery.recovery_plan(recovery.DEFAULT_INTERRUPTED)
    assert len(rows) == 17
    assert sum(len(row["seeds"]) for row in rows) == 51
    assert failure["candidate_dispatches"] == 52
    assert candidate["family_id"] == "Y01F06"
    assert candidate["public_mode"] == "weight_resident_barrier"
