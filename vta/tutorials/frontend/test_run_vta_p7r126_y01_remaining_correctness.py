import run_vta_p7r126_y01_remaining_correctness as p7r126


def test_complement_is_exact_disjoint_exhaustive_partition():
    rows = p7r126.select_complement(p7r126.DEFAULT_P7R124, p7r126.DEFAULT_P7R125)
    assert len(rows) == len({row["candidate_id"] for row in rows}) == 18
    assert {row["family_id"] for row in rows} == p7r126.EXPECTED_FAMILIES
    assert all(
        [row["public_mode"] for row in rows if row["family_id"] == family]
        == list(p7r126.MODE_ORDER)
        for family in sorted(p7r126.EXPECTED_FAMILIES)
    )


def test_complement_keeps_no_performance_labels_and_no_replacement():
    rows = p7r126.select_complement(p7r126.DEFAULT_P7R124, p7r126.DEFAULT_P7R125)
    assert all(row["performance_label"] is None for row in rows)
    assert all(row["board_status"] == "not_dispatched" for row in rows)
    assert all(row["identity"]["template_name"] == "conv2d_packed_residency.vta" for row in rows)


def test_all_candidates_bind_three_seed_fsim_and_command_signatures():
    rows = p7r126.select_complement(p7r126.DEFAULT_P7R124, p7r126.DEFAULT_P7R125)
    bindings, static = p7r126.qualification_bindings(p7r126.DEFAULT_P7R124, rows)
    assert set(bindings) == {row["candidate_id"] for row in rows}
    assert all(static[candidate_id]["status"] == "ok" for candidate_id in bindings)
    assert all(len(row["command_signature_sha256"]) == 64 for row in bindings.values())
    assert all(row["command_signature"]["submissions"] >= 1 for row in bindings.values())


def test_summary_counts_complete_families_without_generalization():
    rows = p7r126.select_complement(p7r126.DEFAULT_P7R124, p7r126.DEFAULT_P7R125)
    bindings, _ = p7r126.qualification_bindings(p7r126.DEFAULT_P7R124, rows)
    synthetic = []
    for index, row in enumerate(rows):
        passed = index < 3
        synthetic.append(
            {
                "candidate_id": row["candidate_id"],
                "family_id": row["family_id"],
                "public_mode": row["public_mode"],
                "status": "passed" if passed else "failed",
                "seeds": [
                    {"correct": passed, "mismatch_count": 0 if passed else seed + 1}
                    for seed in range(3)
                ],
            }
        )
    summary = p7r126.summarize(synthetic, rows, bindings)
    assert summary["complete_three_mode_correct_family_count"] == 1
    assert summary["association"]["causal_or_generalization_claim"] is False
    assert len(summary["association"]["identity_rows"]) == 18
