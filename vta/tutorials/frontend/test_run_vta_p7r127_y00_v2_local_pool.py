import run_vta_p7r127_y00_v2_local_pool as p7r127


def candidates():
    return p7r127.build_candidates(p7r127.load_jsonl(p7r127.DEFAULT_CANDIDATES))


def test_y00_v2_pool_is_eight_complete_three_mode_families():
    rows = candidates()
    assert len(rows) == len({row["candidate_id"] for row in rows}) == 24
    assert {row["family_id"] for row in rows} == {"Y00F{:02d}".format(i) for i in range(8)}
    assert all(
        {row["public_mode"] for row in rows if row["family_id"] == family} == set(p7r127.MODES)
        for family in {row["family_id"] for row in rows}
    )


def test_pool_contains_no_performance_labels_or_board_dispatch():
    rows = candidates()
    assert all(row["performance_label"] is None for row in rows)
    assert all(row["board_status"] == "not_dispatched" for row in rows)
    assert all(row["identity"]["template_name"] == "conv2d_packed_residency.vta" for row in rows)


def test_structural_coverage_order_is_deterministic_complete_and_mode_invariant():
    rows = candidates()
    first = p7r127.structural_coverage_order(rows)
    second = p7r127.structural_coverage_order(list(reversed(rows)))
    assert first == second
    assert len(first) == 8
    assert len({row["family_id"] for row in first}) == 8
    assert all(row["vector"] == p7r127.structural_vector(next(
        candidate for candidate in rows if candidate["family_id"] == row["family_id"]
    )) for row in first)


def test_complete_family_filter_requires_all_three_static_fsim_and_command_passes():
    rows = candidates()
    static = [{"candidate_id": row["candidate_id"], "status": "ok"} for row in rows]
    fsim = [
        {
            "candidate_id": row["candidate_id"],
            "status": "passed",
            "seeds": [{"correct": True}] * 3,
            "command_evidence": {"status": "available_three_seed_consistent"},
        }
        for row in rows
    ]
    assert len(p7r127.complete_families(rows, static, fsim)) == 8
    fsim[0]["status"] = "failed"
    assert len(p7r127.complete_families(rows, static, fsim)) == 7
