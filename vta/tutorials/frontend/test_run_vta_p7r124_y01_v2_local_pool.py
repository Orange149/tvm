import run_vta_p7r124_y01_v2_local_pool as p7r124


def candidates():
    return p7r124.build_candidates(p7r124.load_jsonl(p7r124.DEFAULT_CANDIDATES))


def test_builds_eight_by_three_unique_v2_pool():
    rows = candidates()
    assert len(rows) == len({row["candidate_id"] for row in rows}) == 24
    assert {row["family_id"] for row in rows} == {"Y01F{:02d}".format(i) for i in range(8)}
    assert {row["public_mode"] for row in rows} == set(p7r124.MODES)


def test_selection_input_contains_no_forbidden_latency_or_tophub_material():
    raw = p7r124.DEFAULT_CANDIDATES.read_bytes().lower()
    assert b"latency_ms" not in raw
    assert b"board_latency" not in raw
    assert b"tophub" not in raw


def test_frozen_structural_pair_ranking_is_complete_and_deterministic():
    first = p7r124.pair_ranking(candidates())
    second = p7r124.pair_ranking(candidates())
    assert first == second
    assert len(first) == 28
    assert sorted(row["pre_registered_rank"] for row in first) == list(range(28))
