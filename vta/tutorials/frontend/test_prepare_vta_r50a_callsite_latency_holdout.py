from prepare_vta_r50a_callsite_latency_holdout import choose_pair


def candidate(family, mode, candidate_id):
    return {"family_id": family, "public_mode": mode, "candidate_id": candidate_id}


def correct(candidate_id):
    return {
        "candidate_id": candidate_id,
        "status": "passed",
        "seeds": [{"correct": True}, {"correct": True}, {"correct": True}],
    }


def test_choose_pair_uses_registry_order_and_exclusions():
    candidates = [
        candidate("f0", "original", "o0"),
        candidate("f0", "input_stationary", "i0"),
        candidate("f1", "original", "o1"),
        candidate("f1", "input_stationary", "i1"),
    ]
    correctness = [correct(name) for name in ("o0", "i0", "o1", "i1")]
    assert [row["candidate_id"] for row in choose_pair(candidates, correctness, set())] == [
        "o0", "i0"
    ]
    assert [
        row["candidate_id"]
        for row in choose_pair(candidates, correctness, {"i0"})
    ] == ["o1", "i1"]
