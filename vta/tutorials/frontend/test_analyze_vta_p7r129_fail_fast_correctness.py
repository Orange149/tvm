from analyze_vta_p7r129_fail_fast_correctness import aggregate, analyze, replay_identity


def _seed(seed, correct, load=10):
    return {
        "seed": seed,
        "correct": correct,
        "runtime_profile": {
            "load_buffer_2d_bytes": load,
            "load_buffer_2d_calls": 2,
            "store_buffer_2d_bytes": 3,
            "store_buffer_2d_calls": 1,
            "driver_run_insns": 4,
        },
    }


def _row(candidate, results):
    seeds = [0, 20250901, 20260910]
    return {
        "geometry": "G",
        "candidate_id": candidate,
        "family_id": "F",
        "public_mode": "original",
        "status": "passed" if all(results) else "failed",
        "seeds": [_seed(seed, correct) for seed, correct in zip(seeds, results)],
    }


def test_first_failure_stops_but_pass_requires_all_three():
    failed = replay_identity(_row("a", [False, False, False]))
    passed = replay_identity(_row("b", [True, True, True]))
    assert failed["fail_fast_invocations"] == 1
    assert passed["fail_fast_invocations"] == 3
    assert failed["decision_matches"] and passed["decision_matches"]


def test_late_failure_is_not_promoted():
    record = replay_identity(_row("a", [True, False, False]))
    assert record["fail_fast_invocations"] == 2
    assert record["fail_fast_decision"] == "failed"
    assert record["decision_matches"]


def test_aggregate_counts_saved_invocations_and_traffic():
    records = [
        replay_identity(_row("a", [False, False, False])),
        replay_identity(_row("b", [True, True, True])),
    ]
    result = aggregate(records)
    assert result["full_invocations"] == 6
    assert result["fail_fast_invocations"] == 4
    assert result["full"]["load_buffer_2d_bytes"] == 60
    assert result["fail_fast"]["load_buffer_2d_bytes"] == 40


def test_analysis_rejects_duplicate_candidate_ids():
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        analyze([_row("a", [True, True, True]), _row("a", [False, False, False])])
