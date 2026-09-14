from run_vta_resnet50_callsite_greedy_board import summarize_proposal


KEYS = (
    "driver_run_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_wgt_calls",
    "synchronize_calls",
)


def profile(base):
    return {key: base.get(key, 0) for key in KEYS}


def rows(candidate_latencies):
    expected = {
        "driver_run_calls": 16,
        "load_buffer_2d_wgt_bytes": -393216,
        "load_buffer_2d_wgt_calls": -432,
        "synchronize_calls": 16,
    }
    baseline_profile = profile({
        "driver_run_calls": 52,
        "load_buffer_2d_bytes": 75012096,
        "load_buffer_2d_calls": 12908,
        "load_buffer_2d_wgt_bytes": 33865728,
        "load_buffer_2d_wgt_calls": 6332,
        "synchronize_calls": 52,
    })
    candidate_profile = dict(baseline_profile)
    for key, value in expected.items():
        candidate_profile[key] += value
    candidate_profile["load_buffer_2d_bytes"] -= 393216
    candidate_profile["load_buffer_2d_calls"] -= 432
    correctness = []
    for _ in range(3):
        correctness.extend((
            {"mode": "current_incumbent", "paired_equal": True,
             "runtime_profile_complete": baseline_profile},
            {"mode": "one_node_proposal", "paired_equal": True,
             "runtime_profile_complete": candidate_profile},
        ))
    timing = []
    for index, candidate_ms in enumerate(candidate_latencies):
        timing.extend((
            {"round": index, "mode": "current_incumbent", "latency_ms": 10.0,
             "runtime_profile_complete": {key: 2 * value for key, value in baseline_profile.items()}},
            {"round": index, "mode": "one_node_proposal", "latency_ms": candidate_ms,
             "runtime_profile_complete": {key: 2 * value for key, value in candidate_profile.items()}},
        ))
    return correctness, timing, expected


def test_summarize_proposal_accepts_only_seven_of_seven_strict_win():
    correctness, timing, expected = rows([9.0] * 7)
    result = summarize_proposal(1, 57, [0, 0, 0, 0], [1, 0, 0, 0],
                                correctness, timing, expected)
    assert result["accepted"]
    assert result["paired_wins"] == 7


def test_summarize_proposal_rejects_one_round_loss():
    correctness, timing, expected = rows([9.0] * 6 + [11.0])
    result = summarize_proposal(1, 57, [0, 0, 0, 0], [1, 0, 0, 0],
                                correctness, timing, expected)
    assert not result["accepted"]
    assert result["checks"]["strictly_negative_median_delta"]
    assert not result["checks"]["seven_of_seven_candidate_wins"]
