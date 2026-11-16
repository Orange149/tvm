from analyze_c3_five_tradeoff import summarize
from c3_five_tradeoff_policy import PROGRAMS, INCUMBENT, NEW, SEEDS


def rows(times):
    return [{"round": r, "program": name, "seed": s, "call_kind": "timing", "status": "PASS",
             "correct": True, "arguments_unchanged": True, "boot_id": "one-boot",
             "invocation_id": "%d:%s:%d" % (r, name, s), "latency_ms": times[name]}
            for r in range(6) for name in PROGRAMS for s in SEEDS]


def seal(rule_winner):
    return {"winners": {"rule": rule_winner, "bytes_only": INCUMBENT, "incumbent_only": INCUMBENT},
            "prefixes": {"rule": [NEW], "bytes_only": ["R12-G"], "incumbent_only": []}}


def test_fast_reference_candidate_cannot_replace_frozen_incumbent_winner():
    result = summarize(rows({INCUMBENT: 5., "R12-G": 5.2, NEW: 4.5, "Input-7x7": 5.1}), seal(INCUMBENT))
    assert result["empirical_oracle"] == NEW
    assert not result["node24_new_graph_eligible"]
    assert result["deploy_program"] == INCUMBENT


def test_small_increment_does_not_lower_predefined_gate():
    result = summarize(rows({INCUMBENT: 5., "R12-G": 5.2, NEW: 4.95, "Input-7x7": 5.1}), seal(NEW))
    assert not result["node24_new_graph_eligible"]
    assert result["rule_vs_bytes"]["same_effective_budget"]


def test_fixed_winner_good_against_both_controls_enters_graph():
    result = summarize(rows({INCUMBENT: 5., "R12-G": 5.2, NEW: 4.5, "Input-7x7": 5.1}), seal(NEW))
    assert result["node24_new_graph_eligible"]
    assert result["prediction_new_faster_than_R12"]["direction_correct"]
    assert result["methods"]["rule"]["effective_selection_calls"] == 6
    assert result["methods"]["incumbent_only"]["effective_selection_calls"] == 3
