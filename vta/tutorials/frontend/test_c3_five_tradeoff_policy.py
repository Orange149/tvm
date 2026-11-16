import pytest

from c3_five_tradeoff_policy import PROGRAMS, SEEDS, INCUMBENT, NEW, prefixes, choose_visible


def features():
    data = {name: {"local_qualified": True, "inp_bytes": 100, "acc_bytes": 10,
                  "out_bytes": 10, "mac": 112, "reset": 112, "submissions": 1,
                  "normal_completion": True, "independent_windows": 0,
                  "wgt_arena_bytes": 252 * 1024, "wgt_bytes": 720 * 1024}
            for name in PROGRAMS}
    data[NEW]["independent_windows"] = 21
    data[INCUMBENT]["wgt_bytes"] = 612 * 1024
    data["R12-G"]["wgt_arena_bytes"] = 234 * 1024
    data["Input-7x7"]["wgt_bytes"] = 1152 * 1024
    return data


def test_opportunity_and_bytes_have_same_budget_different_prefix():
    proposal = prefixes(features())
    assert proposal["rule"] == [NEW]
    assert proposal["bytes_only"] == ["R12-G"]
    assert proposal["budget_new_candidates"] == 1


def test_unknown_safe_overlap_does_not_pass():
    data = features()
    data[NEW]["independent_windows"] = 0
    with pytest.raises(ValueError, match="conditions"):
        prefixes(data)


def test_equal_exposure_and_tie_falls_back_to_incumbent():
    panel = [{"program": name, "seed": seed, "call_kind": "selection", "correct": True,
              "status": "PASS", "latency_ms": 5.0} for name in PROGRAMS for seed in SEEDS]
    assert choose_visible(panel, [NEW]) == INCUMBENT
    for row in panel:
        if row["program"] == NEW:
            row["latency_ms"] = 4.0
        if row["program"] == "Input-7x7":
            row["latency_ms"] = 1.0
    assert choose_visible(panel, [NEW]) == NEW
    assert choose_visible(panel, ["R12-G"]) == INCUMBENT


def test_confirmation_cannot_choose_winner():
    panel = [{"program": name, "seed": seed, "call_kind": "confirmation", "correct": True,
              "status": "PASS", "latency_ms": 5.0} for name in (INCUMBENT, NEW) for seed in SEEDS]
    with pytest.raises(ValueError, match="non-selection"):
        choose_visible(panel, [NEW])
