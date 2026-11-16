import json
import pytest
from run_vta_c3_completion_probe import analyze_trace


def result():
    return dict(candidate_id="x", family_id="M2", residence_mode="input_weight_resident_barrier",
                phase="diagnosis", seed=0, correct=True, delayed_correct=True,
                runtime_profile_complete={"driver_run_calls": 2, "driver_run_insns": 204})


def test_long_and_short_submissions_bound_to_same_operator():
    log = "\n".join("[C3_COMPLETION] " + json.dumps({"insn_count": n}) for n in (200, 4))
    matched = analyze_trace(log, [result()])
    assert [r["submit_position"] for r in matched] == [1, 2]
    assert all(r["candidate_id"] == "x" for r in matched)


def test_missing_or_extra_submissions_rejected():
    for counts in ((204,), (100, 100, 4)):
        log = "\n".join("[C3_COMPLETION] " + json.dumps({"insn_count": n}) for n in counts)
        with pytest.raises(ValueError, match="trace"):
            analyze_trace(log, [result()])


def test_mismatched_instruction_count_rejected():
    log = "\n".join("[C3_COMPLETION] " + json.dumps({"insn_count": n}) for n in (200, 5))
    with pytest.raises(ValueError, match="instruction"):
        analyze_trace(log, [result()])
