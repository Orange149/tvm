import pytest
from audit_vta_c3_reuse_completion import check_dispatch_accounting, delayed_evidence, check_logical_dma


def test_every_dispatch_counted_even_wrong_answer():
    result = dict(candidate_id="x", seed=0, phase="timing", correct=False,
                  complete_operator_calls=1, hidden_warmup_calls=0)
    timeline = [dict(event="dispatch_begin", candidate_id="x", seed=0, phase="timing")]
    assert check_dispatch_accounting([result], timeline) == 1
    with pytest.raises(ValueError):
        check_dispatch_accounting([], timeline)


def test_future_identity_or_hidden_warmup_rejected():
    result = dict(candidate_id="y", seed=0, phase="timing",
                  complete_operator_calls=1, hidden_warmup_calls=0)
    timeline = [dict(event="dispatch_begin", candidate_id="x", seed=0, phase="timing")]
    with pytest.raises(ValueError):
        check_dispatch_accounting([result], timeline)
    result.update(candidate_id="x", hidden_warmup_calls=1)
    with pytest.raises(ValueError):
        check_dispatch_accounting([result], timeline)


def test_delayed_read_evidence_consistent():
    result = dict(correct=False, actual_sha256="wrong", expected_sha256="right", delayed_sha256="right",
                  delayed_correct=True, delayed_mismatch_count=0, changed_after_return=True)
    assert delayed_evidence([result])["became_correct_without_second_execution"] == 1
    result["changed_after_return"] = False
    with pytest.raises(ValueError):
        delayed_evidence([result])


def test_bytes_not_divided_by_submission():
    candidate = dict(local_qualification={"semantics": {"totals": {"bytes": 100}}},
                     fsim_qualification={"command_features": {"values": {"submissions": 2}}})
    result = dict(runtime_profile_complete={"bytes": 100, "driver_run_calls": 2})
    check_logical_dma(result, candidate)
    result["runtime_profile_complete"]["bytes"] = 50
    with pytest.raises(ValueError):
        check_logical_dma(result, candidate)
