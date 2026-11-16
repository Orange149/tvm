"""Fixed protocol, evidence boundaries, and three-point NNLS regression tests."""

import numpy as np
import pytest

import c3_residency_tradeoff as p


def rows(latencies=None):
    latencies = latencies or {name: 5 + i for i, name in enumerate(p.PROGRAMS)}
    return [{"call_kind": "timing", "round": r, "program": name, "seed": seed,
             "invocation_id": "%d:%s:%d" % (r, name, seed), "boot_id": "boot-one",
             "status": "PASS", "correct": True, "arguments_unchanged": True,
             "latency_ms": latencies[name] + (j - 1) * 0.01}
            for r in range(6) for name in p.PROGRAMS for j, seed in enumerate(p.SEEDS)]


def test_williams_and_seed_balance():
    assert p.balance()["within_row_predecessor_balanced"]
    for pos in range(3):
        assert sorted(p.seed_order(r)[pos] for r in range(6)) == sorted(p.SEEDS * 2)
    assert sum(order.index("F13-B") < order.index("R13-G") for order in p.ORDERS) == 3


def test_capacity_and_request_units():
    assert [(r + 1) * 18 for r in (0, 8, 12, 13)] == [18, 162, 234, 252]
    assert [p.expectations("R%d-G" % r)["wgt_loads"] for r in (0, 8, 12, 13)] == [112, 64, 40, 34]
    assert p.expectations("R13-G")["wgt_payload_bytes"] == 612 * 1024
    assert p.expectations("R0-G") != p.expectations("Input-7x7")


def test_non_reset_is_not_profiler_total():
    signature = p.expectations("F13-B")
    profile = {"push_gemm_op_calls": 1008, "driver_run_calls": 1}
    assert p.verify_call_profile("F13-B", profile, signature)["mac"] == 896
    with pytest.raises(ValueError):
        p.verify_call_profile("F13-B", {**profile, "push_gemm_op_calls": 896}, signature)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_boot", "zero", "fail"])
def test_invalid_rows_cannot_become_samples(change):
    data = rows()
    if change == "missing":
        data.pop()
    elif change == "duplicate":
        data[-1] = data[0]
    elif change == "wrong_boot":
        data[0]["boot_id"] = "another-boot"
    elif change == "zero":
        data[0]["latency_ms"] = 0
    else:
        data[0]["correct"] = False
    with pytest.raises(ValueError):
        p.round_medians(data)


def test_pairing_uses_round_gains_not_ratio_of_aggregates():
    rm = {r: {"a": a, "b": b} for r, (a, b) in enumerate(
        [(1, 2), (1, 2), (50, 100), (50, 60), (100, 110), (100, 110)])}
    result = p.compare(rm, "a", "b")
    assert result["wins"] == 6
    assert result["median_improvement_pct"] == pytest.approx((50 + 100 / 6) / 2)


def test_nnls_interpolates_but_does_not_validate_command_generalization():
    features = {n: p.expectations(n) for n in p.PROGRAMS}
    model = p.fit_model({"R0-G": 8, "R13-G": 5, "F13-B": 7}, features)
    assert model["rank"] == 3
    assert model["coefficients_ms"] == pytest.approx([5 - 17 / 13 - 2 / 7, 17 / 13, 2 / 7])
    check = p.validate_model(model, {"R8-G": 5 + 15 / 13, "R12-G": 5 + 3 / 13}, features)
    assert check["passed"]
    assert all(x["absolute_error_ms"] < 1e-12 for x in check["rows"])


def test_rank_failure_does_not_replace_feature():
    features = {n: p.expectations(n) for n in p.PROGRAMS}
    features["F13-B"]["mac"] = 112
    model = p.fit_model({"R0-G": 8, "R13-G": 5, "F13-B": 7}, features)
    assert model["status"] == "MODEL_UNIDENTIFIABLE"
    assert "coefficients_ms" not in model
    assert p.validate_model(model, {}, {})["passed"] is None


def test_active_constraints_and_negative_prediction_result_are_preserved():
    features = {n: p.expectations(n) for n in p.PROGRAMS}
    model = p.fit_model({"R0-G": 4, "R13-G": 5, "F13-B": 4}, features)
    assert model["active_zero_constraints"]
    assert min(model["coefficients_ms"]) >= 0
    check = p.validate_model(model, {"R8-G": 20, "R12-G": 20}, features)
    assert not check["passed"]
    assert all(x["signed_relative_error_pct"] < 0 for x in check["rows"])


def test_sealed_files_are_not_overwritten(tmp_path):
    target = tmp_path / "model.json"
    p.write_new(target, {"status": "sealed"})
    with pytest.raises(FileExistsError):
        p.write_new(target, {"status": "changed"})


def test_frozen_adapter_does_not_expand_original_domain():
    from c3_residency_tradeoff_local import make_probe
    from vta.top.vta_node24_partial import Node24WeightSlots
    with pytest.raises(ValueError):
        Node24WeightSlots(8, True)
    assert callable(make_probe(8))
    with pytest.raises(ValueError):
        make_probe(9)
