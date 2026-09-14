"""Pure contract tests for P7R120 preparation."""

from prepare_vta_p7r120_yolo_rpc_contract import MODES, make_timing_orders, select_six


def _rows():
    candidates, static, fsim = [], [], []
    for family in ("Y02B00", "Y02B01"):
        for mode, number in MODES.items():
            identity = {
                "schema": "c3_candidate_id_v2",
                "implementation_mode": number,
                "complete_config_entity": {"entity": []},
            }
            from c3_candidate_identity import canonical_json_bytes
            import hashlib

            candidate_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
            # Test identities need uniqueness without changing their semantics.
            identity["test_family"] = family
            identity["test_mode"] = mode
            candidate_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "identity": identity,
                    "family_id": family,
                    "public_mode": mode,
                }
            )
            static.append({"candidate_id": candidate_id, "status": "ok"})
            fsim.append(
                {
                    "candidate_id": candidate_id,
                    "status": "passed",
                    "seeds": [{"correct": True}] * 3,
                }
            )
    return candidates, static, fsim


def test_selects_exactly_two_complete_three_mode_families():
    candidates, static, fsim = _rows()
    selected = select_six(candidates, static, fsim)
    assert len(selected) == 6
    assert {(row["family_id"], row["public_mode"]) for row in selected} == {
        (family, mode)
        for family in ("Y02B00", "Y02B01")
        for mode in MODES
    }


def test_local_failure_closes_contract():
    candidates, static, fsim = _rows()
    fsim[0]["status"] = "failed"
    import pytest

    with pytest.raises(ValueError, match="FSim"):
        select_six(candidates, static, fsim)


def test_timing_orders_charge_every_candidate_once_per_round_and_rotate():
    ids = ["c{}".format(index) for index in range(6)]
    orders = make_timing_orders(ids)
    assert len(orders) == 7
    assert all(len(order) == 6 and set(order) == set(ids) for order in orders)
    for position in range(6):
        assert {orders[round_index][position] for round_index in range(6)} == set(ids)
    assert orders[6] == orders[0]
