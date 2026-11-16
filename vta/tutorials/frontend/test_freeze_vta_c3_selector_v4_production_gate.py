import copy

import pytest

import c3_selector_v4_diverse
from freeze_vta_c3_selector_v4_production_gate import build_contract, sha256


def upstream():
    return {
        "status": "FROZEN_FOR_FUTURE_LATENCY_UNSEEN_TASKS",
        "performance_labels_read_by_freeze": False,
        "board_contacted": False,
    }, {
        "selector_source_sha256": sha256(c3_selector_v4_diverse.__file__),
    }


def test_contract_freezes_gate_before_ordering_without_labels():
    contract, manifest = upstream()
    value = build_contract(contract, manifest, "upstream")
    assert value["pipeline"][1] == "completion_policy_and_composability_hard_gate"
    assert value["pipeline"][2] == "label_free_low_movement_tile_diversity_order"
    assert value["performance_labels_read_by_freeze"] is False
    assert value["legacy_completion_policy_action"] == "reject_and_regenerate_as_mode11"


def test_contract_rejects_unfrozen_or_drifted_upstream():
    contract, manifest = upstream()
    bad = copy.deepcopy(contract)
    bad["performance_labels_read_by_freeze"] = True
    with pytest.raises(ValueError, match="not label isolated"):
        build_contract(bad, manifest, "upstream")
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["selector_source_sha256"] = "drift"
    with pytest.raises(ValueError, match="source drift"):
        build_contract(contract, bad_manifest, "upstream")
