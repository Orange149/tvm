import pytest

from build_vta_hardware_certificate_ledger import (
    HARDWARE_CERTIFICATE_LEDGER_SCHEMA,
    certificate_identity,
    hardware_fingerprint,
)
from plan_vta_hardware_certified_dispatch import dispatch_plan, route_candidate


def contract():
    return {
        "workload_id": "E00",
        "workload": ["conv", 1],
        "board": {"boot_id": "boot", "fpga_state": "operating", "udmabuf_bytes": 4096},
        "source_guards_sha256": {"schedule": "hash"},
    }


def candidate(total_after=120):
    return {
        "candidate_id": "candidate",
        "workload_id": "E00",
        "residence_mode": "input_stationary",
        "config_index": 7,
        "tir_sha256": "tir",
        "identity": {
            "complete_config_entity": {
                "code_hash": None,
                "entity": [["tile_h", "sp", [-1, 1]]],
            }
        },
        "locally_eligible": True,
        "input_dma_reduction_fraction": 0.5,
        "original": {"dma": {"load_buffer_2d_bytes": 180, "store_buffer_2d_bytes": 20,
                               "load_buffer_2d_calls": 18, "store_buffer_2d_calls": 2}},
        "input_stationary": {"dma": {"load_buffer_2d_bytes": total_after - 20,
                                       "store_buffer_2d_bytes": 20,
                                       "load_buffer_2d_calls": 8, "store_buffer_2d_calls": 2}},
    }


def certificate_key(item):
    probe = {"mode": item["residence_mode"], "config_index": item["config_index"],
             "complete_config_entity": item["identity"]["complete_config_entity"],
             "tir_sha256": item["tir_sha256"]}
    return certificate_identity(contract(), probe, hardware_fingerprint(contract())["key"])[0]


def test_unknown_candidate_is_correctness_only_and_failed_is_rejected():
    item = candidate()
    assert route_candidate(item, contract(), {})["route"] == "correctness_canary_only"
    failed = {certificate_key(item): {"status": "failed"}}
    assert route_candidate(item, contract(), failed)["route"] == "reject_hardware"


def test_non_pareto_candidate_abstains_before_hardware_lookup():
    assert route_candidate(candidate(total_after=220), contract(), {})["route"] == "abstain_keep_original"


def test_static_rejection_does_not_require_dma_records():
    item = candidate()
    item.update(locally_eligible=False, reject_reason="analytic_acc_capacity")
    item["original"] = {"dma": None}
    item["input_stationary"] = {"dma": None}
    routed = route_candidate(item, contract(), {})
    assert routed["route"] == "reject_static"
    assert routed["reason"] == "analytic_acc_capacity"
    assert routed["dma_pareto"] == {"certified": False, "reductions": None}


def test_dma_pareto_uses_the_selected_residency_mode():
    item = candidate()
    item["residence_mode"] = "paper_inspired_hybrid"
    item["paper_inspired_hybrid"] = item.pop("input_stationary")
    assert route_candidate(item, contract(), {})["route"] == "correctness_canary_only"


def test_incumbent_is_first():
    ledger = {"schema": HARDWARE_CERTIFICATE_LEDGER_SCHEMA, "certificates": []}
    records, counts = dispatch_plan([candidate()], contract(), ledger)
    assert records[0]["route"] == "protected_incumbent"
    assert counts == {"protected_incumbent": 1, "correctness_canary_only": 1}


def test_dispatch_lookup_is_stable_when_config_index_changes():
    certified = candidate()
    passed = {"certificate_key": certificate_key(certified), "status": "passed"}
    renumbered = candidate()
    renumbered["config_index"] = 98123
    renumbered["identity"]["complete_config_entity"]["index"] = 98123
    assert route_candidate(
        renumbered, contract(), {passed["certificate_key"]: passed}
    )["route"] == "allow_timing"


def test_dispatch_entity_change_does_not_reuse_certificate():
    certified = candidate()
    passed = {"certificate_key": certificate_key(certified), "status": "passed"}
    changed = candidate()
    changed["identity"]["complete_config_entity"]["entity"] = [
        ["tile_h", "sp", [-1, 2]]
    ]
    assert route_candidate(
        changed, contract(), {passed["certificate_key"]: passed}
    )["route"] == "correctness_canary_only"


@pytest.mark.parametrize("schema", [None, "c3_vta_hardware_certificate_ledger_v1"])
def test_dispatch_rejects_unversioned_and_v1_ledgers(schema):
    ledger = {"certificates": []}
    if schema is not None:
        ledger["schema"] = schema
    with pytest.raises(ValueError, match="refusing legacy or unversioned ledger"):
        dispatch_plan([candidate()], contract(), ledger)
