"""Unit tests for P4h local FSim command-signature collection."""

import json

from collect_vta_fsim_command_signatures import (
    STATUS,
    command_signature,
    parse_queue_records,
    select_representatives,
    summarize,
)


def queue_line(submit, reason, insn, uop, load, store, queue="0xabc"):
    record = {
        "submit": submit,
        "queue_id": queue,
        "reason": reason,
        "insn_capacity": 67108864,
        "uop_capacity": 67108864,
        "submit_threshold": 67108864,
        "submit_threshold_configured": False,
        "insn_bytes": insn,
        "uop_bytes": uop,
        "insn_peak": insn,
        "uop_peak": uop,
        "load_bytes": load,
        "store_bytes": store,
        "finalize_us": 1.0,
        "device_run_us": 99.0,
        "timeout": 0,
    }
    return "noise [VTA_QUEUE] " + json.dumps(record)


def test_parser_keeps_all_valid_records_and_freezes_malformed():
    stderr = "\n".join(
        [queue_line(1, "explicit_sync", 64, 16, 32, 0), "[VTA_QUEUE] {bad", queue_line(2, "explicit_sync", 48, 0, 0, 16)]
    )
    records, malformed = parse_queue_records(stderr)
    assert [row["submit"] for row in records] == [1, 2]
    assert len(malformed) == 1


def test_signature_excludes_address_and_timing_and_checks_mode4_drains():
    text = "\n".join(
        [
            queue_line(1, "explicit_sync", 64, 16, 100, 0, "0x111"),
            queue_line(2, "explicit_sync", 96, 32, 200, 0, "0x111"),
            queue_line(3, "explicit_sync", 48, 8, 0, 40, "0x111"),
        ]
    )
    records, malformed = parse_queue_records(text)
    assert not malformed
    result = command_signature(records, implementation_mode=4, expected_residency_drains=2)
    structural = result["structural"]
    assert structural["drain_check"]["matches_p3e"] is True
    assert structural["finish"]["source_derived_count"] == 3
    assert structural["replay"]["observed_in_queue_json"] is False
    encoded = json.dumps(structural)
    assert "0x111" not in encoded
    assert "device_run_us" not in encoded


def test_signature_hash_stable_across_queue_address_and_timing():
    one, _ = parse_queue_records(queue_line(1, "explicit_sync", 64, 16, 100, 20, "0x1"))
    two_line = queue_line(1, "explicit_sync", 64, 16, 100, 20, "0x999").replace(
        '"device_run_us": 99.0', '"device_run_us": 12345.0'
    )
    two, _ = parse_queue_records(two_line)
    assert command_signature(one, 0, 0)["sha256"] == command_signature(two, 0, 0)["sha256"]


def test_selection_requires_exact_p3e_tir_and_source_link():
    def candidate(identifier, workload, mode, index, tir):
        return {
            "candidate_id": identifier,
            "workload_id": workload,
            "residence_mode": mode,
            "implementation_mode": 4 if mode == "weight_stationary_barrier" else 0,
            "status": "ok",
            "tir_sha256": tir,
            "identity": {"workload": ["task"], "complete_config_entity": {"entity": []}},
            "debug": {"config_index": index},
        }

    p4b, p4g, p3e = [], [], []
    for number, workload in enumerate(("W00", "W02", "W09")):
        index = number + 10
        original = candidate("o" + workload, workload, "original", index, "ot" + workload)
        barrier = candidate(
            "b" + workload, workload, "weight_stationary_barrier", index, "bt" + workload
        )
        barrier["source_p4b_original_candidate_id"] = original["candidate_id"]
        p4b.append(original)
        p4g.append(barrier)
        p3e.append(
            {
                "workload_id": workload,
                "config_index": index,
                "original": {"tir_sha256": original["tir_sha256"], "input_bytes": 1, "weight_bytes": 2, "store_calls": 3},
                "mode4": {"tir_sha256": barrier["tir_sha256"], "input_bytes": 4, "weight_bytes": 5, "store_calls": 6, "sync_static_execution_multiplicity": [number + 1, 1]},
            }
        )
    selected = select_representatives(p4b, p4g, p3e)
    assert len(selected) == 6
    assert [row["implementation_mode"] for row in selected] == [0, 4, 0, 4, 0, 4]


def test_summary_never_upgrades_status_or_claim_boundary():
    rows = [
        {
            "status": STATUS,
            "outcome": "passed_local_signature",
            "correctness": "passed",
            "diagnostic_record_count": 2,
            "implementation_mode": 4,
            "command_signature": {"structural": {"drain_check": {"matches_p3e": True}}},
        }
    ]
    result = summarize(rows)
    assert result["status"] == "fsim_dry_run"
    assert result["claim_boundary"]["board_qualified_capacity"] is False
    assert result["claim_boundary"]["performance_measurement"] == "not_collected"
