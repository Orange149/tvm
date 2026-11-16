import copy

import pytest

from freeze_vta_c3_selector_v4_diverse import build_contract
from test_c3_selector_v4_diverse import row


def task_row(task_id, program_id, request_bytes, tile_h, tile_w):
    value = row(program_id, request_bytes, tile_h, tile_w)
    value["execution_task_id"] = task_id
    return value


def test_freeze_contract_separates_incumbent_and_marks_development_exposure():
    rows = [
        task_row("task", "inc", 500, 1, 1),
        task_row("task", "a", 100, 1, 4),
        task_row("task", "b", 100, 8, 4),
    ]
    contract = build_contract(rows, {"task": {"incumbent_program_id": "inc"}}, "manifest")
    assert contract["task_orders"]["task"]["new_candidate_count"] == 2
    assert "inc" not in contract["task_orders"]["task"]["selector_output"]["ordered_candidate_ids"]
    assert contract["development_source_latency_exposed"] is True
    assert contract["development_source_can_be_holdout_again"] is False
    assert contract["performance_labels_read_by_freeze"] is False


def test_freeze_rejects_latency_leakage_and_task_identity_mismatch():
    rows = [task_row("task", "inc", 500, 1, 1), task_row("task", "a", 100, 1, 4)]
    leaked = copy.deepcopy(rows)
    leaked[1]["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="labels are forbidden"):
        build_contract(leaked, {"task": {"incumbent_program_id": "inc"}}, "manifest")
    with pytest.raises(ValueError, match="policy tasks differ"):
        build_contract(rows, {"other": {"incumbent_program_id": "inc"}}, "manifest")

