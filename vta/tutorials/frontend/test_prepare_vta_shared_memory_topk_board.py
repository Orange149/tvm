import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_vta_shared_memory_topk_board as prepare  # noqa: E402
import run_vta_shared_memory_topk_board_boot as board  # noqa: E402


def test_candidate_threads_and_balanced_orders():
    candidate = (
        "cpu-00-00_t2__vta-01-12_t1__cpu-13-14_t3__"
        "vta-15-17_t1__cpu-18-20_t4"
    )
    assert prepare.candidate_threads(candidate) == [2, 1, 3, 1, 4]
    base = ["c{}".format(index) for index in range(9)]
    orders = prepare.rotated_orders(base)
    assert len({tuple(order) for order in orders}) == 3
    assert all(set(order) == set(base) for order in orders)


def test_interleaved_union_is_configuration_diverse():
    contract = {
        "strategies": {
            "S0": ["a::copy", "b::copy", "c::copy"],
            "S1": ["a::copy", "c::copy", "b::copy"],
            "S2": ["a::shared", "c::shared", "b::shared"],
        },
        "union_configuration_ids": [
            "a::copy", "b::copy", "c::copy",
            "a::shared", "c::shared", "b::shared",
        ],
    }
    order = prepare.interleaved_union(contract)
    assert order[:3] == ["a::copy", "a::shared", "b::copy"]
    assert set(order) == set(contract["union_configuration_ids"])


def test_runner_metrics_cover_arbitrary_stage_count():
    rows = []
    for index in range(6):
        row = {"completion_index": index, "completion_ms": index * 7.5}
        for stage in range(5):
            row["stage{}_get_ms".format(stage)] = stage + 0.1
            row["stage{}_set_ms".format(stage)] = stage + 0.2
        rows.append(row)
    assert board.completion_window_ii(rows, 2) == 7.5
    # Four edges: (0.1+1.2) + (1.1+2.2) + (2.1+3.2) + (3.1+4.2)
    assert abs(board.boundary_api_ms(rows, 5) - 17.2) < 1.0e-12


def test_snapshot_high_water_requires_steady_state():
    assert board.snapshot_high_water([
        {"phase": "after_steady_state", "high_water_bytes": 1234}
    ]) == 1234


def test_deployment_manifest_is_frozen_and_complete():
    path = (
        Path(__file__).resolve().parent
        / "report_out/stage_tile_cotuning/compile_context_audit/"
        "c1_shared_memory_topk_board_prepare_run8/deployment_manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "board_ready_frozen_before_new_performance_labels"
    assert len(payload["packages"]) == 5
    assert len(payload["configurations"]) == 9
    assert len(payload["shared_plans"]) == 4
    assert sum(row["requires_upload"] for row in payload["packages"].values()) == 1
    expected = {row["configuration_id"] for row in payload["configurations"]}
    assert all(set(order) == expected for order in payload["boot_configuration_orders"])
