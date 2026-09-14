"""Unit tests for the frozen ResNet18 full-graph selection/execution helpers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import freeze_vta_c3_resnet18_fullgraph_selections as selections
import freeze_vta_c3_resnet18_fullgraph_fallback as fallback
import run_vta_c3_resnet18_fullgraph_programs as board
import run_vta_c3_resnet18_from_scratch_ours_fullgraph as ours
import run_vta_from_scratch_autotvm_fullgraph as xgb


def test_graph_correctness_fallback_requires_six_stable_equal_observations():
    diagnostic = {
        "programs": {
            "safe": {
                "observations": 6,
                "stock_equal_observations": 6,
                "mismatch_counts": [0] * 6,
                "stable_hash_by_seed": True,
            },
            "unsafe": {
                "observations": 6,
                "stock_equal_observations": 0,
                "mismatch_counts": [1000] * 6,
                "stable_hash_by_seed": False,
            },
        }
    }
    assert fallback.program_is_graph_correct(diagnostic, "safe")
    assert not fallback.program_is_graph_correct(diagnostic, "unsafe")
    assert fallback.program_is_graph_invalid(diagnostic, "unsafe")
    assert not fallback.program_is_graph_invalid(diagnostic, "safe")


def test_fallback_route_signature_binds_workload_and_candidate_order():
    routes = [
        {"workload_id": "R18-H1", "candidate_id": "a"},
        {"workload_id": "R18-H2", "candidate_id": "b"},
        {"workload_id": "R18-H3", "candidate_id": "c"},
    ]
    assert fallback.route_signature(routes) == fallback.route_signature(list(routes))
    changed = [dict(row) for row in routes]
    changed[-1]["candidate_id"] = "fallback-original"
    assert fallback.route_signature(routes) != fallback.route_signature(changed)


def test_parent_aware_rpc_state_ignores_per_connection_child():
    control = object.__new__(ours.ParentAwareCleanStartBoard)
    control.rpc_processes = lambda: [
        {"PID": "102", "PPID": "101", "CWD": "/runtime"},
        {"PID": "101", "PPID": "1", "CWD": "/runtime"},
    ]
    assert control.rpc_state() == {"PID": "101", "CWD": "/runtime"}


def test_parent_aware_rpc_waits_until_connection_child_exits(monkeypatch):
    control = object.__new__(ours.ParentAwareCleanStartBoard)
    observations = iter(
        [
            [
                {"PID": "102", "PPID": "101", "CWD": "/runtime"},
                {"PID": "101", "PPID": "1", "CWD": "/runtime"},
            ],
            [{"PID": "101", "PPID": "1", "CWD": "/runtime"}],
        ]
    )
    control.rpc_processes = lambda: next(observations)
    monkeypatch.setattr(ours.time, "sleep", lambda _: None)
    assert control.wait_idle_rpc("/runtime", timeout_seconds=1) == {
        "PID": "101",
        "CWD": "/runtime",
    }


def test_selected_ids_requires_one_terminal_candidate_per_workload():
    rows = [
        {
            "workload_id": workload_id,
            "policy": "ours_dma_multifidelity",
            "seed": 57001,
            "requested_budget": 50,
            "target": {"final_best_candidate_id": workload_id + "-best"},
        }
        for workload_id in selections.WORKLOADS
    ]
    assert selections.selected_ids(
        rows, "ours_dma_multifidelity", 57001, 50
    ) == {workload_id: workload_id + "-best" for workload_id in selections.WORKLOADS}


def test_selected_ids_rejects_missing_valid_terminal_candidate():
    rows = [
        {
            "workload_id": workload_id,
            "policy": "cheng_minimum_access",
            "seed": 57001,
            "requested_budget": 50,
            "target": {"final_best_candidate_id": None},
        }
        for workload_id in selections.WORKLOADS
    ]
    with pytest.raises(RuntimeError, match="did not find a valid candidate"):
        selections.selected_ids(rows, "cheng_minimum_access", 57001, 50)


def test_balanced_orders_cover_every_program_once_per_round():
    ids = ["stock", "xgb", "cheng", "ml2", "ours"]
    orders = board.balanced_orders(ids, 7)
    assert len(orders) == 7
    assert all(len(order) == len(ids) and set(order) == set(ids) for order in orders)
    assert len({tuple(order) for order in orders}) == 7


def test_selected_fullgraph_failure_keeps_common_nonspliceable_artifacts(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    stock = build / "stock_reference"
    stock.mkdir(parents=True)
    for name in ("graph.json", "params.bin", "graphlib.so"):
        (stock / name).write_bytes(b"placeholder")
    (build / "summary.json").write_text(
        json.dumps(
            {
                "status": "selected_fullgraphs_built_once_before_board",
                "programs": [],
                "policy_to_program": {},
            }
        )
    )
    (build / "artifact_hashes.json").write_text("{}\n")
    monkeypatch.setattr(board, "verify_artifacts_compatible", lambda *_: None)
    monkeypatch.setattr(
        board,
        "CleanStartBoard",
        lambda *_: (_ for _ in ()).throw(RuntimeError("preflight failed")),
    )
    output = tmp_path / "board-output"
    args = SimpleNamespace(
        build_dir=str(build),
        output_dir=str(output),
        host="board",
        ssh_known_hosts="known-hosts",
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        board.run(args)

    expected = {
        "contract.json",
        "timeline.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_hashes.json",
        "invalid_session.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert board.read_json(output / "summary.json")["session_spliced"] is False
    assert board.read_json(output / "summary.json")["status"] == (
        "invalid_entire_session_fail_closed"
    )


def test_ours_clean_start_failure_keeps_common_nonspliceable_artifacts(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "contract.json").write_text(
        json.dumps(
            {
                "status": "frozen_before_resnet18_fullgraph_build_fpga_or_latency"
            }
        )
    )
    (source / "artifact_hashes.json").write_text("{}\n")
    monkeypatch.setattr(ours, "verify_artifacts_compatible", lambda *_: None)
    monkeypatch.setattr(
        ours,
        "clean_start",
        lambda *_: (_ for _ in ()).throw(RuntimeError("clean start failed")),
    )
    output = tmp_path / "ours-output"
    args = SimpleNamespace(
        source_contract=str(source),
        output_dir=str(output),
        candidate_budget=13,
    )
    with pytest.raises(RuntimeError, match="clean start failed"):
        ours.run(args)

    expected = {
        "contract.json",
        "timeline.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_hashes.json",
        "invalid_session.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert ours.read_json(output / "summary.json")["session_spliced"] is False


def test_xgb_cli_refuses_existing_output_without_mutating_it(tmp_path, monkeypatch):
    output = tmp_path / "existing-output"
    output.mkdir()
    sentinel = output / "user-evidence.txt"
    sentinel.write_text("immutable\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vta_from_scratch_autotvm_fullgraph.py",
            "--target-contract",
            "unused",
            "--output-dir",
            str(output),
            "--ssh-known-hosts",
            "unused",
        ],
    )

    with pytest.raises(FileExistsError):
        xgb.main()

    assert sentinel.read_text() == "immutable\n"
    assert not (output / "invalid_session.json").exists()


def test_xgb_cli_failure_commits_common_nonspliceable_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "new-output"

    def fail_after_contract(args):
        root = type(output)(args.output_dir)
        root.mkdir()
        (root / "contract.json").write_text("{}\n")
        (root / "timeline.jsonl").write_text("", encoding="utf-8")
        (root / "results.jsonl").write_text("", encoding="utf-8")
        raise RuntimeError("injected RPC failure")

    monkeypatch.setattr(xgb, "run", fail_after_contract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vta_from_scratch_autotvm_fullgraph.py",
            "--target-contract",
            "unused",
            "--output-dir",
            str(output),
            "--ssh-known-hosts",
            "unused",
        ],
    )

    with pytest.raises(RuntimeError, match="injected RPC failure"):
        xgb.main()

    expected = {
        "contract.json",
        "timeline.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_hashes.json",
        "invalid_session.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "invalid_entire_session_fail_closed"
    assert summary["session_spliced"] is False
