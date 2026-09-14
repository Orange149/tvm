from pathlib import Path
from types import SimpleNamespace

import pytest

import run_vta_c3_resnet18_board_pool as board


ROOT = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
POOL = ROOT / "20260914_p7r482_resnet18_board_pool_freeze_run01"
CROSS = ROOT / "20260914_p7r483_resnet18_cross_compile_run01"


def test_complete_pool_rpc_lifetime_covers_long_non_spliced_session():
    assert board.POOL_RPC_SESSION_TIMEOUT_SECONDS >= 3600


def test_clean_start_ssh_prefers_key_and_retains_password_fallback():
    control = board.CleanStartBoard("board", "/tmp/known-hosts")
    options = " ".join(control.options)
    assert "PreferredAuthentications=publickey,password" in options
    assert "PubkeyAuthentication=yes" in options
    assert "StrictHostKeyChecking=yes" in options


def test_frozen_pool_and_all_retained_binaries_are_exactly_bound():
    candidates, certificates, hashes = board.verify_inputs(POOL, CROSS)
    assert len(candidates) == 50
    assert len(certificates) == 50
    assert len(hashes["binaries"]) == 50
    assert all(certificates[row["candidate_id"]]["status"] == "passed" for row in candidates)


def test_oracle_is_rejected_until_every_correct_candidate_has_five_samples():
    rows = [{"candidate_id": "a", "latency_ms": 1.0} for _ in range(5)]
    with pytest.raises(ValueError, match="incomplete"):
        board.summarize_workload_timing(rows, ["a", "b"])


def test_oracle_is_only_constructed_from_complete_pool():
    rows = []
    for candidate_id, latency in (("a", 2.0), ("b", 1.0)):
        rows.extend({"candidate_id": candidate_id, "latency_ms": latency} for _ in range(5))
    summary = board.summarize_workload_timing(rows, ["a", "b"])
    assert summary["pool_oracle_candidate_id"] == "b"
    assert summary["pool_oracle_latency_ms"] == 1.0


def test_runtime_cost_is_explicitly_logical_dma_not_axi():
    rows = [{"runtime_profile_complete": {"load_buffer_2d_bytes": 10, "store_buffer_2d_bytes": 2, "load_buffer_2d_calls": 3, "store_buffer_2d_calls": 1, "driver_run_calls": 2}}]
    assert board.profile_cost(rows) == {"logical_load_bytes": 10, "logical_store_bytes": 2, "logical_dma_calls": 4, "fpga_kernel_invocations": 2}


def test_failed_session_still_commits_common_audit_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "failed_board_session"
    monkeypatch.setattr(
        board,
        "verify_inputs",
        lambda *_: ([{"candidate_id": "candidate-0"}], {}, {"frozen": "hash"}),
    )
    monkeypatch.setattr(board.vta, "get_env", lambda: SimpleNamespace(TARGET="sim"))
    args = SimpleNamespace(
        board_pool_dir="unused-pool",
        cross_compile_dir="unused-cross",
        output_dir=str(output),
    )

    with pytest.raises(RuntimeError, match="requires VTA TARGET"):
        board.run(args)

    expected = {
        "contract.json",
        "timeline.jsonl",
        "results.jsonl",
        "summary.json",
        "artifact_hashes.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    summary = board.read_json(output / "summary.json")
    assert summary["status"] == "invalid_entire_session_fail_closed"
    assert summary["session_spliced"] is False
    events = board.read_jsonl(output / "timeline.jsonl")
    assert [row["event"] for row in events] == ["W0", "session_invalidated"]
    assert board.read_jsonl(output / "results.jsonl") == []
    ledger = board.read_json(output / "artifact_hashes.json")["artifacts"]
    assert expected - {"artifact_hashes.json"} <= set(ledger)
