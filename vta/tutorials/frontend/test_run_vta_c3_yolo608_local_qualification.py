import json
from pathlib import Path

import pytest

import run_vta_c3_yolo608_local_qualification as subject


HERE = Path(__file__).resolve().parent
FROZEN = HERE / (
    "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/"
    "20260924_c3_yolo608_candidate_pools_run01"
)


def test_frozen_yolo608_pool_is_label_blind_and_complete():
    hashes, pools = subject.verify_input(FROZEN)
    assert "contract.json" in hashes
    assert len(pools) == 11
    candidates = [row for _, pool in pools for row in pool["candidates"]]
    assert len(candidates) == 88
    assert len({row["candidate_id"] for row in candidates}) == 88


def test_workload_allowlist_is_exact_ordered_and_fail_closed():
    pools = [
        ("a.json", {"workload_id": "a"}),
        ("b.json", {"workload_id": "b"}),
    ]
    assert [row[1]["workload_id"] for row in subject.select_pools(pools, ["b", "a"])] == [
        "b", "a",
    ]
    with pytest.raises(ValueError, match="duplicates"):
        subject.select_pools(pools, ["a", "a"])
    with pytest.raises(ValueError, match="unknown workload"):
        subject.select_pools(pools, ["missing"])


def test_arena_certificate_uses_hardware_byte_capacities():
    fingerprint = {"inp_buff_size": 32, "wgt_buff_size": 64, "acc_buff_size": 128}
    good = subject.arena_certificate(
        {"scope_arena_span_sum_upper_bound_bytes": {
            "local.inp_buffer": 16, "local.wgt_buffer": 64, "local.acc_buffer": 96,
        }},
        fingerprint,
    )
    assert good["status"] == "passed"
    assert good["domains"]["local.wgt_buffer"]["headroom_bytes"] == 0
    bad = subject.arena_certificate(
        {"scope_arena_span_sum_upper_bound_bytes": {"local.inp_buffer": 33}},
        fingerprint,
    )
    assert bad["status"] == "failed"


def test_distinct_programs_requires_both_tir_and_command_identity():
    candidates = [
        {"candidate_id": name, "candidate_identity_fields": {"name": name}}
        for name in ("a", "b", "c")
    ]
    static = [
        {"candidate_id": "a", "lowered_tir_sha256": "tir0"},
        {"candidate_id": "b", "lowered_tir_sha256": "tir0"},
        {"candidate_id": "c", "lowered_tir_sha256": "tir1"},
    ]
    fsim = [
        {"candidate_id": name, "workload_id": "w", "status": "passed",
         "command_features": {"values": {"same": True}}}
        for name in ("a", "b", "c")
    ]
    programs = subject.distinct_programs(candidates, static, fsim)
    assert len(programs) == 2
    assert sorted(len(row["equivalent_candidate_ids"]) for row in programs) == [1, 2]


def test_formal_output_is_immutable_and_latency_blind_when_present():
    output = subject.DEFAULT_OUTPUT
    if not output.exists():
        return
    summary = json.loads((output / "summary.json").read_text())
    assert summary["board_contacted"] is False
    assert summary["latency_values_read"] is False
    assert summary["reference_eligible_workload_count"] >= 2
    ledger = json.loads((output / "artifact_hashes.json").read_text())
    for relative, expected in ledger["artifacts"].items():
        assert subject.sha256_file(output / relative) == expected
