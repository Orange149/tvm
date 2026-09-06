import json
from pathlib import Path

import run_cpu_vta_pipeline_v1_p8_latency_scaling as latency


PACKAGE = latency.DEFAULT_PACKAGE


def manifest():
    return json.loads((Path(PACKAGE) / "manifest.json").read_text(encoding="utf-8"))


def row(mode, frame_id, latency_ms):
    stages = len(manifest()["stages"])
    result = {
        "input_index": frame_id % 2,
        "raw_outputs": [{"fnv1a64": "a" if frame_id % 2 == 0 else "b"}],
        "total_latency_ms": latency_ms,
    }
    for index in range(stages):
        result["stage{}_ms".format(index)] = 10.0 + index
        result["stage{}_set_ms".format(index)] = 0.1
        result["stage{}_get_ms".format(index)] = 0.2
    if mode == "L1":
        result.update(
            {
                "p8_framework_materialization_bytes": 0,
                "p8_total_slot_wait_ms": 0.01,
                "p8_boundaries": [
                    {
                        "edge_index": edge,
                        "slot_id": 0,
                        "physical_address": "0x{:x}".format(0x1000 + edge * 0x100000),
                        "boundary_bytes": 256,
                        "physical_addresses": ["0x{:x}".format(0x1000 + edge * 0x100000)],
                        "tensor_bytes": [256],
                    }
                    for edge in range(stages - 1)
                ],
            }
        )
    return result


def test_protocol_is_serial_and_has_six_heterogeneous_edges():
    payload = latency.build_protocol(manifest(), warmup=5, runs_per_block=10)
    assert payload["vta_island_count"] == 3
    assert len(payload["heterogeneous_edges"]) == 6
    assert payload["measurement"]["same_serial_schedule_in_both_modes"] is True
    assert payload["ownership"]["dual_slot_overlap_not_measured"] is True
    assert payload["ordinary_copy_policy"] == "runner_default"


def test_copy_policy_environment_preserves_runner_default():
    assert latency._copy_policy_environment("runner_default") == ""
    assert latency._copy_policy_environment("safe_byte_copy") == (
        "AXU5EVB_DRIVER_SAFE_COPY=1 "
    )
    assert latency._copy_policy_environment("memcpy") == "AXU5EVB_DRIVER_SAFE_COPY=0 "


def test_only_whitelisted_legacy_session_is_classified_as_memcpy():
    artifact = next(iter(latency.LEGACY_MEMCPY_SESSION_ARTIFACTS))
    assert latency._session_copy_policy({"artifact_sha256": artifact}) == "memcpy"
    assert latency._session_copy_policy({"artifact_sha256": "unknown"}) is None


def test_runner_args_only_l1_enables_managed_single_slot():
    l0 = latency.runner_args(manifest(), "L0", 5, 10, "l0.jsonl", "p0")
    l1 = latency.runner_args(manifest(), "L1", 5, 10, "l1.jsonl", "p1")
    assert "--serial" in l0 and "--serial" in l1
    assert "--p8-managed-slots" not in l0
    assert l1[l1.index("--p8-managed-slots") + 1] == "1"


def test_summary_supports_arbitrary_alternating_stage_count():
    blocks = [
        {"rows": [row("L1", index, 100.0 + index) for index in range(4)], "profile": {}},
        {"rows": [row("L1", index, 102.0 + index) for index in range(4)], "profile": {}},
    ]
    summary = latency.summarize_mode("L1", blocks, manifest())
    assert summary["latency_ms_median"] == 102.5
    assert len(summary["boundary_api_service_by_edge"]) == 6
    assert summary["every_frame_has_all_edges"] is True
    assert summary["slot_ids_by_edge"] == {str(edge): [0] for edge in range(6)}
    assert summary["slot_physical_ranges_non_overlapping"] is True


def test_cross_boot_does_not_make_formal_claim_from_one_boot():
    session = {
        "boot_id": "boot-a",
        "mode_summaries": {"L0": {"latency_ms_median": 100.0}, "L1": {"latency_ms_median": 90.0}},
        "effects": {"latency_reduction_ms": 10.0, "latency_reduction_percent": 10.0},
        "single_boot_gate": {"passed": True},
    }
    summary = latency.build_cross_boot_summary([session])
    assert summary["independent_boot_count"] == 1
    assert summary["formal_latency_claim_allowed"] is False
