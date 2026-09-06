import json
from pathlib import Path

import run_cpu_vta_pipeline_v1_p8c as p8c


PACKAGE = (
    Path(__file__).parent
    / "report_out/resource_aware_maxplus/v1_p7_top20_board_20260904/packages/"
    "01_cpu-00-02_t4__vta-03-17_t1__cpu-18-20_t1/package"
)


def _manifest():
    return json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))


def _row(mode, completion, frame_id):
    row = {
        "completion_index": frame_id,
        "completion_ms": completion,
        "input_index": frame_id % 2,
        "raw_outputs": [{"fnv1a64": "a" if frame_id % 2 == 0 else "b"}],
        "total_latency_ms": 10.0 + frame_id,
        "stage0_ms": 2.0,
        "stage0_get_ms": 0.1,
        "stage1_ms": 3.0,
        "stage1_set_ms": 0.1,
        "stage1_get_ms": 0.1,
        "stage2_ms": 4.0,
        "stage2_set_ms": 0.1,
    }
    if mode != "B0":
        row.update(
            {
                "p8_framework_materialization_bytes": 0,
                "p8_total_slot_wait_ms": 0.1,
                "p8_boundaries": [
                    {
                        "edge_index": edge,
                        "slot_id": frame_id % (1 if mode == "B1" else 2),
                        "generation": frame_id + 1,
                        "physical_address": "0x{:x}".format(
                            0x1000
                            + edge * 0x1000
                            + (frame_id % (1 if mode == "B1" else 2)) * 0x200
                        ),
                        "boundary_bytes": 256,
                        "producer_wait_ms": 0.05,
                        "consumer_wait_ms": 0.05,
                    }
                    for edge in (0, 1)
                ],
            }
        )
    return row


def _profile():
    return {
        "mem_copy_from_host_bytes": 100,
        "mem_copy_to_host_bytes": 50,
        "load_buffer_2d_bytes": 1000,
        "store_buffer_2d_bytes": 200,
        "driver_submit_mmio_us": 10,
        "driver_poll_wait_us": 20,
        "synchronize_calls": 2,
    }


def test_protocol_freezes_balanced_mode_order_and_two_edges():
    protocol = p8c.build_protocol(_manifest(), warmup=5, runs_per_block=10)
    assert protocol["measurement"]["block_order"] == ["B0", "B1", "B2", "B2", "B1", "B0"]
    assert [edge["direction"] for edge in protocol["edges"]] == [
        "cpu_to_vta",
        "vta_to_cpu",
    ]
    assert protocol["candidate_throughput_used_for_fit"] is False


def test_runner_args_separate_b0_b1_b2():
    manifest = _manifest()
    b0 = p8c._runner_args(manifest, "B0", 5, 10, "b0.jsonl", "p0")
    b1 = p8c._runner_args(manifest, "B1", 5, 10, "b1.jsonl", "p1")
    b2 = p8c._runner_args(manifest, "B2", 5, 10, "b2.jsonl", "p2")
    assert "--p8-managed-slots" not in b0 and "--serial" not in b0
    assert b1[b1.index("--p8-managed-slots") + 1] == "1" and "--serial" in b1
    assert b2[b2.index("--p8-managed-slots") + 1] == "2" and "--serial" not in b2
    assert all(args[args.index("--warmup-runs") + 1] == "5" for args in (b0, b1, b2))


def test_summarize_uses_completion_intervals_and_checks_slots():
    blocks = []
    for occurrence in range(2):
        rows = [_row("B2", occurrence * 100 + index * 5.0, index) for index in range(4)]
        blocks.append({"rows": rows, "profile": _profile()})
    summary = p8c.summarize_mode("B2", blocks, [802816, 100352])
    assert summary["pipeline_ii_ms_median"] == 5.0
    assert summary["pipeline_fps_from_median_ii"] == 200.0
    assert summary["framework_materialization_bytes_per_frame"] == 0
    assert summary["boundary_api_service_ms_median"]["total"] == 0.4
    assert summary["slot_ids_by_edge"] == {"0": [0, 1], "1": [0, 1]}
    assert summary["slot_tokens_unique_within_block"] is True
    assert summary["slot_addresses_aligned"] is True
    assert summary["slot_physical_ranges_non_overlapping"] is True


def test_summarize_uses_complete_window_for_bursty_completions():
    # The interval median is 1 ms, but nine completed frames span 306 ms.
    completion = [0.0, 1.0, 2.0, 102.0, 103.0, 203.0, 204.0, 205.0, 306.0]
    blocks = [
        {
            "rows": [_row("B0", value, index) for index, value in enumerate(completion)],
            "profile": _profile(),
        }
    ]
    summary = p8c.summarize_mode("B0", blocks, [802816, 100352])
    assert summary["pipeline_interval_ms_median"] == 1.0
    assert summary["pipeline_ii_ms_median"] == 38.25
    assert summary["pipeline_fps"] == 1000.0 / 38.25
    assert summary["throughput_estimator"] == "median_of_block_completion_window_ii_v1"


def test_b0_reports_both_boundary_get_set_materialization():
    blocks = [
        {"rows": [_row("B0", index * 4.0, index) for index in range(4)], "profile": _profile()}
    ]
    summary = p8c.summarize_mode("B0", blocks, [802816, 100352])
    assert summary["framework_materialization_bytes_per_frame"] == 1806336


def _session(boot_id, b0_ii, b2_ii, gate=True):
    return {
        "kind": "cpu_vta_pipeline_v1_p8c_session",
        "candidate_id": "candidate",
        "boot_id": boot_id,
        "mode_summaries": {
            "B0": {
                "pipeline_ii_ms_median": b0_ii,
                "pipeline_fps_from_median_ii": 1000.0 / b0_ii,
                "boundary_api_service_ms_median": {
                    "cpu_to_vta": 0.8,
                    "vta_to_cpu": 0.7,
                    "total": 1.5,
                },
            },
            "B2": {
                "pipeline_ii_ms_median": b2_ii,
                "pipeline_fps_from_median_ii": 1000.0 / b2_ii,
                "boundary_api_service_ms_median": {
                    "cpu_to_vta": 0.02,
                    "vta_to_cpu": 0.02,
                    "total": 0.04,
                },
                "slot_wait_by_edge_ms": {
                    "1": {"producer_median": 1.0, "producer_p95": 2.0}
                },
            },
        },
        "single_boot_gate": {"passed": gate},
    }


def test_cross_boot_summary_does_not_treat_frames_as_independent_boots():
    summary = p8c.build_cross_boot_summary(
        [_session("boot-a", 100.0, 96.0), _session("boot-b", 98.0, 95.0)]
    )
    assert summary["independent_boot_count"] == 2
    assert summary["aggregate"]["delta_ii_ms_mean"] == 3.5
    assert summary["aggregate"]["boundary_api_service_reduction_percent_mean"] > 97.0
    assert summary["aggregate"]["delta_ii_ms_mean_ci95"] is None
    assert summary["gates"]["formal_performance_claim_allowed"] is False


def test_cross_boot_claim_requires_three_positive_paired_boot_effects():
    summary = p8c.build_cross_boot_summary(
        [
            _session("boot-a", 100.0, 96.0),
            _session("boot-b", 98.0, 95.0),
            _session("boot-c", 99.0, 97.0),
        ]
    )
    assert summary["aggregate"]["delta_ii_ms_mean"] == 3.0
    assert summary["aggregate"]["delta_ii_ms_mean_ci95"]["lower"] > 0.0
    assert summary["gates"]["formal_performance_claim_allowed"] is True


def test_cross_boot_failed_functional_gate_blocks_claim():
    summary = p8c.build_cross_boot_summary(
        [
            _session("boot-a", 100.0, 96.0),
            _session("boot-b", 98.0, 95.0, gate=False),
            _session("boot-c", 99.0, 97.0),
        ]
    )
    assert summary["gates"]["all_boot_functional_gates_passed"] is False
    assert summary["gates"]["formal_performance_claim_allowed"] is False
