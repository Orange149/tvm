#!/usr/bin/env python3
"""Local semantic tests for the P7 physical-profile plan."""

import json
import sys

import pytest

import run_cpu_vta_pipeline_v1_p7_profile as p7
from collect_cpu_vta_pipeline_v1_preflight import storage_gate


def _inputs():
    ranking = p7.load_artifact(p7.DEFAULT_RANKING, p7.RANKING_KIND)
    manifest = p7.load_artifact(p7.DEFAULT_MANIFEST, p7.MANIFEST_KIND)
    return ranking, manifest


def test_p7_plan_has_budgeted_matched_matrices():
    ranking, manifest = _inputs()
    plan = p7.build_plan(ranking, manifest)
    experiments = plan["experiments"]

    assert len(experiments["p7b1_cpu_memory_baseline"]) == 36
    assert len(experiments["p7b1_cpu_stage_pairs"]) == 16
    assert len(experiments["p7b2_runtime"]["device_cases"]) == 6
    assert len(experiments["p7b2_runtime"]["boundary_cases"]) == 6
    assert len(experiments["p7c_shared_ddr_contention"]) == 12
    assert plan["measurement_policy"]["memory_target_traffic_bytes_per_sample"] == 128 * 1024 * 1024
    assert plan["measurement_policy"]["memory_wall_time_cv_max"] == 0.10
    assert plan["measurement_policy"]["cpu_stage_isolated_wall_time_cv_max"] == 0.10
    assert plan["measurement_policy"]["cpu_stage_slowdown_ci_relative_halfwidth_max"] == 0.25
    for case in experiments["p7b1_cpu_memory_baseline"]:
        multiplier = 2 if case["operation"] == "copy" else 1
        assert (
            case["operand_bytes_per_stream"] * multiplier
            == case["target_active_working_set_bytes"]
        )
        assert case["worker_lifecycle"] == "persistent_process_threads"
        assert "cache_misses" in case["outputs"]


def test_cpu_pairs_are_traceable_to_frozen_top20_without_labels():
    ranking, manifest = _inputs()
    plan = p7.build_plan(ranking, manifest)

    assert plan["source_artifacts"]["frozen_top20"]["candidate_throughput_consumed"] is False
    for case in plan["experiments"]["p7b1_cpu_stage_pairs"]:
        assert case["source_frozen_candidate"]["candidate_id"]
        assert 1 <= case["source_frozen_candidate"]["rank"] <= 20
        assert case["modes"] == ["a_only", "b_only", "a_b_concurrent"]
        assert case["concurrent_sampling"] == "one_ready_start_barrier_per_scored_sample"
        for stage in (case["stage_a"], case["stage_b"]):
            assert stage["threads"] in p7.THREAD_CHOICES
            assert stage["cpu_affinity"] == list(range(stage["threads"]))

    encoded = json.dumps(plan)
    assert '"predicted_fps"' not in encoded
    assert '"pipeline_fps"' not in encoded
    assert '"measured_fps"' not in encoded


def test_cpu_pair_segments_resolve_to_one_real_multi_stage_package():
    ranking, manifest = _inputs()
    plan = p7.build_plan(ranking, manifest)
    catalog = p7._package_catalog(p7.DEFAULT_PACKAGE_ROOT)
    assert len(catalog) >= 4
    for case in plan["experiments"]["p7b1_cpu_stage_pairs"]:
        resolved = p7._resolve_pair_package(
            case["stage_a"]["segment_id"],
            case["stage_b"]["segment_id"],
            manifest,
            catalog,
        )
        assert resolved["stage_a"]["device"] == "cpu"
        assert resolved["stage_b"]["device"] == "cpu"
        assert resolved["stage_a"]["input_schema"]["arity"] >= 1
        assert resolved["stage_b"]["input_schema"]["arity"] >= 1


def test_bootstrap_slowdown_interval_detects_clear_effect():
    interval = p7._bootstrap_ratio_ci([10.0] * 20, [12.0] * 20, "fixed", samples=200)
    assert interval["lower_95"] == 1.2
    assert interval["upper_95"] == 1.2


def test_boundary_cases_use_three_actual_sizes_per_direction():
    ranking, manifest = _inputs()
    plan = p7.build_plan(ranking, manifest)
    cases = plan["experiments"]["p7b2_runtime"]["boundary_cases"]
    available = {
        direction: {
            int(item["logical_bytes"])
            for item in manifest["boundaries"]
            if item["direction"] == direction
        }
        for direction in ("cpu_to_vta", "vta_to_cpu")
    }
    for direction in available:
        selected = [case for case in cases if case["direction"] == direction]
        assert [case["quantile"] for case in selected] == ["low", "median", "high"]
        assert len({case["logical_bytes"] for case in selected}) == 3
        assert all(case["logical_bytes"] in available[direction] for case in selected)


def test_p7b2_subplan_uses_real_stages_and_identifiable_ownership():
    ranking, manifest = _inputs()
    parent = p7.build_plan(ranking, manifest)
    catalog = p7._combined_package_catalog(
        p7.DEFAULT_PACKAGE_ROOT, p7.DEFAULT_P5A_PACKAGE_ROOT
    )
    plan = p7.build_p7b2_plan(parent, manifest, catalog)

    assert plan["parent_p7_plan_artifact_sha256"] == parent["artifact_sha256"]
    assert len(plan["host_empty_cases"]) == 2
    assert len(plan["vta_runtime_cases"]) == 4
    assert len(plan["boundary_representatives"]) == 3
    assert [
        item["cpu_to_vta_observation"]["logical_bytes"]
        for item in plan["boundary_representatives"]
    ] == [100352, 200704, 802816]
    assert [
        item["vta_to_cpu_observation"]["logical_bytes"]
        for item in plan["boundary_representatives"]
    ] == [100352, 401408, 1605632]
    assert plan["candidate_throughput_used_for_fit"] is False
    assert "alpha_cpu" in plan["identifiability_correction"]["removed_fit"]


def test_boundary_through_origin_fit_uses_mib_units_and_nonnegative_slope():
    fit = p7._fit_boundary_through_origin(
        [
            {
                "direction": "cpu_to_vta",
                "logical_bytes": 1 << 20,
                "service_ms_median": 3.0,
            },
            {
                "direction": "cpu_to_vta",
                "logical_bytes": 2 << 20,
                "service_ms_median": 6.0,
            },
            {
                "direction": "vta_to_cpu",
                "logical_bytes": 1 << 20,
                "service_ms_median": 99.0,
            },
        ],
        "cpu_to_vta",
    )

    assert fit["point_count"] == 2
    assert fit["ms_per_mib"] == pytest.approx(3.0)
    assert fit["fit_mape"] == pytest.approx(0.0)
    assert fit["fit_max_ape"] == pytest.approx(0.0)
    assert fit["formal_parameter_admitted"] is False

    with pytest.raises(ValueError, match="no boundary observations"):
        p7._fit_boundary_through_origin([], "cpu_to_vta")


def test_runner_has_host_empty_queue_floor_mode():
    source = (
        p7.ROOT / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc"
    ).read_text(encoding="utf-8")
    for token in (
        "--host-empty",
        "--host-empty-stage-count",
        "--host-empty-inner-repeats",
        "RunHostEmpty",
        "reference_correctness_passed",
        "--process-cpu-affinity",
        "process_cpu_affinity_effective",
        "all_existing_threads_future_threads_inherit",
    ):
        assert token in source


def test_p7c_subplan_uses_separated_real_vta_segments_and_disjoint_cores():
    ranking, manifest = _inputs()
    parent = p7.build_plan(ranking, manifest)
    p7b2 = p7.load_artifact(
        p7.DEFAULT_P7B2_SESSION,
        "cpu_vta_pipeline_v1_p7b2_runtime_qualification_session",
    )
    catalog = p7._combined_package_catalog(
        p7.DEFAULT_PACKAGE_ROOT, p7.DEFAULT_P5A_PACKAGE_ROOT
    )
    plan = p7.build_p7c_plan(parent, manifest, catalog, p7b2)

    assert plan["actual_run_mode_count"] == 12
    assert len(plan["case_groups"]) == 4
    assert {
        group["vta_pressure_class"]: group["vta_source"]["segment_id"]
        for group in plan["case_groups"]
    } == {"compute_heavy": "vta:01:03", "dma_heavy": "vta:15:19"}
    assert (
        plan["vta_pressure_evidence"]["dma_heavy"]["dma_mib_per_run_ms"]
        > 2.0
        * plan["vta_pressure_evidence"]["compute_heavy"]["dma_mib_per_run_ms"]
    )
    for group in plan["case_groups"]:
        assert group["cpu_workload"]["process_cpu_affinity"] == [0, 1, 2]
        assert group["vta_process_cpu_affinity"] == [3]
        assert group["cpu_workload"]["traffic_bytes_per_sample"] == 128 * 1024 * 1024
        if group["cpu_pressure_class"] == "cache_resident":
            assert group["cpu_workload"]["operand_bytes_per_stream"] == 256 * 1024


def test_p7c_review_derives_signal_counts_from_summary(tmp_path):
    observations = []
    for cpu_class, vta_class, cpu_signal, vta_signal in (
        ("cache_resident", "compute_heavy", True, False),
        ("cache_resident", "dma_heavy", False, False),
        ("streaming", "compute_heavy", False, True),
        ("streaming", "dma_heavy", False, True),
    ):
        observations.append(
            {
                "cpu_pressure_class": cpu_class,
                "vta_pressure_class": vta_class,
                "vta_segment_id": "vta:test",
                "cpu_slowdown": 1.06 if cpu_signal else 1.01,
                "vta_slowdown": 1.06 if vta_signal else 1.01,
                "cpu_signal": cpu_signal,
                "vta_signal": vta_signal,
                "aggregate_effective_bandwidth_GBps": 3.0,
                "isolated_ms": {"vta_stage": 10.0},
                "concurrent_ms": {"vta_stage": 10.6},
                "vta_component_delta_ms": {
                    "driver_run": 0.1,
                    "run_minus_driver": 0.5,
                },
            }
        )
    summary = {
        "observations": observations,
        "streaming_excess_over_cache_control": {
            "compute_heavy": {"vta_slowdown_ratio_streaming_over_cache": 1.05},
            "dma_heavy": {"vta_slowdown_ratio_streaming_over_cache": 1.05},
        },
        "artifact_sha256": "summary",
        "source_session_artifact_sha256": "session",
    }
    output = tmp_path / "review.md"

    p7.write_p7c_review(output, summary)

    review = output.read_text(encoding="utf-8")
    assert "CPU 有 1 组、VTA 有 2 组信号" in review
    assert review.count("VTA signal:") == 2
    assert "不是实测 DDR bandwidth" in review


def test_source_instrumentation_audit_passes():
    ranking, manifest = _inputs()
    plan = p7.build_plan(ranking, manifest)
    audit = p7.audit_plan(
        plan,
        (p7.ROOT / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc").read_text(),
        (p7.ROOT / "vta/apps/native_deploy/ramps_cpu_memory_microbench.cc").read_text(),
        (p7.ROOT / "vta/runtime/runtime.cc").read_text(),
    )
    assert audit["passed"]
    assert all(audit["checks"].values())


def test_orchestrator_synchronizes_processes_and_audits_results(tmp_path):
    helper = tmp_path / "barrier_helper.py"
    helper.write_text(
        """
import argparse, json, pathlib, time
p = argparse.ArgumentParser()
p.add_argument('--ready-file', required=True)
p.add_argument('--start-file', required=True)
p.add_argument('--barrier-token', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
pathlib.Path(a.ready_file).write_text(a.barrier_token)
while not pathlib.Path(a.start_file).exists() or pathlib.Path(a.start_file).read_text().strip() != a.barrier_token:
    time.sleep(0.001)
pathlib.Path(a.output).write_text(json.dumps({'wall_ms': 1.0, 'checksum': 7, 'correct': True}) + '\\n')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_a = tmp_path / "a.jsonl"
    output_b = tmp_path / "b.jsonl"
    pair = p7.run_synchronized_pair(
        [sys.executable, str(helper), "--output", str(output_a)],
        [sys.executable, str(helper), "--output", str(output_b)],
        tmp_path / "barrier",
        timeout_s=5,
    )
    assert pair["process_a"]["returncode"] == 0
    assert pair["process_b"]["returncode"] == 0
    assert p7.summarize_jsonl(output_a)["correctness_passed"]
    assert p7.summarize_jsonl(output_b)["determinism_passed"]


def test_preflight_storage_gate_requires_space_and_clean_current_boot():
    healthy = {
        "sd_requested_path": "/mnt/sd",
        "sd_resolved_path": "/mnt/sd",
        "sd_mount": "/mnt/sd",
        "sd_filesystem_device": "/dev/mmcblk1p2",
        "sd_available_kb": str(256 * 1024),
        "current_boot_storage_error_count": "0",
    }
    assert storage_gate(healthy)
    assert not storage_gate({**healthy, "sd_available_kb": "1024"})
    assert not storage_gate({**healthy, "current_boot_storage_error_count": "1"})
    assert storage_gate(
        {
            **healthy,
            "sd_resolved_path": "/media/sd-mmcblk1p2",
            "sd_mount": "/media/sd-mmcblk1p2",
        }
    )


def test_repo_relative_path_accepts_relative_and_absolute_inputs():
    relative = "vta/tutorials/frontend/report_out/resource_aware_maxplus/example.json"
    absolute = p7.ROOT / relative
    assert p7._repo_relative_path(relative) == relative
    assert p7._repo_relative_path(absolute) == relative


def test_three_boot_summary_admits_only_repeated_component_signals():
    ranking, manifest = _inputs()
    plan = p7.build_plan(ranking, manifest)
    summary = p7.build_p7b1_multiboot_summary(
        plan,
        [
            p7.REPORT_ROOT / "v1_p7_session_boot{}_p7b1_memory.json".format(boot)
            for boot in (1, 2, 3)
        ],
        [
            p7.REPORT_ROOT / "v1_p7_session_boot{}_p7b1_cpu_pairs.json".format(boot)
            for boot in (1, 2, 3)
        ],
        [
            p7.REPORT_ROOT / "v1_p7_session_boot2_p7b1_memory_attempt1_failed.json",
            p7.REPORT_ROOT / "v1_p7_session_boot3_p7b1_memory_attempt1_failed.json",
        ],
    )

    assert summary["passed"]
    assert len(set(summary["board_boot_ids"])) == 3
    assert summary["summary"]["admitted_exact_observation_count"] == {"a": 14, "b": 13}
    assert summary["cpu_concurrency_exact_observations_ready"]
    assert not summary["generalized_slowdown_surface_ready"]
    assert not summary["formal_physical_profile_generated"]
    assert len(summary["failed_memory_attempts_retained"]) == 2
    assert not any(item["passed"] for item in summary["failed_memory_attempts_retained"])
