"""Accounting, causal-timing and inference tests; no board or network access."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import analyze_cpu_vta_residual_validation as analysis


def rows(mode="pipeline", count=307):
    result = []
    spacing = 100 if mode == "pipeline" else 200
    for frame in range(count):
        base = frame * spacing
        row = {"mode": mode, "completion_index": frame, "frame_id": frame,
               "stage_count": 3, "completion_ms": base + 191,
               "total_latency_ms": 191, "stage_cpu_time_scope": "disabled_concurrent_pipeline",
               "pmu_available": False}
        for stage, (offset, duration) in enumerate(((0, 80), (80, 60), (140, 50))):
            start = base + offset
            row.update({f"stage{stage}_start_ms": start, f"stage{stage}_end_ms": start + duration,
                        f"stage{stage}_ms": duration, f"stage{stage}_set_ms": 0,
                        f"stage{stage}_run_ms": duration, f"stage{stage}_get_ms": 0,
                        f"stage{stage}_run_start_ms": start, f"stage{stage}_run_end_ms": start + duration,
                        f"stage{stage}_scheduled_service_ms": duration,
                        f"stage{stage}_process_cpu_ms": None})
        row["stage1_vta_mutex_wait_ms"] = 0.001
        row["stage1_vta_mutex_held_ms"] = 60
        result.append(row)
    return result


def session(condition, kind, block, service=80, stage=0, boot="pilot"):
    metric = (analysis.analyze_pipeline(rows()) if kind == "pipeline" else
              analysis.analyze_serial(rows("serial", 20)))
    metric["stages"][stage]["run_ms"]["mean"] = service
    metric["stages"][stage]["busy_fraction"] = 0.99
    metric["stages"][stage]["completion_ii_ms"] = metric.get("ii_ms", 100)
    return {"boot_id": boot, "block": block, "condition": condition,
            "topology": ["cpu_first", "vta", "cpu_last"], "threads": [4, 1, 4],
            "runner_sha256": "a" * 64, "runtime_bindings": {"libvta": "b" * 64},
            "program_identity": [{"module": "c" * 64}], "metrics": metric}


class AccountingTests(unittest.TestCase):
    def test_pipeline_uses_300_completions_299_intervals(self):
        metric = analysis.analyze_pipeline(rows())
        self.assertEqual((metric["discarded_rows"], metric["scored_rows"], metric["completion_gaps"]), (7, 300, 299))
        self.assertEqual(metric["ii_ms"], 100)
        first = metric["stages"][0]
        self.assertEqual(first["completion_span_ms"], 29900)
        self.assertEqual(first["span_aligned_busy_mean_ms"], 80)
        self.assertEqual(first["mean_gap_ms"], 20)
        self.assertAlmostEqual(first["busy_fraction"], 0.8)
        self.assertAlmostEqual(first["span_accounting_error_ms"], 0)
        self.assertEqual(metric["ddr_attribution"], "unknown")

    def test_cross_frame_cpu_overlap_not_same_frame_sum(self):
        overlap = analysis.analyze_pipeline(rows())["cpu_overlap"]
        self.assertEqual(overlap["overlap_run_ms"], 299 * 40)
        self.assertAlmostEqual(overlap["first_cpu_fraction"], 11960 / 24000)

    def test_incomplete_or_misordered_window_rejected(self):
        for bad in (rows(count=306), rows(count=308), list(reversed(rows()))):
            with self.assertRaises(ValueError):
                analysis.analyze_pipeline(bad)
        bad = rows()
        bad[10]["completion_index"] += 1
        with self.assertRaises(ValueError):
            analysis.analyze_pipeline(bad)

    def test_nonfinite_or_double_counted_services_rejected(self):
        for field, value in (("stage0_run_ms", math.nan), ("stage0_ms", 81)):
            bad = rows()
            bad[8][field] = value
            with self.assertRaises(ValueError):
                analysis.analyze_pipeline(bad)

    def test_serial_outer_and_qualification_discard(self):
        metric = analysis.analyze_serial(rows("serial", 20))
        self.assertEqual(metric["scored_rows"], 15)
        self.assertEqual(metric["outer_residual_ms"]["mean"], 1)
        self.assertEqual(metric["m1_ms"], 80)
        self.assertEqual(analysis.analyze_serial(rows("serial", 10), discard=0)["scored_rows"], 10)

    def test_exclusive_cpu_pool_bound_can_exceed_stage_max(self):
        data = rows("serial", 20)
        for row in data:
            row["stage_cpu_time_scope"] = "exclusive_process_cpu_time"
            for i, demand in enumerate((200, 20, 160)):
                row[f"stage{i}_process_cpu_ms"] = demand
        metric = analysis.analyze_serial(data)
        resource = metric["resource_lower_bound"]
        self.assertEqual(resource["stage_service_max_ms"], 80)
        self.assertEqual(resource["sum_process_cpu_mean_ms"], 380)
        self.assertEqual(resource["exclusive_process_cpu_capacity_ms"], 95)
        self.assertEqual(metric["m1_ms"], 95)
        self.assertEqual(resource["affinity_hall_bound"], "unknown")
        data[8]["stage1_process_cpu_ms"] = None
        incomplete = analysis.analyze_serial(data)["resource_lower_bound"]
        self.assertIsNone(incomplete["exclusive_process_cpu_capacity_ms"])
        self.assertFalse(incomplete["complete_cpu_demand_observation"])

    def test_queue_blocking_and_residence_are_separate(self):
        data = rows()
        data[7]["queue_timings"] = [{"edge_index": 0, "push_request_ms": 710,
                                      "enqueue_ms": 730, "dequeue_ms": 750, "push_return_ms": 731}]
        data[8]["queue_timings"] = copy.deepcopy(data[7]["queue_timings"])
        queue = analysis.analyze_pipeline(data)["queue"]
        self.assertEqual(queue["edges"][0]["producer_block_ms"]["mean"], 20)
        self.assertEqual(queue["edges"][0]["residence_ms"]["mean"], 20)
        self.assertEqual(queue["edges"][0]["push_call_ms"]["count"], 1)
        data[7]["queue_timings"][0]["dequeue_ms"] = 720
        with self.assertRaises(ValueError):
            analysis.analyze_pipeline(data)

    def test_identity_layers_and_same_boot_pair(self):
        a = session("P0", "serial", 0)
        b = session("P0", "pipeline", 0)
        self.assertTrue(analysis.identity_compatibility(a, b)["compatible"])
        b["runner_sha256"] = "d" * 64
        self.assertFalse(analysis.identity_compatibility(a, b)["compatible"])
        a["metrics"]["session_identity"] = a.copy()
        b["metrics"]["session_identity"] = b.copy()
        with self.assertRaises(ValueError):
            analysis.compare_services(a["metrics"], b["metrics"])


class SelectionAndInferenceTests(unittest.TestCase):
    def test_selects_largest_absolute_cpu_service_change(self):
        sessions = []
        for block in range(3):
            for kind in ("serial", "pipeline"):
                sessions.extend((session("P0", kind, block, 80), session("P1", kind, block, 60),
                                 session("P2", kind, block, 120)))
        selected = analysis.select_cpu_contrast(sessions)
        self.assertEqual(selected["intervention"], "P2")
        self.assertEqual(selected["change_fraction"], 0.5)
        self.assertEqual(selected["intervention_parallel_amplification"], 1)
        self.assertTrue(selected["trajectory_consistent"])
        sessions[1]["runner_sha256"] = "changed"
        # A single invalid P1 serial arm cannot invalidate the independent P2 arm.
        self.assertEqual(analysis.select_cpu_contrast(sessions)["intervention"], "P2")

    def test_missing_matching_serial_blocks_does_not_select(self):
        sessions = [session(name, "pipeline", block, 80 if name == "P0" else 120)
                    for block in range(3) for name in ("P0", "P1")]
        self.assertIsNone(analysis.select_cpu_contrast(sessions))

    def test_three_boot_t_ci_uses_boot_means(self):
        records = [{"boot_id": str(boot), "block": block, "control_ii_ms": 100,
                    "contrast_ii_ms": 100 - (10 + boot)} for boot in range(3) for block in range(2)]
        result = analysis.analyze_replicates(records)
        expected = analysis.T95_DF2 / math.sqrt(3)
        self.assertAlmostEqual(result["delta_ii_ms_mean_ci95"]["lower"], 11 - expected)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["independent_boot_count"], 3)
        with self.assertRaises(ValueError):
            analysis.analyze_replicates(records + [records[0]])

    def test_stage_effect_can_pass_without_ii_gain(self):
        records = []
        for boot in range(3):
            for block in range(2):
                a = session("P0", "pipeline", block, 80, boot=str(boot))
                b = session("P1", "pipeline", block, 100 + boot, boot=str(boot))
                records.append({"boot_id": str(boot), "block": block, "stage_index": 0,
                                "control": a, "contrast": b})
        result = analysis.analyze_replicates(records)
        self.assertTrue(result["stage_gate_passed"])
        self.assertEqual(result["stage_service"]["direction"], "regression")
        self.assertFalse(result["ii_gate_passed"])
        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["identity_verified"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            analysis.write_report(path, result)
            self.assertIn("阶段效应不要求吞吐同时改善", path.read_text())

    def test_nested_pilot_report_keeps_proxy_resource_and_ii_distinct(self):
        sessions = [session(name, kind, block, 80 if name == "P0" else 120)
                    for block in range(3) for name in ("P0", "P1") for kind in ("serial", "pipeline")]
        selection = analysis.select_cpu_contrast(sessions)
        replicated = analysis.analyze_replicates([
            {"boot_id": str(boot), "block": block, "stage_index": 0,
             "control_ii_ms": 100, "contrast_ii_ms": 100,
             "control_stage_run_ms": 80, "contrast_stage_run_ms": 120}
            for boot in range(3) for block in range(2)])
        result = {"sessions": sessions, "selection": selection,
                  "telemetry_gate": {"primary_telemetry": False, "paired_relative_changes": [0.03, -0.01]},
                  "historical_e0": {"frozen_proxy": {"support": []}, "existing_replay": []},
                  "replication": {"03:17": replicated}}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "pilot.md"
            analysis.write_report(target, result)
            text = target.read_text()
            for expected in ("旧 proxy", "同块 stage max", "同块资源 M1", "实际 II", "主测量关闭",
                             "三次独立启动", "阶段服务验收门：通过", "II 验收门：未通过"):
                self.assertIn(expected, text)

    def test_small_or_inconsistent_boot_effect_fails(self):
        records = [{"boot_id": str(boot), "block": block, "control_ii_ms": 100,
                    "contrast_ii_ms": 96} for boot in range(3) for block in range(2)]
        result = analysis.analyze_replicates(records)
        self.assertTrue(result["ci95_excludes_zero"])
        self.assertFalse(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
