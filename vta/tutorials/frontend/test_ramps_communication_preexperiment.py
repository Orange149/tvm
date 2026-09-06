#!/usr/bin/env python3
"""Unit tests for the RAMPS communication pre-experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from run_ramps_communication_preexperiment import (
    completion_cycle_ms,
    deploy_command,
    direct_communication_gate,
    direct_profile_statistics,
    exact_pair_permutation,
    model_cycles,
    timing_record_from_result,
)


class ModelCycleTest(unittest.TestCase):
    def test_zero_communication_reduces_b1_to_b0(self):
        values, _ = model_cycles(
            run_ms=[10.0, 8.0, 4.0],
            communication_ms=[0.0, 0.0, 0.0],
            devices=["cpu", "vta", "cpu"],
            threads=[2, 1, 2],
            queue_depth=2,
        )
        self.assertEqual(values["b0_compute_only"], values["b1_communication"])
        self.assertEqual(values["b2_shared"], values["b3_full"])

    def test_multiple_vta_stages_share_mutex(self):
        values, bottleneck = model_cycles(
            run_ms=[3.0, 8.0, 2.0, 7.0, 3.0],
            communication_ms=[0.0] * 5,
            devices=["cpu", "vta", "cpu", "vta", "cpu"],
            threads=[1, 1, 1, 1, 1],
            queue_depth=2,
        )
        self.assertEqual(values["b0_compute_only"], 8.0)
        self.assertEqual(values["b2_shared"], 15.0)
        self.assertEqual(values["b3_full"], 15.0)
        self.assertEqual(bottleneck, "vta_mutex")

    def test_communication_changes_full_model(self):
        values, bottleneck = model_cycles(
            run_ms=[5.0, 5.0],
            communication_ms=[4.0, 4.0],
            devices=["cpu", "vta"],
            threads=[1, 1],
            queue_depth=1,
        )
        self.assertEqual(values["b0_compute_only"], 5.0)
        self.assertEqual(values["b1_communication"], 9.0)
        self.assertEqual(values["b2_shared"], 5.0)
        self.assertEqual(values["b3_full"], 9.0)
        self.assertIn(bottleneck, {"stage_worker", "vta_mutex"})

    def test_fifo_constraint_can_be_critical(self):
        values, bottleneck = model_cycles(
            run_ms=[10.0, 10.0],
            communication_ms=[0.0, 0.0],
            devices=["cpu", "cpu"],
            threads=[1, 1],
            queue_depth=1,
        )
        self.assertEqual(values["b3_full"], 10.0)
        self.assertIn(bottleneck, {"stage_worker", "fifo_0_1"})


class ResultParsingTest(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for frame in range(7):
            base = frame * 12.0
            rows.append(
                {
                    "mode": "pipeline",
                    "frame_id": frame,
                    "stage_count": 2,
                    "stage0_set_ms": 1.0,
                    "stage0_run_ms": 5.0,
                    "stage0_get_ms": 1.0,
                    "stage0_start_ms": base,
                    "stage0_end_ms": base + 7.0,
                    "stage1_set_ms": 2.0,
                    "stage1_run_ms": 6.0,
                    "stage1_get_ms": 1.0,
                    "stage1_start_ms": base + 7.0,
                    "stage1_end_ms": base + 16.0,
                }
            )
        return rows

    def test_completion_cycle_uses_adjacent_completion_interval(self):
        self.assertEqual(completion_cycle_ms(self.rows(), 2), 12.0)

    def test_timing_record_parses_stage_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native_result.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in self.rows()),
                encoding="utf-8",
            )
            record = timing_record_from_result(
                candidate_id="candidate",
                group_id="1_2",
                pair_id="pair01",
                session="session001",
                source_path=path,
                devices=["cpu", "vta"],
                threads=[2, 1],
                queue_depth=2,
                skip_first=2,
                correctness_passed=True,
            )
            self.assertEqual(record.measured_cycle_ms, 12.0)
            self.assertEqual(record.stage_run_ms, [5.0, 6.0])
            self.assertEqual(record.stage_communication_ms, [2.0, 3.0])

    def test_exact_pair_permutation(self):
        # All four positive effects have a one-sided exact p-value of 1/16.
        self.assertEqual(exact_pair_permutation([1.0, 2.0, 3.0, 4.0]), 1.0 / 16.0)

    def test_deploy_command_keeps_ssh_option_in_one_token(self):
        args = SimpleNamespace(
            board="root@192.168.1.150",
            runs=25,
            queue_depth=2,
            ssh_timeout_s=60,
            scp_timeout_s=600,
            serial_timeout_s=180,
            pipeline_timeout_s=240,
            fetch_timeout_s=120,
            output_dir="/tmp/ramps-output",
            rpc_baseline_result="/tmp/baseline.jsonl",
        )
        command = deploy_command(
            args,
            Path("/tmp/result"),
            Path("/tmp/package"),
            "candidate",
            [3, 1, 4],
            "/var/volatile/candidate",
            skip_run=True,
        )
        options = [token for token in command if token.startswith("--ssh-option=")]
        self.assertEqual(len(options), 6)
        self.assertNotIn("--ssh-option", command)


class DirectProfilerAblationTest(unittest.TestCase):
    @staticmethod
    def record(candidate_id, group_id, measured, run_ms, direct_copy_ms, islands):
        devices = ["cpu", "vta", "cpu"]
        return SimpleNamespace(
            candidate_id=candidate_id,
            group_id=group_id,
            measured_cycle_ms=measured,
            vta_island_count=islands,
            stages=[SimpleNamespace(device=device) for device in devices],
            metadata={
                "measured_stage_run_ms": run_ms,
                "direct_communication_profile": {
                    "pipeline": {
                        "direct_copy_ms": direct_copy_ms,
                        "coherence_ms": 0.0,
                    }
                },
            },
        )

    def test_direct_copy_is_added_to_the_serialized_vta_resource(self):
        records = [
            self.record("a", "resnet:i1_1_4", 14.0, [10.0, 8.0, 2.0], 4.0, 1),
            self.record("b", "resnet:i2_2_8", 22.0, [12.0, 14.0, 3.0], 6.0, 2),
        ]
        result = direct_profile_statistics(records, bootstrap=20, seed=1)
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["rows"][0]["shared_compute_ms"], 10.0)
        self.assertEqual(result["rows"][0]["communication_aware_ms"], 12.0)
        self.assertGreater(result["mae_improvement_ms"], 0.0)
        self.assertIn("regret_at_1", result["communication_aware"])
        self.assertIn("regret_at_100", result["communication_aware"])
        self.assertIn("evaluations_to_oracle_95pct", result["communication_aware"])
        self.assertIn("evaluations_to_oracle_98pct", result["communication_aware"])

    def test_direct_gate_does_not_claim_full_b3_validation(self):
        gate = direct_communication_gate(
            {
                "record_count": 200,
                "base": {"spearman": 0.7},
                "communication_aware": {"spearman": 0.9},
                "grouped_bootstrap": {"ci95_ms": [1.0, 3.0], "p_one_sided": 0.01},
            }
        )
        self.assertTrue(gate["communication_mechanism_supported"])
        self.assertFalse(gate["continue_old_set_get_board_preexperiment"])
        self.assertFalse(gate["full_b3_validated"])


if __name__ == "__main__":
    unittest.main()
