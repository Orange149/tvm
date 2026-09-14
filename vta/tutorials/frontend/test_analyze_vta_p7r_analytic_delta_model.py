#!/usr/bin/env python3

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_vta_p7r_analytic_delta_model.py")
SPEC = importlib.util.spec_from_file_location("delta_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(workload, mode, target, feature):
    values = {name: 0.0 for name in MODULE.BYTE_CALL_FEATURES + MODULE.REQUEST_FEATURES + MODULE.COMMAND_FEATURES}
    values["load_bytes"] = feature
    return {
        "workload_id": workload,
        "residence_mode": mode,
        "mechanism_candidate_id": "{}-{}-{}".format(workload, mode, feature),
        "delta_t_ms": target,
        "delta_features": values,
    }


class DeltaModelTest(unittest.TestCase):
    def test_nonnegative_physical_coefficient(self):
        rows = [
            row("A", "input", -0.4, -4.0),
            row("A", "input", -0.2, -2.0),
            row("B", "input", 0.2, 2.0),
            row("B", "input", 0.4, 4.0),
        ]
        model = MODULE.fit_proxy(rows, ["input"], ["load_bytes"])
        self.assertGreaterEqual(model["raw_unit_coefficients_ms"]["load_bytes"], 0.0)
        self.assertAlmostEqual(MODULE.predict_proxy(model, row("C", "input", 0.0, 3.0)), 0.3, places=5)

    def test_leave_one_workload_out_predictions_cover_every_row(self):
        rows = []
        for workload, offset in (("A", 0.0), ("B", 0.1), ("C", -0.1)):
            rows.extend([
                row(workload, "input", -0.3 + offset, -3.0),
                row(workload, "weight", 0.2 + offset, 2.0),
            ])
        workloads, _, folds, predictions, summary = MODULE.evaluate(rows)
        self.assertEqual(workloads, ["A", "B", "C"])
        self.assertEqual(len(predictions), len(rows))
        for held_out, fold in folds.items():
            self.assertNotIn(held_out, fold["train_workloads"])
        self.assertEqual(set(summary), set(MODULE.MODEL_FEATURES))

    def test_ranking_metrics(self):
        rows = [
            {"mechanism_candidate_id": "a", "delta_t_ms": -0.4, "predicted": -0.3},
            {"mechanism_candidate_id": "b", "delta_t_ms": -0.2, "predicted": -0.1},
            {"mechanism_candidate_id": "c", "delta_t_ms": 0.1, "predicted": -0.4},
            {"mechanism_candidate_id": "d", "delta_t_ms": 0.2, "predicted": 0.2},
        ]
        metrics = MODULE.ranking_metrics(rows, "predicted")
        self.assertTrue(math.isclose(metrics["regret_at_1_ms"], 0.5))
        self.assertEqual(metrics["regret_at_2_ms"], 0.0)
        self.assertEqual(metrics["regret_at_4_ms"], 0.0)

    def test_immutable_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "existing"
            path.mkdir()
            with self.assertRaises(FileExistsError):
                MODULE.write_outputs(path, [], {}, [])


if __name__ == "__main__":
    unittest.main()
