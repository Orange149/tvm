import tempfile
import unittest
from pathlib import Path

from analyze_vta_dual_resource_score import (
    average_ranks,
    fit_dual,
    predict_dual,
    spearman,
    static_terms,
)


def make_model(cpu_stages, vta_stages=2):
    total = sum(stage["core_work_ms"] for stage in cpu_stages) + 20.0
    return {
        "static_score_ms": 70.0,
        "total_host_core_work_ms": total,
        "vta_stage_count": vta_stages,
        "cpu_stages": cpu_stages,
        "resource_load_ms": {
            "direct_boundary_communication": 2.0,
            "fluid_prefix_max": 30.0,
            "shared_ddr_physical_demand": 10.0,
            "total_host_physical_pool": 40.0,
            "single_vta_plus_mutex_boundaries": 70.0,
        },
    }


class DualResourceScoreTest(unittest.TestCase):
    def test_static_terms_scale_cpu_pressure_by_effective_cores(self):
        model = make_model(
            [
                {
                    "segment_id": "cpu:a",
                    "run_ms": 40.0,
                    "inclusive_service_ms": 41.0,
                    "core_work_ms": 40.0,
                },
                {
                    "segment_id": "cpu:b",
                    "run_ms": 40.0,
                    "inclusive_service_ms": 41.0,
                    "core_work_ms": 120.0,
                },
            ]
        )
        terms = static_terms(model)
        first, second = terms["cpu_terms"]
        self.assertAlmostEqual(first["effective_cores"], 1.0)
        self.assertAlmostEqual(second["effective_cores"], 3.0)
        self.assertGreater(second["pressure_ms"], first["pressure_ms"])

    def test_predict_dual_selects_cpu_vta_and_other(self):
        terms = static_terms(
            make_model(
                [
                    {
                        "segment_id": "cpu:a",
                        "run_ms": 20.0,
                        "inclusive_service_ms": 20.0,
                        "core_work_ms": 60.0,
                    }
                ]
            )
        )
        self.assertEqual(predict_dual(terms, 0.0, 0.0)["active_resource"], "vta")
        self.assertEqual(predict_dual(terms, 4.0, 0.0)["active_resource"], "cpu")
        terms["other_bound_ms"] = 500.0
        self.assertEqual(predict_dual(terms, 4.0, 2.0)["active_resource"], "other")

    def test_fit_dual_recovers_synthetic_coefficients(self):
        records = []
        for index, (cpu_pressure, vta_pressure) in enumerate(
            [(20.0, 80.0), (100.0, 10.0), (60.0, 40.0), (10.0, 100.0)]
        ):
            terms = {
                "cpu_terms": [
                    {
                        "segment_id": f"cpu:{index}",
                        "service_ms": 50.0,
                        "pressure_ms": cpu_pressure,
                    }
                ],
                "vta_base_ms": 55.0,
                "vta_pressure_ms": vta_pressure,
                "other_bound_ms": 20.0,
            }
            observed = predict_dual(terms, 0.4, 0.1)["score_ms"]
            records.append({"static_terms": terms, "observed_ii_ms": observed})
        fit = fit_dual(records, step=0.01)
        self.assertAlmostEqual(fit["beta_cpu"], 0.4, places=2)
        self.assertAlmostEqual(fit["beta_vta"], 0.1, places=2)

    def test_rank_ties_and_spearman(self):
        self.assertEqual(average_ranks([1.0, 1.0, 3.0]), [1.5, 1.5, 3.0])
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
