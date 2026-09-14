"""Host-only regression checks for the G0 interval and protocol audit."""

import copy
import json
import shlex
import unittest

from analyze_stage_memory_g0_trace import audit, intersect
from run_stage_memory_g0_discovery import summarize, union_ms
from run_stage_memory_g0_pair import analyze as analyze_pair
from analyze_stage_memory_g0_sweep import characterize
from run_stage_memory_g0_sweep import make_plan, TRACE, BASE


def fixture():
    manifest = {"stages": [{"index": 0, "device": "cpu"}, {"index": 1, "device": "vta"}]}
    rows = []
    for frame in range(6):
        base = frame * 20.0
        row = {"frame_id": frame, "completion_ms": base + 25, "total_latency_ms": 25,
               "raw_outputs": [{"fnv1a64": "frozen"}], "p8_managed_slot_count": 2,
               "p8_framework_materialization_bytes": 0, "p8_total_slot_wait_ms": 0,
               "p8_boundaries": [{"edge_index": 0, "slot_id": frame % 2,
                                  "generation": frame // 2 + 1, "tensor_count": 1,
                                  "tensor_bytes": [64], "boundary_bytes": 64,
                                  "physical_addresses": [hex(4096 + (frame % 2) * 64)]}]}
        for i, timestamps in enumerate([[0, .1, .2, .3, .3, 8, 8.2],
                                         [8.3, 8.4, 8.5, 8.6, 9, 24, 24.2]]):
            for name, value in zip(["queue_pop", "slots_acquired", "start", "binding_done",
                                    "run_start", "run_end", "end"], timestamps):
                row[f"stage{i}_{name}_ms"] = base + value
            row[f"stage{i}_set_end_ms"] = row[f"stage{i}_binding_done_ms"]
            row[f"stage{i}_run_ms"] = timestamps[-2] - timestamps[-3]
            row[f"stage{i}_vta_mutex_wait_ms"] = .1 if i == 1 else None
        row["stage1_vta_mutex_acquire_ms"] = base + 8.9
        row["stage1_vta_mutex_release_ms"] = base + 24.1
        rows.append(row)
    return rows, manifest


class G0Tests(unittest.TestCase):
    @unittest.skipUnless((TRACE / "readiness_audit.json").exists(), "frozen artifacts unavailable")
    def test_frozen_census_is_complete_deterministic_and_owner_filtered(self):
        trace = json.loads((TRACE / "readiness_audit.json").read_text())
        dma = json.loads((BASE / "stage_memory_experiments/experiment_ab.json").read_text())
        plan = make_plan(trace, dma)
        self.assertEqual(plan, make_plan(trace, dma))
        self.assertEqual(len(plan["cells"]), 60)
        self.assertEqual(len(plan["excluded_directions"]), 2)
        self.assertEqual(len({(c["topology"], c["active"], c["ready"]) for c in plan["cells"]}), 22)
        selected = [c for c in plan["cells"] if c["topology"] == "D" and c["active"] == 0 and c["ready"] == 3]
        self.assertEqual({c["natural_eligible_encounters"] for c in selected}, {62})
        self.assertEqual({c["quantile"] for c in selected}, {.25, .5, .75})

    @staticmethod
    def pair_rows():
        rows = []
        for index, policy in enumerate(["allow", "wait", "wait", "allow"]):
            base = index * 100
            release = base + (12 if policy == "wait" else 5)
            rows.append({"sample": index, "warmup": False, "block": 0, "policy": policy,
                         "offset_ms": 3, "eligible_ms": base + 5, "active_at_eligible": True,
                         "makespan_ms": release + 10.5 - (base + 2),
                         "outputs_match_reference": True,
                         "sides": [{"binding_ms": base, "release_ms": base + 1,
                                    "start_ms": base + 2, "end_ms": base + 12},
                                   {"binding_ms": base + .5, "release_ms": release,
                                    "start_ms": release + .5, "end_ms": release + 10.5}]})
        return rows

    def test_pair_makespan_is_not_sum_or_fps(self):
        result = analyze_pair(self.pair_rows())
        self.assertEqual(result["allow_makespan_median_ms"], 13.5)
        self.assertEqual(result["wait_makespan_median_ms"], 20.5)
        self.assertEqual(result["action_label"], "unclassified")

    def test_pair_treatment_exposure_is_measured_separately(self):
        result = characterize(self.pair_rows(), {"natural_age_min_ms": 2, "natural_age_max_ms": 4})
        self.assertEqual(result["allow"]["positive_graph_overlap"], 2)
        self.assertEqual(result["wait"]["positive_graph_overlap"], 0)
        self.assertEqual(result["allow"]["age_in_natural_observed_range"], 2)

    def test_pair_wait_must_follow_active_done(self):
        rows = self.pair_rows()
        rows[1]["sides"][1]["release_ms"] = 105
        with self.assertRaises(AssertionError):
            analyze_pair(rows)

    def test_pair_rejects_bad_output_and_order(self):
        rows = self.pair_rows()
        rows[0]["outputs_match_reference"] = False
        with self.assertRaises(AssertionError):
            analyze_pair(rows)
        rows = self.pair_rows()
        rows[2]["policy"] = "allow"
        with self.assertRaises(AssertionError):
            analyze_pair(rows)

    def test_union_does_not_double_count(self):
        self.assertEqual(union_ms([(0, 10), (2, 4), (8, 12), (15, 20)]), 17)
        self.assertEqual(union_ms(intersect([(0, 10)], [(2, 5), (4, 8)])), 6)

    def test_shell_continuation(self):
        invocation = " \\\n --runs 8 \\\n --vta-runtime-profile-dir ''"
        self.assertEqual(shlex.split(invocation.replace("\\\n", "")),
                         ["--runs", "8", "--vta-runtime-profile-dir", ""])

    def test_direction_and_no_causal_claim(self):
        rows, manifest = fixture()
        result = audit(rows, manifest)
        cpu_to_vta, vta_to_cpu = result["directed_encounters"]
        self.assertEqual(cpu_to_vta["encounters"], 0)
        self.assertEqual(vta_to_cpu["encounters"], 2)
        self.assertEqual(vta_to_cpu["not_blocked_by_vta_owner"], 2)
        self.assertAlmostEqual(vta_to_cpu["samples"][0]["active_age_ms"], 11.3)
        self.assertIsNone(result["opportunity_ceiling"])

    def test_protocol_violations_fail(self):
        rows, manifest = fixture()
        bad = copy.deepcopy(rows)
        bad[2]["p8_boundaries"][0]["generation"] = 1
        with self.assertRaises(AssertionError):
            audit(bad, manifest)
        bad = copy.deepcopy(rows)
        bad[0]["stage0_slots_acquired_ms"] = 90
        with self.assertRaises(AssertionError):
            audit(bad, manifest)

    def test_missing_old_readiness_is_not_zero_opportunity(self):
        rows, manifest = fixture()
        for row in rows:
            for i in range(2):
                del row[f"stage{i}_binding_done_ms"]
                del row[f"stage{i}_run_start_ms"]
                del row[f"stage{i}_run_end_ms"]
        result = summarize(rows, manifest, ("frozen",), 2)
        self.assertEqual(result["fps"], 50)
        self.assertTrue(all(not e["readiness_observed"] for e in result["directed_encounters"]))

    def test_output_mismatch_fails(self):
        rows, manifest = fixture()
        rows[3]["raw_outputs"][0]["fnv1a64"] = "bad"
        with self.assertRaises(AssertionError):
            summarize(rows, manifest, ("frozen",), 2)


if __name__ == "__main__":
    unittest.main()
