"""Local structural/manifest/formula tests; no fabricated board result ledger."""

from collections import Counter, defaultdict
from copy import deepcopy
import unittest

import c3_official_window_protocol as protocol


def audit(oid, category, delta=10, same=False, network="resnet18", node=10):
    # Symbolic identities are unit fixtures, never compiler/board certificates.
    return {"object_id": oid, "network": network, "earliest_graph_node": node,
            "semantic_hash": "unit-semantic-" + oid, "classes": [category],
            "delta_mac_insns": delta, "official_baseline_bound": True,
            "local_gates_passed": True,
            "programs": {"B": {"program_id": oid + "_B", "execution_sha256": oid + "-original", "abi_sha256": oid + "-abi"},
                         "A": {"program_id": oid + "_A", "execution_sha256": oid + ("-original" if same else "-aggregated"), "abi_sha256": oid + "-abi"}}}


class FreezeTests(unittest.TestCase):
    def test_class_first_choices_structure_sort_no_refill(self):
        lower = audit("lower", 1, delta=1)
        later = audit("later", 1, delta=20, network="yolov3-tiny", node=1)
        earlier = audit("earlier", 1, delta=20, network="resnet18", node=5)
        dual = audit("dual", 2, delta=15)
        dual["classes"] = [2, 3]
        control = audit("control", 4, delta=0, same=True)
        pool = protocol.freeze_pool([later, lower, control, dual, earlier])
        self.assertEqual(pool["class_first_choices"], {"1": "earlier", "2": "dual", "3": "dual", "4": "control"})
        self.assertEqual([row["object_id"] for row in pool["objects"]], ["earlier", "dual", "control"])
        self.assertEqual(pool["objects"][1]["selection_classes"], [2, 3])
        self.assertEqual(len(pool["programs"]), 5)
        self.assertEqual(pool["coverage_funnel"]["dedup_NA_controls"], 1)
        self.assertTrue(pool["board_allowed"])

    def test_empty_class_not_filled_and_unbound_row_not_selected(self):
        unbound = audit("unbound", 2, 100)
        unbound["official_baseline_bound"] = False
        pool = protocol.freeze_pool([audit("one", 1), audit("other", 1), unbound])
        self.assertEqual(len(pool["objects"]), 1)
        self.assertIsNone(pool["class_first_choices"]["2"])
        self.assertIsNone(pool["class_first_choices"]["3"])
        self.assertIsNone(pool["class_first_choices"]["4"])

    def test_all_identical_stops_before_call_manifest(self):
        pool = protocol.freeze_pool([audit("tight", 4, 0, same=True)])
        self.assertEqual(pool["status"], "STOP_ALL_EXECUTION_REPRESENTATIONS_IDENTICAL")
        self.assertFalse(pool["board_allowed"])
        self.assertEqual(len(pool["programs"]), 1)
        self.assertEqual(pool["objects"][0]["comparison_status"], "DEDUP_NOT_APPLICABLE")
        with self.assertRaisesRegex(ValueError, "not admitted"):
            protocol.build_call_manifest(pool, "unit")

    def test_local_gate_pending_does_not_replace_first_choice(self):
        selected = audit("preferred", 1, 20)
        selected["local_gates_passed"] = False
        pool = protocol.freeze_pool([selected, audit("backup", 1, 10)])
        self.assertEqual(pool["objects"][0]["object_id"], "preferred")
        self.assertFalse(pool["board_allowed"])
        self.assertEqual(pool["status"], "FROZEN_POOL_LOCAL_GATES_PENDING")

    def test_mathematical_semantics_alone_do_not_deduplicate(self):
        first = audit("first", 1)
        second = audit("second", 2)
        second["semantic_hash"] = first["semantic_hash"]
        pool = protocol.freeze_pool([first, second])
        self.assertEqual(len(pool["objects"]), 2)
        self.assertEqual(len(pool["programs"]), 4)

    def test_exact_execution_dedup_after_class_choices_does_not_refill(self):
        first = audit("first", 1, 20)
        duplicate = audit("other-network-label", 2, 30, network="yolov3-tiny")
        duplicate["programs"] = deepcopy(first["programs"])
        spare = audit("spare", 2, 1)
        pool = protocol.freeze_pool([first, duplicate, spare])
        self.assertEqual(len(pool["objects"]), 1)
        self.assertEqual(len(pool["programs"]), 2)
        self.assertEqual(pool["objects"][0]["selection_classes"], [1,2])
        self.assertEqual(pool["class_first_choices"]["2"], duplicate["object_id"])
        self.assertFalse(any(row["object_id"] == "spare" for row in pool["objects"]))

    def test_unsupported_unclassified_object_retained_in_coverage(self):
        unsupported = audit("unknown", 1)
        unsupported.update(eligible=False, classes=[], rejection_reason="SUPPORT_UNRESOLVED")
        pool = protocol.freeze_pool([unsupported, audit("supported", 1)])
        self.assertEqual(pool["coverage_funnel"]["audited_objects"], 2)
        self.assertEqual(pool["coverage_funnel"]["structurally_eligible_objects"], 1)

    def test_frozen_identity_and_performance_selection_gate(self):
        row = audit("one", 1)
        row["performance_labels_used"] = True
        with self.assertRaisesRegex(ValueError, "performance"):
            protocol.freeze_pool([row])
        pool = protocol.freeze_pool([audit("one", 1)])
        pool["objects"][0]["delta_mac_insns"] = 999
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.build_call_manifest(pool, "unit")


class ManifestTests(unittest.TestCase):
    def pool(self):
        return protocol.freeze_pool([audit("fit", 1), audit("wrap", 2),
                                     audit("segment", 3), audit("compact", 4, 0, same=True)])

    def test_exact_60K_budget_qualification_ABA_all_unique_programs(self):
        pool = self.pool()
        manifest = protocol.build_call_manifest(pool, "unit_attempt")
        k = len(pool["programs"])
        self.assertEqual(k, 7)
        self.assertEqual(manifest["target_calls"], 60*k)
        self.assertLessEqual(manifest["target_calls"], 480)
        self.assertEqual(len(manifest["calls"]), manifest["target_calls"])
        self.assertEqual(len(manifest["sessions"]), 10*k)
        phases = Counter(row["phase"] for row in manifest["calls"])
        self.assertEqual(phases, {"qualification": 6*k, "diagnostic": 18*k, "formal": 36*k})
        self.assertEqual(Counter(row["call_kind"] for row in manifest["calls"]), manifest["expected_call_counts"])
        self.assertEqual(len({row["invocation_id"] for row in manifest["calls"]}), 60*k)
        for program in pool["programs"]:
            calls = [row for row in manifest["calls"] if row["program_id"] == program["program_id"] and row["phase"] == "qualification"]
            self.assertEqual([row["seed"] for row in calls[:3]], list(protocol.SEEDS))
            self.assertEqual([row["weight_version"] for row in calls[3:]], ["W0", "W1", "W0"])
            self.assertEqual([row["aba_label"] for row in calls[3:]], ["W0", "W1", "W0_return"])

    def test_every_fresh_session_six_once_calls_and_fixed_rotations(self):
        manifest = protocol.build_call_manifest(self.pool(), "unit")
        cells = defaultdict(list)
        for row in manifest["calls"]:
            cells[row["session_id"]].append(row)
        self.assertEqual(len(cells), len(manifest["sessions"]))
        for sid, cell in cells.items():
            self.assertEqual([row["position"] for row in cell], [1,2,3,4,5,6])
            self.assertEqual([row["helper_count_after"] for row in cell], [1,2,3,4,5,6])
            self.assertTrue(all(row["complete_operator_calls"] == 1 and row["hidden_warmup_calls"] == 0 for row in cell))
            self.assertTrue(all(row["fresh_rpc"] for row in cell))
            self.assertEqual(len({row["program_id"] for row in cell}), 1)
            if cell[0]["phase"] != "qualification":
                rotation = cell[0]["rotation"]
                expected = list(protocol.SEEDS[rotation:] + protocol.SEEDS[:rotation])
                self.assertEqual([row["seed"] for row in cell], expected*2)
            if cell[0]["phase"] == "diagnostic":
                self.assertEqual([row["capture_dump"] for row in cell], [True,False,False,True,True,True])
                self.assertTrue(all(row["mode"] == "detailed" for row in cell))
            else:
                self.assertTrue(all(row["mode"] == "minimal" for row in cell))
            self.assertTrue(all(row["timer_result_is_formal_sample"] == (row["call_kind"] == "timing") for row in cell))

    def test_object_rotation_B_A_balance_and_dedup_control_no_A_call(self):
        pool = self.pool()
        manifest = protocol.build_call_manifest(pool, "unit")
        formal = [row for row in manifest["sessions"] if row["phase"] == "formal"]
        objects = pool["objects"]
        for r in range(6):
            sessions = [row for row in formal if row["round"] == r+1]
            object_order = list(dict.fromkeys(row["object_id"] for row in sessions))
            expected = [row["object_id"] for row in objects[r%4:] + objects[:r%4]]
            if r == 5:
                expected = [row["object_id"] for row in reversed(objects)]
            self.assertEqual(object_order, expected)
        for obj in objects:
            order = [[row["program_id"] for row in formal if row["round"] == r and row["object_id"] == obj["object_id"]] for r in range(1,7)]
            if obj["comparison_applicable"]:
                for round_index, programs in enumerate(order):
                    expected = [obj["baseline_program"], obj["aggregate_program"]]
                    self.assertEqual(programs, expected if round_index % 2 == 0 else expected[::-1])
                self.assertEqual(sum(row[0] == obj["baseline_program"] for row in order), 3)
                self.assertEqual(sum(row[0] == obj["aggregate_program"] for row in order), 3)
            else:
                self.assertEqual(order, [[obj["baseline_program"]]]*6)
                self.assertFalse(any(row["program_id"] == obj["programs"]["A"]["program_id"] for row in manifest["calls"]))

    def test_manifest_reproducible_and_incomplete_attempt_rejected(self):
        pool = self.pool()
        first = protocol.build_call_manifest(pool, "unit")
        self.assertEqual(first, protocol.build_call_manifest(pool, "unit"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            protocol.analyse(pool, first, [])


class SubmitAndFormulaTests(unittest.TestCase):
    def ledger(self, count=1):
        return {"schema": "c3_official_window_submit_ledger_v1", "mode": "minimal", "overflow": False,
                "attempted_submits": count, "recorded_submits": count,
                "submits": [{"ordinal": i+1, "reason": "unit", "insns": 7+i,
                             "insn_bytes": 16*(7+i), "serialized_uop_bytes": 4*i,
                             "completion": True, "timeout": 0} for i in range(count)]}

    def test_submit_capacity_units_mode_and_completion(self):
        self.assertEqual(len(protocol.canonical_submit_sequence(self.ledger(16), "minimal")), 16)
        for mutate in (lambda x:x.update(overflow=True), lambda x:x.update(recorded_submits=0),
                       lambda x:x.update(mode="detailed"), lambda x:x["submits"][0].update(completion=False),
                       lambda x:x["submits"][0].update(timeout=1), lambda x:x["submits"][0].update(insn_bytes=7),
                       lambda x:x["submits"][0].update(serialized_uop_bytes=3)):
            value = self.ledger()
            mutate(value)
            with self.assertRaises(ValueError):
                protocol.canonical_submit_sequence(value, "minimal")
        with self.assertRaises(ValueError):
            protocol.canonical_submit_sequence(self.ledger(17), "minimal")

    def test_round_median_mad_and_pair_formula_exact_ties(self):
        # Pure arithmetic examples, not generated experimental observations.
        baseline = [10,20,30,40,50,60]
        candidate = [9,18,27,36,45,60]
        stats = protocol.round_statistics(baseline)
        self.assertEqual(stats["T_ms"], 35)
        self.assertEqual(stats["MAD_ms"], 15)
        self.assertEqual((stats["min_ms"], stats["max_ms"]), (10,60))
        pair = protocol.paired_statistics(candidate, baseline)
        self.assertAlmostEqual(pair["G_pct"], 10)
        self.assertEqual((pair["wins"],pair["ties"],pair["losses"]), (5,1,0))
        self.assertEqual(pair["median_ms_saved"], 2.5)
        with self.assertRaises(ValueError):
            protocol.round_statistics([1,2,3,4,5,float("nan")])

    def test_once_call_contract_rejects_hidden_call_counter_output_and_mode(self):
        # Contract unit fixture only; never an experimental call ledger.
        pool = protocol.freeze_pool([audit("one", 1)])
        expected = protocol.build_call_manifest(pool, "unit")["calls"][0]
        actual = {**expected, "status":"PASS", "correct":True, "arguments_unchanged":True,
                  "expected_sha256":"unit-reference", "output_sha256":"unit-reference",
                  "runtime_metadata":self.ledger(), "helper_started_after":1, "helper_completed_after":1}
        protocol.validate_call(expected, actual)
        for edit in (lambda x:x.update(complete_operator_calls=2), lambda x:x.update(hidden_warmup_calls=1),
                     lambda x:x.update(helper_count_after=2), lambda x:x.update(helper_completed_after=0),
                     lambda x:x.update(output_sha256="unit-wrong-output"),
                     lambda x:x["runtime_metadata"].update(mode="detailed")):
            changed = deepcopy(actual)
            edit(changed)
            with self.assertRaises(ValueError):
                protocol.validate_call(expected, changed)


if __name__ == "__main__":
    unittest.main()
