"""C3-S fail-closed static eligibility tests, without TVM or a board."""
from pathlib import Path
import unittest

from audit_shared_buffer_storage import classify, finalize_bundles, graph_entries, verify_hashes


def graph(alias=False):
    return {"nodes": [{"op": "null", "name": "data0", "inputs": []},
                      {"op": "tvm_op", "name": "op", "inputs": [[0, 0, 0]],
                       "attrs": {"func_name": "fused_compute"}}],
            "node_row_ptr": [0, 1, 2], "arg_nodes": [0], "heads": [[1, 0, 0]],
            "attrs": {"shape": ["list_shape", [[1, 64], [1, 64]]],
                      "dltype": ["list_str", ["float32", "float32"]],
                      "storage_id": ["list_int", [0, 0 if alias else 1]]}}


def tensor(alias=False, params=(), linked="absent", direction="cpu_to_vta"):
    t = classify(graph_entries(graph(alias), 12), "data0", set(params), linked,
                 {"shape": [1, 64], "dtype": "float32"}, direction)
    t["contract_bytes"] = 256
    return t


def edge(tensors, consumer=1):
    return {"consumer_stage": consumer, "tensors": tensors}


class StorageTests(unittest.TestCase):
    def test_exclusive_input(self):
        t = tensor()
        self.assertEqual(t["status"], "eligible")
        self.assertEqual(t["readers"][0]["node"], 1)
        self.assertEqual(t["writers"], [])
        e = finalize_bundles([edge([t])])[0]
        self.assertEqual(e["predicted_saved_bytes"], 256)

    def test_internal_alias_cross_frame_rejected(self):
        self.assertEqual(tensor(True)["status"], "rejected")

    def test_parameter_rejected(self):
        self.assertEqual(tensor(params=["data0"])["status"], "rejected")

    def test_linked_or_unknown_fail_closed(self):
        for status in ("present", "unknown"):
            self.assertEqual(tensor(linked=status)["status"], "unknown")

    def test_other_direction_rejected(self):
        self.assertEqual(tensor(direction="vta_to_cpu")["status"], "rejected")

    def test_cross_edge_conflict(self):
        edges = finalize_bundles([edge([tensor()]), edge([tensor()])])
        self.assertTrue(all(e["source"] == "external" for e in edges))

    def test_same_sid_different_executors_is_not_alias(self):
        edges = finalize_bundles([edge([tensor()], 1), edge([tensor()], 3)])
        self.assertTrue(all(e["source"] == "pool-anchor" for e in edges))

    def test_partial_bundle_all_external(self):
        bad = tensor(True)
        bad["storage_id"] = 7
        e = finalize_bundles([edge([tensor(), bad])])[0]
        self.assertEqual(e["source"], "external")
        self.assertEqual(e["predicted_saved_bytes"], 0)

    def test_mismatch_and_nop(self):
        entries = graph_entries(graph(), 12)
        t = classify(entries, "data0", set(), "absent", {"shape": [4], "dtype": "int8"}, "cpu_to_vta")
        self.assertEqual(t["status"], "rejected")
        entries[0]["readers"][0]["func"] = "__nop"
        t = classify(entries, "data0", set(), "absent", {"shape": [1, 64], "dtype": "float32"}, "cpu_to_vta")
        self.assertEqual(t["status"], "unknown")

    def test_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_hashes(Path(__file__).parent, {Path(__file__).name: "0" * 64})


if __name__ == "__main__":
    unittest.main()
