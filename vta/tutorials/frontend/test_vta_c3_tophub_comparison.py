"""Small CPU-only adapter tests; no board access or existing artifact writes."""
import gc
from pathlib import Path
import unittest
import weakref

import run_vta_c3_tophub_reuse_comparison as comparison
import run_vta_c3_tophub_reuse_retained_timer as retained


class AdapterTests(unittest.TestCase):
    def test_resolved_outer_extent_does_not_change_inner_tile(self):
        source = {"entity": [["tile_h", "sp", [7, 8]], ["oc_nthread", "ot", 2]]}
        current = {"entity": [["tile_h", "sp", [-1, 8]], ["oc_nthread", "ot", 2]]}
        self.assertEqual(comparison.tiling(source), comparison.tiling(current))
        current["entity"][0][2][-1] = 7
        self.assertNotEqual(comparison.tiling(source), comparison.tiling(current))

    def test_helper_module_survives_returned_handle_deletion(self):
        class Module:
            pass
        original = retained.ORIGINAL_LOAD
        before = len(retained.LOADED_MODULES)
        try:
            retained.ORIGINAL_LOAD = lambda *_: Module()
            module = retained.retained_load(None, Path("timer.so"))
            ref = weakref.ref(module)
            del module
            gc.collect()
            self.assertIsNotNone(ref())
        finally:
            del retained.LOADED_MODULES[before:]
            retained.ORIGINAL_LOAD = original
        gc.collect()
        self.assertIsNone(ref())

    def test_recovery_adapter_binding_and_no_timing_reuse(self):
        original = retained.ORIGINAL_VERIFY
        try:
            retained.ORIGINAL_VERIFY = lambda _: {"schema": "c3_tophub_four_mode_v1", "source_bindings": {}}
            contract = retained.bound_verify(Path("unused"))
            self.assertTrue(contract["timer_module_retained_for_whole_session"])
            self.assertIn(str(Path(retained.__file__).resolve()), contract["source_bindings"])
            self.assertIn("no previous timing reused", contract["recovery_scope"])
        finally:
            retained.ORIGINAL_VERIFY = original


if __name__ == "__main__":
    unittest.main()
