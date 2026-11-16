"""Baseline provenance and serialization safeguards, not timing experiments."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import tvm
from tvm import autotvm

from c3_official_window_baseline import (
    assert_effective_config, key, normalize_llvm, restore_external_global_bindings,
    select_records,
)
from test_c3_valid_window import fixture


class BaselineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.selected = select_records()

    def test_exact_source_and_workload(self):
        self.assertEqual(len(self.records), 54)
        row = min(self.selected.values(), key=lambda r: r["line"])
        self.assertEqual(row["line"], 2)
        self.assertEqual(row["source_model"], "pynq_1x16_i8w8a32_15_15_18_17")
        self.assertEqual(row["search_budget"], "unknown")
        wrong_resolution = deepcopy(row["workload"])
        wrong_resolution[1][1][2:4] = [224, 224]
        self.assertNotIn(key(wrong_resolution), self.selected)

    def test_config_entity_not_only_index(self):
        row = min(self.selected.values(), key=lambda r: r["line"])
        config = autotvm.task.ConfigEntity.from_json_dict(row["complete_config_entity"])
        self.assertEqual(assert_effective_config(row, config), row["complete_config_entity"])
        changed = deepcopy(row["complete_config_entity"])
        for entity in changed["entity"]:
            if entity[0] == "oc_nthread":
                entity[2] = 1
        self.assertEqual(changed["index"], row["complete_config_entity"]["index"])
        with self.assertRaisesRegex(ValueError, "EFFECTIVE_MISMATCH"):
            assert_effective_config(row, autotvm.task.ConfigEntity.from_json_dict(changed))
        with self.assertRaisesRegex(ValueError, "FALLBACK"):
            assert_effective_config(row, SimpleNamespace(is_fallback=True))

    def test_roundtrip_keeps_enclosing_globalvar(self):
        gv = tvm.ir.GlobalVar("kept_global")
        original = tvm.IRModule({gv: fixture()})
        loaded = tvm.ir.load_json(tvm.ir.save_json(original))
        self.assertFalse(gv.same_as(loaded.get_global_var("kept_global")))
        restored = restore_external_global_bindings(original, loaded)
        self.assertTrue(gv.same_as(restored.get_global_var("kept_global")))
        self.assertTrue(tvm.ir.structural_equal(original, restored, map_free_vars=False))

    def test_roundtrip_changed_function_rejected(self):
        original = tvm.IRModule({"kept_global": fixture(6)})
        changed = tvm.IRModule({"kept_global": fixture(5)})
        with self.assertRaisesRegex(ValueError, "CHANGED_CONTENT"):
            restore_external_global_bindings(original, changed)

    def test_llvm_normalization_keeps_identity_graph_and_fields(self):
        original = '!1 = !{!"0xabc.w4.b0", !2, i64 0}\n!2 = !{!"0xabc", !3, i64 0}\n'
        renamed = original.replace("0xabc", "0x123456")
        self.assertEqual(normalize_llvm(original), normalize_llvm(renamed))
        split_alias = renamed.replace('!"0x123456",', '!"0xffff",')
        self.assertNotEqual(normalize_llvm(original), normalize_llvm(split_alias))
        self.assertNotEqual(normalize_llvm(original), normalize_llvm(original.replace("i64 0", "i64 1")))
        call = 'call i32 @VTAUopPush(i32 0, i32 0, i32 2, i32 3, i32 4)'
        self.assertNotEqual(normalize_llvm(call), normalize_llvm(call.replace("i32 4", "i32 5")))


if __name__ == "__main__":
    unittest.main()
