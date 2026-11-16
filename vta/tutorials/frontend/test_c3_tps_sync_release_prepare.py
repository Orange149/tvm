"""Synthetic and sealed-artifact gates; no compilation or target invocation."""
import os
import json
from collections import Counter
import unittest

from c3_tps_sync_release_prepare import (
    HOLD, OFFICIAL, QUALIFIED, canonical_config, compiled_module_contract,
    complete_function, ordered_reductions,
)


def load(domain, address, origin):
    return {"kind": "load", "domain": domain, "cells": [(address, origin)]}


def reset(*destinations):
    return {"kind": "kernel", "mode_reset": [0, 1],
            "expanded": [[0, 1, dst, 0, 0, 0, 0, 0] for dst in destinations]}


def mac(dst, src, wgt):
    return {"kind": "kernel", "mode_reset": [0, 0],
            "expanded": [[0, 0, dst, src, wgt, 0, 0, 0]]}


def store(*addresses):
    return {"kind": "store", "cells": [(dst, (3, dst)) for dst in addresses]}


def sequence(events):
    return ordered_reductions(events, lambda event: event["expanded"], lambda event: event["cells"])


class PrepareContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TVM_LIBRARY_PATH"] = str(OFFICIAL / "runtime_local_final")
        import vta  # Register frozen VTA intrinsics before loading its parent IR.

    def setUp(self):
        self.prefix = [load(2, 0, (0, 10)), load(2, 1, (0, 11)),
                       load(1, 0, (1, 20)), reset(0, 1)]

    def test_loop_interleaving_preserves_each_acc_reduction(self):
        left = [mac(0, 0, 0), mac(0, 1, 0), mac(1, 0, 0), mac(1, 1, 0)]
        right = [mac(0, 0, 0), mac(1, 0, 0), mac(0, 1, 0), mac(1, 1, 0)]
        self.assertEqual(sequence(self.prefix + left + [store(0, 1)]),
                         sequence(self.prefix + right + [store(0, 1)]))

    def test_same_product_multiset_different_reduction_is_detected(self):
        first = sequence(self.prefix + [mac(0, 0, 0), mac(0, 1, 0), store(0)])
        second = sequence(self.prefix + [mac(0, 1, 0), mac(0, 0, 0), store(0)])
        self.assertEqual(sorted(first[(3, 0)]), sorted(second[(3, 0)]))
        self.assertNotEqual(first, second)

    def test_same_sram_address_with_different_version_is_visible(self):
        first = sequence(self.prefix + [mac(0, 0, 0), store(0)])
        second = sequence(self.prefix + [load(2, 0, (0, 99)), mac(0, 0, 0), store(0)])
        self.assertNotEqual(first, second)

    def test_missing_reset_and_duplicate_store_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "WITHOUT_RESET"):
            sequence(self.prefix[:3] + [mac(0, 0, 0), store(0)])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OBSERVATION"):
            sequence(self.prefix + [mac(0, 0, 0), store(0), store(0)])

    def test_config_index_is_not_semantics_but_schedule_fields_are(self):
        value = {"index": 4, "code_hash": None, "entity": [["tile_ci", "sp", [-1, 4]]]}
        other = {**value, "index": 99}
        self.assertEqual(canonical_config(value), canonical_config(other))
        other["entity"] = [["tile_ci", "sp", [-1, 8]]]
        self.assertNotEqual(canonical_config(value), canonical_config(other))
        self.assertEqual(value["index"], 4)

    def test_real_sealed_dictattrs_function_selection(self):
        import tvm
        module = tvm.ir.load_json((QUALIFIED / "TPS_B_detailed/parent.json").read_text())
        symbol = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
        gv, function = complete_function(module, symbol)
        self.assertEqual(str(gv.name_hint), symbol)
        self.assertEqual(str(function.attrs["global_symbol"]), symbol)
        self.assertEqual(len(function.params), 4)

    def test_sealed_tps_prepare_chain_semantics_reductions_and_initializer_binding(self):
        import tvm
        from c3_official_window_baseline import buffer_abi
        from c3_official_window_method import audit_function
        from c3_official_window_reference import from_workload, postprocess_from_output_terms, semantic_gates
        from c3_tps_aggregation_prepare import compare_initializer_bindings
        from c3_valid_window import certify_trace, dma_cells, expanded_uops

        directory = QUALIFIED / "TPS_B_detailed"
        symbol = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
        parent = tvm.ir.load_json((directory / "parent.json").read_text())
        final = tvm.ir.load_json((directory / "final.json").read_text())
        _, function = complete_function(parent, symbol)
        _, final_function = complete_function(final, symbol)
        abi = buffer_abi(function)
        self.assertEqual(abi, buffer_abi(final_function))
        self.assertEqual(abi, json.loads((directory / "reference_contract.json").read_text())["abi"])
        audit = audit_function(function)
        expected = json.loads((directory / "logical_certificate.json").read_text())
        self.assertEqual(audit["counts"], expected["baseline_counts"])
        semantic = certify_trace(audit["events"])
        workload = ["conv2d_packed.vta", ["TENSOR", abi[0]["shape"], "int8"],
                    ["TENSOR", abi[1]["shape"], "int8"], [1, 1], [1, 1, 1, 1],
                    [1, 1], "NCHW1n16c", "int32"]
        reference = from_workload(workload, abi, postprocess_from_output_terms(semantic["output_terms"], abi))
        self.assertEqual(semantic_gates(reference, semantic["output_terms"]),
                         json.loads((directory / "independent_semantics.json").read_text()))
        ordered = ordered_reductions(audit["events"], expanded_uops, dma_cells)
        self.assertEqual(len(ordered), 3136)
        for output, terms, _ in semantic["output_terms"]:
            self.assertEqual(len(ordered[tuple(output)]), 72)
            self.assertEqual(Counter(ordered[tuple(output)]),
                             Counter({(tuple(inp), tuple(wgt)): count for inp, wgt, count in terms}))
        # The independently archived minimal-mode parent must retain the exact
        # per-output sequence, not merely the same product multiplicities.
        minimal = QUALIFIED / "TPS_B_minimal"
        _, minimal_function = complete_function(tvm.ir.load_json((minimal / "parent.json").read_text()), symbol)
        minimal_audit = audit_function(minimal_function)
        self.assertEqual(ordered, ordered_reductions(minimal_audit["events"], expanded_uops, dma_cells))
        llvm = (directory / "module.ll").read_text()
        compiled = compiled_module_contract(final, symbol, llvm)
        self.assertEqual(compiled["abi"], abi)
        self.assertEqual(compiled["compiled_cache_contract"],
                         json.loads((directory / "compiled_cache_contract.json").read_text()))
        current_contract = compiled["alu_initializer_contract"]
        self.assertEqual(current_contract, json.loads((directory / "alu_initializer_contract.json").read_text()))
        comparison = compare_initializer_bindings(
            llvm, (minimal / "module.ll").read_text(), current_contract,
            json.loads((minimal / "alu_initializer_contract.json").read_text()))
        self.assertEqual(comparison["status"], "PER_TIR_SCOPE_INITIALIZER_AND_CACHE_SHARING_EQUAL")
        self.assertEqual(len(comparison["bindings"]), audit["counts"]["static_kernel_scopes"])

    def test_run02_serialized_module_owns_its_function_and_compiled_binding(self):
        import tvm
        from unittest.mock import patch

        run = HOLD / "20261016_c3_tps_sync_release_local_run02"
        directory = run / "prepare/O_rebuilt"
        symbol = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
        parent = tvm.ir.load_json((directory / "parent.json").read_text())
        final = tvm.ir.load_json((directory / "final.json").read_text())
        parent_gv, _ = complete_function(parent, symbol)
        final_gv, final_function = complete_function(final, symbol)
        self.assertEqual(str(parent_gv.name_hint), str(final_gv.name_hint))
        self.assertFalse(parent_gv.same_as(final_gv))
        # This is the exact failed lookup, not merely a similar synthetic key.
        with self.assertRaisesRegex(tvm.error.InternalError, "There is no definition"):
            final[parent_gv]
        self.assertTrue(final[final_gv].same_as(final_function))

        # Exercise the actual entry point used by O/R preparation and S's
        # initializer comparison. Neither compilation nor invocation is allowed.
        llvm = (directory / "module.ll").read_text()
        with patch("tvm.build", side_effect=AssertionError("read-only regression compiled")):
            compiled = compiled_module_contract(final, symbol, llvm)
            reloaded = tvm.ir.load_json(tvm.ir.save_json(final))
            self.assertFalse(reloaded.get_global_var(str(final_gv.name_hint)).same_as(final_gv))
            self.assertEqual(compiled_module_contract(reloaded, symbol, llvm), compiled)
        self.assertEqual(compiled["abi"], json.loads((directory / "reference_contract.json").read_text())["abi"])
        self.assertEqual(compiled["alu_initializer_contract"],
                         json.loads((run / "diagnosis.json").read_text())["initializer_contract"])
        with self.assertRaisesRegex(ValueError, "FULL_FUSION_SYMBOL"):
            compiled_module_contract(final, "wrong_symbol", llvm)

    def test_function_selection_rejects_wrong_symbol_and_missing_attrs(self):
        import tvm
        from tvm import tir
        module = tvm.ir.load_json((QUALIFIED / "TPS_B_detailed/parent.json").read_text())
        with self.assertRaisesRegex(ValueError, "FULL_FUSION_SYMBOL"):
            complete_function(module, "wrong_symbol")
        no_attrs = tvm.IRModule({"main": tir.PrimFunc([], tir.Evaluate(0))})
        with self.assertRaisesRegex(ValueError, "FULL_FUSION_SYMBOL"):
            complete_function(no_attrs, "main")

    def test_synthetic_decorated_instrument_accepted_by_pass_context(self):
        os.environ["TVM_LIBRARY_PATH"] = str(OFFICIAL / "runtime_local_final")
        import tvm

        @tvm.instrument.pass_instrument
        class SyntheticArchive:
            def __init__(self):
                self.entered = False

            def enter_pass_ctx(self):
                self.entered = True

        archive = SyntheticArchive()
        with tvm.transform.PassContext(instruments=[archive]):
            self.assertTrue(archive.entered)


if __name__ == "__main__":
    unittest.main()
