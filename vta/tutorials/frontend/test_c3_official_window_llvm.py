"""Offline LLVM gates; never compile or invoke a target."""
import json
from pathlib import Path
import re
import unittest

import tvm
from tvm import tir
import vta  # Registers TIR intrinsics; no simulator or target is invoked.

from c3_official_window_llvm import llvm_alu_contract


def fixture():
    scopes = []
    for imm in (0, -127):
        body = tir.Evaluate(tir.call_extern("int32", "VTAUopPush", 1, 0, 0, 0, 0, 1, 1, imm))
        scopes.append(tir.AttrStmt(tir.const(0), "coproc_uop_scope", tir.StringImm("VTAPushALUOp"), body))
    function = tir.PrimFunc([], tir.SeqStmt(scopes))
    llvm = """@__tvm_static_handle = private global ptr null, align 8
@__tvm_static_handle.2 = private global ptr null, align 8
define i32 @main() {
entry:
  %0 = call i32 @VTAPushALUOp(ptr @__tvm_static_handle, ptr @__tvm_static_init_lambda, ptr null, i32 0)
  %1 = call i32 @VTAPushALUOp(ptr @__tvm_static_handle.2, ptr @__tvm_static_init_lambda.1, ptr null, i32 0)
  ret i32 0
}
define private i32 @__tvm_static_init_lambda(ptr %closure) {
entry:
  %0 = call i32 @VTAUopPush(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0)
  ret i32 0
}
define private i32 @__tvm_static_init_lambda.1(ptr %closure) {
entry:
  %0 = call i32 @VTAUopPush(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 -127)
  ret i32 0
}
"""
    return function, llvm


def swapped_max_immediates(llvm):
    zero = r"(@VTAUopPush\(i32 1, i32 0,[^\n]*?, i32 1, i32 1, i32 )0(\))"
    negative = r"(@VTAUopPush\(i32 1, i32 0,[^\n]*?, i32 1, i32 1, i32 )-127(\))"
    changed, nzero = re.subn(zero, r"\g<1>SWAP_SENTINEL\2", llvm)
    changed, nnegative = re.subn(negative, r"\g<1>0\2", changed)
    if nzero != nnegative or nzero == 0:
        raise AssertionError("fixture must have both MAX immediates")
    return changed.replace("SWAP_SENTINEL", "-127")


class StaticInitializerBinding(unittest.TestCase):
    def test_each_kernel_is_bound(self):
        function, llvm = fixture()
        result = llvm_alu_contract(function, llvm)
        self.assertEqual(result["kernel_scopes"], 2)
        self.assertEqual([r["mode_reset_opcode_use_imm_imm"][-1] for r in result["bindings"]], [0, -127])

    def test_max_immediate_swap_preserves_multiset_but_fails(self):
        function, llvm = fixture()
        changed = swapped_max_immediates(llvm)
        def old_multiset(text):
            return sorted(re.findall(r"@VTAUopPush\(([^)]*)\)", text))
        self.assertEqual(old_multiset(llvm), old_multiset(changed))
        with self.assertRaisesRegex(ValueError, "AUXILIARY_FIELDS_DIFFER"):
            llvm_alu_contract(function, changed)

    def test_callback_exchange_fails(self):
        function, llvm = fixture()
        changed = llvm.replace("ptr @__tvm_static_init_lambda,", "ptr @CALLBACK_SWAP,").replace(
            "ptr @__tvm_static_init_lambda.1,", "ptr @__tvm_static_init_lambda,").replace(
            "ptr @CALLBACK_SWAP,", "ptr @__tvm_static_init_lambda.1,")
        with self.assertRaisesRegex(ValueError, "AUXILIARY_FIELDS_DIFFER"):
            llvm_alu_contract(function, changed)

    def test_loop_peeling_duplicate_calls_do_not_change_scope_order(self):
        function, llvm = fixture()
        duplicate = "  %2 = call i32 @VTAPushALUOp(ptr @__tvm_static_handle, ptr @__tvm_static_init_lambda, ptr null, i32 0)\n"
        changed = llvm.replace("  ret i32 0\n}", duplicate + "  ret i32 0\n}", 1)
        result = llvm_alu_contract(function, changed)
        self.assertEqual(result["dispatch_call_sites"], 3)
        self.assertEqual(result["bindings"][0]["dispatch_call_sites"], 2)

    def test_one_handle_cannot_change_callback(self):
        function, llvm = fixture()
        duplicate = "  %2 = call i32 @VTAPushALUOp(ptr @__tvm_static_handle, ptr @__tvm_static_init_lambda.1, ptr null, i32 0)\n"
        changed = llvm.replace("  ret i32 0\n}", duplicate + "  ret i32 0\n}", 1)
        with self.assertRaisesRegex(ValueError, "CHANGES_INITIALIZER"):
            llvm_alu_contract(function, changed)

    def test_extra_missing_handle_or_push_rejected(self):
        function, llvm = fixture()
        with self.assertRaisesRegex(ValueError, "HANDLE_SOURCE_SCOPE_CENSUS"):
            llvm_alu_contract(function, "@__tvm_static_handle.extra = private global ptr null\n" + llvm)
        missing = llvm.replace("@__tvm_static_handle.2 = private global ptr null, align 8\n", "")
        with self.assertRaisesRegex(ValueError, "HANDLE_SOURCE_SCOPE_CENSUS"):
            llvm_alu_contract(function, missing)
        removed = re.sub(r"^.*@VTAUopPush.*\n", "", llvm, count=1, flags=re.M)
        with self.assertRaisesRegex(ValueError, "UOP_PUSH_CENSUS"):
            llvm_alu_contract(function, removed)

    def test_dispatch_signature_nonconstant_fields_rejected(self):
        function, llvm = fixture()
        with self.assertRaisesRegex(ValueError, "DISPATCH_DIFFERS"):
            llvm_alu_contract(function, llvm.replace("@VTAPushALUOp(", "@VTAPushGEMMOp(", 1))
        with self.assertRaisesRegex(ValueError, "SIGNATURE_DIFFERS"):
            llvm_alu_contract(function, llvm.replace("ptr null, i32 0)", "ptr null, i32 4)", 1))
        with self.assertRaisesRegex(ValueError, "CACHE_SIGNATURE"):
            llvm_alu_contract(function, llvm.replace("ptr null, i32 0)", "ptr null, i32 8)", 1))
        with self.assertRaisesRegex(ValueError, "UOP_MODE_IMMEDIATE"):
            llvm_alu_contract(function, llvm.replace("i32 1, i32 1, i32 -127)", "i32 1, i32 1, i32 %unknown)", 1))

    def test_comments_do_not_create_fake_calls(self):
        function, llvm = fixture()
        commented = "; call i32 @VTAPushALUOp(ptr @bad, ptr @bad, ptr null, i32 0)\n" + llvm
        self.assertEqual(llvm_alu_contract(function, commented)["dispatch_call_sites"], 2)

    def test_pointer_expression_and_unbound_push_rejected(self):
        function, llvm = fixture()
        pointer = llvm.replace("ptr @__tvm_static_handle,", "ptr getelementptr (i8, ptr @__tvm_static_handle, i32 1),", 1)
        with self.assertRaisesRegex(ValueError, "STATIC_INIT_POINTER"):
            llvm_alu_contract(function, pointer)
        stray = "  %99 = call i32 @VTAUopPush(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0)\n"
        with self.assertRaisesRegex(ValueError, "OUTSIDE_BOUND_INITIALIZER"):
            llvm_alu_contract(function, llvm.replace("  ret i32 0\n}", stray + "  ret i32 0\n}", 1))

    def test_real_resnet18_final_tir_and_llvm_then_negative_swap(self):
        run = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261014_c3_official_valid_window_run01/local_final01"
        descriptor = run / "resnet18_224_node012_descriptor.json"
        if not descriptor.is_file():
            self.skipTest("real local capture has not been produced")
        row = json.loads(descriptor.read_text())
        module_path = Path(row["final_module"])
        if not module_path.is_absolute():
            module_path = Path(__file__).resolve().parents[3] / module_path
        module = tvm.ir.load_json(module_path.read_text())
        function = module[row["global_var"]]
        llvm = (run / row["object_id"] / "detailed/module.ll").read_text()
        result = llvm_alu_contract(function, llvm)
        self.assertEqual(result["kernel_scopes"], 8)
        self.assertEqual(result["dispatch_call_sites"], 16)
        self.assertEqual(result["bindings"][3]["mode_reset_opcode_use_imm_imm"], [1, 0, 1, 1, 0])
        self.assertEqual(result["bindings"][-1]["mode_reset_opcode_use_imm_imm"], [1, 0, 1, 1, -127])
        with self.assertRaisesRegex(ValueError, "AUXILIARY_FIELDS_DIFFER"):
            llvm_alu_contract(function, swapped_max_immediates(llvm))


if __name__ == "__main__":
    unittest.main()
