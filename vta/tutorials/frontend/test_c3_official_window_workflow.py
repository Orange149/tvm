"""Reject standalone ABI drift before a target can be invoked."""
import unittest
from c3_official_window_workflow import execution_identity, llvm_cache_contract, native_certificate_for_storage
import json


class StandaloneABI(unittest.TestCase):
    def test_exact_large_dependency_bitmap_serializes_without_decimal_limit(self):
        large = (1 << 20000) | (1 << 112) | 1
        stored = native_certificate_for_storage({"ancestors": [0, large], "status": "PASS"})
        reloaded = json.loads(json.dumps(stored))
        self.assertEqual([int(v, 16) for v in reloaded["ancestors"]], [0, large])

    def test_alu_8byte_closure_rejected_even_if_gemm_4byte(self):
        llvm = ('%x = call i32 @VTAPushGEMMOp(ptr @h0, ptr @init0, ptr %sig, i32 4)\n'
                '%y = call i32 @VTAPushALUOp(ptr @h1, ptr @init1, ptr %sig2, i32 8)')
        with self.assertRaisesRegex(ValueError, "COMPILED_KERNEL_CACHE_SIGNATURE"):
            llvm_cache_contract(llvm)
        repaired = llvm.replace("i32 8)", "i32 4)")
        self.assertEqual(len(llvm_cache_contract(repaired)["sites"]), 2)

    def test_execution_identity_keeps_requests_and_cache_sharing(self):
        abi = [{"kind": "buffer", "shape": [1, 16], "dtype": "int8"}]
        first = [{"kind": "kernel", "scope": 0, "signature": [],
                  "cache_identity": {"scope": 0, "signature": []},
                  "source_scope_sha256": "temporary_name", "uops": []}]
        same = [{**first[0], "source_scope_sha256": "renamed"}]
        self.assertEqual(execution_identity(first, abi), execution_identity(same, abi))
        changed = [{**first[0], "cache_identity": {"scope": 1, "signature": []}}]
        self.assertNotEqual(execution_identity(first, abi), execution_identity(changed, abi))
        self.assertNotEqual(execution_identity(first, abi), execution_identity(first + [{"kind": "sync"}], abi))


if __name__ == "__main__":
    unittest.main()
