"""The ARM gate must reject reordered bodies even when their multiset matches."""
import unittest

from c3_tps_aggregation_prepare import compare_initializer_bindings


def llvm(values):
    return '\n'.join(
        f'define internal i32 @__tvm_static_init_lambda{i}(ptr %closure) {{\n'
        f'entry:\n  call void @VTAUopPush(i32 0, i32 0, i32 {value})\n'
        '  ret i32 0\n}' for i, value in enumerate(values))


def contract(prefix='sim'):
    return {'bindings': [
        {'source_scope_ordinal': i, 'handle': f'@{prefix}_handle{i}',
         'initializer': f'@__tvm_static_init_lambda{i}', 'dispatch': 'VTAPushGEMMOp',
         'signature_bytes': 0, 'dispatch_call_sites': 1,
         'mode_reset_opcode_use_imm_imm': [0, 0, 0, 0, 0]} for i in range(2)]}


class InitializerBindingTest(unittest.TestCase):
    def test_bound_correspondence_accepts_handle_renaming(self):
        result = compare_initializer_bindings(llvm([3, 7]), llvm([3, 7]), contract(), contract('arm'))
        self.assertEqual(result['status'], 'PER_TIR_SCOPE_INITIALIZER_AND_CACHE_SHARING_EQUAL')
        self.assertFalse(result['unordered_multiset_used_as_proof'])

    def test_equal_body_multiset_with_swapped_scopes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'BOUND_INITIALIZER_BODY:0'):
            compare_initializer_bindings(llvm([3, 7]), llvm([7, 3]), contract(), contract('arm'))

    def test_signature_change_is_rejected_even_with_identical_bodies(self):
        changed = contract('arm')
        changed['bindings'][1]['signature_bytes'] = 4
        with self.assertRaisesRegex(ValueError, 'SCOPE_CACHE_CONTRACT:1'):
            compare_initializer_bindings(llvm([3, 7]), llvm([3, 7]), contract(), changed)

    def test_handle_alias_is_rejected_even_with_identical_bodies(self):
        changed = contract('arm')
        changed['bindings'][1]['handle'] = changed['bindings'][0]['handle']
        with self.assertRaisesRegex(ValueError, 'CACHE_SHARING_RELATION:handle'):
            compare_initializer_bindings(llvm([3, 3]), llvm([3, 3]), contract(), changed)


if __name__ == '__main__':
    unittest.main()
