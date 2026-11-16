"""Zero-target tests for actual fallback policy and dispatch provenance."""
import unittest
from unittest.mock import patch
from c3_fallback_identity_local import observe_fallback, config_values, sources, EXPECTED


class FallbackTests(unittest.TestCase):
    def test_real_axu_fallback_query_and_original_schedule(self):
        import tvm, vta
        from tvm import te
        from vta.top import vta_conv2d
        for node in (34, 46):
            source = sources(node)
            actual, queries = [], []
            with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='axu5evb')):
                with tvm.target.Target(vta.get_env().target), observe_fallback(node, actual, queries):
                    x, w = [te.placeholder(shape, name=name, dtype='int8')
                            for shape, name in zip(source.SHAPES[:2], ('X', 'W'))]
                    out = vta_conv2d.conv2d_packed(x, w, (1,1), (1,1,1,1), (1,1), 'NCHW1n16c', 'int32')
                    tail = te.compute(out.shape, lambda *i: tvm.tir.min(tvm.tir.max(out[i], -127), 127).astype('int8'), name='tail', tag='broadcast')
                    vta_conv2d.schedule_conv2d_packed([tail])
            self.assertEqual(len(actual), 1)
            self.assertEqual(config_values(actual[0]['config']), EXPECTED[node])
            self.assertTrue(actual[0]['is_fallback'])
            self.assertTrue(any(q['workload'] == source.WORKLOAD and
                                q['config_object'] == actual[0]['config_object'] for q in queries))

    def test_real_relay_build_never_opens_tophub(self):
        import tvm, vta
        for node in (34,46):
            source=sources(node)
            relay=tvm.relay.parse(source.extract_relay((source.GRAPH/'relay_000.txt').read_text()))
            actual,queries=[],[]
            with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='axu5evb')):
                target=tvm.target.Target(vta.get_env().target,host='llvm')
                tvm.relay.backend.te_compiler.get().clear()
                with patch('tvm.autotvm.tophub.context',side_effect=AssertionError('history lookup')) as guard:
                    with observe_fallback(node,actual,queries) as context:
                        self.assertNotIsInstance(context,tvm.autotvm.FallbackContext)
                        with vta.build_config(disabled_pass={'AlterOpLayout','tir.CommonSubexprElimTIR'}):
                            library=tvm.relay.build(relay,target=target)
                self.assertEqual(guard.call_count,0)
                self.assertTrue(library)
                self.assertEqual(len(actual),1)
                self.assertEqual(config_values(actual[0]['config']),EXPECTED[node])

    def test_legacy_relay_root_fallback_reproduces_hidden_history(self):
        import tvm, vta
        source=sources(34)
        relay=tvm.relay.parse(source.extract_relay((source.GRAPH/'relay_000.txt').read_text()))
        with vta.Environment(dict(vta.get_env().cfg_dict,TARGET='axu5evb')):
            target=tvm.target.Target(vta.get_env().target,host='llvm')
            with tvm.autotvm.FallbackContext():
                with patch('tvm.autotvm.tophub.context',side_effect=AssertionError('history lookup')) as guard:
                    with self.assertRaisesRegex(AssertionError,'history lookup'):
                        tvm.relay.build(relay,target=target)
                    self.assertEqual(guard.call_count,1)

    def test_sim_default_is_not_axu_policy(self):
        import vta
        with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='sim')):
            with self.assertRaisesRegex(ValueError, 'AXU_FALLBACK_POLICY_REQUIRED'):
                with observe_fallback(34, [], []):
                    self.fail('wrong policy accepted')

    def test_real_fusion_tail_and_scope(self):
        for node, shift in ((34, 'right_shift(%57, 6'), (46, 'right_shift(%24, 8')):
            source = sources(node)
            text = source.extract_relay((source.GRAPH/'relay_000.txt').read_text())
            self.assertIn(shift, text)
            self.assertIn('a_min=-127f', text)
            self.assertIn('a_max=127f', text)
        with self.assertRaisesRegex(ValueError, 'FIXED_TWO_NODE_SCOPE'):
            sources(24)

    def test_explicit_dispatch_rejected_and_hook_restored(self):
        import vta
        from vta.top import vta_conv2d
        before = vta_conv2d._schedule_conv2d_packed_impl
        with vta.Environment(dict(vta.get_env().cfg_dict, TARGET='axu5evb')):
            with patch('vta.top.residency_dispatch.active_context', return_value=object()):
                with self.assertRaisesRegex(ValueError, 'EXPLICIT_ROUTE_FORBIDDEN'):
                    with observe_fallback(34, [], []):
                        self.fail('explicit route accepted')
            self.assertIs(vta_conv2d._schedule_conv2d_packed_impl, before)
            with self.assertRaisesRegex(RuntimeError, 'preflight'):
                with observe_fallback(34, [], []):
                    raise RuntimeError('preflight')
            self.assertIs(vta_conv2d._schedule_conv2d_packed_impl, before)


if __name__ == '__main__':
    unittest.main()
