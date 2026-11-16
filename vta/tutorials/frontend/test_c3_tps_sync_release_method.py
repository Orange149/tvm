"""Host-only TPS release contracts; no build or target invocation occurs."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import tvm
from tvm import tir
import vta  # register VTA intrinsics before loading a frozen parent

from c3_tps_sync_release_method import transform, verify_event_delta, certify_joint
from c3_official_window_method import _limits, _trace
from c3_valid_window import statement_sequence
from c3_sync_release_audit import Graph, later_weight_target
from c3_five_tradeoff_commands import decode_submissions


HERE = Path(__file__).resolve().parent
SEALED = HERE / 'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261014_c3_tps_window_complete_run03/TPS_B_detailed'


def dep(kind, source, target):
    return tir.Evaluate(tir.call_intrin('int32', 'tir.vta.coproc_dep_' + kind, source, target))


def fixture(overwrite_input=False, extent=2, intervening_pop=False, prefix='arbitrary'):
    buffers = [tir.decl_buffer((16384,), dtype, name=prefix + name)
               for name, dtype in (('x', 'int8'), ('w', 'int8'), ('add', 'int32'), ('y', 'int8'))]
    params = [tir.Var(prefix + str(i), 'handle') for i in range(4)]
    iv = tir.IterVar(None, tir.Var(prefix + '_vta', 'int32'), tir.IterVar.ThreadIndex, 'vta')
    outer, reduction, dma, base = [tir.Var(prefix + '_' + name, 'int32')
                                  for name in ('output_group', 'reduce', 'dma', 'base')]

    def scope(stage, body):
        return tir.AttrStmt(iv, 'coproc_scope', stage, statement_sequence(body))

    def kernel(reset):
        body = tir.For(base, 0, 4, tir.ForKind.SERIAL,
                       tir.Evaluate(tir.call_intrin('int32', 'tir.vta.uop_push', 0, reset, base,
                                                   0 if reset else reduction * 4 + base, 0, 0, 0, 0)))
        return tir.AttrStmt(iv, 'coproc_uop_scope', tir.StringImm('VTAPushGEMMOp'), body)

    def load(buffer, source, target, domain):
        return tir.Evaluate(tir.call_extern('int32', 'VTALoadBuffer2D', tir.const(0, 'int32'),
                            buffer.data, source, 1, 1, 1, 0, 0, 0, 0, target, domain))

    inputs = tir.For(dma, 0, 4, tir.ForKind.SERIAL,
                    load(buffers[0], reduction * 4 + dma,
                         dma if overwrite_input else reduction * 4 + dma, 2))
    load_body = [dep('pop', 2, 1), inputs, load(buffers[1], reduction, 0, 1), dep('push', 1, 2)]
    if intervening_pop:
        load_body.insert(2, dep('pop', 2, 1))
    body = [scope(2, [kernel(1), dep('push', 2, 1)]),
            tir.For(reduction, 0, extent, tir.ForKind.SERIAL, statement_sequence([
                scope(1, load_body),
                scope(2, [dep('pop', 1, 2), kernel(0), dep('push', 2, 1)])])),
            dep('pop', 2, 1)]
    return tir.PrimFunc(params, tir.For(outer, 0, 2, tir.ForKind.SERIAL, statement_sequence(body)),
                        buffer_map=dict(zip(params, buffers))).with_attr('tir.noalias', True)


def events(function):
    return _trace(function, _limits(None))[0]


def kernel_nodes(function):
    nodes = []
    tir.stmt_functor.post_order_visit(function.body,
        lambda node: nodes.append(node) if isinstance(node, tir.AttrStmt) and node.attr_key == 'coproc_uop_scope' else None)
    return nodes


class TirReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.changed, cls.decisions = transform(cls.original)
        cls.before, cls.after = events(cls.original), events(cls.changed)

    def test_exact_two_second_iteration_moves(self):
        rows = self.decisions[0]['dynamic_moves']
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['current_mac_input_range'] == [0, 4] and
                            row['released_input_range'] == [4, 8] for row in rows))
        self.assertTrue(self.decisions[0]['first_iteration_wait_preserved'])

    def test_no_initializer_clone_or_cache_identity_change(self):
        before, after = kernel_nodes(self.original), kernel_nodes(self.changed)
        self.assertEqual(len(before), len(after))
        self.assertTrue(all(a.same_as(b) for a, b in zip(before, after)))
        self.assertEqual([e for e in self.before if e['kind'] == 'kernel'],
                         [e for e in self.after if e['kind'] == 'kernel'])

    def test_conditions_wrap_only_a_pop(self):
        conditions = []
        tir.stmt_functor.post_order_visit(self.changed.body,
            lambda node: conditions.append(node) if isinstance(node, tir.IfThenElse) else None)
        self.assertEqual(len(conditions), 2)
        for condition in conditions:
            self.assertIsInstance(condition.then_case, tir.Evaluate)
            self.assertEqual(condition.then_case.value.op.name, 'tir.vta.coproc_dep_pop')
            self.assertIsNone(condition.else_case)

    def test_variable_names_not_identity(self):
        _, decisions = transform(fixture(prefix='completely_different'))
        self.assertEqual(decisions[0]['dynamic_moves'], self.decisions[0]['dynamic_moves'])

    def test_active_input_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'NEW_INPUT_OVERWRITES_ACTIVE_MAC_INPUT'):
            transform(fixture(overwrite_input=True))

    def test_crossing_another_pop_not_supported(self):
        with self.assertRaisesRegex(ValueError, 'EXPECTED_ONE_STATIC_SHARED_INPUT_LOAD_SCOPE'):
            transform(fixture(intervening_pop=True))

    def test_three_iteration_reduction_rejected(self):
        with self.assertRaisesRegex(ValueError, 'REDUCTION_MUST_BE_TWO_SERIAL_ITERATIONS'):
            transform(fixture(extent=3))

    def test_cache_identity_change_rejected(self):
        after = deepcopy(self.after)
        next(e for e in after if e['kind'] == 'kernel')['cache_identity']['scope'] += 100
        with self.assertRaisesRegex(ValueError, 'CACHE_IDENTITY_CHANGED'):
            verify_event_delta(self.before, after)

    def test_missing_or_extra_dynamic_move_rejected(self):
        with self.assertRaisesRegex(ValueError, 'EXPECTED_EXACTLY_TWO_DYNAMIC_WAIT_MOVES'):
            verify_event_delta(self.before, self.before)

    def test_adjacent_notification_order_is_not_normalized_away(self):
        after = deepcopy(self.after)
        index = next(i for i in range(len(after) - 1) if after[i]['kind'] == 'push' and
                     after[i + 1]['kind'] == 'pop')
        after[index], after[index + 1] = after[index + 1], after[index]
        with self.assertRaisesRegex(ValueError, 'ORDERED_CONTROL_SEQUENCE_CHANGED'):
            verify_event_delta(self.before, after)

    def test_unconditional_startup_move_rejected(self):
        after = deepcopy(self.before)
        index = next(i for i, e in enumerate(after) if e['kind'] == 'pop' and e['pair'] == [2, 1])
        pop = after.pop(index)
        after.insert(index + 4, pop)
        with self.assertRaisesRegex(ValueError, 'ONLY_MAC_PRODUCED_WAIT_MAY_MOVE'):
            verify_event_delta(self.before, after)


class SealedParentReadOnlyTest(unittest.TestCase):
    def test_existing_parent_has_two_semantic_moves_without_native_number_matching(self):
        module = tvm.ir.load_json((SEALED / 'parent.json').read_text())
        original = next(iter(module.functions.values()))
        changed, decisions = transform(original)
        self.assertEqual(len(decisions[0]['dynamic_moves']), 2)
        for row in decisions[0]['dynamic_moves']:
            self.assertEqual(row['current_mac_input_range'], [0, 1024])
            self.assertEqual(row['released_input_range'], [1024, 2048])
            self.assertEqual(row['protected_weight_range'], [0, 144])
        self.assertTrue(all(a.same_as(b) for a, b in zip(kernel_nodes(original), kernel_nodes(changed))))


class JointNativeReadOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = decode_submissions((SEALED / 'native_call_01.txt').read_text())[0]
        cls.before_certificate = json.loads((SEALED / 'native_certificate_01.json').read_text())
        graph = Graph(cls.before)
        cls.moves = [(token, later_weight_target(graph, token)) for token in graph.tokens
                     if token['lane'] == [2, 1] and cls.before[token['producer']]['description'] == 'GEMM'
                     and 'reset_out: 0' in cls.before[token['producer']]['detail']
                     and later_weight_target(graph, token) is not None]
        assert len(cls.moves) == 2

    def proposal(self):
        """Graph fixture only: no dump or target-execution claim is created."""
        after, cert = deepcopy(self.before), deepcopy(self.before_certificate)
        for token, target in self.moves:
            old = token['consumer']
            a, b = list(after[old]['bits']), list(after[target]['bits'])
            a[1], b[1] = 0, 1
            after[old]['bits'], after[target]['bits'] = a, b
            control = next(c for c in cert['dependency_map'] if c['kind'] == 'pop' and
                           c['lane'] == [2, 1] and c['ordinal'] == token['ordinal'])
            control['instruction'] = target
        return after, cert

    def test_two_actual_dag_moves_are_checked_together(self):
        after, cert = self.proposal()
        result = certify_joint(self.before, self.before_certificate, after, cert)
        self.assertEqual(result['status'], 'JOINT_NATIVE_SYNC_RELEASE_PASSED')
        self.assertEqual(len(result['native_moves']), 2)
        self.assertGreater(result['original_ordered_physical_conflicts_retained'], 0)
        self.assertGreater(len(result['original_unordered_dead_read_conflicts']), 0)
        self.assertFalse(result['proves_performance'])

    def test_stale_native_dependency_binding_rejected(self):
        after, _ = self.proposal()
        with self.assertRaisesRegex(ValueError, 'NATIVE_DEPENDENCY_CERTIFICATE_BINDING_CHANGED'):
            certify_joint(self.before, self.before_certificate, after, self.before_certificate)

    def test_joint_physical_overlap_rejected(self):
        after, cert = self.proposal()
        old = self.moves[1][0]['consumer']
        material = next(m for m in cert['material_map'] if m['instruction'] == old)
        material['native']['sram'] = 0
        with self.assertRaisesRegex(ValueError, 'ORIGINAL_LIVE_MEMORY_ORDER_UNRESOLVED|NATIVE_MATERIAL_ACCESS_CHANGED'):
            certify_joint(self.before, self.before_certificate, after, cert)

    def test_additional_consumer_change_rejected(self):
        after, cert = self.proposal()
        token = next(t for t in Graph(self.before).tokens if t['lane'] == [2, 1] and
                     self.before[t['consumer']]['description'] == 'LOAD INP' and
                     'reset_out: 1' in self.before[t['producer']]['detail'])
        target = later_weight_target(Graph(self.before), token)
        a, b = list(after[token['consumer']]['bits']), list(after[target]['bits'])
        a[1], b[1] = 0, 1
        after[token['consumer']]['bits'], after[target]['bits'] = a, b
        next(c for c in cert['dependency_map'] if c['kind'] == 'pop' and c['lane'] == [2, 1]
             and c['ordinal'] == token['ordinal'])['instruction'] = target
        with self.assertRaisesRegex(ValueError, 'JOINT_MOVE_LOSES_PHYSICAL_ORDER|ONLY_MAC_PRODUCED_WAIT_MAY_MOVE'):
            certify_joint(self.before, self.before_certificate, after, cert)


if __name__ == '__main__':
    unittest.main()
