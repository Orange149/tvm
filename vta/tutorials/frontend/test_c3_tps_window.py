"""V4 fail-closed contracts and finite-queue counterexamples."""
import struct
import unittest

import tvm
from tvm import tir
import vta  # registers frozen intrinsics

from c3_tps_window_hardware import parse_bit
from c3_tps_window_method import migrate,acc_spacing
from c3_tps_window_progress import certify_progress,LANES
from c3_tps_window_native import _simulate
from c3_valid_window import statement_sequence
from c3_official_window_method import _trace, _limits


def insns(rows):
    return [{'index':i,'stage':s,'bits':b} for i,(s,b) in enumerate(rows)]


class FiniteProgressTest(unittest.TestCase):
    def cert(self,rows,capacity=1,limit=1_000_000):
        return certify_progress(insns(rows),{1:capacity,2:capacity,3:capacity},
                                {lane:capacity for lane in LANES},limit)

    def test_forward_progress_with_one_entry_instruction_fifos(self):
        r=self.cert([(1,(0,0,0,1)),(2,(1,0,0,0))])
        self.assertEqual(r['status'],'FINITE_FIFO_PROGRESS_PROVED')
        self.assertEqual(r['states_enumerated'],0)

    def test_two_ready_tokens_are_not_one_bit(self):
        with self.assertRaisesRegex(ValueError,'UNMATCHED'):
            self.cert([(1,(0,0,0,1)),(1,(0,0,0,1)),(2,(1,0,0,0))])

    def test_acyclic_graph_can_deadlock_in_finite_fetch_fifo(self):
        # C waits for the later L; fetch cannot reach L past the full C FIFO.
        rows=[(2,(1,0,0,0)),(2,(0,0,0,0)),(2,(0,0,0,0)),(1,(0,0,0,1))]
        r=self.cert(rows)
        self.assertEqual(r['status'],'REACHABLE_FINITE_FIFO_DEADLOCK')
        self.assertFalse(r['all_token_edges_forward'])

    def test_backward_edge_is_safe_with_sufficient_finite_capacity(self):
        r=self.cert([(2,(1,0,0,0)),(1,(0,0,0,1))],capacity=2)
        self.assertEqual(r['status'],'FINITE_FIFO_PROGRESS_PROVED')
        self.assertEqual(r['method'],'exhaustive_finite_state_exploration')

    def test_ready_release_cycle_rejected(self):
        r=self.cert([(1,(0,1,0,1)),(2,(1,0,1,0))],capacity=2)
        self.assertEqual(r['status'],'REACHABLE_FINITE_FIFO_DEADLOCK')

    def test_resource_limit_is_not_illegality(self):
        r=self.cert([(2,(1,0,0,0)),(1,(0,0,0,1))],capacity=2,limit=2)
        self.assertEqual(r['status'],'ANALYSIS_STATE_LIMIT')

    def test_token_backpressure_does_not_get_forward_theorem(self):
        r=self.cert([(1,(0,0,0,1)),(1,(0,0,0,1)),(2,(1,0,0,0)),(2,(1,0,0,0))])
        self.assertEqual(r['status'],'FINITE_FIFO_PROGRESS_PROVED')
        self.assertEqual(r['method'],'exhaustive_finite_state_exploration')


class BitstreamTest(unittest.TestCase):
    def bit(self,date=b'2026/10/14\0'):
        magic=bytes.fromhex('0ff00ff00ff00ff000')
        b=struct.pack('>H',len(magic))+magic+b'\0\1'
        for key,value in zip(b'abcd',[b'design\0',b'part\0',date,b'00:00:00\0']):
            b+=bytes([key])+struct.pack('>H',len(value))+value
        payload=b'\xff'*80+bytes.fromhex('aa995566')+b'\x20\0\0\0'
        return b+b'e'+struct.pack('>I',len(payload))+payload

    def test_header_dates_do_not_change_payload(self):
        a,pa=parse_bit(self.bit())
        b,pb=parse_bit(self.bit(b'2020/01/01\0'))
        self.assertEqual(pa,pb)
        self.assertNotEqual(a['whole_sha256'],b['whole_sha256'])

    def test_truncation_rejected(self):
        with self.assertRaisesRegex(ValueError,'TRUNCATED'):
            parse_bit(self.bit()[:-1])

    def test_trailing_data_rejected(self):
        with self.assertRaisesRegex(ValueError,'TRAILING'):
            parse_bit(self.bit()+b'\0')


class NativeNopTest(unittest.TestCase):
    def test_compute_nop_is_greedy_cutoff_but_not_serialized_payload(self):
        kernels=[{'identity':'a'},{'identity':'b'}]
        definitions={'a':[1],'b':[2]}
        plain,packed,*_= _simulate(kernels,definitions)
        blocked,packed_blocked,*_= _simulate(kernels,definitions,{1:1})
        self.assertEqual(packed,packed_blocked)
        self.assertEqual(len([i for i in plain if i['kind']=='load']),1)
        self.assertEqual(len([i for i in blocked if i['kind']=='load']),2)
        self.assertEqual([i['source'] for i in blocked if i['kind']=='load'],[0,1])


def dep(kind,a,b):
    return tir.Evaluate(tir.call_intrin('int32','tir.vta.coproc_dep_'+kind,a,b))


def bank(offset,count=8,extent=2,reset=0,notified=True):
    vta_var=tir.Var('vta','int32')
    iv=tir.IterVar(None,vta_var,tir.IterVar.ThreadIndex,'vta')
    i=tir.Var('base_i','int32')
    op=tir.Evaluate(tir.call_intrin('int32','tir.vta.uop_push',0,reset,i+offset,i,0,0,0,0))
    kernel=tir.AttrStmt(iv,'coproc_uop_scope',tir.StringImm('VTAPushGEMMOp'),statement_sequence([
        tir.Evaluate(tir.call_extern('int32','VTAUopLoopBegin',extent,0,0,0)),
        tir.For(i,0,count,tir.ForKind.SERIAL,op),
        tir.Evaluate(tir.call_extern('int32','VTAUopLoopEnd'))]))
    body=statement_sequence([dep('pop',1,2),kernel,dep('push',2,1)]) if notified else kernel
    return tir.AttrStmt(iv,'coproc_scope',2,body)


class MethodTest(unittest.TestCase):
    def func(self,stmts):
        return tir.PrimFunc([],statement_sequence(stmts)).with_attr('tir.noalias',True)

    def test_pair_does_not_require_eight_fragments_or_node_name(self):
        f=self.func([bank(0),bank(64)])
        n,nd=migrate(f,'N');g,gd=migrate(f,'G')
        self.assertEqual(len(nd),1);self.assertEqual(len(gd),1)
        self.assertEqual(gd[0]['joined_uops'],16)
        # These parser fixtures deliberately omit DMA/RESET; their events do
        # not claim full-function semantic qualification.
        before=_trace(f,_limits(None))[0];after=_trace(n,_limits(None))[0]
        self.assertEqual([e for e in before if e['kind']=='kernel'],
                         [e for e in after if e['kind']=='kernel'])
        self.assertEqual(len([e for e in _trace(g,_limits(None))[0] if e['kind']=='kernel']),1)

    def test_internal_reset_is_preserved(self):
        f=self.func([bank(0,reset=1),bank(64)])
        g,decisions=migrate(f,'G')
        self.assertFalse(decisions);self.assertTrue(tvm.ir.structural_equal(f,g))

    def test_first_N_keeps_one_lexical_initializer_site_per_bank(self):
        outer=tir.Var('outer_group','int32')
        f=self.func([tir.For(outer,0,2,tir.ForKind.SERIAL,statement_sequence([bank(0),bank(64)]))])
        n,_=migrate(f,'N',first_only=True)
        def occurrences(node):
            if isinstance(node,tir.AttrStmt):
                return 1 if node.attr_key=='coproc_uop_scope' else occurrences(node.body)
            if isinstance(node,tir.SeqStmt):return sum(occurrences(c) for c in node.seq)
            if isinstance(node,tir.For):return occurrences(node.body)
            if isinstance(node,tir.IfThenElse):
                return occurrences(node.then_case)+(occurrences(node.else_case) if node.else_case is not None else 0)
            return 0
        self.assertEqual(occurrences(n.body),2)
        old=_trace(f,_limits(None))[0];new=_trace(n,_limits(None))[0]
        self.assertEqual([e for e in old if e['kind']=='kernel'],[e for e in new if e['kind']=='kernel'])

    def test_incompatible_hardware_loop_is_preserved(self):
        f=self.func([bank(0),bank(64,extent=3)])
        g,decisions=migrate(f,'G')
        self.assertFalse(decisions);self.assertTrue(tvm.ir.structural_equal(f,g))

    def test_unknown_effect_stops_pair(self):
        f=self.func([bank(0),tir.Evaluate(tir.call_extern('int32','unknown_side_effect')),bank(64)])
        g,decisions=migrate(f,'G')
        self.assertFalse(decisions);self.assertTrue(tvm.ir.structural_equal(f,g))

    def test_capacity_is_checked_before_join(self):
        for total in (8191,8192,8193):
            f=self.func([bank(0,count=4096),bank(64,count=total-4096)])
            _g,decisions=migrate(f,'G')
            self.assertEqual(decisions[0]['status'],
                'LOCAL_DIAGNOSTIC_PROPOSAL' if total<=8192 else 'PRESERVE_BOUNDARY')

    def test_loop_carry_raw_violation_is_rejected(self):
        events=_trace(self.func([bank(0,count=2)]),_limits(None))[0]
        with self.assertRaisesRegex(ValueError,'PIPELINE_DISTANCE'):
            acc_spacing(events,4)

    def test_bank_loop_requires_provenance(self):
        axis=tir.Var('any_extent_two','int32')
        f=self.func([tir.For(axis,0,2,tir.ForKind.SERIAL,bank(axis*64))])
        g,decisions=migrate(f,'G')
        self.assertTrue(tvm.ir.structural_equal(f,g))
        self.assertEqual(decisions[0]['status'],'SUPPORT_UNRESOLVED')


if __name__=='__main__':unittest.main()
