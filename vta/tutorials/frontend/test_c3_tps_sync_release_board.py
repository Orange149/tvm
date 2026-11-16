"""Budget, ordering, failure accounting and actual-stream adoption gates."""
import json
import math
from copy import deepcopy
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from c3_tps_sync_release_protocol import manifest, analyse, validate_results, transport_contract
from c3_tps_sync_release_support import accounting, validate_completion
from c3_tps_sync_release_local import capture_helper_call

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ledger=manifest('test', {'R':{'sha':'r'},'S':{'sha':'s'}})
        self.rows=[dict(r,status='PASS',target_calls=1,elapsed_ms=None if r['phase'].startswith('diagnostic_') else 10.) for r in self.ledger['calls']]

    def test_budget(self):
        self.assertEqual(len(self.rows),120)
        self.assertEqual(len(self.ledger['sessions']),20)
        self.assertEqual(Counter(r['program'] for r in self.rows),{'R':60,'S':60})
        self.assertEqual(sum(r['timed'] for r in self.rows),36)
        for p in 'RS':
            for rnd in range(1,7):
                self.assertEqual({r['seed'] for r in self.rows if r['program']==p and r['round']==rnd and r['timed']},{0,20250901,20260910})
        self.assertEqual(self.ledger['round_orders'],[['R','S'],['S','R']]*3)

    def test_rotations_and_no_implicit_calls(self):
        for session in self.ledger['sessions']:
            rows=[r for r in self.rows if r['session_id']==session['session_id']]
            self.assertEqual(len(rows),6)
            if session['phase']!='qualification':
                self.assertEqual([r['seed'] for r in rows[:3]],[r['seed'] for r in rows[3:]])
        self.assertEqual(self.ledger['implicit_calls_allowed'],0)

    def test_missing_duplicate_reordered(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[-1]], [self.rows[1],self.rows[0]]+self.rows[2:], [self.rows[0]]*120):
            with self.assertRaises(ValueError):validate_results(self.ledger,rows)

    def test_identity_and_nonfinite(self):
        for update in ({'program_identity':{}},{'elapsed_ms':math.nan},{'elapsed_ms':math.inf},{'elapsed_ms':0},{'target_calls':2},{'status':'FAIL'}):
            rows=deepcopy(self.rows);rows[0].update(update)
            with self.assertRaises(ValueError):validate_results(self.ledger,rows)

    def test_only_formal_paired_ratios(self):
        from statistics import median
        for row in self.rows:
            if row['timed']:row['elapsed_ms']=10+row['round']-(1 if row['program']=='S' else 0)
        d=analyse(self.ledger,self.rows)
        self.assertEqual(d['primary_comparison'],'S/R')
        self.assertEqual(d['pairwise']['S/R']['wins'],6)
        self.assertAlmostEqual(d['pairwise']['S/R']['Gain_percent'],median(100/(10+r) for r in range(1,7)))
        self.assertTrue(d['engineering_goal_met'])

    def test_negative_result_retained(self):
        for row in self.rows:
            if row['timed'] and row['program']=='S':row['elapsed_ms']=12
        d=analyse(self.ledger,self.rows)
        self.assertLess(d['pairwise']['S/R']['Gain_percent'],0)
        self.assertFalse(d['engineering_goal_met'])

    def test_transport_ports(self):
        t=transport_contract(self.ledger)
        self.assertEqual([r['port'] for r in t['sessions']],list(range(9400,9420)))
        self.assertEqual(t['rpc_connect_attempts_per_session'],1)
        bad=deepcopy(self.ledger);bad['sessions'].pop()
        with self.assertRaises(ValueError):transport_contract(bad)

class AccountingTests(unittest.TestCase):
    def test_throw_with_stale_metadata_is_not_a_completed_call(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'session_01';folder.mkdir()
            (folder/'call_01_begin.json').write_text('{}')
            def fail():raise RuntimeError('target failed')
            def persist(d):(folder/'call_01_execution.json').write_text(json.dumps(d))
            with self.assertRaisesRegex(RuntimeError,'target failed'):
                capture_helper_call(1,fail,lambda:0,lambda:'{}',lambda:'{}',persist)
            a=accounting(Path(tmp))
            self.assertEqual(a['completed_calls']['exact'],0)
            self.assertIsNone(a['started_calls']['exact'])
            self.assertEqual(a['started_calls']['upper_bound'],1)

    def test_completion_kind_contract(self):
        d={'helper_returned':True,'call_index':1,'counts':1,'metadata':{
            'schema':'c3_official_window_once_helper_v1','started_calls':1,'completed_calls':1,
            'packed_calls_per_invocation':1,'adds_synchronize':False,'parameter_count':4,'last_kind':'timer'}}
        self.assertTrue(validate_completion(d,1,'timer'))
        with self.assertRaises(ValueError):validate_completion(d,1,'diagnostic')
        d['metadata']={}
        with self.assertRaises(ValueError):validate_completion(d,1,'timer')

class RealEvidenceTests(unittest.TestCase):
    def test_sealed_actual_pair_without_target_execution(self):
        import tvm
        from c3_tps_sync_release_arm import LOCAL
        from c3_tps_sync_release_method import certify_native_pair
        from c3_official_window_method import audit_function
        from c3_tps_window_progress import LANES
        from c3_tps_sync_release_support import verify_native_storage
        from c3_tps_aggregation_capacity import organization
        events=[];texts=[];certs=[]
        for p,folder in [('R','prepare/R'),('S','S_program')]:
            parent=tvm.ir.load_json((LOCAL/folder/'parent.json').read_text())
            function=parent[list(parent.functions)[0]]
            events.append(audit_function(function)['events'])
            texts.append((LOCAL/f'{p}_detailed/native_call_01.txt').read_text())
            certs.append(json.loads((LOCAL/f'{p}_detailed/native_certificate_01.json').read_text()))
            self.assertEqual(organization(texts[-1], certs[-1])['serialized_uop_bytes'], 8192)
            self.assertTrue(verify_native_storage(certs[-1],[{'data_address':i+1,'byte_offset':0} for i in range(4)]))
        result=certify_native_pair(texts[0],certs[0],texts[1],certs[1],events[0],events[1],
                                  {1:512,2:512,3:512},{lane:1024 for lane in LANES})
        self.assertIn('status',result)
        with self.assertRaisesRegex(ValueError,'NATIVE_DUMP_IDENTITY_CHANGED'):
            certify_native_pair(texts[0]+'x',certs[0],texts[1],certs[1],events[0],events[1],
                                {1:512,2:512,3:512},{lane:1024 for lane in LANES})

if __name__=='__main__':unittest.main()
