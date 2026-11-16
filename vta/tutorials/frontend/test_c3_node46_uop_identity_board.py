"""Budget, ordering, failure accounting and actual-stream adoption gates."""
import json
import math
from copy import deepcopy
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from c3_node46_uop_identity_protocol import manifest, analyse, validate_results, transport_contract
from c3_tps_sync_release_support import accounting, validate_completion
from c3_tps_sync_release_local import capture_helper_call

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ledger=manifest('test', {'B':{'sha':'b'},'A':{'sha':'a'}})
        self.rows=[dict(r,status='PASS',target_calls=1,elapsed_ms=None if r['phase'].startswith('diagnostic_') else 10.) for r in self.ledger['calls']]

    def test_budget(self):
        self.assertEqual(len(self.rows),120)
        self.assertEqual(len(self.ledger['sessions']),20)
        self.assertEqual(Counter(r['program'] for r in self.rows),{'B':60,'A':60})
        self.assertEqual(sum(r['timed'] for r in self.rows),36)
        for p in 'BA':
            for rnd in range(1,7):
                self.assertEqual({r['seed'] for r in self.rows if r['program']==p and r['round']==rnd and r['timed']},{0,20250901,20260910})
        self.assertEqual(self.ledger['round_orders'],[['B','A'],['A','B']]*3)

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
            if row['timed']:row['elapsed_ms']=10+row['round']-(1 if row['program']=='A' else 0)
        d=analyse(self.ledger,self.rows)
        self.assertEqual(d['primary_comparison'],'A/B')
        self.assertEqual(d['pairwise']['A/B']['wins'],6)
        self.assertAlmostEqual(d['pairwise']['A/B']['Gain_percent'],median(100/(10+r) for r in range(1,7)))
        self.assertTrue(d['engineering_goal_met'])

    def test_negative_result_retained(self):
        for row in self.rows:
            if row['timed'] and row['program']=='A':row['elapsed_ms']=12
        d=analyse(self.ledger,self.rows)
        self.assertLess(d['pairwise']['A/B']['Gain_percent'],0)
        self.assertFalse(d['engineering_goal_met'])

    def test_transport_ports(self):
        t=transport_contract(self.ledger)
        self.assertEqual([r['port'] for r in t['sessions']],list(range(9600,9620)))
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

class Node46StorageTests(unittest.TestCase):
    def test_baseline_subset_and_generic_candidate_adoption(self):
        from c3_node46_uop_identity_arm import adopt
        adoption=adopt()
        self.assertEqual(adoption['status'],'NODE46_B_A_LOCAL_QUALIFICATION_ADOPTED')
        self.assertEqual(adoption['adopted_FSim_calls'],28)
        self.assertEqual(adoption['new_FSim_calls'],0)
        self.assertTrue(adoption['baseline_original_stop_preserved'])

    def test_real_binding_and_full_weight_region(self):
        from c3_node46_uop_identity_arm import LOCAL,BASELINE
        from c3_node46_uop_identity_board_support import verify_native_storage,tensor_sizes
        abi=json.loads((LOCAL/'prepare/result.json').read_text())['programs']['R']['abi']
        self.assertEqual(tensor_sizes(abi)[1],884736)
        cert=json.loads((LOCAL/'R_detailed/native_certificate_01.json').read_text())
        storage=[{'data_address':i+1,'byte_offset':0} for i in range(4)]
        self.assertTrue(verify_native_storage(cert,storage,abi))
        broken=deepcopy(cert)
        bases=broken['tensor_domain_dram_bases']
        w=next(r for r in bases if r['tensor']==1)
        x=next(r for r in bases if r['tensor']==0)
        x['dram_base']=(w['dram_base']*256+400000)//16
        with self.assertRaisesRegex(ValueError,'OVERLAP'):verify_native_storage(broken,storage,abi)
        wrong=deepcopy(abi);wrong[1]['shape'][0]=4
        with self.assertRaisesRegex(ValueError,'SIZE_CHANGED'):tensor_sizes(wrong)

    def test_late_projection_and_initializer_binding_contract(self):
        from c3_node46_uop_identity_arm import LOCAL,BASELINE
        from c3_uop_identity_method import llvm_signature_provenance
        from c3_tps_aggregation_prepare import compare_initializer_bindings
        source=LOCAL/'prepare/R'
        llvm=(source/'module.ll').read_text()
        self.assertIn('PASSED',llvm_signature_provenance(llvm,json.loads((source/'late_projection.json').read_text()))['status'])
        contract=json.loads((source/'alu_initializer_contract.json').read_text())
        self.assertIn('EQUAL',compare_initializer_bindings(llvm,llvm,contract,contract)['status'])
        wrong=deepcopy(contract);wrong['bindings'][0]['signature_bytes']=123
        with self.assertRaises(ValueError):compare_initializer_bindings(llvm,llvm,contract,wrong)
        broken=llvm.replace(' = and i32 ', ' = add i32 ')
        with self.assertRaises(ValueError):llvm_signature_provenance(broken,json.loads((source/'late_projection.json').read_text()))

if __name__=='__main__':unittest.main()
