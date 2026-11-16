"""Pure counter/storage negative cases; no RPC connection or target call."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from c3_yolo416_board_session import metadata_gate
from c3_yolo416_board_execute import accounting,warm_contract
from c3_yolo416_comparison_protocol import manifest
from test_c3_yolo416_comparison_protocol import pool
from c3_yolo416_operator_board_adapter import physical_storage


class BoardGateTests(unittest.TestCase):
    def evidence(self,kind='operator'):
        return {'counts':1,'helper_returned':True,'metadata':{
            'started_calls':1,'completed_calls':1,'packed_calls_per_invocation':1,
            'parameter_count':4 if kind=='operator' else 0,'adds_synchronize':False,
            'last_kind':'timer','schema':'c3_official_window_once_helper_v1' if kind=='operator'
            else 'c3_yolo416_graph_once_v1'}}

    def test_exact_one_fused_or_graph_invocation(self):
        for kind in ('operator','graph'):metadata_gate(self.evidence(kind),1,kind,False)
        for field,value in [('completed_calls',0),('started_calls',2),('parameter_count',True),
                            ('packed_calls_per_invocation',True),('adds_synchronize',True),
                            ('last_kind','diagnostic')]:
            e=self.evidence();e['metadata'][field]=value
            with self.assertRaises(ValueError):metadata_gate(e,1,'operator',False)
        e=self.evidence();e['helper_returned']=False
        with self.assertRaises(ValueError):metadata_gate(e,1,'operator',False)

    def test_partial_records_never_count_as_certified(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);u=root/'session_001';u.mkdir()
            (u/'call_01_begin.json').write_text('{')
            result=accounting(root)
            self.assertEqual(result['registered_attempts'],1);self.assertEqual(result['certified'],0)
            self.assertEqual(result['helper_completed_observed_lower_bound'],0)
            self.assertEqual(result['execution_unresolved_units'],['session_001'])
            self.assertEqual(len(result['invalid_records']),1)
            (u/'call_01_begin.json').write_text(json.dumps({'position':1,'call_index':1,'object_id':'x','variant':'TH'}))
            (u/'call_01_execution.json').write_text(json.dumps(self.evidence()))
            (u/'call_01_end.json').write_text('{')
            result=accounting(root)
            self.assertEqual(result['helper_completed_observed_lower_bound'],1)
            self.assertEqual(result['certified'],0)

    def test_cumulative_counts_not_summed_per_call(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);u=root/'session_001';u.mkdir()
            for i in (1,2):
                e=self.evidence();e['counts']=i;e['metadata']['started_calls']=i;e['metadata']['completed_calls']=i
                (u/f'call_{i:02d}_begin.json').write_text(json.dumps({'call_index':i,'position':i,'object_id':'x','variant':'TH'}))
                (u/f'call_{i:02d}_execution.json').write_text(json.dumps(e))
                (u/f'call_{i:02d}_end.json').write_text('{"status":"PASS"}')
            r=accounting(root)
            self.assertEqual(r['helper_completed_observed_lower_bound'],2);self.assertEqual(r['certified'],2)

    def test_physical_ranges_use_each_memory_type_units(self):
        abi=[{'shape':shape,'dtype':dtype} for shape,dtype in
             (([16],'int8'),([256],'int8'),([16],'int32'),([16],'int8'))]
        storage=[{'data_address':i+1,'byte_offset':0,'device_type':12} for i in range(4)]
        native={'tensor_domain_dram_bases':[{'tensor':i,'domain':domain,'dram_base':base}
            for i,domain,base in ((0,2,256),(1,1,32),(2,3,192),(3,4,1024))]}
        self.assertEqual(physical_storage(native,storage,abi)['intervals'],
                         [(4096,4112),(8192,8448),(12288,12352),(16384,16400)])
        bad=deepcopy(native);bad['tensor_domain_dram_bases'][1]['dram_base']=16
        with self.assertRaises(ValueError):physical_storage(bad,storage,abi)
        alias=deepcopy(storage);alias[1]['data_address']=alias[0]['data_address']
        with self.assertRaises(ValueError):physical_storage(native,alias,abi)

    def test_warm_path_is_own_program_rotation_and_all_six_positions(self):
        m=manifest('a',pool(0));session=next(s for s in m['sessions']
            if s['phase']=='formal' and s['round']==2 and s['variant']=='TH')
        submit={'schema':'c3_official_window_submit_ledger_v1','mode':'detailed','overflow':False,
            'attempted_submits':1,'recorded_submits':1,'submits':[{'ordinal':1,'completion':True,
                'timeout':0,'insns':10,'insn_bytes':160,'serialized_uop_bytes':16,'reason':'explicit_sync'}]}
        prior=[dict(r,helper={'submits':submit}) for r in m['calls'] if r['phase'].startswith('diagnostic_')]
        contract=warm_contract(session,prior)
        self.assertEqual([r['position'] for r in contract['calls']],list(range(1,7)))
        self.assertEqual([r['seed'] for r in contract['calls']],[20250901,20260910,0]*2)
        with self.assertRaises(ValueError):warm_contract(dict(session,variant='other'),prior)
        own=[r for r in prior if r['variant']=='TH' and r['rotation']==1]
        with self.assertRaises(ValueError):warm_contract(session,prior+[own[0]])


if __name__=='__main__':unittest.main()
