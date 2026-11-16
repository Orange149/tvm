from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from c3_yolo416_graph_local import ledger,helper_gate,make_cases,accounting


class ProtocolTests(unittest.TestCase):
    def test_twenty_four_calls_and_six_cases_per_worker(self):
        rows=ledger();self.assertEqual(len(rows),24)
        for v in ('TH','P'):
            for m in ('detailed','minimal'):
                unit=[r for r in rows if r['variant']==v and r['mode']==m]
                self.assertEqual([r['call_index'] for r in unit],list(range(1,7)))
                self.assertEqual([r['seed'] for r in unit],[0,20250901,20260910,0,0,0])
                self.assertTrue(all(r['GraphExecutor_calls']==1 and not r['performance_sample'] for r in unit))

    def valid(self):
        return {'helper_returned':True,'counts':1,'metadata':{
            'schema':'c3_yolo416_graph_once_v1','started_calls':1,'completed_calls':1,
            'last_kind':'diagnostic','parameter_count':0,'packed_calls_per_invocation':1,'adds_synchronize':False}}

    def test_helper_contract_rejects_stale_extra_or_boolean_counts(self):
        good=self.valid();helper_gate(good,1)
        for key,value in (('completed_calls',0),('started_calls',2),('adds_synchronize',True),
                          ('parameter_count',1),('packed_calls_per_invocation',True)):
            bad=deepcopy(good);bad['metadata'][key]=value
            with self.assertRaises(ValueError):helper_gate(bad,1)

    def test_begin_without_execution_is_not_a_completed_call(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'TH_detailed';folder.mkdir()
            (folder/'call_01_begin.json').write_text(json.dumps(ledger()[0]))
            a=accounting(root)
            self.assertEqual(a['registered'],1)
            self.assertEqual(a['helper_completed_observed_lower_bound'],0)
            self.assertEqual(a['all_certificates_passed'],0)
            self.assertEqual(a['unresolved_workers'],['TH_detailed'])

    def test_weight_observability_and_shared_reference_calls(self):
        class FakeSpec:
            abi=[{}, {}, {}, {}]
            def reference(self,inputs):
                return np.array([float(np.mean(inputs[0]))+float(inputs[1][0])+float(inputs[2][0])],dtype='float32')
        nodes=[{'op':'null','name':s} for s in ('data','w','a')]
        for i in range(12):
            nodes.append({'op':'tvm_op','name':'f'+str(i),'inputs':[[i+2 if i else 0,0,0],[1,0,0],[2,0,0]],
                         'attrs':{'func_name':'fake','num_inputs':'3','num_outputs':'1','flatten_data':'0'}})
        graph={'nodes':nodes,'node_row_ptr':list(range(16)),
               'attrs':{'shape':['list_shape',[[1,3,416,416],[1],[1]]+[[1]]*12],
                        'dltype':['list_str',['float32','int8','int32']+['float32']*12]},
               'heads':[[14,0,0]]*8}
        def no_CPU(*args):raise AssertionError('unexpected fake CPU function')
        cases,name,evidence=make_cases(graph,{'w':np.array([1],dtype='int8'),'a':np.array([0],dtype='int32')},
                                      {'fake':FakeSpec()},no_CPU)
        self.assertEqual(len(cases),6);self.assertEqual(name,'w')
        self.assertFalse(evidence['selection_uses_target_or_timing'])
        self.assertFalse(np.array_equal(cases[3]['heads'][0],cases[4]['heads'][0]))
        np.testing.assert_array_equal(cases[3]['heads'][0],cases[5]['heads'][0])
        self.assertEqual(len(cases[0]['convolutions']),12)


if __name__=='__main__':unittest.main()
