"""Namespace/storage contract tests; never execute a target."""
from copy import deepcopy
import unittest
from c3_yolo416_graph_trace import compose,storage_bindings


def fixture():
    shapes=[[1]]*8;dtypes=['int8','int8','int32','int8']*2
    nodes=[{'op':'null','name':str(i)} for i in range(8)]
    for i in (3,7):
        nodes[i]={'op':'tvm_op','name':'f'+str(i),'inputs':[[i-3,0,0],[i-2,0,0],[i-1,0,0]],
                  'attrs':{'func_name':'shared','num_inputs':'3','num_outputs':'1','flatten_data':'0'}}
    graph={'nodes':nodes,'node_row_ptr':list(range(9)),'heads':[[7,0,0]],
           'attrs':{'shape':['list_shape',shapes],'dltype':['list_str',dtypes],
                    'storage_id':['list_int',[0,1,2,3,4,5,6,3]]}}
    trace=[{'kind':'load','tensor':i,'domain':d} for i,d in enumerate((2,1,3))]
    trace+=[{'kind':'kernel','scope':0,'signature':[],
             'cache_identity':{'scope':0,'signature':[]},'uops':[{'args':[0,0,0,0,0,0,0,0],'loops':[]}]}]
    trace+=[{'kind':'store','tensor':3,'domain':4},{'kind':'sync'}]
    functions={'shared':{'abi':[{'shape':[1],'dtype':d} for d in dtypes[:4]],'events':trace}}
    return graph,functions


class CompositionTests(unittest.TestCase):
    def test_shared_handle_distinct_tensor_versions(self):
        g,f=fixture();events,mapping=compose(g,f,2)
        kernels=[e for e in events if e['kind']=='kernel']
        self.assertEqual(kernels[0]['cache_identity'],kernels[1]['cache_identity'])
        self.assertEqual(kernels[0]['uops'],f['shared']['events'][3]['uops'])
        self.assertEqual([c['tensor_entries'] for c in mapping['calls']],[[0,1,2,3],[4,5,6,7]])
        self.assertEqual([e['tensor'] for e in events if e['kind']=='load'],[0,1,2,4,5,6])
        self.assertEqual(len(mapping['scope_bindings']),1)

    def test_different_functions_cannot_alias_scope_zero(self):
        g,f=fixture();g['nodes'][7]['attrs']['func_name']='other';f['other']=deepcopy(f['shared'])
        events,mapping=compose(g,f,2)
        self.assertEqual([e['scope'] for e in events if e['kind']=='kernel'],[0,1])

    def test_unknown_sync_or_input_output_alias_rejected(self):
        g,f=fixture();f['shared']['events'].pop()
        with self.assertRaisesRegex(ValueError,'COMPLETED_FUSED_SYNC'):compose(g,f,2)
        g,f=fixture();g['attrs']['storage_id'][1][3]=0
        with self.assertRaisesRegex(ValueError,'STORAGE_ALIAS'):compose(g,f,2)

    def native(self,g,mapping):
        rows=[]
        for c in mapping['calls']:
            for entry,domain in zip(c['tensor_entries'],(2,1,3,4)):
                sid=g['attrs']['storage_id'][1][entry]
                unit={1:256,2:16,3:64,4:16}[domain]
                rows.append({'tensor':entry,'domain':domain,'dram_base':(sid+1)*4096//unit})
        return {'tensor_domain_dram_bases':rows}

    def test_memory_type_units_and_reused_pool(self):
        g,f=fixture();_,mapping=compose(g,f,2)
        result=storage_bindings(g,mapping,self.native(g,mapping))
        self.assertEqual(result['status'],'GRAPH_NATIVE_STORAGE_REUSE_BOUND')
        self.assertEqual(len(result['observed_pool_physical_intervals']),7)

    def test_changed_or_overlapping_pool_physical_bases_rejected(self):
        g,f=fixture();_,mapping=compose(g,f,2);native=self.native(g,mapping)
        native['tensor_domain_dram_bases'][-1]['dram_base']+=1
        with self.assertRaisesRegex(ValueError,'BASE_CHANGED'):storage_bindings(g,mapping,native)
        native=self.native(g,mapping)
        native['tensor_domain_dram_bases'][1]['dram_base']=4096//256
        with self.assertRaisesRegex(ValueError,'RANGES_OVERLAP'):storage_bindings(g,mapping,native)

    def test_real_TH_P_graph_composition(self):
        from pathlib import Path
        import tvm,vta
        from c3_tps_sync_release_local import HOLD,read
        from c3_official_window_method import _trace,_limits
        source=HOLD/'20261017_c3_yolo416_tps_prepare_run02'
        for variant in ('TH','P'):
            functions={}
            for row in read(source/variant/'functions.json'):
                m=tvm.ir.load_json(Path(row['parent']).read_text())
                trace,_=_trace(list(m.functions.values())[0],_limits(None))
                functions[row['symbol']]={'abi':row['abi'],'events':trace}
            events,mapping=compose(read(source/variant/'graph.json'),functions)
            self.assertEqual(len(mapping['calls']),12)
            self.assertEqual(len(mapping['scope_bindings']),66)
            calls={r['node']:r for r in mapping['calls']}
            self.assertEqual(calls[29]['symbol'],calls[57]['symbol'])
            self.assertNotEqual(calls[29]['tensor_entries'][:3],calls[57]['tensor_entries'][:3])
            self.assertEqual(sum(e['kind']=='sync' for e in events),12)


if __name__=='__main__':unittest.main()
