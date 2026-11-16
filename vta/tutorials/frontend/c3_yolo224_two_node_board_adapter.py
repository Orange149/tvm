"""Fixed YOLO224 graph, byte-identical base ARM library, two replacements.

Reuse the qualified graph buffer checks; the constructor and native proof
explicitly retain the original serial functions and all 22 submissions.
"""
import json
from pathlib import Path
import numpy as np
import tvm, vta
from tvm import relay
from tvm.contrib import graph_executor
from c3_tps_sync_release_local import read, write, sha, require
from c3_official_window_reference import from_workload
from c3_yolo416_graph_host_reference import HostCPUCallback
from c3_yolo416_graph_board_adapter import GraphAdapter as BaseAdapter, allocation_snapshot
from c3_yolo224_two_node_local_v2 import make_cases, make_route, traces
from c3_yolo224_two_node_native_v2 import certify_graph


class GraphAdapter(BaseAdapter):
    def __init__(self, remote, descriptor, artifacts, output, diagnostic):
        self.remote, self.descriptor, self.output = remote, descriptor, output
        self.retained=[]; self.expected_files={}; loaded={}
        specs=[('storage_helper.so',artifacts['storage_helper']),
               ('graph_once_helper.so',artifacts['graph_once_helper']),
               ('route_helper.so',artifacts['route_helper']),
               ('cpu_reference_helper.so',artifacts['cpu_reference_helper']),
               ('base.so',artifacts['base_arm']),
               ('node34.so',descriptor['replacement_modules']['34']),
               ('node46.so',descriptor['replacement_modules']['46'])]
        if diagnostic:
            specs += [('graph_observer_helper.so',artifacts['graph_observer_helper']),
                      ('rpc_observer_helper.so',artifacts['rpc_observer_helper'])]
        write(output/'DMA_before_reference.json',allocation_snapshot(remote))
        for name,item in specs:
            require(sha(item['path'])==item['sha256'],'GRAPH_UPLOAD_IDENTITY_CHANGED')
            remote.upload(item['path'],target=name); loaded[name]=remote.load_module(name)
            self.retained.append(loaded[name]); self.expected_files[name]=item['sha256']
        get=remote.get_function; self.storage_query=get('c3.tps_window.buffer_storage')
        source=Path(descriptor['graph_prepared']); graph=read(source/'graph.json')
        self.source,self.graph=source,graph
        self.params={str(k):v.numpy() for k,v in relay.load_param_dict((source/'params.bin').read_bytes()).items()}
        self.functions=read(source/'functions.json')
        convolutions={s:from_workload(r['workload'],r['abi'],r['reference']['postprocess'])
                      for s,r in self.functions.items()}
        require(all(spec.contract()==self.functions[s]['reference'] for s,spec in convolutions.items()),
                'GRAPH_REFERENCE_CONTRACT_CHANGED')
        cpu_manifest=read(source/'CPU_manifest.json')
        require(len(cpu_manifest)==21 and all(r['raw_pointer_gate']['status']==
                'CPU_RAW_POINTER_REFERENCE_SUPPORTED' for r in cpu_manifest.values()),'CPU_TIR_REFERENCE_GATE')
        # The original ARM library is used directly, without rebuilding any CPU
        # function. The source TIR gate and archive identity are retained; no
        # unavailable original ARM LLVM proof is invented. Every CPU reference
        # call also checks that it leaves the submission ledger unchanged.
        self.cpu=HostCPUCallback(loaded['base.so'],cpu_manifest,remote.cpu(0),
            get('vta.runtime.official_submit_ledger'),get('c3.yolo416.cpu_reference_call'))
        golden,weight_name,observable=make_cases(graph,self.params,convolutions,self.cpu)
        local=Path(descriptor['local_unit'])/(descriptor['actual_variant']+'_detailed')
        frozen=read(local/'reference_protocol.json')
        require(weight_name==frozen['weight_parameter'] and observable==frozen['observability'],
                'ARM_REFERENCE_WEIGHT_OBSERVABILITY_CHANGED')
        write(output/'reference_protocol.json',{'weight_parameter':weight_name,'observability':observable,
            'CPU_reference_calls':self.cpu.calls,'GraphExecutor_reference_calls':0,'VTA_reference_calls':0,
            'CPU_component':'exact original graph ARM CPU functions; separate CPU argument storage',
            'integer_component':'independent NumPy convolution and fused postprocess',
            'original_ARM_LLVM_available':False,'original_ARM_binary':artifacts['base_arm']})
        self.cases={(c['seed'],'W0'):c for c in golden[:3]}; self.cases[0,'W1']=golden[4]
        self.weight_name=weight_name
        write(output/'DMA_after_reference.json',allocation_snapshot(remote))
        sizes={}
        for sid,shape,dtype in zip(graph['attrs']['storage_id'][1],graph['attrs']['shape'][1],graph['attrs']['dltype'][1]):
            size=int(np.prod(shape))*np.dtype(dtype).itemsize
            sizes[sid]=max(sizes.get(sid,0),(size+255)//256*256)
        reserve=sum(sizes.values())+16*1024**2; state=allocation_snapshot(remote)
        require(state['high_water_bytes']+reserve<=201326592,'GRAPH_DMA_ADMISSION_FAILED')
        write(output/'DMA_admission.json',{'snapshot':state,'graph_pool_aligned_upper_bound':sum(sizes.values()),
            'additional_workspace_reserve':16*1024**2,'fits':True,'not_actual_allocator_peak_proof':True})
        replacements={34:loaded['node34.so'],46:loaded['node46.so']}
        self.target=make_route(source,descriptor['actual_variant'],loaded['base.so'],replacements,diagnostic,get)
        self.proxy=None; module=self.target
        rowptr=graph['node_row_ptr']; self.mapping={'calls':[]}
        for node,n in enumerate(graph['nodes']):
            symbol=n.get('attrs',{}).get('func_name')
            if symbol not in self.functions:continue
            entries=[rowptr[p]+o for p,o,v in n['inputs']]+[rowptr[node]]
            self.mapping['calls'].append({'node':node,'symbol':symbol,'tensor_entries':entries})
        require(len(self.mapping['calls'])==12,'GRAPH_CALL_COVERAGE')
        if diagnostic:
            text='\n'.join('%d %s 3 1'%(r['node'],r['symbol']) for r in self.mapping['calls'])
            self.proxy=get('c3.yolo416.rpc_observe_module')(self.target,text); module=self.proxy
        self.executor=graph_executor.create((source/'graph.json').read_text(),module,[remote.ext_dev(0),remote.cpu(0)])
        self.executor.load_params((source/'params.bin').read_bytes())
        self.data=self.executor.get_input('data'); self.parameters={name:self.executor.get_input(name) for name in self.params}
        self.buffers=[self.data,*self.parameters.values()]
        self.storage=json.loads(self.storage_query(*self.buffers)); self.outputs=[self.executor.get_output(i) for i in range(8)]
        ids=graph['attrs']['storage_id'][1]
        null_ids={ids[rowptr[i]] for i,n in enumerate(graph['nodes']) if n['op']=='null'}
        for i,shape in ((0,[1,255,14,14]),(4,[1,255,7,7])):
            nid,nout,version=graph['heads'][i]; entry=rowptr[nid]+nout
            require(version==0 and graph['attrs']['shape'][1][entry]==shape and
                    graph['attrs']['dltype'][1][entry]=='float32' and ids[entry] not in null_ids,
                    'GRAPH_SENTINEL_ABI_OR_PARAMETER_ALIAS')
        self.metadata=get('c3.yolo416.graph_metadata'); self.counts=lambda:json.loads(self.metadata())['completed_calls']
        self.timer=get('c3.yolo416.graph_timed_once'); self.diagnostic=get('c3.yolo416.graph_diagnostic_once')
        get('c3.yolo416.graph_reset_counts')(); meta=json.loads(self.metadata())
        require(meta['started_calls']==meta['completed_calls']==0,'GRAPH_HELPER_NOT_FRESH')
        write(output/'helper_initial_state.json',meta); write(output/'DMA_after_executor.json',allocation_snapshot(remote))
        self.definitions=None; self.bases=None
        if diagnostic:self.events=traces(source,descriptor['actual_variant'])

    def certify(self,row,dump,execution,commands,tokens):
        from c3_yolo416_graph_trace import storage_bindings
        certificate,self.definitions=certify_graph(self.graph,self.functions,self.events,dump,
            execution['submits'],self.definitions,commands,tokens)
        bases=[]
        for call,mapping in zip(certificate['calls'],self.mapping['calls']):
            for fragment in call['submissions']:
                if fragment['kind']!='material':continue
                native=fragment['native']
                for item in native['tensor_domain_dram_bases']:
                    require(0<=item['tensor']<4,'FUNCTION_TENSOR_ID')
                    bases.append({**item,'tensor':mapping['tensor_entries'][item['tensor']]})
                native.pop('cache_definitions',None)
                native['ancestors']=[hex(x) for x in native['ancestors']]
        require(self.bases is None or self.bases==bases,'GRAPH_DRAM_STORAGE_CHANGED')
        self.bases=bases
        storage=storage_bindings(self.graph,self.mapping,{'tensor_domain_dram_bases':bases},self.observed)
        return {'native':certificate,'storage':storage,'integer_comparisons':self.snapshot_rows}

    def local_signature(self,position):
        unit=Path(self.descriptor['local_unit'])/(self.descriptor['actual_variant']+'_detailed')
        return read(unit/('call_%02d_end.json'%position))['helper']['submits']
