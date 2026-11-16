"""Whole graph in one fresh RPC unit; no transport ownership or hidden run."""
import json
from pathlib import Path
import numpy as np
import tvm,vta
from tvm import relay
from tvm.contrib import graph_executor
from c3_tps_sync_release_local import read,write,sha,require
from c3_official_window_reference import from_workload,array_sha
from c3_yolo416_graph_host_reference import HostCPUCallback
from c3_yolo416_graph_local_v2 import make_cases


def allocation_snapshot(remote):
    row=json.loads(remote.get_function('vta.runtime.udmabuf_allocation_snapshot')())
    require(row['schema']=='vta_udmabuf_allocation_snapshot_v1','ALLOCATION_SNAPSHOT_SCHEMA')
    require(row['high_water_bytes']<=201326592 and
            (not row['initialized'] or row['capacity_bytes']==201326592),'ACTUAL_DMA_POOL_INVALID')
    return row


class GraphAdapter:
    def __init__(self,remote,descriptor,artifacts,output,diagnostic):
        self.remote,self.descriptor,self.output=remote,descriptor,output
        self.retained=[];self.expected_files={};loaded={}
        specs=[('storage_helper.so',artifacts['storage_helper']),
               ('graph_once_helper.so',artifacts['graph_once_helper']),
               ('cpu_reference_helper.so',artifacts['cpu_reference_helper']),
               ('reference.so',{'path':descriptor['reference_module'],'sha256':sha(descriptor['reference_module'])}),
               ('target.so',{'path':descriptor['arm_module'],'sha256':sha(descriptor['arm_module'])})]
        if diagnostic:
            specs += [('graph_observer_helper.so',artifacts['graph_observer_helper']),
                      ('rpc_observer_helper.so',artifacts['rpc_observer_helper'])]
        write(output/'DMA_before_reference.json',allocation_snapshot(remote))
        for name,item in specs:
            require(sha(item['path'])==item['sha256'],'GRAPH_UPLOAD_IDENTITY_CHANGED')
            remote.upload(item['path'],target=name);loaded[name]=remote.load_module(name)
            self.retained.append(loaded[name]);self.expected_files[name]=item['sha256']
        get=remote.get_function;self.storage_query=get('c3.tps_window.buffer_storage')
        source=Path(descriptor['graph_prepared']);compiled=Path(descriptor['compiled_source'])
        graph=read(source/'graph.json');self.graph,self.source=graph,source
        self.params={str(k):v.numpy() for k,v in relay.load_param_dict((source/'params.bin').read_bytes()).items()}
        units=read(Path(descriptor['operator_qualification'])/'objects.json');convolutions={}
        for row in read(compiled/'functions.json'):
            i=next(i for i,u in enumerate(units) if u['variant']==descriptor['variant'] and u['symbol']==row['symbol'])
            reference=read(Path(descriptor['operator_qualification'])/'objects'/('%02d'%i)/'prepare/R/reference_contract.json')
            spec=from_workload(row['workload'],row['abi'],reference['postprocess'])
            require(spec.contract()==reference,'GRAPH_OPERATOR_REFERENCE_CHANGED');convolutions[row['symbol']]=spec
        cpu_manifest=read(descriptor['CPU_reference_manifest'])
        require(len(cpu_manifest)==21 and all(r.get('raw_pointer_llvm_gate',{}).get('status')==
                'CPU_RAW_POINTER_LLVM_CONFIRMED' for r in cpu_manifest.values()),'CPU_REFERENCE_LLVM_NOT_QUALIFIED')
        self.cpu=HostCPUCallback(loaded['reference.so'],cpu_manifest,remote.cpu(0),
            get('vta.runtime.official_submit_ledger'),get('c3.yolo416.cpu_reference_call'))
        golden,weight_name,observable=make_cases(graph,self.params,convolutions,self.cpu)
        frozen=read(Path(descriptor['local_unit'])/(descriptor['variant']+'_detailed')/'reference_protocol.json')
        require(weight_name==frozen['weight_parameter'] and observable==frozen['observability'],
                'ARM_GRAPH_WEIGHT_OBSERVABILITY_RULE_CHANGED')
        write(output/'reference_protocol.json',{'weight_parameter':weight_name,'observability':observable,
              'CPU_reference_calls':self.cpu.calls,'GraphExecutor_reference_calls':0,'VTA_reference_calls':0,
              'CPU_component':'same frozen ARM CPU code, separate reference SO; CPU argument storage',
              'integer_component':'independent NumPy convolution and fused postprocess'})
        self.cases={(c['seed'],'W0'):c for c in golden[:3]};self.cases[0,'W1']=golden[4]
        self.weight_name=weight_name
        write(output/'DMA_after_reference.json',allocation_snapshot(remote))
        # Conservatively account all graph storage pools and an additional
        # 16 MiB for CPU workspace growth before constructing the executor.
        sizes={}
        for sid,shape,dtype in zip(graph['attrs']['storage_id'][1],graph['attrs']['shape'][1],graph['attrs']['dltype'][1]):
            size=int(np.prod(shape))*np.dtype(dtype).itemsize
            sizes[sid]=max(sizes.get(sid,0),(size+255)//256*256)
        reserve=sum(sizes.values())+16*1024**2
        state=allocation_snapshot(remote)
        require(state['high_water_bytes']+reserve<=201326592,'GRAPH_DMA_ADMISSION_BEFORE_EXECUTOR_FAILED')
        write(output/'DMA_admission.json',{'snapshot':state,'graph_pool_aligned_upper_bound':sum(sizes.values()),
              'additional_workspace_reserve':16*1024**2,'fits':True,
              'not_a_proof_of_actual_allocator_peak':True})
        self.target=loaded['target.so'];self.proxy=None;module=self.target
        self.mapping=read(source/'namespace.json')
        if diagnostic:
            text='\n'.join('%d %s 3 1'%(r['node'],r['symbol']) for r in self.mapping['calls'])
            self.proxy=get('c3.yolo416.rpc_observe_module')(self.target,text);module=self.proxy
        self.executor=graph_executor.create((source/'graph.json').read_text(),module,[remote.ext_dev(0),remote.cpu(0)])
        self.executor.load_params((source/'params.bin').read_bytes())
        self.data=self.executor.get_input('data')
        self.parameters={name:self.executor.get_input(name) for name in self.params}
        self.buffers=[self.data,*self.parameters.values()]
        self.storage=json.loads(self.storage_query(*self.buffers));self.outputs=[self.executor.get_output(i) for i in range(8)]
        rowptr=graph['node_row_ptr'];ids=graph['attrs']['storage_id'][1]
        null_ids={ids[rowptr[i]] for i,n in enumerate(graph['nodes']) if n['op']=='null'}
        for i,shape in ((0,[1,255,26,26]),(4,[1,255,13,13])):
            nid,nout,version=graph['heads'][i];entry=rowptr[nid]+nout
            require(version==0 and graph['attrs']['shape'][1][entry]==shape and
                    graph['attrs']['dltype'][1][entry]=='float32' and ids[entry] not in null_ids,
                    'GRAPH_SENTINEL_HEAD_ABI_OR_PARAMETER_ALIAS')
        self.metadata=get('c3.yolo416.graph_metadata')
        self.counts=lambda:json.loads(self.metadata())['completed_calls']
        self.timer=get('c3.yolo416.graph_timed_once');self.diagnostic=get('c3.yolo416.graph_diagnostic_once')
        get('c3.yolo416.graph_reset_counts')()
        meta=json.loads(self.metadata());require(meta['started_calls']==meta['completed_calls']==0,
                                                 'GRAPH_HELPER_NOT_FRESH')
        write(output/'helper_initial_state.json',meta)
        write(output/'DMA_after_executor.json',allocation_snapshot(remote))
        self.definitions=None;self.bases=None
        if diagnostic:self.events=read(source/'events.json')

    def prepare_call(self,row):
        case=self.cases[row['seed'],row['weight']]
        self.data.copyfrom(case['data']);self.parameters[self.weight_name].copyfrom(case['weight'])
        for i in (0,4):
            self.outputs[i].copyfrom(np.full_like(case['heads'][i],np.nan))
            require(np.isnan(self.outputs[i].numpy()).all(),'GRAPH_OUTPUT_SENTINEL_NOT_WRITTEN')
        if self.proxy is not None:self.proxy['c3_yolo416_observer_begin']()
        return {'data_sha256':array_sha(case['data']),'weight_sha256':array_sha(case['weight']),
                'expected_head_sha256':[array_sha(a) for a in case['heads']],'storage':self.storage}

    def invoke(self,diagnostic):
        return (self.diagnostic if diagnostic else self.timer)(self.executor.module)

    def check(self,row):
        case=self.cases[row['seed'],row['weight']];heads=[a.numpy() for a in self.outputs]
        require(all(np.array_equal(a,b) for a,b in zip(heads,case['heads'])),'FPGA_GRAPH_RAW_OUTPUT_MISMATCH')
        require(np.array_equal(self.data.numpy(),case['data']),'GRAPH_INPUT_DATA_MUTATED')
        for name,buffer in self.parameters.items():
            expected=case['weight'] if name==self.weight_name else self.params[name]
            require(np.array_equal(buffer.numpy(),expected),'GRAPH_READONLY_TRAINED_PARAMETER_MUTATED:'+name)
        require(json.loads(self.storage_query(*self.buffers))==self.storage,'GRAPH_LONG_PARAMETER_STORAGE_CHANGED')
        self.observed=None;self.snapshot_rows=[]
        if self.proxy is not None:
            require(self.proxy['c3_yolo416_observer_end']()==12,'GRAPH_OBSERVER_CALL_COUNT')
            self.observed=json.loads(self.proxy['c3_yolo416_observer_metadata']())
            require(len(self.observed)==len(case['convolutions'])==12,'GRAPH_OBSERVER_REFERENCE_COUNT')
            for i,(metadata,golden) in enumerate(zip(self.observed,case['convolutions'])):
                require(metadata['symbol']==golden['symbol'] and metadata['inputs_before_count']==
                    metadata['inputs_after_count']==3 and metadata['outputs_count']==1,'GRAPH_OBSERVER_ABI_CHANGED')
                for j,wanted in enumerate(golden['inputs']):
                    for kind in (0,1):
                        snapshot=self.proxy['c3_yolo416_observer_snapshot'](i,kind,j)
                        require(np.array_equal(snapshot.numpy(),wanted),'GRAPH_ACTUAL_INTEGER_INPUT_MISMATCH')
                snapshot=self.proxy['c3_yolo416_observer_snapshot'](i,2,0)
                require(np.array_equal(snapshot.numpy(),golden['output']),'GRAPH_ACTUAL_INTEGER_OUTPUT_MISMATCH')
                self.snapshot_rows.append({'node':int(metadata['node']),'symbol':metadata['symbol'],
                    'input_sha256':[array_sha(a) for a in golden['inputs']],
                    'output_sha256':array_sha(golden['output']),'all_elements_compared':True})
        return {'correct':True,'all_8_raw_outputs_equal':True,'head_sha256':[array_sha(a) for a in heads],
                'actual_storage':self.storage,'integer_intermediate_comparisons':len(self.snapshot_rows),
                'DMA_after_call':allocation_snapshot(self.remote)}

    def certify(self,row,dump,execution,commands,tokens):
        from c3_tps_window_native import certify_native
        from c3_yolo416_fifo_certificate import certify_action_graph
        from c3_five_tradeoff_commands import decode_submissions
        from c3_official_window_workflow import native_certificate_for_storage
        from c3_tps_aggregation_capacity import organization
        from c3_yolo416_graph_trace import storage_bindings
        native=certify_native(self.events,dump,execution['submits'],self.definitions)
        self.definitions=native['cache_definitions'];require(native['submission_count']==12,'GRAPH_ACTUAL_SUBMISSION_COUNT')
        if self.bases is None:self.bases=native['tensor_domain_dram_bases']
        require(self.bases==native['tensor_domain_dram_bases'],'GRAPH_DRAM_STORAGE_CHANGED')
        storage=storage_bindings(self.graph,self.mapping,native,self.observed)
        progress=[certify_action_graph(rs,commands,tokens) for rs in decode_submissions(dump)]
        require(len(progress)==12 and all(p['status']=='FINITE_FIFO_PROGRESS_PROVED' for p in progress),
                'GRAPH_FIFO_PROGRESS_UNPROVED')
        return {'native':native_certificate_for_storage(native),'progress':progress,'storage':storage,
                'integer_comparisons':self.snapshot_rows,'organization':organization(dump,native)}

    def local_signature(self,position):
        unit=Path(self.descriptor['local_unit'])/(self.descriptor['variant']+'_detailed')
        return read(unit/('call_%02d_end.json'%position))['helper']['submits']

    def release(self):
        self.remote=self.cpu=self.executor=self.target=self.proxy=None
        self.parameters={};self.buffers=[];self.outputs=[];self.retained=[]
        self.metadata=self.counts=self.timer=self.diagnostic=self.storage_query=None
