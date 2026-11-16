"""One qualified fused operator in one fresh RPC unit; no transport ownership."""
import ctypes
import json
import math
from pathlib import Path
import numpy as np
import tvm,vta
from c3_tps_sync_release_local import read,write,require,sha,initialize_helper
from c3_official_window_reference import array_sha


def physical_storage(native,storage,abi):
    require(len(storage)==len(abi)==4,'FOUR_BUFFER_STORAGE_REQUIRED')
    require(len({s['data_address'] for s in storage})==4 and all(s['data_address']>0
        and s['byte_offset']==0 and s['device_type']==12 for s in storage),'OPERATOR_STORAGE_ALIAS_OR_OFFSET')
    bases=native['tensor_domain_dram_bases'];widths={1:256,2:16,3:64,4:16}
    require(sorted((b['tensor'],b['domain']) for b in bases)==[(0,2),(1,1),(2,3),(3,4)],
            'OPERATOR_TENSOR_DOMAIN_CHANGED')
    sizes={i:math.prod(a['shape'])*np.dtype(a['dtype']).itemsize for i,a in enumerate(abi)}
    intervals=sorted((b['dram_base']*widths[b['domain']],
                     b['dram_base']*widths[b['domain']]+sizes[b['tensor']]) for b in bases)
    require(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),'OPERATOR_PHYSICAL_STORAGE_OVERLAP')
    return {'status':'FOUR_MODEL_TENSORS_PHYSICAL_BINDING_PASSED','intervals':intervals}


class OperatorAdapter:
    def __init__(self,remote,descriptor,artifacts,output,diagnostic):
        self.remote,self.descriptor,self.output=remote,descriptor,output
        self.retained=[];self.expected_files={};self.cases={}
        for name,item in [('once_helper.so',artifacts['helper']),('storage_helper.so',artifacts['storage_helper']),
                          ('target.so',{'path':descriptor['arm_module'],'sha256':sha(descriptor['arm_module'])})]:
            require(sha(item['path'])==item['sha256'],'UPLOAD_IDENTITY_CHANGED')
            remote.upload(item['path'],target=name);self.retained.append(remote.load_module(name))
            self.expected_files[name]=item['sha256']
        self.module=self.retained[-1];get=remote.get_function
        self.timer=get('c3.official_window.fused_timed_once');self.diagnostic=get('c3.official_window.fused_diagnostic_once')
        self.metadata=get('c3.official_window.metadata');self.counts=get('c3.official_window.counts')
        self.storage_query=get('c3.tps_window.buffer_storage')
        initialize_helper(get('c3.official_window.reset_counts'),self.counts,self.metadata,
            lambda v:write(output/'helper_initial_state.json',v))
        for row in descriptor['model_cases']['cases']:
            with np.load(row['path']) as file:
                arrays=[file['a%d'%i].copy() for i in range(4)];expected=file['expected'].copy()
            require([array_sha(a) for a in arrays[:3]]==row['input_sha256'] and
                    array_sha(expected)==row['expected_sha256'],'MODEL_CASE_CONTENT_CHANGED')
            self.cases[row['seed'],row['weight']]=(arrays,expected)
        self.buffers=tuple(tvm.nd.array(a,remote.ext_dev(0)) for a in self.cases[0,'W0'][0])
        self.storage=json.loads(self.storage_query(*self.buffers))
        self.handles=[int(ctypes.cast(a.handle,ctypes.c_void_p).value) for a in self.buffers]
        self.definitions=None;self.bases=None
        if diagnostic:
            from c3_official_window_method import audit_function
            from c3_valid_window import certify_trace
            parent=tvm.ir.load_json(Path(descriptor['parent_module']).read_text())
            self.events=audit_function(list(parent.functions.values())[0])['events']
            self.products=certify_trace(self.events)['output_terms']

    def prepare_call(self,row):
        arrays,expected=self.cases[row['seed'],row['weight']]
        for buffer,value in zip(self.buffers,arrays):buffer.copyfrom(value)
        require(np.all(self.buffers[-1].numpy()!=expected),'OPERATOR_OUTPUT_SENTINEL_MISSING')
        return {'input_sha256':[array_sha(a) for a in arrays[:3]],'expected_sha256':array_sha(expected),
                'storage':self.storage,'host_RPC_wrapper_handles':self.handles}

    def invoke(self,diagnostic):
        function=self.diagnostic if diagnostic else self.timer
        return function(self.module,self.descriptor['symbol'],*self.buffers)

    def check(self,row):
        arrays,expected=self.cases[row['seed'],row['weight']];actual=self.buffers[-1].numpy()
        require(np.array_equal(actual,expected),'FPGA_FULL_FUSION_OUTPUT_MISMATCH')
        require(all(np.array_equal(a.numpy(),b) for a,b in zip(self.buffers[:-1],arrays[:-1])),
                'READONLY_MODEL_ARGUMENT_CHANGED')
        storage=json.loads(self.storage_query(*self.buffers))
        require(storage==self.storage and self.handles==[
            int(ctypes.cast(a.handle,ctypes.c_void_p).value) for a in self.buffers],
            'ACTUAL_MODEL_PARAMETER_STORAGE_CHANGED')
        return {'correct':True,'output_sha256':array_sha(actual),'reference_sha256':array_sha(expected),
                'actual_storage':storage}

    def certify(self,row,dump,execution,commands,tokens):
        from c3_tps_window_native import certify_native
        from c3_yolo416_fifo_certificate import certify_action_graph
        from c3_valid_window import certify_trace
        from c3_five_tradeoff_commands import decode_submissions
        from c3_official_window_workflow import native_certificate_for_storage
        from c3_tps_aggregation_capacity import organization
        native=certify_native(self.events,dump,execution['submits'],self.definitions)
        self.definitions=native['cache_definitions'];require(native['submission_count']==1,'OPERATOR_MULTI_SUBMIT_UNEXPECTED')
        require(certify_trace(self.events,ancestors=native['ancestors'])['output_terms']==self.products,
                'OPERATOR_NATIVE_DEPENDENCY_SEMANTICS_CHANGED')
        storage=physical_storage(native,json.loads(self.storage_query(*self.buffers)),self.descriptor['abi'])
        if self.bases is None:self.bases=native['tensor_domain_dram_bases']
        require(self.bases==native['tensor_domain_dram_bases'],'OPERATOR_BACKEND_STORAGE_CHANGED')
        progress=[certify_action_graph(rs,commands,tokens) for rs in decode_submissions(dump)]
        require(len(progress)==1 and all(p['status']=='FINITE_FIFO_PROGRESS_PROVED' for p in progress),
                'OPERATOR_FIFO_PROGRESS_UNPROVED')
        return {'native':native_certificate_for_storage(native),'progress':progress,'storage':storage,
                'organization':organization(dump,native)}

    def local_signature(self,position):
        unit=Path(self.descriptor['local_unit'])
        return read(unit/'R_detailed'/('call_%02d_end.json'%position))['submit_ledger']

    def release(self):
        self.remote=None;self.module=None;self.retained=[];self.buffers=()
        self.timer=self.diagnostic=self.metadata=self.counts=self.storage_query=None
