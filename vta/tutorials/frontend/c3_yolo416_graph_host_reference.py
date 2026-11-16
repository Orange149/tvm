"""ARM CPU reference using regular CPU storage, not the nonfreeing DMA pool.

Requires a separately certified raw-pointer-compatible CPU function and the
frozen runtime CPUPtr fallback. No target-module or runtime rewrite occurs.
"""
import json
import re
import numpy as np
from c3_yolo416_graph_cpu_reference import cpu_instruction_gate,require


def raw_pointer_gate(fn):
    from tvm import tir
    base=cpu_instruction_gate(fn);forbidden=[]
    def visit(n):
        if isinstance(n,tir.Call) and n.op.name=='tir.call_extern' and n.args and isinstance(n.args[0],tir.StringImm):
            name=n.args[0].value
            if name.startswith('VTA') and name not in ('VTABufferCPUPtr','VTATLSCommandHandle'):
                forbidden.append(name)
    tir.stmt_functor.post_order_visit(fn.body,visit)
    require(not forbidden,'CPU_REFERENCE_REQUIRES_REAL_DMA_BUFFER_OR_UNKNOWN_VTA_EXTERN')
    return {'status':'CPU_RAW_POINTER_REFERENCE_SUPPORTED','instruction_gate':base,
            'runtime_precondition':'VTABufferCPUPtr returns unknown raw CPU pointer unchanged',
            'no_DMA_argument_allocation':True,'workspace_allocation_requires_separate_accounting':True}


def raw_pointer_llvm_gate(llvm,symbol):
    from c3_yolo416_graph_binding import project
    from c3_official_window_llvm import _function_bodies,_without_comments,_SYMBOL,_unquote
    _,ownership=project(llvm,symbol);bodies=_function_bodies(_without_comments(llvm));seen=set()
    for name in ownership['reachable_functions']:
        seen.update(_unquote(t)[1:] for t in re.findall(_SYMBOL,bodies[name])
                    if _unquote(t).startswith('@VTA'))
    require(seen<= {'VTABufferCPUPtr','VTATLSCommandHandle'},'CPU_LLVM_REQUIRES_REAL_DMA_BUFFER_OR_VTA_EXECUTION')
    return {'status':'CPU_RAW_POINTER_LLVM_CONFIRMED','reachable_VTA_symbols':sorted(seen),
            'only_reachable_function_bodies_checked':True}


class HostCPUCallback:
    def __init__(self,module,manifest,device,ledger,invoke):
        require(type(device.device_type) is int and device.device_type%128==1 and device.device_id==0,
                'HOST_REFERENCE_REQUIRES_CPU0_STORAGE')
        require(callable(ledger) and callable(invoke),'HOST_REFERENCE_HELPERS_REQUIRED')
        self.module,self.manifest,self.device,self.ledger,self.invoke=module,manifest,device,ledger,invoke
        self.calls=[]

    def __call__(self,symbol,inputs,outputs,flatten):
        import tvm
        require(symbol in self.manifest,'UNBOUND_CPU_REFERENCE_SYMBOL')
        row=self.manifest[symbol]
        require(row.get('raw_pointer_gate',{}).get('status')=='CPU_RAW_POINTER_REFERENCE_SUPPORTED',
                'CPU_RAW_POINTER_REFERENCE_GATE_MISSING')
        require(row.get('instruction_gate',{}).get('status')=='NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION',
                'CPU_REFERENCE_INSTRUCTION_GATE_MISSING')
        abi=row['abi'];require(len(abi)==len(inputs)+len(outputs) and all(a['kind']=='buffer' for a in abi),
                               'CPU_REFERENCE_PARAMETER_ABI')
        args=[]
        for a,x in zip(abi,inputs):
            require(str(x.dtype)==a['dtype'] and x.size==int(np.prod(a['shape'])),'CPU_REFERENCE_INPUT_ABI')
            require(flatten or list(x.shape)==a['shape'],'CPU_REFERENCE_INPUT_SHAPE')
            args.append(tvm.nd.array(np.ascontiguousarray(x).reshape(a['shape']),self.device))
        for a,d in zip(abi[len(inputs):],outputs):
            require(a['dtype']==d['dtype'] and int(np.prod(a['shape']))==int(np.prod(d['shape'])),
                    'CPU_REFERENCE_OUTPUT_ABI')
            require(flatten or a['shape']==d['shape'],'CPU_REFERENCE_OUTPUT_SHAPE')
            args.append(tvm.nd.empty(a['shape'],a['dtype'],self.device))
        before=json.loads(self.ledger())
        self.invoke(self.module,symbol,*args)
        require(json.loads(self.ledger())==before,'CPU_REFERENCE_CHANGED_SUBMISSION_LEDGER')
        require(all(np.ascontiguousarray(a.numpy().reshape(x.shape)).tobytes()==np.ascontiguousarray(x).tobytes()
                    for a,x in zip(args,inputs)),'CPU_REFERENCE_MUTATED_READONLY_PARAMETER')
        result=[a.numpy().reshape(d['shape']) for a,d in zip(args[len(inputs):],outputs)]
        self.calls.append({'symbol':symbol,'trusted_CPU_function_calls':1,'VTA_submissions':0,
                           'argument_storage':'CPU RAM; temporary descriptor views only'})
        return result
