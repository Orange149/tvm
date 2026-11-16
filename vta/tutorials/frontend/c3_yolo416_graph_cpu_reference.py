"""Local FSim callback for the explicitly trusted, unchanged CPU graph path."""
import json
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def cpu_instruction_gate(fn):
    from tvm import tir
    forbidden=[]
    intrinsics={'tir.vta.uop_push','tir.vta.coproc_sync','tir.vta.coproc_dep_push','tir.vta.coproc_dep_pop'}
    externs={'VTAUopPush','VTAPushGEMMOp','VTAPushALUOp','VTALoadBuffer2D','VTAStoreBuffer2D',
             'VTADepPush','VTADepPop','VTASynchronize'}
    def visit(n):
        if not isinstance(n,tir.Call):return
        if n.op.name in intrinsics:forbidden.append(n.op.name)
        if n.op.name=='tir.call_extern' and n.args and isinstance(n.args[0],tir.StringImm) and n.args[0].value in externs:
            forbidden.append(n.args[0].value)
    tir.stmt_functor.post_order_visit(fn.body,visit)
    require(not forbidden,'CPU_REFERENCE_WOULD_ENQUEUE_VTA_INSTRUCTIONS')
    return {'status':'NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION'}


class CPUCallback:
    def __init__(self,module,manifest):
        import tvm
        self.module,self.manifest=module,manifest
        self.device=tvm.device('ext_dev',0)
        self.ledger=tvm.get_global_func('vta.runtime.official_submit_ledger')
        self.calls=[]

    def __call__(self,symbol,inputs,outputs,flatten):
        import tvm
        require(symbol in self.manifest,'UNBOUND_CPU_REFERENCE_SYMBOL')
        row=self.manifest[symbol]
        require(row.get('instruction_gate',{}).get('status')=='NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION',
                'CPU_REFERENCE_INSTRUCTION_GATE_MISSING')
        abi=row['abi']
        require(len(abi)==len(inputs)+len(outputs) and all(a['kind']=='buffer' for a in abi),
                'CPU_REFERENCE_PARAMETER_ABI')
        args=[]
        for a,x in zip(abi,inputs):
            require(str(x.dtype)==a['dtype'] and x.size==int(np.prod(a['shape'])), 'CPU_REFERENCE_INPUT_ABI')
            require(flatten or list(x.shape)==a['shape'],'CPU_REFERENCE_INPUT_SHAPE')
            args.append(tvm.nd.array(np.ascontiguousarray(x).reshape(a['shape']),self.device))
        for a,d in zip(abi[len(inputs):],outputs):
            require(a['dtype']==d['dtype'] and int(np.prod(a['shape']))==int(np.prod(d['shape'])),
                    'CPU_REFERENCE_OUTPUT_ABI')
            require(flatten or a['shape']==d['shape'],'CPU_REFERENCE_OUTPUT_SHAPE')
            args.append(tvm.nd.empty(a['shape'],a['dtype'],self.device))
        function=self.module.get_function(symbol,True)
        require(function is not None,'CPU_REFERENCE_FUNCTION_MISSING')
        before=json.loads(self.ledger())
        function(*args) # one CPU function, never a GraphExecutor/VTA convolution
        after=json.loads(self.ledger())
        require(before==after,'CPU_REFERENCE_CHANGED_SUBMISSION_LEDGER')
        require(all(np.ascontiguousarray(a.numpy().reshape(x.shape)).tobytes()==np.ascontiguousarray(x).tobytes()
                    for a,x in zip(args,inputs)),
                'CPU_REFERENCE_MUTATED_READONLY_PARAMETER')
        result=[a.numpy().reshape(d['shape']) for a,d in zip(args[len(inputs):],outputs)]
        self.calls.append({'symbol':symbol,'trusted_CPU_function_calls':1,'VTA_submissions':0})
        return result
