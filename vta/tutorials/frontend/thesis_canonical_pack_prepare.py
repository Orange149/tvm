#!/usr/bin/env python3
"""Prepare packed P1/P2 IR and check its integer layout algebra on LLVM."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
import tvm
from tvm import autotvm, relay
from tvm.contrib import graph_executor
from vta.top.graphpack import graph_pack
from thesis_canonical_graph_split import UnitGraph, Replace, operation
from thesis_supplement_run import sha256, write_new_json


def typed(function):
    return relay.transform.InferType()(tvm.IRModule.from_expr(function))


def unpack(data, shape):
    no, co, h, w, ni, ci = map(int, shape)
    return relay.reshape(relay.transpose(data, axes=[0,4,1,5,2,3]), newshape=[no*ni,co*ci,h,w])


def pack(data, shape, batch, channel):
    n,c,h,w=map(int,shape)
    return relay.transpose(relay.reshape(data,newshape=[n//batch,batch,c//channel,channel,h,w]),axes=[0,2,4,5,1,3])


class PackedConvCPU(relay.ExprMutator):
    def visit_call(self, call):
        args=[self.visit(arg) for arg in call.args]
        if operation(call,'nn.conv2d'):
            match=re.fullmatch(r'NCHW(\d+)n(\d+)c',str(call.attrs.data_layout))
            if not match or str(call.attrs.kernel_layout)!=f'OIHW{match[2]}o{match[2]}i':
                raise ValueError('Unexpected packed convolution convention')
            batch,channel=map(int,match.groups())
            data=unpack(args[0],call.args[0].checked_type.shape)
            weight=unpack(args[1],call.args[1].checked_type.shape)
            convolution=relay.nn.conv2d(data,weight,strides=call.attrs.strides,padding=call.attrs.padding,
                dilation=call.attrs.dilation,groups=call.attrs.groups,channels=call.attrs.channels,
                kernel_size=call.attrs.kernel_size,data_layout='NCHW',kernel_layout='OIHW',out_dtype=call.attrs.out_dtype)
            no,co,h,w,ni,ci=map(int,call.checked_type.shape)
            return pack(convolution,[no*ni,co*ci,h,w],batch,channel)
        return relay.Call(call.op,args,call.attrs,call.type_args,call.span)


def prepare(canonical,qualification,output):
    output.mkdir(parents=True,exist_ok=False)
    graph=UnitGraph(tvm.ir.load_json(canonical.read_text()))
    contract={'scope':'Local packed IR and independent NCHW LLVM convolution substitution; no VTA/simulator or hardware qualification',
              'canonical_qir_sha256':sha256(canonical),'batch_factor':1,'channel_factor':16,'weight_bits':8,
              'extra_quantize_calls':0,'programs':[{'name':'P1','VTA_units':[3,17]},{'name':'P2','VTA_units':[3,15]}],
              'diagnostics':'Reuse exact zero/uniform/normal input/intermediate bytes from prior frozen host qualification',
              'stage02_qualified':False,'board_accessed':False}
    write_new_json(output/'contract.json',contract)
    results=[]
    for label,end in [('P1',18),('P2',16)]:
        directory=output/label
        directory.mkdir()
        original,inputs,outputs=graph.stage(3,end)
        function=original['main']
        replacements={var:relay.Call(tvm.ir.Op.get('annotation.bitpack_start'),[var]) for var in function.params}
        wrapped=Replace(replacements).visit(function.body)
        wrapped=relay.Tuple([relay.Call(tvm.ir.Op.get('annotation.bitpack_end'),[field]) for field in wrapped.fields])
        wrapped=typed(relay.Function(list(function.params),wrapped))['main']
        packed=typed(graph_pack(wrapped,1,16,8,start_name=None,stop_name=None,boundary_bridge=True))
        if not tvm.ir.structural_equal(packed['main'].checked_type,original['main'].checked_type):
            raise ValueError('Packing changed the external dtype/shape interface')
        counts=Counter()
        relay.analysis.post_order_visit(packed['main'].body,lambda n:counts.update([str(n.attrs.data_layout)]) if operation(n,'nn.conv2d') else None)
        if dict(counts)!={'NCHW1n16c':15 if label=='P1' else 14}:
            raise ValueError('Packed convolution count differs from the canonical unit range')
        (directory/'packed_qir.json').open('x').write(tvm.ir.save_json(packed))
        (directory/'packed_qir.txt').open('x').write(packed.astext(show_meta_data=False))
        emulated=typed(PackedConvCPU().visit(packed['main']))
        (directory/'llvm_algebra_qir.json').open('x').write(tvm.ir.save_json(emulated))
        with autotvm.apply_history_best([]),tvm.transform.PassContext(opt_level=3,disabled_pass=['AlterOpLayout']):
            factory=relay.build(emulated,target='llvm')
        factory.get_lib().export_library(str(directory/'llvm_algebra.so'))
        exe=graph_executor.GraphModule(factory['default'](tvm.cpu()))
        checks=[]
        for case in ['zeros','uniform','normal']:
            case_dir=directory/case
            case_dir.mkdir()
            with np.load(qualification/case/'canonical_boundaries.npz',allow_pickle=False) as saved:
                for i,node in enumerate(inputs):
                    values=saved[f'node_{graph.ids[node]}']
                    exe.set_input(f'edge3_{i}',values)
                    np.save(case_dir/f'input{i}.npy',values,allow_pickle=False)
                expected=[saved[f'node_{graph.ids[node]}'] for node in outputs]
            exe.run()
            for i,reference in enumerate(expected):
                actual=exe.get_output(i).numpy()
                np.save(case_dir/f'output{i}.npy',actual,allow_pickle=False)
                np.save(case_dir/f'reference{i}.npy',reference,allow_pickle=False)
                if actual.dtype!=reference.dtype or actual.shape!=reference.shape or not np.array_equal(actual,reference):
                    raise ValueError('Packing algebra output changed integer semantics')
                checks.append({'input':case,'output_index':i,'node_id':graph.ids[outputs[i]],
                               'dtype':str(actual.dtype),'shape':list(actual.shape),'integer_exact':True})
        result={'program':label,'VTA_unit_begin':3,'VTA_unit_end_exclusive':end,'packed_convolutions':sum(counts.values()),
                'input_arity':len(inputs),'output_arity':len(outputs),'checks':checks,
                'output_sources':['executor_output' if node not in inputs else 'live_input' for node in outputs]}
        write_new_json(directory/'checks.json',result)
        results.append(result)
        print(label+': packed layout algebra integer-exact on three synthetic inputs',flush=True)
    summary={'status':'LOCAL_PACKED_LAYOUT_ALGEBRA_PASS','programs':results,'extra_quantize_calls':0,
             'board_accessed':False,'VTA_lowering_or_execution_qualified':False,'stage02_qualified':False}
    write_new_json(output/'summary.json',summary)
    write_new_json(output/'artifact_hashes.json',{str(p.relative_to(output)):{'sha256':sha256(p),'bytes':p.stat().st_size}
                                                for p in sorted(output.rglob('*')) if p.is_file()})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-qir',type=Path,required=True)
    p.add_argument('--host-qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.canonical_qir,args.host_qualification,args.output),indent=2))
