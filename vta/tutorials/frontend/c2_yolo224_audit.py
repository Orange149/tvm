#!/usr/bin/env python3
"""Certify the actual YOLO graph, ARM CPU reader and allocation plan."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from c2_yolo224_prepare import ROOT,HERE,sha,write
from audit_shared_buffer_storage import audit_package
from make_shared_buffer_plan import make_plan
from c3_input_early_release_audit import SDK_OBJDUMP,POOL_LAUNCH,direct_calls,certify_wait_loop,instructions


def llvm_functions(text):
    return {m.group(1):m.group(0) for m in re.finditer(r'^define [^\n]*?@([^ (]+)\([^\n]*\)[^{\n]*\{\n.*?^\}',text,re.M|re.S)}


def audit_reader(out):
    package=out/'package'; stage=package/'stages/stage1'
    graph=json.loads((stage/'graph.json').read_text())
    nodes=graph['nodes'];rows=graph['node_row_ptr'];attrs={k:v[1] for k,v in graph['attrs'].items()}
    input_node=next(i for i,n in enumerate(nodes) if n['name']=='stage1_in0')
    entry=rows[input_node];sid=attrs['storage_id'][entry]
    readers=[(i,port) for i,n in enumerate(nodes) for port,r in enumerate(n['inputs']) if r[:2]==[input_node,0]]
    if len(readers)!=1 or [i for i,s in enumerate(attrs['storage_id']) if s==sid]!=[entry]:
        raise ValueError('input is not exclusive or has multiple readers')
    node,port=readers[0];reader=nodes[node]['attrs']['func_name']
    if port!=0 or nodes[node]['attrs']['num_inputs']!='1' or nodes[node]['attrs']['num_outputs']!='1':
        raise ValueError('unknown last-reader ABI')
    if any(h[0]==input_node for h in graph['heads']):raise ValueError('input exported as output')
    functions=llvm_functions((stage/'module.ll').read_text())
    compute=functions[reader+'_compute_']
    callbacks=re.findall(r'@(__tvm_parallel_lambda[.\d]*)',compute)
    if len(set(callbacks))!=1 or '@__TVMBackendParallelLaunch' not in compute:
        raise ValueError('synchronous CPU parallel callback not resolved')
    callback=callbacks[0];leaf=functions[callback]
    # Callback has no calls except LLVM intrinsics and no escaping function pointers.
    real_calls=[line for line in leaf.splitlines() if re.search(r'\bcall\b',line) and '@llvm.' not in line]
    if real_calls:raise ValueError('CPU callback contains unknown call: '+str(real_calls))
    hardware=('VTALoadBuffer2D','VTAStoreBuffer2D','VTAPushGEMMOp','VTAPushALUOp','VTASynchronize','VTADeviceRun')
    if any(x in functions[reader]+compute+leaf for x in hardware):raise ValueError('reader enqueues hardware reads')
    evidence=out/'reader_evidence';attempt=1
    while evidence.exists():
        attempt+=1;evidence=out/f'reader_evidence_attempt{attempt:02d}'
    evidence.mkdir()
    def dump(label,path,*args):
        result=subprocess.run([str(SDK_OBJDUMP),*args,str(path)],check=True,capture_output=True,text=True)
        (evidence/(label+'.txt')).write_text(result.stdout)
        return result.stdout
    library=stage/'graphlib.so';runtime=package/'libtvm_runtime.so';driver=package/'libvta.so'
    wrapper=dump('wrapper_arm',library,'-d','--disassemble='+reader)
    body=dump('compute_arm',library,'-d','--disassemble='+reader+'_compute_')
    # LLVM-private callbacks have no ELF symbol. Resolve the actual callback
    # passed in x0 to ParallelLaunch, then bound it by the next .text function.
    pair=re.search(r'adrp\s+x0,\s*([0-9a-f]+)[^\n]*\n[^\n]*add\s+x0, x0, #0x([0-9a-f]+)',body)
    if not pair:raise ValueError('ARM callback address not resolved')
    callback_address=int(pair[1],16)+int(pair[2],16)
    symbols=dump('symbol_table',library,'-t')
    following=[int(m[1],16) for m in re.finditer(r'^([0-9a-f]+)\s+.*?\bF\s+\.text\s+',symbols,re.M)
               if int(m[1],16)>callback_address]
    if not following:raise ValueError('ARM callback extent not bounded')
    callback_stop=min(following)
    arm_leaf=dump('callback_arm',library,'-d','--start-address='+hex(callback_address),'--stop-address='+hex(callback_stop))
    reloc=dump('relocations',library,'-R')
    if direct_calls(wrapper)!=['VTATLSCommandHandle',reader+'_compute_']:
        raise ValueError('unreviewed wrapper ARM call graph: '+str(direct_calls(wrapper)))
    if direct_calls(body)!=['VTABufferCPUPtr','VTABufferCPUPtr'] or sum(x['mnemonic']=='blr' for x in instructions(body))!=1:
        raise ValueError('unreviewed compute ARM call graph')
    if '__TVMBackendParallelLaunch' not in reloc:raise ValueError('parallel launch relocation missing')
    code=[x for x in instructions(arm_leaf) if x['mnemonic']!='nop']
    if not code or code[-1]['mnemonic']!='ret' or any(x['mnemonic'] in ('bl','blr','br','svc','hvc','smc') for x in code):
        raise ValueError('callback ARM contains unknown call/effect')
    launch=dump('parallel_launch_arm',runtime,'-d','--disassemble=TVMBackendParallelLaunch')
    pool=dump('thread_pool_arm',runtime,'-d','--disassemble='+POOL_LAUNCH)
    if not any(POOL_LAUNCH in symbol for symbol in direct_calls(launch)):
        raise ValueError('TVMBackendParallelLaunch does not call certified pool')
    wait=certify_wait_loop(pool)
    pointer=dump('cpu_pointer_arm',driver,'-d','--disassemble=VTABufferCPUPtr')
    if any(x in pointer for x in hardware):raise ValueError('CPU pointer resolution enqueues device work')
    for label,body_text in [('wrapper_llvm',functions[reader]),('compute_llvm',compute),('callback_llvm',leaf)]:
        (evidence/(label+'.ll')).write_text(body_text+'\n')
    result={'schema':'c2-yolo224-last-reader-v1','status':'static_qualified_dynamic_pending',
        'input_node':input_node,'input_entry':entry,'storage_id':sid,'node':node,'function':reader,
        'callback':callback,'callback_ARM_range':[hex(callback_address),hex(callback_stop)],
        'input_shape':attrs['shape'][entry],'input_dtype':attrs['dltype'][entry],
        'single_direct_reader':readers,'input_storage_alias_entries':[entry],
        'reader_output_entry':rows[node],'reader_output_shape':attrs['shape'][rows[node]],
        'reader_output_dtype':attrs['dltype'][rows[node]],'reader_has_no_hardware_commands':True,
        'callback_has_no_external_calls':True,'synchronous_wait':wait,
        'identities':{'graph':sha(stage/'graph.json'),'library':sha(library),
                      'runtime':sha(runtime),'driver':sha(driver),'llvm':sha(stage/'module.ll')},
        'dynamic_requirements':['actual reader input/output disjoint in virtual and physical ranges',
            'all later operator input/output ranges disjoint from original input during qualification',
            'all eight outputs byte-identical; generation release exactly once; controlled overwrite'],
        'evidence':{p.name:sha(p) for p in evidence.iterdir()}}
    write(out/'last_reader_certificate.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve();package=out/'package'
    paths=[p for p in package.rglob('*') if p.is_file() and (p.name in ('manifest.json','libtvm_runtime.so','libvta.so','c2_yolo224_runner')
           or p.name in ('graph.json','params.params','graphlib.so'))]
    checks=out/'package_checks.sha256'
    checks.write_text(''.join(sha(p)+'  '+str(p.relative_to(package))+'\n' for p in sorted(paths)))
    audit=audit_package(package,checks);audit['audit_source_sha256']=sha(HERE/'audit_shared_buffer_storage.py')
    if not (out/'storage_audit.json').exists():write(out/'storage_audit.json',audit)
    elif json.loads((out/'storage_audit.json').read_text())!=json.loads(json.dumps(audit)):raise ValueError('audit identity drift')
    build=json.loads((out/'runtime/build_manifest.json').read_text())
    plan=make_plan(audit,build)
    if not (out/'shared_buffer_plan.json').exists():write(out/'shared_buffer_plan.json',plan)
    elif json.loads((out/'shared_buffer_plan.json').read_text())!=plan:raise ValueError('shared plan drift')
    certificate=audit_reader(out)
    print(json.dumps({'graph_pool_bytes':audit['graph_pool_aligned_bytes'],
        'new_external_K2_bytes':audit['external_k2_bytes'],'anchor_saved':audit['predicted_saved_bytes'],
        'last_reader':certificate['function'],'static_last_reader':True}),flush=True)


if __name__=='__main__':main()
