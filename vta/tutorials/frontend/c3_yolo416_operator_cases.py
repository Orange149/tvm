"""Export model-parameter operator cases from an independent graph reference.

No GraphExecutor.run or compiled VTA convolution. Actual trained W and ADD
remain fixed across three input seeds. A shared function uses its earliest
graph call for the one-operator benchmark; both graph calls stay documented.
"""
import argparse
from pathlib import Path
import sys
import traceback
from c3_tps_sync_release_local import HERE,RUNTIME,read,write,sha,verify,require


def export(root,qualified):
    root.mkdir(exist_ok=False);status='STOPPED_OPERATOR_CASE_INPUT'
    try:
        import numpy as np
        import tvm,vta
        from tvm import relay
        from vta.testing import simulator
        from c3_yolo416_graph_cpu_reference import CPUCallback
        from c3_yolo416_graph_local_v2 import make_cases,SEEDS
        from c3_official_window_reference import from_workload,array_sha,output_sentinel
        seal=read(qualified/'completion_manifest.json')
        require(read(qualified/'status.json')['status']==
            'WHOLE_TH_P_GRAPH_LOCAL_QUALIFIED_ARM_AND_FPGA_QUALIFICATION_PENDING','WHOLE_GRAPH_NOT_QUALIFIED')
        bindings={str(qualified/p):d for p,d in seal['artifact_sha256'].items()}
        bindings.update(read(qualified/'freeze.json')['bindings'])
        for p in HERE.glob('c3_*.py'):bindings[str(p.resolve())]=sha(p)
        verify(bindings);write(root/'freeze.json',{'bindings':bindings,'VTA_target_calls':0,'GraphExecutor_calls':0})
        contract=read(qualified/'contract.json');compiled=Path(contract['compiled']);prepared=Path(contract['prepared'])
        operators=Path(contract['operators']);units=read(operators/'objects.json')
        graph=read(prepared/'TH/graph.json');rows=read(compiled/'TH/functions.json')
        params={str(k):v.numpy() for k,v in relay.load_param_dict((prepared/'TH/params.bin').read_bytes()).items()}
        require(Path(simulator.LIBS[0]._name).resolve()==(RUNTIME/'libvta_fsim.so').resolve(),'CASE_RUNTIME_CHANGED')
        specs={}
        for row in rows:
            refs=[]
            for variant in ('TH','P'):
                i=next(i for i,u in enumerate(units) if u['variant']==variant and u['symbol']==row['symbol'])
                refs.append(read(operators/'objects'/('%02d'%i)/'prepare/R/reference_contract.json'))
            require(refs[0]==refs[1],'TH_P_COMPLETE_REFERENCE_CONTRACT_DIFFERS')
            spec=from_workload(row['workload'],row['abi'],refs[0]['postprocess'])
            require(spec.contract()==refs[0],'OPERATOR_CASE_REFERENCE_CHANGED');specs[row['symbol']]=spec
        lib=tvm.runtime.load_module(str(prepared/'TH/reference_fsim.so'))
        cpu=CPUCallback(lib,read(prepared/'TH/CPU_manifest.json'))
        status='STOPPED_OPERATOR_CASE_REFERENCE'
        cases,weight_name,observable=make_cases(graph,params,specs,cpu)
        require(len(cases)==6,'GRAPH_REFERENCE_CASE_COUNT_CHANGED')
        ledger=tvm.get_global_func('vta.runtime.official_submit_ledger')
        require(read(qualified/'TH_detailed/reference_protocol.json')['observability']==observable,
                'GRAPH_REFERENCE_WEIGHT_RULE_CHANGED')
        require(__import__('json').loads(ledger())['attempted_submits']==0,'CASE_EXPORT_ENQUEUED_VTA')
        table=[];all_calls=[]
        # Three distinct seeds, same actual model parameters. Full graph W1
        # cases are used only to verify the frozen observability rule above.
        for row in rows:
            folder=root/row['object_id'];folder.mkdir();spec=specs[row['symbol']]
            observations=[]
            for seed,case in zip(SEEDS,cases[:3]):
                entries=[(i,g) for i,g in enumerate(case['convolutions']) if g['symbol']==row['symbol']]
                require(len(entries)==len(row['nodes']),'SHARED_GRAPH_FUNCTION_MAPPING_CHANGED')
                for (i,g),node in zip(entries,row['nodes']):
                    all_calls.append({'seed':seed,'node':node,'symbol':row['symbol'],
                        'reference_position':i,'input_sha256':[array_sha(a) for a in g['inputs']],
                        'expected_sha256':array_sha(g['output'])})
                g=entries[0][1];arrays=[x.copy() for x in g['inputs']]
                require(np.array_equal(spec.reference(arrays),g['output']),'OPERATOR_GRAPH_REFERENCE_MISMATCH')
                expected=g['output'];sentinel=output_sentinel(expected)
                path=folder/('seed_%d_W0.npz'%seed)
                np.savez_compressed(path,**{'a%d'%i:a for i,a in enumerate([*arrays,sentinel])},expected=expected)
                observations.append({'seed':seed,'weight':'W0','path':str(path),
                    'input_sha256':[array_sha(a) for a in arrays],'expected_sha256':array_sha(expected)})
            require(all(x['input_sha256'][1:]==observations[0]['input_sha256'][1:] for x in observations),
                    'MODEL_WEIGHT_OR_ADD_CHANGED_ACROSS_SEEDS')
            first=np.load(observations[0]['path']);arrays=[first['a%d'%i].copy() for i in range(4)]
            arrays[1],expected=spec.observable_weights(arrays)
            require(not np.array_equal(expected,first['expected']),'OPERATOR_W0_W1_NOT_OBSERVABLE')
            arrays[3]=output_sentinel(expected);path=folder/'seed_0_W1.npz'
            np.savez_compressed(path,**{'a%d'%i:a for i,a in enumerate(arrays)},expected=expected)
            observations.append({'seed':0,'weight':'W1','path':str(path),
                'input_sha256':[array_sha(a) for a in arrays[:3]],'expected_sha256':array_sha(expected)})
            table.append({'object_id':row['object_id'],'symbol':row['symbol'],'graph_nodes':row['nodes'],
                'benchmark_call_node':row['nodes'][0],'abi':row['abi'],'cases':observations,
                'W0_and_ADD_provenance':'unchanged trained parameters from the frozen complete graph',
                'input_provenance':'independent NumPy integer graph reference with frozen local CPU component'})
        write(root/'operator_cases.json',table);write(root/'all_graph_call_provenance.json',all_calls)
        write(root/'reference_protocol.json',{'CPU_reference_calls':cpu.calls,'GraphExecutor_calls':0,
              'VTA_target_calls':0,'whole_graph_observable_weight':weight_name,'rule':observable,
              'source_graph':str(prepared/'TH'),'reference_CPU_is_shared_trusted_code':True})
        require(len(table)==11 and len(all_calls)==36,'OPERATOR_CASE_COVERAGE_CHANGED')
        verify(bindings);status='TRAINED_MODEL_OPERATOR_CASES_EXPORTED_NO_TARGET_EXECUTION'
    except Exception as e:
        write(root/'failure_controller.json',{'type':type(e).__name__,'message':str(e)});traceback.print_exc()
    write(root/'status.json',{'status':status,'VTA_target_calls':0,'GraphExecutor_calls':0,'FPGA_calls':0})
    print(status,flush=True)
    return 0 if status.startswith('TRAINED_MODEL_OPERATOR_CASES_EXPORTED') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--qualified',type=Path,required=True)
    a=p.parse_args();sys.exit(export(a.output.resolve(),a.qualified.resolve()))
