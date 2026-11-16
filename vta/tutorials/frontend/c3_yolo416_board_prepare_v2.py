"""Freeze TH/P identities, trained-model cases and complete board call ledger.

This entry point has no board transport or target invocation. All local and
ARM stages must already be sealed successfully. No timing informs its pool.
"""
import argparse
from pathlib import Path
import sys
import traceback
from c3_tps_sync_release_local import HERE,REPO,read,write,sha,verify,require
from c3_tps_aggregation_prepare import OFFICIAL,STORAGE,common_runtime_bindings,compare_initializer_bindings
from c3_yolo416_comparison_protocol import manifest
from c3_yolo416_board_transport import transport_contract


def bound(path,digest=None):
    path=Path(path).resolve();actual=sha(path)
    require(digest is None or digest==actual,'ARTIFACT_IDENTITY_CHANGED:'+str(path))
    return {'path':str(path),'sha256':actual}


def prepare(root,arm,cases,host,known_hosts):
    root.mkdir(exist_ok=False)
    try:
        import tvm,vta
        from c3_official_window_llvm import llvm_alu_contract
        from c3_official_window_baseline import normalize_llvm
        bindings={}
        for folder in (arm,cases):
            seal=read(folder/'completion_manifest.json')
            require(seal.get('success') is True,'AUXILIARY_STAGE_NOT_SEALED_SUCCESSFULLY')
            bindings.update({str(folder/p):d for p,d in seal['artifact_sha256'].items()})
            bindings[str(folder/'completion_manifest.json')]=sha(folder/'completion_manifest.json')
        require(read(arm/'status.json')['status']=='ALL_TH_P_ARM_MODULES_AND_GRAPH_HELPERS_BUILT_FPGA_PENDING',
                'ARM_PREPARATION_INCOMPLETE')
        require(read(cases/'status.json')['status']=='TRAINED_MODEL_OPERATOR_CASES_EXPORTED_NO_TARGET_EXECUTION',
                'TRAINED_MODEL_OPERATOR_CASES_MISSING')
        source=read(arm/'source_contract.json');operators=Path(source['operators'])
        graph_local=Path(source['whole_graph']);compiled=Path(source['compiled']);graph_prepared=Path(source['graph_prepared'])
        require(str(graph_local/'status.json') in read(cases/'freeze.json')['bindings'],'CASE_GRAPH_LINEAGE_MISSING')
        bindings.update(read(arm/'freeze.json')['bindings']);bindings.update(read(cases/'freeze.json')['bindings'])
        common=read(OFFICIAL/'runtime_contract.json');bindings.update(common_runtime_bindings(common))
        runtime=common['runtime'][1]
        helper=common['helper']['arm'];storage_build=read(STORAGE/'arm_build.json')
        hardware=read(operators/'hardware_archive.json');verify(hardware['source_bindings'])
        bit=hardware['bitstreams'][0]
        artifacts={'runtime':bound(runtime['library'],runtime['binary_sha256']),
            'helper':bound(helper['binary'],helper['binary_sha256']),
            'storage_helper':bound(STORAGE/'storage_arm.so',storage_build['binary_sha256']),
            'runtime_tvm':bound(REPO/'build_axu_aarch64/libtvm_runtime.so'),
            'rpc_server':bound(REPO/'build_axu_aarch64/tvm_rpc'),
            'bitstream':bound(bit['path'],bit['whole_sha256']),
            'hardware_archive':bound(operators/'hardware_archive.json')}
        for name in ('graph_once','graph_observer','rpc_observer','cpu_reference'):
            artifacts[name+'_helper']=bound(arm/(name+'_arm.so'))
        for value in artifacts.values():bindings[value['path']]=value['sha256']
        units=read(operators/'objects.json');case_rows=read(cases/'operator_cases.json')
        require(len(units)==22 and len(case_rows)==11,'OPERATOR_POOL_COVERAGE_CHANGED')
        descriptors={};pool=[];equivalence=[]
        for row in read(compiled/'TH/functions.json'):
            oid=row['object_id'];programs={};mods={};llvms={}
            case=next(c for c in case_rows if c['object_id']==oid)
            require(case['abi']==row['abi'] and case['benchmark_call_node']==row['nodes'][0],
                    'MODEL_CASE_ABI_OR_GRAPH_CALL_CHANGED')
            for variant in ('TH','P'):
                i=next(i for i,u in enumerate(units) if u['variant']==variant and u['object_id']==oid)
                item=units[i];module_dir=arm/'operators'/('%02d'%i)
                contract=read(module_dir/'contract.json')
                require(contract['status']=='ARM_INITIALIZERS_MATCH_FSIM_FPGA_PENDING' and
                        sha(module_dir/'module.so')==contract['module_sha256'],'ARM_OPERATOR_NOT_BOUND')
                mods[variant]=tvm.ir.load_json(Path(item['final']).read_text())
                llvms[variant]=(module_dir/'module.ll').read_text()
                descriptor={'kind':'operator','variant':variant,'object_id':oid,'symbol':item['symbol'],
                    'abi':item['abi'],'arm_module':str(module_dir/'module.so'),'parent_module':item['parent'],
                    'local_unit':str(operators/'objects'/('%02d'%i)),
                    'model_cases':case,'execution_identity':{'final_IR_sha256':sha(item['final']),
                        'arm_llvm_sha256':sha(module_dir/'module.ll')},
                    'source_contract':str(module_dir/'contract.json')}
                descriptors[oid+'/'+variant]=descriptor
                programs[variant]={'module':bound(module_dir/'module.so'),'symbol':item['symbol'],
                    'abi':item['abi'],'model_cases_manifest_sha256':sha(cases/'operator_cases.json'),
                    'execution_identity':descriptor['execution_identity']}
            same_tir=tvm.ir.structural_equal(mods['TH'],mods['P'],map_free_vars=True)
            same_llvm=normalize_llvm(llvms['TH'])==normalize_llvm(llvms['P'])
            same=same_tir or same_llvm
            if same:
                fns={v:list(mods[v].functions.values())[0] for v in ('TH','P')}
                proof=compare_initializer_bindings(llvms['TH'],llvms['P'],
                    llvm_alu_contract(fns['TH'],llvms['TH']),llvm_alu_contract(fns['P'],llvms['P']))
                equality={'source_final_IR_structurally_equal':same_tir,
                          'normalized_final_LLVM_equal':same_llvm,'initializer_comparison':proof}
                path=root/(oid+'_equivalence.json');write(path,equality)
                identity={'structural_equivalence_proof_sha256':sha(path)}
                for v in programs:programs[v]['execution_identity']=identity
            equivalence.append(same)
            pool.append({'object_id':oid,'kind':'operator','programs':programs,
                         'representatives':{'TH':'TH','P':'TH' if same else 'P'}})
        graph_programs={}
        for variant in ('TH','P'):
            module_dir=arm/'graph'/variant;contract=read(module_dir/'contract.json')
            require(sha(module_dir/'module.so')==contract['module_sha256'],'ARM_GRAPH_MODULE_CHANGED')
            descriptor={'kind':'graph','variant':variant,'object_id':'whole_graph',
                'arm_module':str(module_dir/'module.so'),'reference_module':str(module_dir/'reference_arm.so'),
                'CPU_reference_manifest':str(module_dir/'CPU_reference_manifest.json'),
                'graph_prepared':str(graph_prepared/variant),'local_unit':str(graph_local),
                'compiled_source':str(compiled/variant),'operator_qualification':str(operators)}
            descriptors['whole_graph/'+variant]=descriptor
            graph_programs[variant]={'module':bound(module_dir/'module.so'),
                'graph_sha256':sha(graph_prepared/variant/'graph.json'),
                'params_sha256':sha(graph_prepared/variant/'params.bin'),
                'execution_identity':{'arm_llvm_sha256':sha(module_dir/'module.ll')}}
        graph_same=all(equivalence)
        require((graph_prepared/'TH/graph.json').read_bytes()==(graph_prepared/'P/graph.json').read_bytes(),
                'GRAPH_STRUCTURE_DIFFERS')
        from c3_yolo416_parameter_identity import compare as compare_parameters
        write(root/'TH_P_parameter_identity.json',compare_parameters(
            graph_prepared/'TH/params.bin',graph_prepared/'P/params.bin'))
        if graph_same:
            common_id={'all_11_fused_execution_representations_equal':True,
                'graph_sha256':sha(graph_prepared/'TH/graph.json'),'params_sha256':sha(graph_prepared/'TH/params.bin')}
            for v in graph_programs:graph_programs[v]['execution_identity']=common_id
        pool.append({'object_id':'whole_graph','kind':'graph','programs':graph_programs,
                     'representatives':{'TH':'TH','P':'TH' if graph_same else 'P'}})
        artifacts['programs']=descriptors;write(root/'artifacts.json',artifacts)
        ledger=manifest(root.name,pool);write(root/'call_manifest.json',ledger)
        write(root/'transport_contract.json',transport_contract(ledger))
        for p in [*HERE.glob('c3_*.py'),*HERE.glob('c3_*.cc'),*HERE.glob('test_c3_yolo416*.py'),Path(known_hosts)]:
            bindings[str(p.resolve())]=sha(p)
        for p in root.rglob('*'):
            if p.is_file():bindings[str(p.resolve())]=sha(p)
        verify(bindings)
        write(root/'freeze.json',{'status':'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING','bindings':bindings,
            'host':host,'known_hosts':str(Path(known_hosts).resolve()),'target_calls':ledger['total_target_calls'],
            'sessions':len(ledger['sessions']),'formal_samples':ledger['formal_samples'],
            'I_equals_P':True,'new_FSim_calls':0,'FPGA_calls':0,'no_retry':True,
            'no_SD_write_reboot_or_external_service_termination':True})
        return True
    except Exception as e:
        write(root/'prepare_failure.json',{'type':type(e).__name__,'message':str(e),'FPGA_calls':0})
        traceback.print_exc();return False


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--arm',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--host',default='192.168.1.247')
    p.add_argument('--known-hosts',default='/home/orange/.ssh/known_hosts');a=p.parse_args()
    ok=prepare(a.output.resolve(),a.arm.resolve(),a.cases.resolve(),a.host,a.known_hosts)
    print({'success':ok});sys.exit(0 if ok else 1)
