"""Freeze real F / unchanged fixed oc2 TPS T / identity reuse U; no old samples.

Only the two already selected YOLO224 complete fused operators are compared.
The generic historical session worker is reused without relabeling its program
slots. Its ready-state string is an internal compatibility constant only.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import traceback
from c3_tps_sync_release_local import HERE, REPO, HOLD, read, write, sha, verify, require, RUNTIME, HELPER, PYTHON, env_for_worker
from c3_fallback_identity_protocol import manifest, analyse, transport_contract
from c3_tps_broader_baseline_compare import bound, bind_seal, arm_build

LOCAL = HOLD/'20261019_c3_fallback_identity_local_run03'


def qualified_unit(unit, bindings):
    if (unit/'completion_manifest.json').exists():
        bind_seal(unit, bindings)
    else:
        require(unit.parent == LOCAL, 'LOCAL_QUALIFICATION_SEAL_MISSING')
    for mode in ('detailed', 'minimal'):
        summary=read(unit/('R_'+mode)/'summary.json')
        require(summary['status']=='FULL_LOCAL_QUALIFICATION_PASSED' and summary['target_calls']==7 and len(summary['calls'])==7 and
                all(c['status']=='PASS' and c['correct'] is True for c in summary['calls']),
                'SEVEN_CALL_MODE_QUALIFICATION_REQUIRED')
    # These sources are identical across F/T/U; historical timing rows are not adopted.
    original=read(unit/'freeze.json')['bindings']
    for path in (RUNTIME/'libvta_fsim.so', HELPER):
        require(original[str(path)]==sha(path), 'LOCAL_COMMON_RUNTIME_CHANGED')
        bindings[str(path)]=sha(path)


def prepare(root, host, known_hosts):
    import numpy as np
    from c3_tps_aggregation_board import make_cases
    from c3_official_window_reference import array_sha
    root.mkdir(exist_ok=False)
    try:
        with (root/'tool_preflight.log').open('x') as log:
            code=subprocess.run([str(PYTHON),'-m','unittest','test_c3_fallback_identity_local',
                'test_c3_yolo416_qualification_v2','test_c3_fallback_identity_protocol',
                'test_c3_fallback_identity_compare','test_c3_fallback_physical','-v'],env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT).returncode
        require(code==0,'ZERO_TARGET_TOOL_PREFLIGHT_FAILED')
        bindings={}
        require(bind_seal(LOCAL, bindings)['status']=='TWO_AXU_FALLBACKS_LOCAL_QUALIFIED',
                'ACTUAL_FALLBACK_QUALIFICATION_REQUIRED')
        verify(read(LOCAL/'freeze.json')['bindings'])
        programs,pool,common={},{},None
        pool=[]
        for node in (34,46):
            unit=LOCAL/('node%d'%node)
            d=read(unit/'prepare/result.json')['programs']['R']
            verify(d['artifact_bindings'])
            qualified_unit(unit, bindings)
            dispatch=read(unit/'prepare/R/fallback_dispatch.json')
            require(dispatch['history_used'] is False and dispatch['explicit_residency_used'] is False and
                    all(c['is_fallback'] and c['target_policy']=='axu5evb' for c in dispatch['actual_configs']),
                    'REAL_AXU_FALLBACK_PROVENANCE_REQUIRED')
            arm=arm_build(unit/'prepare/R',root/('node%d_F_arm'%node))
            reference=read(d['reference_contract'])
            old=HOLD/('20261017_c3_node%d_uop_identity_board_run01'%node)
            bind_seal(old,bindings)
            old_artifacts=read(old/'artifacts/prepared.json')
            keys=('runtime','helper','storage_helper','runtime_tvm','rpc_server','bitstream','hardware_archive')
            current={k:old_artifacts[k] for k in keys}
            if common is None:common=current
            require(all(current[k]['sha256']==common[k]['sha256'] for k in keys), 'COMMON_ARM_BASE_DIFFERS')
            variants={'F':{'arm_module':str(arm),'unit':unit}}
            adoption={}
            for variant,old_name in (('T','B'),('U','A')):
                prior=old_artifacts['programs'][old_name]
                mapping={prior[k]:digest for k,digest in prior['sha256'].items()}
                verify(mapping);bindings.update(mapping)
                prior_unit=Path(prior['qualified_directory']).parents[1]
                qualified_unit(prior_unit,bindings)
                require(read(prior['reference_contract'])==reference and prior['abi']==d['abi'],
                        'F_T_U_FULL_FUSION_OR_ABI_DIFFERS')
                variants[variant]={'arm_module':prior['arm_module'],'unit':prior_unit}
                adoption[variant]={'original_name':old_name,'unit':str(prior_unit),
                                   'ARM_contract':bound(prior['arm_contract']), 'timing_samples_adopted':0}
            write(root/('node%d_qualification_adoption.json'%node),adoption)
            cases,ids=make_cases(reference)
            require(ids['0:W0']['reference']!=ids['0:W1']['reference'],'W0_W1_OUTPUTS_NOT_DISTINCT')
            folder=root/('node%d_cases'%node);folder.mkdir()
            rows=[]
            for (seed,weight),(arrays,expected) in cases.items():
                path=folder/('%d_%s.npz'%(seed,weight))
                require(np.all(arrays[-1]!=expected),'OUTPUT_SENTINEL_MISSING')
                np.savez(path,**{'a%d'%i:a for i,a in enumerate(arrays)},expected=expected)
                rows.append({'seed':seed,'weight':weight,'path':str(path),
                             'input_sha256':[array_sha(a) for a in arrays[:3]],'expected_sha256':array_sha(expected)})
            model_cases={'cases':rows,'origin':'independent seeds, equal for F/T/U; no trained graph timing'}
            object_id='yolo224_node%d'%node
            identities={}
            for variant,info in variants.items():
                local=info['unit'];module=Path(info['arm_module'])
                parent=local/'prepare/R/parent.json';final=local/'prepare/R/final.json'
                execution={'module_sha256':sha(module),'final_IR_sha256':sha(final)}
                descriptor={'kind':'operator','variant':variant,'object_id':object_id,
                            'symbol':d['symbol'],'abi':d['abi'],'arm_module':str(module),
                            'parent_module':str(parent),'local_unit':str(local),'model_cases':model_cases,
                            'execution_identity':execution}
                programs[object_id+'/'+variant]=descriptor
                identities[variant]={'variant':variant,'module':bound(module),'source_final':bound(final),
                                     'source_local':str(local),'semantic_reference_sha256':sha(d['reference_contract']),
                                     'execution_identity':execution}
            require(len({json.dumps(i['execution_identity'],sort_keys=True) for i in identities.values()})==3,
                    'F_T_U_STRUCTURAL_ALIAS_REQUIRES_NEW_LEDGER')
            pool.append({'object_id':object_id,'kind':'operator','programs':identities})
        for item in common.values():
            require(sha(item['path'])==item['sha256'],'COMMON_ARM_IDENTITY_CHANGED')
            bindings[item['path']]=item['sha256']
        write(root/'artifacts.json',{**common,'programs':programs})
        ledger=manifest(root.name,pool)
        require(ledger['total_target_calls']==360 and len(ledger['sessions'])==60 and
                ledger['formal_samples']==108,'FIXED_THREE_WAY_BUDGET')
        write(root/'call_manifest.json',ledger)
        write(root/'transport_contract.json',transport_contract(ledger))
        write(root/'comparison_semantics.json',{'network':'YOLOv3-tiny-224','nodes':[34,46],
            'F':'actual locally adapted AXU5EVB default safe fallback; not upstream original',
            'T':'sealed fixed oc2/h1 local adapted TPS, without identity reuse',
            'U':'same T with content-equivalent cache identity reuse only',
            'T_over_F':'existing TPS route comparison','U_over_T':'identity method increment',
            'U_over_F':'combined software route, not solely new method attribution',
            'whole_graph_or_network_FPS_measured':False,'old_samples_adopted':0,
            'stop_on_first_failure_no_retry':True,'no_SD_writes_reboot_external_service_termination':True,
            'generic_session_ready_status_legacy_string_is_not_a_TopHub_claim':True})
        for path in [*HERE.glob('c3_*.py'),*HERE.glob('test_c3_fallback*.py'),Path(known_hosts)]:
            bindings[str(path.resolve())]=sha(path)
        for path in root.rglob('*'):
            if path.is_file():bindings[str(path.resolve())]=sha(path)
        verify(bindings)
        write(root/'freeze.json',{'status':'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING',
            'actual_comparison':'F_T_U_TWO_YOLO224_OPERATORS','bindings':bindings,'host':host,
            'known_hosts':str(Path(known_hosts).resolve()),'FPGA_calls':0,
            'target_calls':360,'formal_samples':108,'sessions':60})
        print(json.dumps({'status':'F_T_U_READY','planned_calls':360,'FPGA_calls':0}),flush=True)
        return 0
    except Exception as error:
        write(root/'prepare_failure.json',{'type':type(error).__name__,'message':str(error),'FPGA_calls':0})
        traceback.print_exc()
        return 1


def report(root):
    from c3_yolo416_board_execute import accounting
    require(not (root/'completion_manifest.json').exists(),'ATTEMPT_ALREADY_SEALED')
    account=accounting(root/'board')
    success=read(root/'board/summary.json')['status']=='COMPLETE_F_T_U_SINGLE_BOOT_ATTEMPT'
    result={'status':'F_T_U_COMPLETE' if success else 'F_T_U_STOPPED','success':success,'accounting':account}
    if success:
        verify(read(root/'freeze.json')['bindings'])
        rows=read(root/'board/results.json')
        result['statistics']=analyse(read(root/'call_manifest.json'),rows)
        require(account['certified']==360 and not account['execution_unresolved_units'] and not account['invalid_records'],
                'CALL_ACCOUNTING_INCOMPLETE')
    write(root/'result.json',result)
    write(root/'completion_manifest.json',{'status':result['status'],'success':success,'paused':True,
        'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print(json.dumps({k:v for k,v in result.items() if k!='accounting'},ensure_ascii=False),flush=True)
    return 0 if success else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','board','report']);p.add_argument('output',type=Path)
    p.add_argument('--host',default='192.168.1.247');p.add_argument('--known-hosts',default='/home/orange/.ssh/known_hosts')
    args=p.parse_args()
    if args.phase=='prepare':sys.exit(prepare(args.output.resolve(),args.host,args.known_hosts))
    if args.phase=='board':
        from c3_fallback_identity_execute import execute
        sys.exit(execute(args.output.resolve()))
    sys.exit(report(args.output.resolve()))
