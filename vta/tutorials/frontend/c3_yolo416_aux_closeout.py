"""Close build-only ARM or reference-case export stages without target calls."""
import argparse
from pathlib import Path
import shutil
from c3_tps_sync_release_local import HERE,read,write,sha,verify,require


def seal(root,stage):
    require(not (root/'completion_manifest.json').exists(),'ALREADY_SEALED')
    status=read(root/'status.json')['status'];p=root/'freeze.json'
    bindings=read(p)['bindings'] if p.exists() else {};verify(bindings)
    wanted={'arm':'ALL_TH_P_ARM_MODULES_AND_GRAPH_HELPERS_BUILT_FPGA_PENDING',
            'cases':'TRAINED_MODEL_OPERATOR_CASES_EXPORTED_NO_TARGET_EXECUTION'}[stage]
    success=status==wanted
    outer=root/'outer_worker.json'
    if outer.exists():
        record=read(outer)
        success=success and record['returncode']==0 and record['resource_limit'] is None
    if success:
        require(not list(root.glob('failure*.json')),'FAILURE_STATE_PRESENT')
        if stage=='arm':
            from c3_yolo416_arm_prepare import elf_gate
            tasks=read(root/'tasks.json');require(len(tasks)==24,'ARM_TASK_COVERAGE')
            for i,t in enumerate(tasks):
                worker=read(root/('%02d_worker.json'%i));contract=read(root/t['key']/'contract.json')
                require(worker['returncode']==0 and worker['resource_limit'] is None,'ARM_WORKER_FAILED')
                require(contract['status']=='ARM_INITIALIZERS_MATCH_FSIM_FPGA_PENDING' and
                        contract['module_sha256']==sha(root/t['key']/'module.so'),'ARM_CONTRACT_CHANGED')
                proofs=read(root/t['key']/'initializer_comparison.json')
                require(len(proofs)==(1 if t['kind']=='operator' else 11) and all(
                    r['status']=='PER_TIR_SCOPE_INITIALIZER_AND_CACHE_SHARING_EQUAL' for r in proofs.values()),
                    'ARM_INITIALIZERS_NOT_EQUAL')
                elf_gate(root/t['key']/'module.so')
                if t['kind']=='graph':
                    require(sha(root/t['key']/'module.so')==sha(root/t['key']/'reference_arm.so'),
                            'ARM_REFERENCE_LIBRARY_CHANGED')
                    require(len(read(root/t['key']/'CPU_reference_manifest.json'))==21,'CPU_REFERENCE_COVERAGE')
            helper=read(root/'helpers_build.json');require(helper['returncode']==0 and helper['resource_limit'] is None,
                                                         'ARM_HELPERS_BUILD_FAILED')
            for name in ('graph_once','graph_observer','cpu_reference','rpc_observer'):elf_gate(root/(name+'_arm.so'))
        else:
            rows=read(root/'operator_cases.json');require(len(rows)==11,'MODEL_CASE_OBJECT_COVERAGE')
            require(len(read(root/'all_graph_call_provenance.json'))==36,'MODEL_CASE_CALLSITE_COVERAGE')
            for row in rows:
                require(len(row['cases'])==4 and row['benchmark_call_node']==row['graph_nodes'][0],
                        'MODEL_CASE_ROLE_CHANGED')
                a=row['cases'];require([x['seed'] for x in a[:3]]==[0,20250901,20260910],
                                      'MODEL_CASE_SEEDS_CHANGED')
                require(all(x['input_sha256'][1:]==a[0]['input_sha256'][1:] for x in a[:3]),
                        'MODEL_WEIGHT_OR_ADD_NOT_FIXED')
                require(a[3]['expected_sha256']!=a[0]['expected_sha256'],'W0_W1_NOT_OBSERVABLE')
    write(root/'closeout_validation.json',{'status':status,'success':success,'FPGA_calls':0,
          'new_VTA_target_calls':0,'new_GraphExecutor_calls':0,'performance_samples':0})
    text=['# YOLO416 '+('ARM构建' if stage=='arm' else '真实模型算子用例')+'封存','', '状态：'+status+'。','']
    if success and stage=='arm':
        text+=['22个独立融合函数及TH/P两份完整图完成ARM构建，逐作用域initializer、签名和缓存共享关系与FSim一致。',
               'CPU参考及诊断适配helper完成ARM编译，真实FPGA资格仍待执行。']
    elif success:
        text+=['11种函数使用完整图原训练权重及ADD，三个seed只改变网络输入；算子输入来自独立图参考。',
               '每个函数固定最早图调用作为单算子测量对象，节点29/57的两处真实参数身份仍完整记录并在整图中保留。',
               'W0→W1→W0的W1由固定独立参考规则选取且输出可观察。CPU参考单独调用受信CPU代码及NumPy卷积，没有运行GraphExecutor或VTA目标函数。']
    else:
        text+=['首次失败或限制处停止，不重试、不补调用、不形成性能结论。']
    text+=['FPGA调用0、正式样本0。构建或参考计算的本地时长不作为加速证据。']
    (root/'REPORT.md').open('x').write('\n\n'.join(text)+'\n')
    snapshots=root/'source_snapshots';snapshots.mkdir()
    for path,digest in bindings.items():
        path=Path(path)
        if path.parent==HERE and path.suffix in ('.py','.cc'):
            shutil.copyfile(path,snapshots/path.name);require(sha(snapshots/path.name)==digest,'SNAPSHOT_CHANGED')
    verify(bindings)
    write(root/'completion_manifest.json',{'status':'SEALED_'+status,'success':success,'FPGA_calls':0,'new_VTA_target_calls':0,
          'new_GraphExecutor_calls':0,'performance_samples':0,
          'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    return success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('arm','cases'));p.add_argument('output',type=Path)
    a=p.parse_args();print({'success':seal(a.output.resolve(),a.stage)})
