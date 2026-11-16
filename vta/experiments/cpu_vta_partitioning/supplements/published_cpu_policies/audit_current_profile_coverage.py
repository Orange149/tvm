"""List matched-input gaps for existing current layouts; no board or replay.

This audit deliberately does not synthesize missing per-kernel CPU traces or
split unmeasured first/last host CPU work from a pipeline performance target.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read, write, sha256
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/matched_device_profile_20261112_01/input_coverage'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_matched_device_profile'


def cpu_support(stage, module_hash, network, profiles):
    found=[(key,value) for key,value in profiles.items()
        if value['identity']['network']==network and value['identity']['program']==stage['program']
        and value['identity']['threads']==stage['threads']
        and value['identity']['module_sha256']==module_hash]
    exact=[key for key,value in found if value['configuration']['stages'][0]['worker_core_map']==stage['worker_core_map']]
    if exact:
        return dict(kind='matching_CPU_phase_structure',profiles=exact,
                    other_executor_and_measurement_provenance_still_required=True)
    if found:
        return dict(kind='phase_structure_requires_explicit_homogeneous_core_permutation',
                    profiles=[key for key,_ in found],
                    not_an_exact_affinity_measurement=True)
    if stage['threads']==1:
        return dict(kind='single_thread_no_fork_join_but_only_aggregate_cost',
                    per_kernel_CPU_trace_missing=True,aggregate_not_a_measured_continuous_busy_interval=True)
    return dict(kind='missing_multithread_fork_join_structure',
                aggregate_shares_do_not_determine_kernel_order=True)


def run():
    if OUT.exists():raise FileExistsError('preserve the offline coverage audit')
    profiles_path=ROOT/'artifacts/cross_network_model_01/phase_validation/ISOLATED_CPU_INPUTS.json'
    inputs=read(profiles_path)
    if not inputs['no_pipeline_FPS']:raise ValueError('CPU projection contains pipeline data')
    sources={str(profiles_path):sha256(profiles_path),str(Path(__file__)):sha256(__file__)}
    for path, expected in inputs['sources'].items():
        if sha256(path)!=expected:raise ValueError('isolated CPU source changed')
    pool_path=ROOT/'artifacts/window_tail_check_20261112_01/POOL.json'
    pool=read(pool_path);sources[str(pool_path)]=sha256(pool_path)
    rows=[]
    for network,configs in pool.items():
        path=ROOT/'artifacts/window_tail_check_20261112_01'/network/'capsule/plans/programs.json'
        programs=read(path);sources[str(path)]=sha256(path)
        for cid,cfg in configs.items():
            for s in cfg['stages']:
                if s['device']!='cpu':continue
                row=dict(network=network,configuration=cid,program=s['program'],threads=s['threads'],
                    requested_map=s['worker_core_map'],module_sha256=programs[s['program']]['files']['module.so']['sha256'])
                row.update(cpu_support(s,row['module_sha256'],network,inputs['profiles']))
                rows.append(row)
    distinct_missing=sorted({(r['network'],r['program'],r['threads']) for r in rows
                            if r['kind']=='missing_multithread_fork_join_structure'})
    host=dict(current_matched_VTA_profiles=3,
        available=['device completion brackets','driver wall and thread CPU',
                   'CPU time between consecutive device calls',
                   'aggregate isolated run CPU','sum of pre-first and post-last host CPU residual'],
        unknown=['individual pre-first versus post-last host CPU work',
                 'CPU demand variation under interference','exact host nonrunning causes'],
        first_last_split_not_identifiable_from_aggregate_CPU=True,
        old_profiles_for_other_modules_not_substitutable=True)
    result=dict(rows=rows,distinct_missing_multithread_signatures=distinct_missing,
        host_input_scope=host,new_native_sessions=0,new_DNN_inferences=0,
        no_new_prediction_or_DP=True,sources=sources,
        conclusion='Complete temporal predictor is not justified from these records alone. Missing structure must be measured or stated as a separate approximation, not inferred from pipeline FPS.')
    OUT.mkdir();write(OUT/'RESULTS.json',result);csv_write(OUT/'CPU_coverage.csv',rows)
    lines=['下一步输入可用性核查（纯离线，不增加推理）',
        '对象仍是尾部验证的四个冻结代表；未假称整个Top20已经获得时序输入。',
        '已有粗粒度阶段profile足以继续运行原静态DP，但不能等同于完整线程／设备依赖轨迹。',
        '', '确实缺少的多线程内核结构（同模块、同线程数）：']
    lines += ['%s %s %s线程'%v for v in distinct_missing]
    lines += ['',
        'YOLO当前静态第一名主要缺CPU2–4三线程、CPU12–13三线程的逐次并行任务结构。旧CPU0–2三／四线程轨迹属于其他模块，不能替代。',
        '新VTA探针已经给出三个当前模块的设备完成与CPU时间，但第一设备调用之前／最后设备调用之后的主机CPU份额仍只有合计，不能根据实际FPS任选分摊。',
        '下一项应优先复用既有隔离采集器补这些受影响签名，或先用明确声明的两端放置敏感性检验是否值得补齐；本核查不自动发起补测。',
        '现有时序模型已实现主机／设备分离，不需要再起一个复杂调度模拟器。即使局部回放变准，也仍需未测候选验证及全域输入成本评估，不能宣布Top20改善。',
        '没有覆盖的输入保留未知；本次不从流水时长回填系数、不自动扩大到全部子图。']
    (REPORT/'INPUT_COVERAGE_zh.txt').write_text('\n'.join(lines)+'\n')
    write(OUT/'MANIFEST.json',dict(sources=sources,files={str(p):sha256(p) for p in
        (OUT/'RESULTS.json',OUT/'CPU_coverage.csv',REPORT/'INPUT_COVERAGE_zh.txt')}))
    print('Offline input gaps:',distinct_missing)


if __name__=='__main__':run()
