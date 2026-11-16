"""Historical evidence is audited here, never supplied as a ranking input."""
import csv
from copy import deepcopy
import json
from pathlib import Path
from .common import ROOT,PROJECT,PARENT,SOURCE,api

PAPER_ROOT=PROJECT.parents[2]/'paper/my_paper/analysis'

def _verify_run(path):
    c=api();state=c.read(path/'state.json');receipts=list((path/'accepted').rglob('*.json'))
    if state['status']!='COMPLETE' or not receipts:raise ValueError('historical run is not complete: '+str(path))
    count=0;performances=[]
    for p in receipts:
        r=c.read(p)
        if 'files' not in r or 'task' not in r:raise ValueError('receipt lacks original evidence')
        c.verify_inventory(path,r['files']);count+=len(r['files'])
        raw=path/r['raw'];task=r['task']
        for leaf,expected in [('boot_id',r['boot_id']),('task_hash',c.digest(task))]:
            if (raw/leaf).is_file() and (raw/leaf).read_text().strip()!=expected:raise ValueError('historical task/boot differs')
        if task['kind']=='performance':performances.append(dict(task=task,receipt=str(p),receipt_sha256=c.sha256(p)))
    return dict(run=str(path),boot_id=state['boot_id'],bundle=state['bundle'],receipts=len(receipts),
                verified_raw_files=count,performances=performances)

def _matched_pairs(runs,kind):
    """A reused mechanism needs unchanged execution apart from its intervention."""
    c=api();groups={}
    for run in runs.values():
        for entry in run['performances']:
            task=entry['task'];cfg=deepcopy(task['configuration'])
            cfg.pop('id',None);cfg.pop('context_id',None)
            if kind=='host':
                treatment=tuple(cfg.pop('host_mask'))
                for row in cfg['stages']:
                    if row['device']=='vta':
                        row.pop('mask',None);row.pop('signature',None)
            else:
                treatment=cfg.pop('execution_mode');cfg.pop('core_partition',None)
            key=c.digest(dict(configuration=cfg,environment=task['environment'],executor_sha256=task['executor_sha256'],
                input_indices=task['input_indices'],warmup=task['warmup'],scored=task['scored'],tail=task['tail'],boot_id=run['boot_id']))
            groups.setdefault(key,{}).setdefault(str(treatment),[]).append(dict(receipt=entry['receipt'],sha256=entry['receipt_sha256']))
    pairs=[dict(identity=k,variants=v) for k,v in groups.items() if len(v)>=2]
    if not pairs:raise ValueError('no actual identity-matched '+kind+' intervention evidence')
    return pairs

def audit(output):
    from cpu_policies.open_resources import domain
    from .search import inputs
    c=api();output=Path(output).resolve();out=output/'evidence';out.mkdir(parents=True,exist_ok=True)
    programs,records,ddr,hashes=inputs()
    sources={}
    def source(relative):
        p=PAPER_ROOT/relative
        if not p.is_file():raise ValueError('missing historical evidence: '+str(p))
        sources[str(p)]=dict(sha256=c.sha256(p),bytes=p.stat().st_size)
        return str(p)
    cases=[]
    def add(key,question,paths,finding,use,limit,paper):
        cases.append(dict(id=key,question=question,sources=[source(p) for p in paths],
            finding=finding,next_model_use=use,limitations=limit,literature=paper))
    add('baseline_history','为什么不能把旧中间版本都当论文基线？',
        [str(ROOT/'runs/resnet18_published_policy_dp_pilot_01/analysis/READOUT_zh.txt')],
        '旧单线程前五项10.923FPS、前缀方法11.841FPS、固定核组10.948FPS；固定组未自动奏效。',
        '旧单线程混入项目容量项；正式Tarnawski重新明确设备负载目标。',
        '旧FPS只作开发记录，不与本批相减；前缀方法不作为外部基线。','Tarnawski平台适配；Pipe-it执行组织')
    add('thread_curves','增加线程是否以更多核时换取更短隔离服务？',
        ['20261108_thread_work/README.md',str(ROOT.parent/'multithreading/runs/resnet18_multithread_short_02/analysis.json')],
        '不同子图线程曲线不同；历史约62%来自四线程共享开放配置，不能改称不重叠2+2分组收益。',
        '阶段分别选择线程数，同时保留服务时间和计算核时。','仅限已测子图；不缩放到YOLO。','Pipe-it；Packrat')
    add('isolated_gap','隔离CPU计算进入流水后为何明显变慢？',
        ['20261107_run_scope_diagnosis/isolated_to_pipeline.csv','20261107_run_scope_diagnosis/caller_roi_stats.csv','20261107_run_scope_diagnosis/verification.json'],
        '四线程前段51.252→83.010ms，中段18.263→41.789ms；CPU核时仅增8.88%和6.57%。',
        '需要检查调度竞争、同步及调用线程热点；容量下界不是阶段响应时间。',
        '等待核时不能直接加到墙钟；同步、缓存和调度占比仍未知。','Packrat附录A；GCAPS任务资源拆分')
    add('host_hotspot','固定计算配置后，主机所在核心是否改变慢化？',
        ['20261108_thread_work/runs/resnet18_thread_work_probe_01/analysis/cases.csv',
         '20261108_thread_work/runs/resnet18_thread_work_probe_01/analysis/matched_effects.csv'],
        '匹配配置中主机核心1约10.598FPS、核心2约12.805FPS；主机放核心3时后段变慢。',
        '按非均匀线程份额和主机允许集合评价热点竞争。','核心2不是普适规则；短窗口单启动。','GCAPS第4节；本文核心容量适配')
    add('group_serial','同一资源放置下，同组串行是否有效？',
        ['20261108_resource_envelope/runs/resnet18_resource_behavior_mechanism_01/analysis/matched_effects.csv'],
        '共享主机时串行收益两轮约3.89%和7.57%；主机固定2时约0.58%和1.09%。',
        '执行器的组互斥与静态组服务语义需要匹配。','收益依赖主机位置；不能单独归因于评分公式。','Pipe-it组互斥；本文单VTA复用')
    add('failed_envelope','均分资源包络是否已经足够？',
        ['20261108_resource_envelope/README.md'],
        '均分包络严格Top20最佳11.240FPS，原静态名单12.025FPS；误差下降没有保证选点改善。',
        '保留为失败分支，检查线程份额与约束交互。','不得只保留正向结果。','固定线程组组织思想的适配边界')
    add('failed_corun','持续饱和干扰profile能否通用地修正？',
        ['20261108_controlled_contention/analysis.json','20261108_controlled_contention/VERIFICATION.json'],
        'CPU共跑仅部分切图满足当时条件；加入VTA共跑在验证切图选点退化。',
        '不将饱和慢化比例作为全域通用系数。','持续共跑强度高于真实帧流，未接入完整DP。','Packrat受控干扰的适用性讨论')
    add('frame_stream','共同帧流微基准和覆盖筛选是否解决全域问题？',
        ['20261108_cpu_frame_stream/analysis.json','20261108_dp_frame_stream_top20/analysis.json',
         '20261108_autotvm_diverse_top20/REPORT_zh.md'],
        '小池共同帧流有改善，完整DP池仅改变部分预算选点；覆盖筛选不等于模型更准确。',
        '作为代价与适用范围反例，不自动扩大整组profile。','历史开发回放，不能称未见配置独立确认。','本文探索；AutoTVM覆盖策略')
    add('factorial','CPU细化是否需要组合使用？',
        ['20261108_model_ablation/README.md'],
        '完整模型约13.001/13.249FPS，均分基础约11.913/11.919；单项不保证有益。',
        '本轮分开CPU细化和主机需求，并保留单项补充对照。','旧八组始终计入主机需求。','本文三因素开发消融')
    add('host_demand','显式计入主机需求和集中约束分别有何作用？',
        ['20261109_host_work_ablation/README.md'],
        '需求计入有正向选点；集中项改善一次预算，但二十次预算下降1.26%。',
        '主模型关闭集中项，集中项只作补充。','当前Top20同分，效果包含候选过滤与确定性并列选择。','GCAPS资源拆分思想；本文容量适配')
    verified={}
    for name in ('resnet18_resource_behavior_mechanism_01','resnet18_thread_work_probe_01'):
        verified[name]=_verify_run(ROOT/'runs'/name)
    matched={kind:_matched_pairs(verified,kind) for kind in ('host','gate')}
    # Current-executor matched gate/host evidence exists. The old overlap pilot
    # used input16/group affinity and the contention diagnostic another binary.
    # Supply only the two missing current-executor overlap interventions.
    mechanism=[]
    for label,second in [('two_thread_overlap',[0,1]),('two_thread_split',[2,3])]:
        rows=[domain.stage('cpu',0,3,2,[0,1]),domain.stage('vta',3,5,1,[0,1,2,3]),
              domain.stage('cpu',5,8,2,second),domain.stage('vta',8,20,1,[0,1,2,3]),domain.stage('cpu',20,21,1,[0])]
        cfg=domain.candidate(rows,'shared4','off');domain.validate_candidate(cfg,programs)
        mechanism.append(dict(label=label,configuration=cfg))
    curves=[]
    for r in records.values():
        row=r['configuration']['stages'][0]
        if row['device']=='cpu' and (row['begin'],row['end']) in ((0,3),(5,8),(20,21)):
            curves.append(dict(program=row['program'],threads=row['threads'],**r['cost'],
                source_component_key=next(k for k,v in records.items() if v is r),
                source_boot_id=r['measurement']['boot_id']))
    c.write(out/'EVIDENCE_CHAIN.json',dict(cases=cases,source_files=sources,verified_runs=verified,
        isolated_cost_hashes=hashes,old_performance_not_ranking_input=True),replace=True)
    inherited=PAPER_ROOT/'20261108_thread_work/INHERITED_PROFILE_COST.json'
    c.write(out/'INHERITED_PROFILE_COST.json',c.read(inherited),replace=True)
    c.write(out/'MECHANISM_AUDIT.json',dict(required_new_configurations=mechanism,
        reused=['same-group serialization matched pair','host-core placement matched pair'],
        identity_matched_receipts=matched,
        why_overlap_needs_two='old input16/group-affinity or different diagnostic executable is not current identity',
        old_FPS_not_reused_as_new_measurements=True),replace=True)
    with (out/'evidence_chain.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in row.items()} for row in cases)
    with (out/'thread_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in curves for k in r}));w.writeheader();w.writerows(curves)
    baseline=dict(source='Efficient Algorithms for Device Placement of DNN Graph Operators, NeurIPS 2020 §§3,5',
        URL='https://proceedings.neurips.cc/paper/2020/file/b14680dec683e744ada1f2fe08614086-Paper.pdf',
        objective='max(max_c sum_CPU_stages_on_c isolated_service, sum_all_VTA_islands isolated_service, DDR_sum)',
        CPU_threads=1,CPU_mapping='physical singleton; same-core stages serialized',
        host_CPU_capacity=False,thread_weights=False,team_service_guard=False,host_concentration_guard=False,
        adaptations=['compiled fused interval costs replace additive node costs','one physical VTA reused by up to three islands',
                     'shared DDR planning constraint','Top-K wrapper; original author system not reproduced'],
        corrected_old_baseline='old single_core inherited enhanced CPU/host capacity; it is not this frozen baseline',
        methods_as_sources={'Pipe-it':'fixed compute teams and group mutual exclusion',
                           'GCAPS':'CPU management operations separated from device execution; not its preemption or WCET analysis'},
        excluded=['2023 TVM paper','YOLO','Bayesian optimization'])
    c.write(out/'BASELINE_CONTRACT.json',baseline,replace=True)
    lines=['# ResNet18证据演进与Tarnawski基线核对','',
        '历史数据只用于解释规律；新排名不读取本表或旧FPS。探索正例和失败分支均保留。','',
        '|问题|已有观察|后续用途|限制|','|---|---|---|---|']
    for r in cases:lines.append('|'+ '|'.join(r[k] for k in ('question','finding','next_model_use','limitations'))+'|')
    lines+=['','## 本轮基线契约','',baseline['objective'],'',
        'CPU单线程、物理核心映射及同核串行；不计主机CPU容量。阶段服务包含已有输入输出墙钟口径。',
        '融合区间成本、DDR、单VTA复用及Top-K明确标为平台适配，不称原作者系统原样复现。','',
        '## 待补机制证据','',
        '同组串行和主机热点已有匹配原始凭据；两线程重叠／分开仅补两项当前程序、输入0的匹配执行。',
        '旧62%结果不是不重叠固定分组收益；不同程序或输入的旧FPS不混成本批数据。','',
        '逐项来源与哈希见EVIDENCE_CHAIN.json；基线开关见BASELINE_CONTRACT.json。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    from model_ablation.common import identity as old_model
    from host_work_ablation.common import identity as old_host
    from frame_stream_top20.common import identity as old_frame
    c.write(output/'PRE_IMPLEMENTATION_IDENTITIES.json',dict(model_ablation=old_model(),host_work_ablation=old_host(),frame_stream_top20=old_frame()),replace=True)
    return dict(report=str(out/'REPORT_zh.md'),historical_cases=len(cases),
                verified_receipts=sum(v['receipts'] for v in verified.values()),
                verified_raw_files=sum(v['verified_raw_files'] for v in verified.values()),mechanism_additions=2)
