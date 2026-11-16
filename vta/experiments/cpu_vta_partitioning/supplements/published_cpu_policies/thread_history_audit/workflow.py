from collections import Counter
from pathlib import Path
from itertools import product
import csv,json,shutil,time
from cpu_vta_partitioning.common import digest,read,sha256,write,now
from phase_model.simulator import rational,ceiling
from model_semantics_audit.replay import validate_isolated
from model_semantics_audit.common import replay_input_guard
from scheduler_gap_audit.curves import csv_write
from .common import ROOT,PARENT,REPORT,freeze,protocol,sources
from .simulator import HistoryReplay


def scenarios(iso):
    return [dict(case=case,placement=placement,runnable_wait=wait,control_mode=control)
        for case,placement,wait,control in product(sorted(iso['cases']),protocol()['placements'],protocol()['runnable_wait'],protocol()['control_modes'])]


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seal=read(PARENT/'INPUT_FREEZE.json')
    for p,h in seal['sources'].items():
        if sha256(p)!=h:raise ValueError('frozen model source changed')
    for name,h in seal['files'].items():
        if sha256(PARENT/name)!=h:raise ValueError('frozen model input changed')
    iso=validate_isolated(read(PARENT/'ISOLATED_INPUTS.json'));diag=read(PARENT/'DIAGNOSTIC_CONTROL_INPUTS.json')
    if diag.get('origin')!='observed_pipeline_control_CPU' or diag.get('diagnostic_only') is not True or diag.get('forbidden_for_static_prediction') is not True:
        raise ValueError('diagnostic controls must remain separate from isolated inputs')
    freeze(output/'ISOLATED_INPUTS.json',iso);freeze(output/'DIAGNOSTIC_CONTROL_INPUTS.json',diag)
    old_plan=read(PARENT/'PLAN.json');old_seal=read(PARENT/'REPLAY_FREEZE.json');references={}
    for s in old_plan['specs']:
        if s['host_mode']!='greedy':continue
        name='replays/'+s['id']+'.json'
        if sha256(PARENT/name)!=old_seal['files'][name]:raise ValueError('old greedy reference changed')
        references[s['case']+'|'+s['control_mode']]=read(PARENT/name)['result']['replay']
    freeze(output/'BASELINE_PREDICTIONS.json',references)
    specs=scenarios(iso)
    if len(specs)!=36 or len({digest(s) for s in specs})!=36:raise ValueError('expected 36 deterministic scenarios')
    freeze(output/'PLAN.json',dict(protocol=protocol(),scenarios=specs,sources=sources(),
        parent_input_freeze_sha256=sha256(PARENT/'INPUT_FREEZE.json'),
        files={n:sha256(output/n) for n in ('ISOLATED_INPUTS.json','DIAGNOSTIC_CONTROL_INPUTS.json','BASELINE_PREDICTIONS.json')},
        pipeline_results_decoded=False))
    return dict(scenarios=36,new_native_sessions=0)


def replay(output):
    output=Path(output);started=time.monotonic();new=cached=0;files={};useful={}
    with replay_input_guard(output):
        plan=read(output/'PLAN.json')
        if plan['sources']!=sources():raise ValueError('new model source changed after plan freeze')
        for p,h in plan['files'].items():
            if sha256(output/p)!=h:raise ValueError('projection changed')
        iso=validate_isolated(read(output/'ISOLATED_INPUTS.json'));diag=read(output/'DIAGNOSTIC_CONTROL_INPUTS.json')
        baseline=read(output/'BASELINE_PREDICTIONS.json')
        for index,spec in enumerate(plan['scenarios'],1):
            case=iso['cases'][spec['case']];relative='replays/history_'+digest(spec)[:24]+'.json';p=output/relative
            identity=dict(plan_sha256=sha256(output/'PLAN.json'),spec=spec)
            if p.exists():
                item=read(p)
                if item['input_identity']!=digest(identity) or item['content_sha256']!=digest({k:v for k,v in item.items() if k!='content_sha256'}):raise ValueError('history replay evidence changed')
                cached+=1
            else:
                tick=time.monotonic();controls=diag['cases'][spec['case']] if spec['control_mode']=='diagnostic_observed' else None
                result=HistoryReplay(case['stages'],placement=spec['placement'],runnable_completion_wait=spec['runnable_wait'],
                    controls=controls,gate=case['configuration']['execution_mode'],host_order='before',**protocol()['window']).run()
                if spec['placement']=='instant' and spec['runnable_wait'] and result['replay']!=baseline[spec['case']+'|'+spec['control_mode']]:
                    raise ValueError('six original frozen baseline predictions not exactly restored')
                item=dict(spec=spec,result=result,seconds=time.monotonic()-tick,input_identity=digest(identity),
                    pipeline_performance_decoded=False,diagnostic_only=spec['control_mode']=='diagnostic_observed')
                item['content_sha256']=digest(item);freeze(p,item);new+=1
            amount=item['result']['replay']['useful_CPU_ns'];key=(spec['case'],spec['control_mode'])
            if key in useful and useful[key]!=amount:raise ValueError('placement or waiting changed finite CPU demand')
            useful[key]=amount;files[relative]=sha256(p)
            print('线程历史离线回放',index,'/36',spec['case'],spec['placement'],spec['runnable_wait'],spec['control_mode'],flush=True)
        freeze(output/'REPLAY_FREEZE.json',dict(files=files,plan_sha256=sha256(output/'PLAN.json'),sources=sources(),
            all_36_before_native_performance_decode=True,new_native_sessions=0))
    write(output/'LAST_REPLAY_COST.json',dict(seconds=time.monotonic()-started,new=new,cached=cached,
        original_replay_seconds=sum(read(output/n)['seconds'] for n in files)),replace=True)
    return dict(replays=36,new=new,cached=cached,new_native_sessions=0)


def validate(output):
    output=Path(output);plan=read(output/'PLAN.json');seal=read(output/'REPLAY_FREEZE.json')
    if plan['sources']!=sources() or seal['sources']!=sources() or sha256(output/'PLAN.json')!=seal['plan_sha256']:
        raise ValueError('offline history source/plan changed')
    for p,h in {**plan['files'],**seal['files']}.items():
        if sha256(output/p)!=h:raise ValueError('offline history evidence changed')
    return dict(validated=True,replays=len(seal['files']),new_native_sessions=0,new_DNN_inferences=0)


def summarize_prediction(item):
    spec=item['spec'];r=item['result'];window=r['window_cpu'];gaps=window['gaps'];host=[0,0,0,0];caller=wait=0
    for row in window['rows']:
        core=list(map(rational,row['core_CPU']));meta=(row['role'],row['stage'],row['slot'])
        if row['role']=='stage' and row['stage']%2==1:
            host=[a+b for a,b in zip(host,core)]
        if meta==('stage',0,0):
            caller+=sum(core)
            if row['synthetic_wait']:wait+=sum(core)
    joins=[k for k in r['kernel_joins'] if k['stage']==0]
    # All frozen cases alternate CPU/VTA with a CPU first; assert in prepare.
    return dict(**spec,predicted_II_ms=r['replay']['interval_ns']/1e6,predicted_front_ms=r['replay']['stage_service_ns'][0]/1e6,
        host_core0_fraction=float(host[0]/sum(host)),host_core_ms_per_completion=[float(v/gaps/1000000) for v in host],
        front_caller_core_ms_per_completion=float(caller/gaps/1000000),front_caller_wait_core_ms_per_completion=float(wait/gaps/1000000),
        process_core_occupancy=[float(rational(v)/rational(window['duration'])) for v in window['core_totals']],
        noncaller_last_fraction=sum(0 not in k['last_slots'] for k in joins)/len(joins),
        caller_in_last_tie_fraction=sum(0 in k['last_slots'] for k in joins)/len(joins),
        last_ties=sum(len(k['last_slots'])>1 for k in joins),
        placement_changes=sum(v['count'] for v in r['placement_changes']),
        predicted_wait_work_not_added_to_II=True)


def analyze(output):
    output=Path(output);validate(output);seal=read(output/'REPLAY_FREEZE.json')
    # Only after all scenario predictions are sealed, open native evidence.
    native_path=ROOT/'artifacts/yolo_host_placement_01/analysis/RESULTS.json';native_delivery=read(native_path.parents[1]/'DELIVERY.json')
    if sha256(native_path)!=native_delivery['results_sha256']:raise ValueError('new host-placement receipt changed')
    native=read(native_path)
    gap_path=ROOT/'artifacts/scheduler_gap_audit_01/analysis/RESULTS.json';gap_delivery=read(gap_path.parents[1]/'DELIVERY.json')
    if sha256(gap_path)!=gap_delivery['results_sha256']:raise ValueError('gap measurement source changed')
    gap=read(gap_path)
    reference={r['network']+'/'+r['case']:dict(actual_front_ms=r['clean_pipeline_service_ms'],model_previous_ms=r['model_stage_service_ms'])
        for r in gap['stages'] if r['stage']=='cpu_00_02'}
    reference['yolov3_tiny416/overlap']['actual_front_ms']=native['clean']['stage_service_ms'][0]
    rows=[]
    for name in seal['files']:
        item=read(output/name);r=summarize_prediction(item);actual=reference[r['case']]['actual_front_ms']
        r.update(actual_front_ms=actual,signed_front_error_ms=r['predicted_front_ms']-actual,absolute_front_error_ms=abs(r['predicted_front_ms']-actual))
        rows.append(r)
    overlaps=[r for r in rows if r['case']=='yolov3_tiny416/overlap' and r['runnable_wait']]
    decisions=[]
    for control in protocol()['control_modes']:
        baselines={r['case']:r for r in rows if r['placement']=='instant' and r['runnable_wait'] and r['control_mode']==control}
        for policy in ('retain','wake_boundary'):
            changes={r['case']:r['absolute_front_error_ms']-baselines[r['case']]['absolute_front_error_ms'] for r in rows
                if r['placement']==policy and r['runnable_wait'] and r['control_mode']==control}
            decisions.append(dict(control_mode=control,placement=policy,absolute_error_changes_ms=changes,
                target_error_reduced=changes['yolov3_tiny416/overlap']<0,no_case_error_increased=all(v<=0 for v in changes.values()),
                not_a_selected_or_deployed_rule=True))
    native_placement=dict(host_core0_fraction=native['actual_host_core0_fraction'],
        front_caller_core_ms_per_completion=next(r['all_CPU_ms_per_completion'] for r in native['roles'] if r['role']=='cpu_00_02/caller'),
        core_occupancy=[sum(r['by_core_ns'][c] for r in native['roles'])/native['formal_completion_window_ns'] for c in range(4)],
        native_counting_windows_have_dequeue_and_read_offsets=True,
        quantitative_attribution_allowed=native['quantitative_attribution_allowed'])
    result=dict(rows=rows,reference=reference,native_placement=native_placement,decisions=decisions,
        sources=sources(),source_native_results_sha256=sha256(native_path),source_gap_results_sha256=sha256(gap_path),
        predictions_frozen_before_native_decode=True,replays=36,new_native_sessions=0,new_DNN_inferences=0,
        no_model_rule_automatically_selected=True,recorded_at=now())
    dest=output/'analysis';dest.mkdir(parents=True,exist_ok=True);write(dest/'RESULTS.json',result)
    csv_write(dest/'all_36_scenarios.csv',rows);render(dest,result)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():shutil.copy2(p,REPORT/p.name)
    write(output/'DELIVERY.json',dict(status='OFFLINE_COMPLETE',new_native_sessions=0,new_DNN_inferences=0,
        sources=sources(),results_sha256=sha256(dest/'RESULTS.json'),replay_freeze_sha256=sha256(output/'REPLAY_FREEZE.json'),
        files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},score_changed=False,DP_changed=False,Top20_changed=False))
    return dict(report=str(REPORT/'README.md'),replays=36,new_native_sessions=0,decisions=decisions)


def render(dest,result):
    labels={'instant':'原逐事件选核','retain':'保留初始落核','wake_boundary':'仅潜在唤醒点选核'}
    lines=['# 线程落核历史与调用线程等待：离线对照','',
        '本轮36次电脑端回放，新增板端会话和推理均为0。全部预测在读取本轮分析使用的已有流水FPS及落核计数前冻结。隔离输入和诊断性控制CPU输入保持分离。','',
        '## 规则与来源','',
        '三种选核近似共同使用原隔离工作、依赖、队列和同核连续公平分摊：原逐事件贪心；同一逻辑线程保留初始核心；或只在潜在唤醒边界重选并在等负载时保留上一核心。初次选核均按当前已分配可运行任务数、核心编号确定，不指定核心0、某个DNN或实测比例。',
        '正的非运行区间只能作为可能的唤醒边界，不能证明实际睡眠。实际Linux也可迁移可运行任务，严格保留只是一种敏感性近似。三种近似都没有实现有限调度粒度、PELT或完整CFS。',
        '源码依据：[Linux v5.4 fair.c](https://github.com/torvalds/linux/blob/v5.4/kernel/sched/fair.c)的select_task_rq_fair、wake_affine保留prev_cpu信息；[CFS设计说明](https://www.kernel.org/doc/html/v5.4/scheduler/sched-design-CFS.html)区分理想共享与真实调度。这里借鉴线程身份与落核历史的语义，不声称复现供应商内核。','',
        '## 保留可运行等待时的逐配置结果','',
        '| 配置 | 控制工作口径 | 放置近似 | 预测前段ms | 实测前段ms | 绝对误差ms | 预测主机CPU在核心0的比例 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in result['rows']:
        if not r['runnable_wait']:continue
        lines.append(f"| {r['case']} | {r['control_mode']} | {labels[r['placement']]} | {r['predicted_front_ms']:.3f} | {r['actual_front_ms']:.3f} | {r['absolute_front_error_ms']:.3f} | {100*r['host_core0_fraction']:.2f}% |")
    n=result['native_placement']
    lines+=['','## YOLO重叠配置的资源证据','',
        f"本批原生计数的主机核心0份额为{100*n['host_core0_fraction']:.2f}%，前段调用线程CPU约{n['front_caller_core_ms_per_completion']:.3f}核ms／完成间隔。原生窗口有收集及读取偏移；下表回放使用严格完成时间戳窗口，口径差异保留。",'',
        '| 控制工作口径 | 放置近似 | 可运行等待 | 前段预测ms | 调用线程CPU核ms／间隔 | 其中模型完成／屏障等待CPU核ms／间隔 | 核心0占用 | 最后完成者仅为其他worker的比例 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in result['rows']:
        if r['case']!='yolov3_tiny416/overlap':continue
        lines.append(f"| {r['control_mode']} | {labels[r['placement']]} | {r['runnable_wait']} | {r['predicted_front_ms']:.3f} | {r['front_caller_core_ms_per_completion']:.3f} | {r['front_caller_wait_core_ms_per_completion']:.3f} | {100*r['process_core_occupancy'][0]:.2f}% | {100*r['noncaller_last_fraction']:.2f}% |")
    lines+=['', '关闭可运行等待是模型敏感性对照，不代表板子采用阻塞等待，也不是新的真实性能结果。等待CPU从执行依赖动态产生，不将实测等待量叠加为阶段惩罚。',
        '', '## 跨配置检查','']
    for d in result['decisions']:
        lines.append(f"- {d['control_mode']}／{labels[d['placement']]}：目标误差降低={d['target_error_reduced']}；三个配置误差均不增加={d['no_case_error_increased']}；逐配置绝对误差变化ms={d['absolute_error_changes_ms']}。")
    lines+=['', '诊断性控制口径使用已有流水内CPU时间，不能作为可部署的隔离预测器。仅有零控制口径的输入来自隔离工作，但它明确遗漏控制需求。所有组合完整交付，不挑最接近实测的一项宣称成功。',
        '若某规则只改善当前YOLO配置或仍无法同时解释调用线程热点与主机分布，应保留负向结论；不能据此直接接入DP。尚未建模的实际唤醒、迁移、有限粒度及系统负载仍为缺口。',
        '本轮不自动挑选新规则、补测或生成Top-20。下一步需要依据这里的跨配置误差和资源证据另行确定。']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
