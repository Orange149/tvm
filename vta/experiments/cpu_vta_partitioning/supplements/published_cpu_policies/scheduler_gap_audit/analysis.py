"""Critical-completion evidence, not a sum-of-thread-waits FPS correction."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics as st
from cpu_vta_partitioning.common import read,sha256,verify_inventory,write
from .common import REPORT,protocol
from .prepare import specs
from .runtime import make_task,summary
from .curves import csv_write


def events(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def average(values):return st.mean(values) if values else None


def queue_bounds(r,phase=False):
    prefix='read_' if phase else 'queue_read_'
    q=r['queue_after_ns']-r['queue_before_ns']
    excess=r['wall_start_ns']-r[prefix+'start0_ns']+r[prefix+'end1_ns']-r['wall_end_ns']
    return max(0,q-excess),min(q,r['wall_end_ns']-r['wall_start_ns'])


def call_details(raw,task):
    phases=events(Path(raw)/'phase_trace.jsonl');trace=events(Path(raw)/'wait_trace.jsonl')
    launches=[r for r in trace if r['kind']==0];callbacks=defaultdict(list);barriers=defaultdict(list)
    for r in trace:
        if r['kind']==1:callbacks[r['launch']].append(r)
        elif r['kind']==2:barriers[r['launch']].append(r)
    per_call=[];kernels=[]
    for phase in phases:
        if phase['operation']!='run' or not task['warmup']<=phase['call']<task['warmup']+task['scored']:continue
        chosen=sorted([r for r in launches if r['caller']==phase['tid'] and phase['wall_start_ns']<=r['wall_start_ns']<=r['wall_end_ns']<=phase['wall_end_ns']],key=lambda r:r['wall_start_ns'])
        body_cpu=0;barrier_cpu=0;last_cpu=0;last_queue_lo=0;last_queue_hi=0;caller_tail_q=0;skew=0;pre_q=0
        for index,launch in enumerate(chosen):
            group=callbacks[launch['launch']]
            if len(group)!=launch['tasks']:raise ValueError('incomplete launch in selected run')
            last=max(group,key=lambda r:(r['wall_end_ns'],r['task']))
            lo,hi=queue_bounds(last);last_queue_lo+=lo;last_queue_hi+=hi
            cpu=sum(r['cpu_end_ns']-r['cpu_start_ns'] for r in group)
            bcpu=sum(r['cpu_end_ns']-r['cpu_start_ns'] for r in barriers[launch['launch']])
            body_cpu+=cpu-bcpu;barrier_cpu+=bcpu;last_cpu+=last['cpu_end_ns']-last['cpu_start_ns']
            delta=max(r['wall_start_ns'] for r in group)-min(r['wall_start_ns'] for r in group);skew+=delta
            baseline=launch['worker_queue_baselines'][last['task']]
            if baseline['tid']!=last['tid'] or not baseline['read_start_ns']<=baseline['read_end_ns']<=launch['wall_start_ns']:
                raise ValueError('scored launch lacks matching worker queue baseline')
            if baseline['queue_ns']>last['queue_before_ns']:raise ValueError('worker baseline counter moved backwards')
            pre=last['queue_before_ns']-baseline['queue_ns'];pre_q+=pre
            tail=launch['queue_after_ns']-launch['caller_tail_queue_start_ns']
            if tail<0:raise ValueError('caller completion queue moved backwards')
            caller_tail_q+=tail
            kernels.append(dict(stage=phase['stage'],frame=phase['call'],kernel=index,launch=launch['launch'],
                module=launch['module'],function_offset=launch['function_offset'],tasks=launch['tasks'],
                last_worker=last['task'],last_TID=last['tid'],last_worker_CPU_ms=(last['cpu_end_ns']-last['cpu_start_ns'])/1e6,
                last_worker_body_queue_lower_ms=lo/1e6,last_worker_body_queue_upper_ms=hi/1e6,
                last_worker_pre_entry_queue_counter_ms=pre/1e6,
                pre_entry_counter_includes_read_envelope=True,launch_skew_ms=delta/1e6,
                caller_tail_queue_counter_ms=tail/1e6,
                start_ns=launch['wall_start_ns'],end_ns=launch['wall_end_ns'],
                worker_events=group,launch_event=launch))
        lo,hi=queue_bounds(phase,True)
        per_call.append(dict(stage=phase['stage'],frame=phase['call'],run_wall_ms=(phase['wall_end_ns']-phase['wall_start_ns'])/1e6,
            caller_run_CPU_ms=(phase['cpu_end_ns']-phase['cpu_start_ns'])/1e6,
            caller_run_queue_lower_ms=lo/1e6,caller_run_queue_upper_ms=hi/1e6,
            callback_outside_barrier_CPU_ms=body_cpu/1e6,barrier_CPU_ms=barrier_cpu/1e6,
            sum_last_finisher_CPU_ms=last_cpu/1e6,
            sum_last_finisher_queue_lower_ms=last_queue_lo/1e6,sum_last_finisher_queue_upper_ms=last_queue_hi/1e6,
            sum_last_finisher_pre_entry_counter_ms=pre_q/1e6,
            caller_completion_queue_counter_ms=caller_tail_q/1e6,sum_launch_skew_ms=skew/1e6,
            launches=len(chosen),not_an_additive_stage_wall_decomposition=True))
    return per_call,kernels,phases


def perturbation(iso_clean,iso_observed,pipe_clean,pipe_observed):
    gap=pipe_clean-iso_clean
    disturbance=max(abs(iso_observed-iso_clean),abs(pipe_observed-pipe_clean))
    return dict(isolation_to_pipeline_gap_ms=gap,probe_disturbance_ms=disturbance,
        disturbance_to_gap=disturbance/abs(gap) if gap else None,
        quantitative_attribution_allowed=gap>0 and disturbance<=protocol()['probe_attribution_fraction']*gap,
        no_performance_result_rejected=True,no_automatic_remeasurement=True)


def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);state=read(run_dir/'state.json')
    if state['status']!='COMPLETE':raise ValueError('all bounded sessions must be accepted before final analysis')
    measurements={};source_hashes={};call_rows=[];kernel_rows=[];control_rows=[];phase_rows=[]
    for spec in specs():
        bundle=output/spec['network']/'capsule';task=make_task(bundle,spec)
        path=run_dir/'accepted'/state['boot_id']/(task['task_id']+'.json');item=read(path)
        if item['task']!=task or item['boot_id']!=state['boot_id']:raise ValueError('analysis receipt identity differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],task,bundle)!=item['summary']:raise ValueError('analysis receipt does not replay')
        source_hashes[str(path)]=sha256(path)
        measurements[(spec['network'],spec['scope'],spec['case'],spec['phase'])]=item
        if spec['phase']=='observed':
            calls,kernels,phases=call_details(item['raw'],task)
            common=dict(network=spec['network'],scope=spec['scope'],case=spec['case'])
            call_rows += [{**common,**r} for r in calls]
            kernel_rows += [{**common,**r} for r in kernels]
            for p in phases:
                if not task['warmup']<=p['call']<task['warmup']+task['scored']:continue
                lo,hi=queue_bounds(p,True)
                phase_rows.append(dict(**common,stage=p['stage'],frame=p['call'],operation=p['operation'],
                    TID=p['tid'],start_ns=p['wall_start_ns'],end_ns=p['wall_end_ns'],
                    wall_ms=(p['wall_end_ns']-p['wall_start_ns'])/1e6,CPU_ms=(p['cpu_end_ns']-p['cpu_start_ns'])/1e6,
                    queue_lower_ms=lo/1e6,queue_upper_ms=hi/1e6))
                if p['stage'].startswith('_'):
                    control_rows.append(dict(**common,role=p['stage'],frame=p['call'],
                        CPU_ms=(p['cpu_end_ns']-p['cpu_start_ns'])/1e6,wall_ms=(p['wall_end_ns']-p['wall_start_ns'])/1e6,
                        queue_lower_ms=lo/1e6,queue_upper_ms=hi/1e6))
    predictions=read(output/'PREDICTIONS_FREEZE.json');stage_table=[];case_table=[]
    for key,pred in predictions['cases'].items():
        network,case=key.split('/');clean=measurements[network,'pipeline',case,'clean'];observed=measurements[network,'pipeline',case,'observed']
        n=pred['configuration']['stages'][0]['threads'];component='front_t'+str(n)
        iso=measurements[network,'component',component,'clean']['summary']['cost']
        iso_probe=measurements[network,'component',component,'observed']['summary']['cost']
        valid=perturbation(iso['service_ms'],iso_probe['service_ms'],clean['summary']['stage_service_ms'][0],observed['summary']['stage_service_ms'][0])
        case_table.append(dict(network=network,case=case,configuration=pred['configuration'],
            clean_FPS=clean['summary']['fps'],probe_FPS=observed['summary']['fps'],
            clean_interval_ms=clean['summary']['ii_ms'],probe_interval_ms=observed['summary']['ii_ms'],
            clean_process_core_ms=clean['summary']['process_core_ms_per_completion'],
            model_interval_ms=pred['matched_long_replay']['interval_ns']/1e6,**valid))
        for i,stage in enumerate(pred['configuration']['stages']):
            selected=[r for r in call_rows if r['network']==network and r['scope']=='pipeline' and r['case']==case and r['stage']==stage['program']]
            baseline=[r for r in call_rows if r['network']==network and r['scope']=='component' and r['case']==component]
            fields=('caller_run_CPU_ms','caller_run_queue_lower_ms','caller_run_queue_upper_ms',
                'callback_outside_barrier_CPU_ms','barrier_CPU_ms','sum_last_finisher_CPU_ms',
                'sum_last_finisher_queue_lower_ms','sum_last_finisher_queue_upper_ms',
                'sum_last_finisher_pre_entry_counter_ms','caller_completion_queue_counter_ms','sum_launch_skew_ms')
            stage_table.append(dict(network=network,case=case,stage=stage['program'],device=stage['device'],threads=stage['threads'],mask=stage['mask'],
                isolated_service_ms=iso['service_ms'] if i==0 else pred['isolated_costs'][i]['service_ms'],
                isolated_service_source='this_batch_clean' if i==0 else 'frozen_historical_macro_profile',
                model_stage_service_ms=pred['matched_long_replay']['stage_service_ns'][i]/1e6,
                clean_pipeline_service_ms=clean['summary']['stage_service_ms'][i],
                observed_pipeline_service_ms=observed['summary']['stage_service_ms'][i],
                remaining_model_gap_ms=clean['summary']['stage_service_ms'][i]-pred['matched_long_replay']['stage_service_ns'][i]/1e6,
                quantitative_attribution_allowed=valid['quantitative_attribution_allowed'] if i==0 else None,
                **{k:average([r[k] for r in selected]) for k in fields},
                isolated_callback_outside_barrier_CPU_ms=average([r['callback_outside_barrier_CPU_ms'] for r in baseline]) if i==0 else None,
                isolated_last_finisher_queue_lower_ms=average([r['sum_last_finisher_queue_lower_ms'] for r in baseline]) if i==0 else None,
                isolated_last_finisher_queue_upper_ms=average([r['sum_last_finisher_queue_upper_ms'] for r in baseline]) if i==0 else None,
                isolated_pre_entry_queue_counter_ms=average([r['sum_last_finisher_pre_entry_counter_ms'] for r in baseline]) if i==0 else None,
                unknown_work='non-callback runtime, cache/memory causes, exact wakeup/runnable timeline; no fitted correction'))
    target=output/'analysis';target.mkdir(parents=True,exist_ok=True)
    for leaf,rows in [('stage_errors.csv',stage_table),('cases.csv',case_table),('calls.csv',call_rows),
                      ('control_activity.csv',control_rows),('phase_activity.csv',phase_rows)]:csv_write(target/leaf,rows)
    csv_write(target/'critical_kernels.csv',[{k:v for k,v in r.items() if k not in ('worker_events','launch_event')} for r in kernel_rows])
    write(target/'CRITICAL_TIMELINES.json',kernel_rows,replace=True)
    result=dict(boot_id=state['boot_id'],cases=case_table,stages=stage_table,receipts=source_hashes,
        prediction_sha256=sha256(output/'PREDICTIONS_FREEZE.json'),state=state,
        attribution_is_not_FPS_model=True,all_thread_waits_not_added=True,
        no_new_Top20=True,no_formula_change=True,unknowns_preserved=True)
    write(target/'RESULTS.json',result,replace=True)
    render(output,result,kernel_rows)
    REPORT.mkdir(parents=True,exist_ok=True)
    for path in target.iterdir():shutil.copy2(path,REPORT/path.name)
    for leaf in ('thread_options.csv','subgraph_summary.csv','resnet18_speedup.png','yolov3_tiny416_speedup.png','SUMMARY.json'):
        shutil.copy2(output/'curves'/leaf,REPORT/leaf)
    return dict(status='COMPLETE',report=str(REPORT/'README.md'),cases=case_table)


def render(output,result,kernels):
    target=Path(output)/'analysis';lines=['# 子图线程曲线与剩余阶段误差诊断','',
        '本轮未修改评分、DP、Top-20、阶段模块或运行库，未重启开发板。新批次只使用自身启动身份：`'+result['boot_id']+'`。',
        '','## 已有线程配置空间','',
        '覆盖ResNet18的85个和YOLO的81个CPU子图，共664种真实隔离签名，保留全部1—4线程选项。每个子图独立选择线程数及核心集合；不是每个网络统一线程数。',
        '曲线标签仅描述已有单次观察，不是显著性、等效性或裁剪依据。实际并行参与记录覆盖52项，其他标为未知。',
        '','## 本批正式性能与探针影响','',
        '| 网络 | 方案 | 无探针FPS | 有探针FPS | 前段隔离→流水差ms | 探针最大扰动ms | 可定量归因 |',
        '|---|---|---:|---:|---:|---:|---|']
    labels={'overlap':'四线程重叠配置','split':'三线程分组配置','static_first':'原完整模型静态第一名'}
    for r in result['cases']:
        lines.append(f"| {r['network']} | {labels[r['case']]} | {r['clean_FPS']:.3f} | {r['probe_FPS']:.3f} | {r['isolation_to_pipeline_gap_ms']:.3f} | {r['probe_disturbance_ms']:.3f} | {r['quantitative_attribution_allowed']} |")
    lines+=['','无探针值为本批正式观察。探针超出差值20%时只保留线索，不拒收正式性能、不自动重测。两种YOLO配置同时改变线程数和映射，不能称为单独核心重叠的因果干预。',
        '','## 前段CPU证据','',
        '| 网络／配置 | 隔离服务ms | 原模型时序预测ms | 本批流水服务ms | 屏障外回调CPU：隔离／流水 核ms | 最后完成者回调内排队范围ms | 进入回调前排队计数ms | 调用线程完成尾部排队ms |',
        '|---|---:|---:|---:|---|---|---:|---:|']
    for r in result['stages']:
        if r['stage']!='cpu_00_02':continue
        lines.append(f"| {r['network']}／{labels[r['case']]} | {r['isolated_service_ms']:.3f} | {r['model_stage_service_ms']:.3f} | {r['clean_pipeline_service_ms']:.3f} | {r['isolated_callback_outside_barrier_CPU_ms']:.3f}／{r['callback_outside_barrier_CPU_ms']:.3f} | {r['sum_last_finisher_queue_lower_ms']:.3f}–{r['sum_last_finisher_queue_upper_ms']:.3f} | {r['sum_last_finisher_pre_entry_counter_ms']:.3f} | {r['caller_completion_queue_counter_ms']:.3f} |")
    lines+=['','上述等待指标存在重叠，不能相加成阶段服务或“解释百分比”。回调内排队范围根据自线程计数读取包络给出；回调前计数同时保留逐TID基线读取区间，不冒充精确唤醒轨迹。',
        '屏障外回调CPU仍可能包括内核读数和运行时工作，不等于纯算术。生产/收集活动、VTA主机set/run/get分别记录，生产阻塞与CPU排队分开。',
        '模型预测使用旧隔离输入，在新计时前冻结；同时保留旧短窗口回放和匹配64时间戳回放，没有用本批FPS重新拟合。',
        '','## 如何使用这些结果','',
        '- 优先比较关键worker排队的增长与计算CPU量变化，再观察启动错位及调用线程尾部；不将整个非运行区间叫作调度等待。',
        '- 只有探针质量允许时才讨论数量上的关系；没有完整唤醒轨迹，不能宣称已经重建Linux调度或解释全部差距。',
        '- 两网络不同输入大小、子图及主机活动混杂；本轮支持负载级解释，不支持“网络名称决定策略”。',
        '- 本轮不回填慢化系数，不自动追加共享模型或全域标定。下一步修改必须针对本表揭示的缺项。',
        '','## 证据与成本','',
        '18个原生会话，627次完整DNN推理，63次隔离CPU阶段调用。初始化、空回调及零推理计数自测另行记账。所有输出、实际绑核、boot、任务及原始文件哈希经校验。',
        '完整逐阶段表见stage_errors.csv；逐调用见calls.csv；关键worker及启动记录见critical_kernels.csv和CRITICAL_TIMELINES.json；控制线程见control_activity.csv。']
    (target/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for case in result['cases']:
        rows=[r for r in kernels if r['network']==case['network'] and r['scope']=='pipeline' and r['case']==case['case'] and r['stage']=='cpu_00_02' and r['frame']==8]
        if not rows:continue
        origin=min(r['start_ns'] for r in rows);fig,ax=plt.subplots(figsize=(12,3.5))
        for r in rows:
            for w in r['worker_events']:
                ax.broken_barh([((w['wall_start_ns']-origin)/1e6,(w['wall_end_ns']-w['wall_start_ns'])/1e6)],(w['task']-.3,.6),facecolors='tab:red' if w['tid']==r['last_TID'] else 'tab:blue')
        ax.set(xlabel='ms since first selected launch',ylabel='Logical worker',title=case['network']+' '+case['case']+' callback wall spans (not CPU residency)')
        fig.tight_layout();fig.savefig(target/(case['network']+'_'+case['case']+'_timeline.png'),dpi=160);plt.close(fig)
