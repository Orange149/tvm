"""Separate caller CPU consumed while completing a join from off-CPU delay.

No new predictor: isolated work, existing replay state, and observed pipeline
state remain separate. Native observation has a probe-quality limitation.
"""
from collections import defaultdict
from pathlib import Path
import json
import statistics as st
import shutil

import analyze_resnet_middle_switch as capture
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from phase_model.extract import project_call,average_calls
from phase_model.simulator import rational
from scheduler_gap_audit.curves import csv_write

ROOT=capture.ROOT
OUT=ROOT/'artifacts/caller_wait_budget_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_caller_wait_budget'


def projected_metrics(profile):
    kernels=profile['kernels']
    body=sum((sum((rational(x) for x in k['workers'][0]['compute_segments'])) for k in kernels))
    worker=sum((sum((rational(x) for x in w['compute_segments'])) for k in kernels for w in k['workers']))
    prefix=sum((rational(k['launch_prefix_cpu']) for k in kernels))
    serial=sum((rational(x) for x in profile['serial_cpu_segments']))
    tail=sum((rational(k['isolated_completion_wait_and_return_cpu']) for k in kernels))
    caller=rational(profile['caller_CPU']);wall=rational(profile['run_wall'])
    if body+prefix+serial+tail!=caller:raise ValueError('caller CPU partition not conserved')
    if wall<caller:raise ValueError('caller CPU exceeds run wall')
    return dict(caller_compute_CPU_ms=float(body)/1e6,caller_other_CPU_ms=float(prefix+serial)/1e6,
        completion_and_return_CPU_ms=float(tail)/1e6,caller_off_CPU_ms=float(wall-caller)/1e6,
        caller_run_CPU_ms=float(caller)/1e6,caller_run_wall_ms=float(wall)/1e6,
        all_callback_CPU_ms=float(worker)/1e6,recorded_stage_run_CPU_ms=float(caller+worker-body)/1e6,
        noncaller_bookkeeping_CPU_not_included=True)


def model_metrics(model,index):
    tasks=[r for r in model['tasks'] if r['stage']==index and r['slot']==0 and r['operation'] not in ('set','get')]
    frames=sorted({t['frame'] for t in tasks})
    if frames!=list(range(8,24)):raise ValueError('modeled caller frames differ')
    frame_wall=[]
    for frame in frames:
        rows=sorted((t for t in tasks if t['frame']==frame),key=lambda t:(t['ready_ns'],t['end_ns']))
        if any(a['end_ns']!=b['ready_ns'] for a,b in zip(rows,rows[1:])):
            raise ValueError('modeled caller path is not contiguous')
        frame_wall.append(rows[-1]['end_ns']-rows[0]['ready_ns'])
    avg=lambda operation:sum((rational(t['CPU']) for t in tasks if operation(t)))/16/1e6
    body=avg(lambda t:t['operation']=='worker_compute')
    wait=avg(lambda t:t['operation']=='completion_wait')
    other=avg(lambda t:t['operation'] not in ('worker_compute','completion_wait'))
    caller=body+wait+other;wall=st.mean(frame_wall)/1e6
    queued=sum((rational(t['ready_not_running']) for t in tasks))/16/1e6
    if abs(wall-float(caller+queued))>0.000003:raise ValueError('modeled path wall is not conserved')
    noncaller=sum((rational(t['CPU']) for t in model['tasks'] if t['stage']==index and t['slot']!=0
        and t['operation']=='worker_compute'))/16/1e6
    return dict(caller_compute_CPU_ms=float(body),caller_other_CPU_ms=float(other),
        completion_and_return_CPU_ms=float(wait),caller_off_CPU_ms=float(queued),
        caller_run_CPU_ms=float(caller),caller_run_wall_ms=wall,
        all_callback_CPU_ms=float(body+noncaller),recorded_stage_run_CPU_ms=float(caller+noncaller),
        runtime_return_CPU_not_separately_modeled=True)


def yolo_reference():
    """Reuse the accepted repaired trace, preserving its different boot/probe."""
    from yolo_scheduler_trace_repair.runtime import summary,make_task
    from yolo_scheduler_trace_repair.prepare import validate
    from yolo_scheduler_trace_repair.common import specs
    output=ROOT/'artifacts/yolo_scheduler_trace_repair_01';run=ROOT/'runs/yolo_scheduler_trace_repair_01'
    state=read(run/'state.json');bundle=output/'capsule';validate(output)
    if state['status']!='COMPLETE' or state['accepted_sessions']!=3:raise ValueError('repaired YOLO source incomplete')
    records={};sources={}
    for spec in specs():
        task=make_task(bundle,spec);path=run/'accepted'/state['boot_id']/(task['task_id']+'.json');r=read(path)
        if r['task']!=task or r['boot_id']!=state['boot_id']:raise ValueError('YOLO reference task changed')
        verify_inventory(r['raw'],r['files'])
        if summary(r['raw'],task,bundle)!=r['summary']:raise ValueError('YOLO raw evidence no longer replays')
        records[spec['phase']]=r;sources[str(path)]=sha256(path)
    if any(r['summary']['output_fingerprints']!=records['qualify']['summary']['output_fingerprints'] for r in records.values()):
        raise ValueError('YOLO diagnostic outputs differ from qualification')
    r=records['observed'];raw=Path(r['raw']);doc=r['task'];program='cpu_00_02'
    intr=read(ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json')
    old=next(v for v in intr['CPU'].values() if v['identity']['network']=='yolov3_tiny416'
        and v['identity']['program']==program and v['identity']['threads']==4)
    pv=old['provenance'];source=pv['observed_receipt']
    if sha256(source)!=pv['observed_sha256']:raise ValueError('YOLO isolated source changed')
    original=read(source);verify_inventory(original['raw'],original['files']);sources[source]=pv['observed_sha256']
    if (old['identity']['module_sha256']!=read(bundle/'plans/programs.json')[program]['files']['module.so']['sha256']
        or doc['runtime_sha256']!=pv['runtime_sha256']):raise ValueError('YOLO module/runtime differs')
    phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    trace=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    la=[x for x in trace if x['kind']==0];cb=defaultdict(list);ba=defaultdict(list)
    for x in trace:
        if x['kind'] in (1,2):(cb if x['kind']==1 else ba)[x['launch']].append(x)
    selected=sorted((p for p in phase if p['stage']==program and p['operation']=='run'
        and doc['warmup']<=p['call']<doc['warmup']+doc['scored']),key=lambda p:p['call'])
    if [p['call'] for p in selected]!=list(range(doc['warmup'],doc['warmup']+doc['scored'])):
        raise ValueError('YOLO reference frames incomplete')
    current=average_calls([project_call(p,la,cb,ba) for p in selected])
    if [(k['function_offset'],k['tasks']) for k in current['kernels']]!=[(k['function_offset'],k['tasks']) for k in old['average']['kernels']]:
        raise ValueError('YOLO kernel sequence differs')
    return dict(rows=[dict(program='yolov3_tiny416/cpu_00_02',scope=scope,**projected_metrics(p))
        for scope,p in [('isolated',old['average']),('pipeline_observer',current)]],
        source_receipts=sources,source_boot=state['boot_id'],new_measurements=0,
        threads=4,window={k:doc[k] for k in ('warmup','scored','tail')},
        TASKSTATS_queries_enabled=read(raw/'queue_status.json')['enabled'],
        clean_front_ms=records['clean']['summary']['stage_service_ms'][0],
        observed_front_ms=r['summary']['stage_service_ms'][0],
        different_resource_configuration_boot_and_probe_than_ResNet=True,
        not_a_matched_cross_network_causal_comparison=True)


def run():
    receipts,hashes=capture.verified();observed=receipts['observed'];raw=Path(observed['raw']);doc=observed['task']
    phases=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    trace=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    launches=[r for r in trace if r['kind']==0];callbacks=defaultdict(list);barriers=defaultdict(list)
    for row in trace:
        if row['kind'] in (1,2):(callbacks if row['kind']==1 else barriers)[row['launch']].append(row)
    intr=read(ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json')
    seal=read(ROOT/'artifacts/compiled_gap_validation_01/SELECTION_FREEZE.json')
    ref=seal['prediction_files']['resnet18/'+capture.campaign.CID+'/compiled_output_ring1']
    if sha256(ref['path'])!=ref['sha256']:raise ValueError('frozen model changed')
    model=read(ref['path'])['result'];rows=[]
    for index in (0,2):
        stage=doc['stages'][index];program=stage['program']
        old=next(v for v in intr['CPU'].values() if v['identity']['network']=='resnet18'
            and v['identity']['program']==program and v['identity']['threads']==3)
        pv=old['provenance'];source=pv.get('receipt',pv.get('observed_receipt'));expected=pv.get('sha256',pv.get('observed_sha256'))
        if sha256(source)!=expected:raise ValueError('isolated profile provenance changed')
        r=read(source);verify_inventory(r['raw'],r['files']);hashes[source]=expected
        selected=sorted((p for p in phases if p['stage']==program and p['operation']=='run' and 8<=p['call']<24),key=lambda p:p['call'])
        if [p['call'] for p in selected]!=list(range(8,24)):raise ValueError('observed phase frames differ')
        current=average_calls([project_call(p,launches,callbacks,barriers) for p in selected])
        if [(k['function_offset'],k['tasks']) for k in old['average']['kernels']]!=[(k['function_offset'],k['tasks']) for k in current['kernels']]:
            raise ValueError('CPU work/dependency pattern differs')
        for scope,value in [('isolated',projected_metrics(old['average'])),('model',model_metrics(model,index)),
            ('pipeline_observer',projected_metrics(current))]:
            rows.append(dict(program=program,scope=scope,**value))
    result=dict(rows=rows,yolo_reference=yolo_reference(),source_receipts=hashes,model_reference=ref,source_sha256=sha256(__file__),
        observed_trace_probe_quality_passed=False,exact_remaining_error_causal_partition=False,
        independent_noncaller_bookkeeping_not_measured=True,no_CPU_time_wait_sum_added_to_II=True,
        contains_pipeline_observations=True,diagnostic_only=True,new_native_sessions=0,new_DNN=0,
        formula_DP_and_Top20_unchanged=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'caller_budget.csv',rows)
    render(result);return result


def render(r):
    lines=['CPU子图的额外需求：计算量、完成等待CPU和离核时间必须分开','',
        '本表只分解每帧调用线程的run墙钟：计算CPU＋串行/启动CPU＋完成等待及返回CPU＋离核时间。四项互不重复。',
        '计算核时来自各逻辑线程自己的CPU时间钟；记录到的阶段核时可按不同TID求和，但不把多个线程等待墙钟加到完成间隔中。非调用worker回调外记账仍未单独测量。',
        '隔离、模型和流水探针具有各自原身份。流水探针扰动条件未通过，不能精确归因剩余误差；“完成等待及返回”包含运行时返回工作，不能全部等同于纯自旋。','',
        '子图/口径 | caller计算CPU | 其他CPU | 完成等待及返回CPU | 离核时间 | run墙钟（均ms）']
    for row in r['rows']:
        lines.append(f"{row['program']}/{row['scope']} | {row['caller_compute_CPU_ms']:.3f} | {row['caller_other_CPU_ms']:.3f} | {row['completion_and_return_CPU_ms']:.3f} | {row['caller_off_CPU_ms']:.3f} | {row['caller_run_wall_ms']:.3f}")
    by={(x['program'],x['scope']):x for x in r['rows']};a=by['cpu_00_02','isolated'];b=by['cpu_00_02','pipeline_observer'];m=by['cpu_00_02','model']
    lines+=['','可支持的发现：',
        f"1. CPU0–2的计算回调总CPU从{a['all_callback_CPU_ms']:.3f}到{b['all_callback_CPU_ms']:.3f}核毫秒；记录到的阶段run总CPU却从{a['recorded_stage_run_CPU_ms']:.3f}到{b['recorded_stage_run_CPU_ms']:.3f}核毫秒。固定DNN计算量不等于固定运行时CPU需求。",
        f"2. CPU0–2调用线程的完成等待及返回CPU由{a['completion_and_return_CPU_ms']:.3f}增至{b['completion_and_return_CPU_ms']:.3f} ms。模型已产生{m['completion_and_return_CPU_ms']:.3f} ms等待CPU，所以不能说它完全遗漏这类竞争，也不能把实测值整体再相加。",
        '3. CPU5–7主要表现为离核时间增加。把CPU0–2和CPU5–7统一乘一个慢化比例，会把两种不同现象混在一起。',
        '4. 额外等待与并行内核完成次序、共享核心及其他阶段时序耦合，不能作为每个子图独立不变的局部成本。完成通知、唤醒及实际调度次序仍有未观测部分。',
        '本轮只完善证据与计量口径；没有得到新的可部署评分，没有新增板测，不用诊断数据拟合系数。','',
        '附：复用已验收的YOLO修复轨迹，不增加测量。该四线程前段同样出现完成等待CPU增长；说明这不是只在ResNet出现的现象。']
    for row in r['yolo_reference']['rows']:
        lines.append(f"YOLO/{row['scope']}：caller计算{row['caller_compute_CPU_ms']:.3f} ms，其他CPU{row['caller_other_CPU_ms']:.3f} ms，完成等待及返回CPU{row['completion_and_return_CPU_ms']:.3f} ms，离核{row['caller_off_CPU_ms']:.3f} ms。")
    y=r['yolo_reference']
    lines += [f"YOLO该批前段干净服务{y['clean_front_ms']:.3f} ms，探针服务{y['observed_front_ms']:.3f} ms。它来自旧boot，仍启用TASKSTATS，窗口{y['window']}，与ResNet绑定、线程数、切图和探针均不同；不据此把差异全部归因于DNN结构，也不跨批计算性能增益。",
        '共同规律是CPU子图在流水中会因完成等待而产生额外CPU占用；具体程度依赖子图及共享布局，不应按网络名称设系数。','']
    (OUT/'FINDINGS_zh.txt').write_text('\n'.join(lines),encoding='utf-8');REPORT.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS.json','FINDINGS_zh.txt','caller_budget.csv'):shutil.copy2(OUT/name,REPORT/name)
    plot(r)


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(15.6,4.1),layout='constrained')
    parts=[('caller_compute_CPU_ms','Computation CPU','#3977ad'),
        ('caller_other_CPU_ms','Other CPU work','#9ac1dc'),
        ('completion_and_return_CPU_ms','Completion / return CPU','#e6a03d'),
        ('caller_off_CPU_ms','Off-CPU time','#b5b5b5')]
    scopes=['isolated','model','pipeline_observer']
    for ax,program in zip(axes,('cpu_00_02','cpu_05_07','yolov3_tiny416/cpu_00_02')):
        yolo=program.startswith('yolov3')
        available=r['yolo_reference']['rows'] if yolo else r['rows']
        selected_scopes=['isolated','pipeline_observer'] if yolo else scopes
        rows=[next(row for row in available if row['program']==program and row['scope']==scope) for scope in selected_scopes]
        left=[0.]*len(rows)
        for key,label,color in parts:
            values=[row[key] for row in rows]
            ax.barh(range(len(rows)),values,left=left,label=label,color=color,height=.62)
            left=[a+b for a,b in zip(left,values)]
        for i,row in enumerate(rows):ax.text(left[i]+.4,i,f"{row['caller_run_wall_ms']:.2f}",va='center',fontsize=9)
        labels={'isolated':'Isolated trace','model':'Existing replay','pipeline_observer':'Pipeline observer'}
        ax.set_yticks(range(len(rows)),[labels[x] for x in selected_scopes])
        ax.invert_yaxis();ax.set_xlabel('Caller run wall time per frame (ms)')
        ax.set_title('YOLO CPU units 0-2' if yolo else 'ResNet '+program.replace('cpu_00_02','CPU units 0-2').replace('cpu_05_07','CPU units 5-7'))
        ax.set_xlim(0,max(left)*1.15);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Diagnostic decomposition: observed scheduling was perturbed by instrumentation',fontsize=11)
    for ext in ('svg','png'):
        path=OUT/('caller_wait_budget.'+ext);fig.savefig(path,dpi=180)
        shutil.copy2(path,REPORT/path.name)
    plt.close(fig)


if __name__=='__main__':
    result=run();print(json.dumps(result['rows'],ensure_ascii=False,indent=2))
