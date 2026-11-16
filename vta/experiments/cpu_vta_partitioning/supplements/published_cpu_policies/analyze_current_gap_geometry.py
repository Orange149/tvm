"""Compare nonoverlapping paths of the frozen model and repaired old trace.

No replay, parameters, component costs or native sessions are added. A model
join resumes a continuation immediately; its subsequent CPU queueing belongs
to the caller span, so native completion tails must not be compared with a
constructed zero tail as though they were entirely missing from the model.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import statistics as st
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import clip,Attribution
from yolo_scheduler_trace_repair.events import validate_profile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_gap_geometry_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_current_gap_geometry'


def partition(run_start,run_end,kernels):
    previous=run_start;entry=body=0
    for row in kernels:
        a,b,c=row['launch_start_ns'],row['last_start_ns'],row['last_end_ns']
        if not previous<=a<=b<=c<=run_end:raise ValueError('kernel critical windows overlap or reverse')
        entry+=b-a;body+=c-b;previous=c
    outside=run_end-run_start-entry-body
    if outside<0:raise ValueError('path partition lost time')
    return dict(run_ns=run_end-run_start,critical_entry_ns=entry,
        critical_body_ns=body,caller_and_other_outside_kernel_envelope_ns=outside)


def model_rows(result,frames):
    tasks=result['tasks'];rows=[];paths=[]
    for frame in frames:
        selected=[t for t in tasks if t['stage']==0 and t['frame']==frame]
        start=min(t['ready_ns'] for t in selected if t['operation'] in ('serial','launch_prefix'))
        gets=[t for t in selected if t['operation']=='get']
        if len(gets)!=1:raise ValueError('cannot reconstruct model run/get boundary')
        end=gets[0]['ready_ns'];kernels=[]
        for k in range(12):
            prefixes=[t for t in selected if t['operation']=='launch_prefix' and t['kernel']==k]
            workers=[t for t in selected if t['operation']=='worker_compute' and t['kernel']==k]
            if len(prefixes)!=1 or sorted(t['slot'] for t in workers)!=list(range(4)):
                raise ValueError('model kernel/worker sequence incomplete')
            last=max(workers,key=lambda t:(t['end_ns'],-t['slot']))
            join=next(j for j in result['kernel_joins'] if j['stage']==0 and j['frame']==frame and j['kernel']==k)
            if join['generation']!=0 or last['end_ns']!=join['join_ns']:
                raise ValueError('unexpected barrier or ambiguous join')
            cpu=Fraction(last['CPU']['numerator'],last['CPU']['denominator'])
            path=dict(frame=frame,kernel=k,last_slot=last['slot'],
                launch_start_ns=prefixes[0]['ready_ns'],last_start_ns=last['first_service_ns'],last_end_ns=last['end_ns'],
                last_CPU_ns=float(cpu),last_nonrunning_ns=float(last['end_ns']-last['first_service_ns']-cpu))
            kernels.append(path);paths.append(path)
        rows.append(dict(frame=frame,**partition(start,end,kernels)))
    return rows,paths


def native_rows(receipt,frames):
    raw=Path(receipt['raw']);affinity=read(raw/'affinity.json')
    phases=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    trace=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    launches=[r for r in trace if r['kind']==0];workers=defaultdict(list)
    for r in trace:
        if r['kind']==1:workers[r['launch']].append(r)
    mapping={w['slot']:w['tid'] for w in affinity['stages'][0]['worker_slots']}
    streams=validate_profile(read(raw/'profile.json'),receipt['task'],read(raw/'affinity.json'))['streams']
    attribution=Attribution(streams);caller_stream=streams[str(affinity['stages'][0]['caller_tid'])]
    caller_core=receipt['task']['stages'][0]['worker_core_map'][0]
    rows=[];paths=[];outside_rows=[]
    for frame in frames:
        run=next(p for p in phases if p['stage']=='cpu_00_02' and p['operation']=='run' and p['call']==frame)
        group=sorted([r for r in launches if r['caller']==run['tid'] and
            run['wall_start_ns']<=r['wall_start_ns']<=r['wall_end_ns']<=run['wall_end_ns']],key=lambda r:r['wall_start_ns'])
        if len(group)!=12:raise ValueError('native launch sequence incomplete')
        kernels=[]
        for k,launch in enumerate(group):
            ws=workers[launch['launch']]
            if sorted(w['task'] for w in ws)!=list(range(4)) or any(w['tid']!=mapping[w['task']] for w in ws):
                raise ValueError('native logical thread identity differs')
            last=max(ws,key=lambda w:(w['wall_end_ns'],-w['task']))
            cpu=last['cpu_end_ns']-last['cpu_start_ns']
            path=dict(frame=frame,kernel=k,last_slot=last['task'],function_offset=last['function_offset'],
                launch_start_ns=launch['wall_start_ns'],last_start_ns=last['wall_start_ns'],last_end_ns=last['wall_end_ns'],
                last_CPU_ns=cpu,last_nonrunning_ns=last['wall_end_ns']-last['wall_start_ns']-cpu)
            kernels.append(path);paths.append(path)
        rows.append(dict(frame=frame,**partition(run['wall_start_ns'],run['wall_end_ns'],kernels)))
        cursor=run['wall_start_ns'];on=off=unknown=0;actors=defaultdict(int)
        for a,b in [(r['launch_start_ns'],r['last_end_ns']) for r in kernels]+[(run['wall_end_ns'],run['wall_end_ns'])]:
            span=clip(caller_stream,cursor,a)
            on+=span['on_CPU_ns'];off+=span['off_CPU_ns'];unknown+=span['unknown_ns']
            for wait in span['off']:
                for role,ns in attribution.other_work(wait,caller_core)['by_role_ns'].items():actors[role]+=ns
            cursor=b
        if unknown or on+off!=rows[-1]['caller_and_other_outside_kernel_envelope_ns']:
            raise ValueError('caller outside-envelope scheduler coverage incomplete')
        outside_rows.append(dict(frame=frame,on_CPU_ns=on,off_CPU_ns=off,
            coincident_other_CPU_ns=dict(actors),off_CPU_not_all_runqueue=True))
    return rows,paths,outside_rows


def summary(rows):
    return {k.replace('_ns','_ms'):st.mean(r[k] for r in rows)/1e6 for k in rows[0] if k!='frame'}


def run():
    parent=ROOT/'artifacts/current_model_gap_01';plan=read(parent/'PLAN.json')
    if sha256(parent/'INPUTS.json')!=plan['input_sha256']:raise ValueError('isolated model inputs changed')
    for path,expected in plan['original_model_sources'].items():
        if sha256(path)!=expected:raise ValueError('frozen replay implementation changed')
    sealed=read(parent/'REPLAY_FREEZE.json');sources={};model={}
    # Load and validate both sealed predictions before any native performance.
    for name,expected in sealed['files'].items():
        p=parent/name
        if sha256(p)!=expected:raise ValueError('frozen prediction changed')
        model[p.stem]=read(p)['result'];sources[str(p)]=expected
    native_path=ROOT/'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json'
    delivery=read(native_path.parent.parent/'DELIVERY.json')
    if sha256(native_path)!=delivery['results_sha256']:raise ValueError('repaired trace changed')
    trace=read(native_path)
    if trace['probe_quality_passed'] is not True:raise ValueError('native selected-frame attribution unproven')
    frames=trace['complete_front_frames'];receipt=None
    for path,expected in trace['source_receipts'].items():
        if sha256(path)!=expected:raise ValueError('native source receipt changed')
        r=read(path);verify_inventory(r['raw'],r['files']);sources[path]=expected
        if r['task']['spec']['phase']=='observed':receipt=r
    if receipt is None:raise ValueError('missing accepted observation')
    nrows,npaths,outside=native_rows(receipt,frames);means=dict(native_observed=summary(nrows));all_frames=[];all_paths=[]
    for mode,result in model.items():
        rows,paths=model_rows(result,frames);means[mode]=summary(rows)
        all_frames += [dict(mode=mode,**r) for r in rows];all_paths += [dict(mode=mode,**r) for r in paths]
    all_frames += [dict(mode='native_observed',**r) for r in nrows]
    all_paths += [dict(mode='native_observed',**r) for r in npaths]
    deltas={mode:{k:means[mode][k]-means['native_observed'][k] for k in means[mode]} for mode in model}
    per_kernel=[]
    for k in range(12):
        native=[r for r in npaths if r['kernel']==k]
        for mode in model:
            selected=[r for r in all_paths if r['mode']==mode and r['kernel']==k]
            per_kernel.append(dict(mode=mode,kernel=k,
                native_entry_ms=st.mean(r['last_start_ns']-r['launch_start_ns'] for r in native)/1e6,
                model_entry_ms=st.mean(r['last_start_ns']-r['launch_start_ns'] for r in selected)/1e6,
                native_body_ms=st.mean(r['last_end_ns']-r['last_start_ns'] for r in native)/1e6,
                model_body_ms=st.mean(r['last_end_ns']-r['last_start_ns'] for r in selected)/1e6,
                native_last_callback_CPU_ms=st.mean(r['last_CPU_ns'] for r in native)/1e6,
                model_last_compute_CPU_ms=st.mean(r['last_CPU_ns'] for r in selected)/1e6,
                native_CPU_and_model_CPU_clock_scopes_differ=True,
                critical_slot_agreement=sum(a['last_slot']==b['last_slot'] for a,b in zip(native,selected))/len(frames)))
    outside_summary=dict(on_CPU_ms=st.mean(r['on_CPU_ns'] for r in outside)/1e6,
        off_CPU_ms=st.mean(r['off_CPU_ns'] for r in outside)/1e6,
        coincident_other_CPU_ms={role:sum(r['coincident_other_CPU_ns'].get(role,0) for r in outside)/len(frames)/1e6
            for role in sorted({role for r in outside for role in r['coincident_other_CPU_ns']})},
        coincidence_not_causal_recoverable_latency=True)
    result=dict(means=means,model_minus_native_observed_ms=deltas,matched_frames=frames,kernels=per_kernel,
        native_caller_outside_kernel_envelopes=outside_summary,
        sources=sources,model_sources_are_isolated_only=True,
        model_is_frozen_finite_service_with_refreshed_VTA_completion_profiles=True,
        no_new_model_or_replay=True,new_native_sessions=0,new_DNN=0,
        caller_outside_span_merges_return_and_serial_to_avoid_artificial_zero_tail_comparison=True,
        entry_markers_and_model_first_service_have_distinct_envelopes=True,
        original_clean_stage_service_ms=trace['clean']['stage_service_ms'][0],
        native_observed_not_formal_performance=True,source_sha256=sha256(__file__))
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'frames.csv',all_frames)
    csv_write(OUT/'kernels.csv',per_kernel);csv_write(OUT/'paths.csv',all_paths)
    text=['当前冻结回放与流水轨迹：相同帧的运行路径分解','',
        '只使用现有输入、封存预测及已经修复缓冲截断的轨迹，不重跑模型或DNN。',
        '两种回放仅设备完成语义不同；相同切图、线程/核心、CPU隔离输入和有限服务近似。',
        '原始观测质量条件支持所选帧的诊断；不以探针开启的耗时报告正式FPS。',
        '逐帧相同口径：帧12–23，每帧12次启动。运行墙钟拆成最后完成线程进入前、回调体跨度、剩余调用线程及其他区间。',
        '三个区间不重叠，其和严格等于run墙钟。完成返回尾部与串行区间合并，避免模型把等待移到下一段后误判为全漏计。','',
        '版本 / run ms / 关键线程进入前ms / 关键回调体跨度ms / 剩余调用线程区间ms']
    keys=('run_ms','critical_entry_ms','critical_body_ms','caller_and_other_outside_kernel_envelope_ms')
    for mode,v in means.items():text.append(mode+' / '+' / '.join(f'{v[k]:.3f}' for k in keys))
    text+=['',f"实际调用线程在第三列剩余区间内，运行{outside_summary['on_CPU_ms']:.3f} ms，未运行{outside_summary['off_CPU_ms']:.3f} ms。",
        '同核其他CPU工作重合量：'+str(outside_summary['coincident_other_CPU_ms'])+' ms；不是移除这些工作后可等量恢复的时间。',
        '各内核明细见kernels.csv。回调体跨度同时含运行和未运行时间，不称为纯计算。',
        '模型最后完成线程与真实最后完成线程可能不同；分解对照用于定位错位，不把两者差额逐项当作可添加的代价。',
        '原始回调CPU时钟、模型有效计算CPU工作以及调度on-CPU记录具有不同边界，不能混成一个精确等式。',
        '本表与正式阶段平均服务的窗口范围不同，明确保留帧12–23的条件，不拼接不同窗口或旧FPS。',
        '没有把任何流水时长、CPU排队或FPS输入新模型。结果不改静态评分、DP或Top-20。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in
        [OUT/'RESULTS.json',OUT/'frames.csv',OUT/'kernels.csv',OUT/'paths.csv',REPORT/'FINDINGS_zh.txt']}),replace=True)
    print(json.dumps(dict(means=means,deltas=deltas,new_DNN=0,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
