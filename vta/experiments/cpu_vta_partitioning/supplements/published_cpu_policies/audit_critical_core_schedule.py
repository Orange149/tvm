"""Locate placement errors in a sealed replay using existing task events.

The recorder delegates every scheduling operation unchanged. Its replay must
equal the sealed result before the native trace is read. Observed occupancy
is diagnostic evidence, never input to a new prediction or a fitted penalty.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, digest, verify_inventory
from device_completion_replay import CompletionFinite
from phase_model.simulator import ceiling
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.analysis import queue_bounds
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import clip, Attribution, intersection
from yolo_scheduler_trace_repair.events import validate_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/critical_core_schedule_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_critical_core_schedule'
KERNELS = (2, 6, 10)


class ServiceRecorder:
    """Observe chosen services without adding tasks or changing time."""
    def __init__(self, owner, begin, end):
        self.owner = owner
        self.delegate = owner.scheduler
        self.begin, self.end = begin, end
        self.rows = {c:[] for c in range(4)}

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def consume(self, tasks, duration):
        start = max(self.owner.time, self.begin)
        end = min(self.owner.time + duration, self.end)
        if start < end:
            for core, key in self.delegate.choices.items():
                task = self.owner.active[key]
                meta = task['meta']
                identity = tuple(meta.get(k) for k in ('role','stage','frame','slot','operation','kernel'))
                row = dict(start_ns=ceiling(start),end_ns=ceiling(end),core=core,
                           identity=list(identity),synthetic_wait=task['waiting'])
                if row['end_ns'] == row['start_ns']:
                    continue
                previous = self.rows[core][-1] if self.rows[core] else None
                if (previous and previous['end_ns']==row['start_ns'] and
                        previous['identity']==row['identity'] and
                        previous['synthetic_wait']==row['synthetic_wait']):
                    previous['end_ns'] = row['end_ns']
                else:
                    self.rows[core].append(row)
        return self.delegate.consume(tasks, duration)


def model_role(identity, configuration):
    role, index, frame, slot, operation, kernel = identity
    if role != 'stage':
        return role
    stage = configuration['stages'][index]
    if stage['device']=='vta':
        return stage['program']+'/host'
    return stage['program']+('/caller' if slot==0 else '/worker'+str(slot))


def model_occupancy(recorder, task, core, configuration):
    start, end = task['first_service_ns'], task['end_ns']
    own = other = 0
    actors = defaultdict(int)
    for span in recorder.rows[core]:
        amount = intersection(start,end,span['start_ns'],span['end_ns'])
        if not amount:
            continue
        identity = span['identity']
        if (identity[0],identity[1],identity[3]) == (task['role'],task['stage'],task['slot']):
            own += amount
        else:
            other += amount
            actors[model_role(identity,configuration)] += amount
    if own+other > end-start:
        raise ValueError('recorded model occupancy exceeds a core')
    return dict(model_on_CPU_ns=own,model_other_on_core_ns=other,
                model_idle_ns=end-start-own-other,model_competing_roles_ns=dict(actors))


def run():
    if OUT.exists():
        raise FileExistsError('preserve the existing diagnostic analysis')
    parent = ROOT/'artifacts/current_model_gap_01'
    plan, inputs = read(parent/'PLAN.json'),read(parent/'INPUTS.json')
    if sha256(parent/'INPUTS.json') != plan['input_sha256']:
        raise ValueError('frozen isolated inputs changed')
    sources = {str(parent/'INPUTS.json'):plan['input_sha256']}
    for path, expected in plan['original_model_sources'].items():
        if sha256(path) != expected:
            raise ValueError('frozen model implementation changed')
        sources[path] = expected
    sealed = read(parent/'REPLAY_FREEZE.json')
    path = parent/'independent.json'
    if sha256(path) != sealed['files'][path.name]:
        raise ValueError('sealed replay changed')
    frozen = read(path)['result']
    sources[str(path)] = sha256(path)
    # The diagnostic frame selection is fixed by the previous accepted audit.
    # Only its frame numbers and source hashes are read before replay.
    geometry_path = ROOT/'artifacts/current_gap_geometry_01/RESULTS.json'
    geometry = read(geometry_path)
    frames = geometry['matched_frames']
    front = [t for t in frozen['tasks'] if t['stage']==0 and t['frame'] in frames]
    begin = min(t['ready_ns'] for t in front)
    end = max(t['end_ns'] for t in front)
    started = time.monotonic()
    with isolated_inputs_only():
        replay = CompletionFinite(inputs['case']['stages'],quanta=inputs['quanta_ns'],
            placement=inputs['placement'],completion_model='independent',
            completion_point=inputs['point'],runnable_completion_wait=True,controls=None,
            gate=inputs['case']['configuration']['execution_mode'],host_order='before',**inputs['window'])
        recorder = ServiceRecorder(replay,begin,end)
        replay.scheduler = recorder
        result = replay.run()
    if digest(result) != digest(frozen):
        raise ValueError('observing services changed the sealed replay')
    elapsed = time.monotonic()-started
    OUT.mkdir(parents=True)
    write(OUT/'REPLAY_CHECK.json',dict(exact_sealed_result_reproduced=True,
        original_sha256=sha256(path),result_digest=digest(result),seconds=elapsed,
        model_scheduling_or_inputs_changed=False,source_sha256=sha256(__file__)))
    print(json.dumps(dict(replay_matches=True,seconds=elapsed)),flush=True)
    # The sealed replay was reproduced before inspecting native placement.
    receipt = None
    for path, expected in geometry['sources'].items():
        if '/accepted/' not in path:
            continue
        if sha256(path) != expected:
            raise ValueError('diagnostic source receipt changed')
        value = read(path)
        if value['task']['spec']['phase']=='observed':
            if receipt is not None:
                raise ValueError('ambiguous native diagnostic source')
            receipt=value;sources[path]=expected
    if receipt is None:
        raise ValueError('native diagnostic source missing')
    raw = Path(receipt['raw'])
    verify_inventory(raw,receipt['files'])
    affinity = read(raw/'affinity.json')
    profile = validate_profile(read(raw/'profile.json'),receipt['task'],affinity)
    streams = profile['streams']
    attribution = Attribution(streams)
    tids = {w['slot']:w['tid'] for w in affinity['stages'][0]['worker_slots']}
    mapping = receipt['task']['stages'][0]['worker_core_map']
    trace = [json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    phases = [json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    launches = [r for r in trace if r['kind']==0]
    callbacks = defaultdict(list)
    for r in trace:
        if r['kind']==1:
            callbacks[r['launch']].append(r)
    predicted = {(t['frame'],t['kernel'],t['slot']):t for t in frozen['tasks']
                 if t['stage']==0 and t['operation']=='worker_compute'}
    rows=[]
    for frame in frames:
        runs=[p for p in phases if p['stage']=='cpu_00_02' and p['operation']=='run' and p['call']==frame]
        if len(runs)!=1:
            raise ValueError('native phase identity ambiguous')
        run=runs[0]
        chosen=sorted([l for l in launches if l['caller']==run['tid'] and
            run['wall_start_ns']<=l['wall_start_ns']<=l['wall_end_ns']<=run['wall_end_ns']],key=lambda l:l['wall_start_ns'])
        if len(chosen)!=12:
            raise ValueError('native launch sequence incomplete')
        for kernel in KERNELS:
            group=callbacks[chosen[kernel]['launch']]
            if sorted(w['task'] for w in group)!=list(range(4)):
                raise ValueError('native worker set incomplete')
            worker=max(group,key=lambda w:(w['wall_end_ns'],-w['task']))
            slot=worker['task'];tid=worker['tid'];core=mapping[slot]
            if tid!=tids[slot]:
                raise ValueError('native worker identity differs')
            span=clip(streams[str(tid)],worker['wall_start_ns'],worker['wall_end_ns'])
            if span['unknown_ns']:
                raise ValueError('critical callback event coverage incomplete')
            actors=defaultdict(int);unobserved=uncertainty=0
            for off in span['off']:
                other=attribution.other_work(off,core)
                unobserved+=other['unobserved_ns'];uncertainty+=other['overlap_uncertainty_ns']
                for role,ns in other['by_role_ns'].items():
                    actors[role]+=ns
            lo,hi=queue_bounds(worker)
            if lo<0 or hi<lo:
                raise ValueError('queue read envelope invalid')
            task=predicted[(frame,kernel,slot)]
            row=dict(frame=frame,kernel=kernel,slot=slot,core=core,
                native_on_CPU_ns=span['on_CPU_ns'],native_off_CPU_ns=span['off_CPU_ns'],
                native_preempt_off_ns=span['preempt_off_ns'],
                native_nonpreempt_off_ns=span['nonpreempt_off_ns'],
                native_queue_lower_ns=lo,native_queue_upper_ns=hi,
                native_yield_calls=worker['yield_calls'],native_barrier_calls=worker['barrier_calls'],
                native_competing_roles_ns=dict(actors),unobserved_other_work_ns=unobserved,
                observed_role_overlap_uncertainty_ns=uncertainty,
                native_callback_CPU_ns=worker['cpu_end_ns']-worker['cpu_start_ns'],
                **model_occupancy(recorder,task,core,inputs['case']['configuration']),
                same_worker_different_execution_windows=True,
                occupancy_is_not_a_removable_delay_or_predictor_input=True)
            rows.append(row)
    summaries=[]
    for kernel in KERNELS:
        chosen=[r for r in rows if r['kernel']==kernel]
        scalar_keys=('native_on_CPU_ns','native_off_CPU_ns','native_queue_lower_ns','native_queue_upper_ns',
                     'native_preempt_off_ns','native_nonpreempt_off_ns','model_on_CPU_ns',
                     'model_other_on_core_ns','model_idle_ns','unobserved_other_work_ns')
        summary=dict(kernel=kernel,frames=len(chosen),native_last_core_histogram=dict(Counter(r['core'] for r in chosen)),
                     mean_ms={k:sum(r[k] for r in chosen)/len(chosen)/1e6 for k in scalar_keys},
                     native_yield_calls=sum(r['native_yield_calls'] for r in chosen),
                     native_barrier_calls=sum(r['native_barrier_calls'] for r in chosen))
        for key in ('native_competing_roles_ns','model_competing_roles_ns'):
            roles=sorted({role for r in chosen for role in r[key]})
            summary[key.replace('_ns','_ms')]={role:sum(r[key].get(role,0) for r in chosen)/len(chosen)/1e6 for role in roles}
        summaries.append(summary)
    result=dict(rows=rows,summaries=summaries,sources=sources,
        isolated_model_exactly_reproduced=True,recording_did_not_change_schedule=True,
        new_native_sessions=0,new_DNN=0,offline_replays=1,
        actual_wakeup_timestamps_unobserved=True,new_predictor=False,
        causal_recovered_latency_not_inferred=True,model_or_static_costs_or_Top20_changed=False,
        source_sha256=sha256(__file__),references=[
            'https://www.kernel.org/doc/html/v5.4/accounting/delay-accounting.html',
            'https://www.kernel.org/doc/html/v5.4/scheduler/sched-design-CFS.html'])
    write(OUT/'RESULTS.json',result);csv_write(OUT/'critical_callbacks.csv',rows)
    csv_write(OUT/'summary.csv',summaries)
    spans=[row for core in range(4) for row in recorder.rows[core]]
    csv_write(OUT/'model_services.csv',spans)
    REPORT.mkdir(parents=True,exist_ok=True)
    lines=['关键计算线程：实际同核竞争与冻结回放对照','',
        '只使用已有隔离输入、冻结模型和已验收轨迹；无新板测。记录模型服务区间后，完整结果与原冻结回放逐值一致。',
        '相同诊断帧12–23，逐次选择实际最后完成线程，并在模型中定位同一逻辑线程。两边窗口取各自这次回调的跨度，不冒充同一绝对时刻。','',
        '计算序号 / 实际运行ms / 实际未运行ms / TASKSTATS排队区间ms / 模型同核其他工作ms']
    for r in summaries:
        m=r['mean_ms'];lines.append(f"{r['kernel']} / {m['native_on_CPU_ns']:.3f} / {m['native_off_CPU_ns']:.3f} / {m['native_queue_lower_ns']:.3f}–{m['native_queue_upper_ns']:.3f} / {m['model_other_on_core_ns']:.3f}")
        lines.append('实际同核工作重合ms：'+str(r['native_competing_roles_ms']))
        lines.append('模型同核其他工作ms：'+str(r['model_competing_roles_ms']))
    lines+=['',
        '三个计算序号的实际最后完成回调均未记录Yield或屏障调用；已记录的switch-out均带preempt标志。TASKSTATS排队增量提供独立证据，不能再仅把它们称为无法区分的睡眠。',
        '仍保留读数包络、CPU时钟/调度事件边界差异和未跟踪系统任务。排队计数不是精确唤醒时间，不宣称掌握完整Linux运行队列。',
        '主机工作总核时已经计入；本次暴露的是这些工作在何时、落在哪个核心上执行的近似误差。不能把观测到的主机重合时长再加到模型中重复计费。',
        '相同回调在实际与模型中完成时刻不同，重合量差异用于定位执行次序和落核假设，不等于可直接回填的慢化系数。',
        '隔离profile份额作用域修正与本问题分开；当前回放已使用内核隔离计算CPU工作，仍需要解释具体核心上关键线程的服务次序。',
        '后续优先评估不依赖流水FPS的主机落核不确定性处理，或可事前控制的主机放置；不继续通过填加壁钟残差来扩大模拟器。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    files=[OUT/'REPLAY_CHECK.json',OUT/'RESULTS.json',OUT/'critical_callbacks.csv',OUT/'summary.csv',
           OUT/'model_services.csv',REPORT/'FINDINGS_zh.txt']
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in files}))
    print(json.dumps(dict(summaries=summaries,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    run()
