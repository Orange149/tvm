"""Bound what old callback markers can say about real completion notification.

Uses accepted/repaired records only. No board calls, runtime replacement, score,
or empirical wait penalty. Later TASKSTATS read markers are still not runtime
num_pending decrement markers.
"""
from collections import defaultdict
from pathlib import Path
import json
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from yolo_scheduler_trace.events import clip, Attribution
from yolo_scheduler_trace_repair.events import validate_profile
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/completion_marker_gap_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_completion_marker_gap'


def verified_sources():
    old_path = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json'
    old = read(old_path); hashes = {str(old_path): sha256(old_path)}; yolo = None
    for path, expected in old['source_receipts'].items():
        if sha256(path) != expected: raise ValueError('YOLO receipt changed')
        receipt = read(path); verify_inventory(receipt['raw'], receipt['files']); hashes[path] = expected
        if receipt['task']['spec']['phase'] == 'observed': yolo = receipt
    if yolo is None: raise ValueError('missing original observation')
    raw = Path(yolo['raw'])
    yolo_streams = validate_profile(read(raw / 'profile.json'), yolo['task'], read(raw / 'affinity.json'))['streams']
    freeze_path = ROOT / 'artifacts/resnet_middle_switch_01/analysis/DIAGNOSTIC_RAW_FREEZE.json'
    frozen = read(freeze_path); verify_inventory(frozen['raw'], frozen['files'])
    if frozen['original_campaign_acceptance'] is not False:
        raise ValueError('original rejected session was relabeled')
    import analyze_resnet_middle_switch as resnet
    raw = Path(frozen['raw']); phases = lines(raw / 'phase_trace.jsonl')
    res_streams = resnet.diagnostic_events(read(raw / 'profile.json'), frozen['task'], read(raw / 'affinity.json'), phases)['streams']
    hashes[str(freeze_path)] = sha256(freeze_path)
    for path in [Path(resnet.__file__), Path(__file__), ROOT / 'yolo_scheduler_trace/events.py',
                 ROOT / 'yolo_scheduler_trace_repair/events.py']:
        hashes[str(path)] = sha256(path)
    return [dict(network='yolov3_tiny416', raw=yolo['raw'], task=yolo['task'], streams=yolo_streams,
                 target_stages=[0], quality=old['probe_quality_passed'], queue_reads=True,
                 original_acceptance=True, expected_frames={0: old['complete_front_frames']}),
            dict(network='resnet18', raw=frozen['raw'], task=frozen['task'], streams=res_streams,
                 target_stages=[0, 2], quality=False, queue_reads=False,
                 original_acceptance=False, expected_frames={})], hashes


def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]


def marker_interval(launch, workers, queue_enabled):
    body_end = max(w['wall_end_ns'] for w in workers)
    return_end = launch['wall_end_ns']
    if return_end < body_end: raise ValueError('launch returns before callbacks')
    if queue_enabled:
        if any(w['queue_read_end1_ns'] < w['wall_end_ns'] for w in workers):
            raise ValueError('queue read must follow callback-end marker')
        later = max(w['queue_read_end1_ns'] for w in workers)
    else:
        if any(w['queue_read_end1_ns'] != 0 for w in workers):
            raise ValueError('disabled queue interface has a read marker')
        later = body_end
    if not body_end <= later <= return_end:
        raise ValueError('completion marker order invalid')
    return dict(body_end_ns=body_end, later_observed_end_ns=later, return_end_ns=return_end,
                original_marker_to_return_ns=return_end-body_end,
                known_pre_signal_read_extension_ns=later-body_end,
                unresolved_to_return_ns=return_end-later,
                later_marker_is_runtime_signal=False,
                unresolved_is_upper_limit_on_post_signal_return_not_actual_delay=True)


def extract(source):
    raw = Path(source['raw']); task = source['task']; streams = source['streams']
    phases = lines(raw / 'phase_trace.jsonl'); trace = lines(raw / 'wait_trace.jsonl')
    launches = [r for r in trace if r['kind'] == 0]; callbacks = defaultdict(list)
    for r in trace:
        if r['kind'] == 1: callbacks[r['launch']].append(r)
    affinity = read(raw / 'affinity.json'); rows = []; frames = defaultdict(list); partial = []
    attribution = Attribution(streams)
    for index in source['target_stages']:
        stage = task['stages'][index]
        tids = {w['slot']:w['tid'] for w in affinity['stages'][index]['worker_slots']}
        selected = sorted((p for p in phases if p['stage']==stage['program'] and p['operation']=='run'
                           and task['warmup'] <= p['call'] < task['warmup']+task['scored']), key=lambda p:p['call'])
        if [p['call'] for p in selected] != list(range(task['warmup'], task['warmup']+task['scored'])):
            raise ValueError('incomplete scored phase set')
        for phase in selected:
            a, b = phase['wall_start_ns'], phase['wall_end_ns']
            coverage_tids = streams.keys() if index in source['expected_frames'] else [str(t) for t in tids.values()]
            if any(not clip(streams[t], a, b)['fully_covered'] for t in coverage_tids):
                partial.append(dict(network=source['network'], stage=stage['program'], frame=phase['call']))
                continue
            frames[index].append(phase['call'])
            group = sorted((r for r in launches if r['caller']==phase['tid'] and a<=r['wall_start_ns']<=r['wall_end_ns']<=b),
                           key=lambda r:r['wall_start_ns'])
            expected = {('yolov3_tiny416', 0):12, ('resnet18', 0):13, ('resnet18', 2):10}[(source['network'], index)]
            if len(group)!=expected: raise ValueError('kernel sequence count changed')
            for kernel, launch in enumerate(group):
                workers = sorted(callbacks[launch['launch']], key=lambda r:r['task'])
                if (len(workers)!=len(tids) or [w['task'] for w in workers]!=list(range(len(tids)))
                    or any(w['tid']!=tids[w['task']] for w in workers)):
                    raise ValueError('logical worker mapping changed')
                markers = marker_interval(launch, workers, source['queue_reads'])
                last = max(workers, key=lambda w:(w['wall_end_ns'], -w['task']))
                late = markers['later_observed_end_ns']; end = markers['return_end_ns']
                caller = clip(streams[str(tids[0])], late, end)
                worker = clip(streams[str(last['tid'])], markers['body_end_ns'], end)
                entry = clip(streams[str(last['tid'])], launch['wall_start_ns'], last['wall_start_ns'])
                body = clip(streams[str(last['tid'])], last['wall_start_ns'], last['wall_end_ns'])
                if not caller['fully_covered'] or not worker['fully_covered']:
                    raise ValueError('unexpected uncovered interior')
                actors = defaultdict(int)
                for off in caller['off']:
                    for role, ns in attribution.other_work(off, stage['worker_core_map'][0])['by_role_ns'].items():
                        actors[role] += ns
                path_actors={};path_unknown={}
                for name,span in [('entry',entry),('body',body)]:
                    roles=defaultdict(int);unknown=0
                    for off in span['off']:
                        value=attribution.other_work(off,stage['worker_core_map'][last['task']])
                        for role,ns in value['by_role_ns'].items():roles[role]+=ns
                        unknown+=value['unobserved_ns']
                    path_actors[name]=dict(roles);path_unknown[name]=unknown
                rows.append(dict(network=source['network'], stage=stage['program'], frame=phase['call'], kernel=kernel,
                    launch=launch['launch'], last_body_slot=last['task'], **markers,
                    launch_wall_ns=launch['wall_end_ns']-launch['wall_start_ns'],
                    last_body_entry_wall_ns=last['wall_start_ns']-launch['wall_start_ns'],
                    last_body_entry_on_CPU_ns=entry['on_CPU_ns'],
                    last_body_entry_off_CPU_ns=entry['off_CPU_ns'],
                    last_body_wall_ns=last['wall_end_ns']-last['wall_start_ns'],
                    last_body_CPU_clock_ns=last['cpu_end_ns']-last['cpu_start_ns'],
                    last_body_on_CPU_ns=body['on_CPU_ns'], last_body_off_CPU_ns=body['off_CPU_ns'],
                    stage_run_wall_ns=b-a,
                    caller_post_marker_on_CPU_ns=caller['on_CPU_ns'],
                    caller_post_marker_preempt_off_ns=caller['preempt_off_ns'],
                    caller_post_marker_nonpreempt_off_ns=caller['nonpreempt_off_ns'],
                    last_body_worker_post_body_on_CPU_ns=worker['on_CPU_ns'],
                    last_body_worker_post_body_off_CPU_ns=worker['off_CPU_ns'],
                    coincident_actor_CPU_ns=dict(actors),
                    critical_entry_and_body_actors_CPU_ns=path_actors,
                    critical_entry_and_body_unobserved_ns=path_unknown,
                    original_campaign_accepted=source['original_acceptance'],
                    probe_quality_for_original_gap=source['quality']))
        if index in source['expected_frames'] and frames[index]!=source['expected_frames'][index]:
            raise ValueError('complete frame set changed')
    summary = []
    for index in source['target_stages']:
        program = task['stages'][index]['program']; chosen = [r for r in rows if r['stage']==program]
        frame_ids = frames[index]
        if not frame_ids: raise ValueError('no fully covered frame')
        keys = ('original_marker_to_return_ns', 'known_pre_signal_read_extension_ns', 'unresolved_to_return_ns',
                'caller_post_marker_on_CPU_ns', 'caller_post_marker_preempt_off_ns',
                'caller_post_marker_nonpreempt_off_ns', 'last_body_worker_post_body_off_CPU_ns',
                'launch_wall_ns','last_body_entry_wall_ns','last_body_entry_on_CPU_ns','last_body_entry_off_CPU_ns',
                'last_body_wall_ns','last_body_CPU_clock_ns','last_body_on_CPU_ns','last_body_off_CPU_ns')
        stage_wall=st.mean(next(r['stage_run_wall_ns'] for r in chosen if r['frame']==f) for f in frame_ids)
        serial=stage_wall-sum(r['launch_wall_ns'] for r in chosen)/len(frame_ids)
        if serial<0 or any(r['launch_wall_ns']!=r['last_body_entry_wall_ns']+r['last_body_wall_ns']+
                          r['original_marker_to_return_ns'] for r in chosen):
            raise ValueError('critical path does not conserve run wall')
        summary.append(dict(network=source['network'], stage=program, complete_frames=frame_ids,
            kernels=len(chosen), queue_reads=source['queue_reads'],
            stage_run_wall_ms=stage_wall/1e6, outside_launch_wall_ms=serial/1e6,
            **{k.replace('_ns','_ms_per_frame'):sum(r[k] for r in chosen)/len(frame_ids)/1e6 for k in keys},
            critical_path_coincident_actor_CPU_ms_per_frame={part:{role:
                sum(r['critical_entry_and_body_actors_CPU_ns'][part].get(role,0) for r in chosen)/len(frame_ids)/1e6
                for role in sorted({role for r in chosen for role in r['critical_entry_and_body_actors_CPU_ns'][part]})}
                for part in ('entry','body')},
            critical_path_unobserved_actor_ms_per_frame={part:
                sum(r['critical_entry_and_body_unobserved_ns'][part] for r in chosen)/len(frame_ids)/1e6
                for part in ('entry','body')},
            probe_quality_for_original_gap=source['quality'], exact_runtime_signal_observed=False))
    return rows, summary, partial


def isolated_yolo_geometry():
    """Same clock/body markers, without inventing isolation scheduler events."""
    index_path=ROOT/'artifacts/caller_wait_budget_01/RESULTS.json'
    old=read(index_path); receipts=old['yolo_reference']['source_receipts']
    matches=[]
    for path, expected in receipts.items():
        if sha256(path)!=expected:raise ValueError('isolated index receipt changed')
        r=read(path)
        if r['task']['spec']['scope']!='pipeline':matches.append((path,expected,r))
    if len(matches)!=1:raise ValueError('expected the one frozen isolated reference')
    path,expected,r=matches[0];verify_inventory(r['raw'],r['files']);raw=Path(r['raw']);task=r['task']
    phases=lines(raw/'phase_trace.jsonl');trace=lines(raw/'wait_trace.jsonl')
    launches=[a for a in trace if a['kind']==0];workers=defaultdict(list)
    for a in trace:
        if a['kind']==1:workers[a['launch']].append(a)
    chosen=sorted((p for p in phases if p['stage']=='cpu_00_02' and p['operation']=='run'
        and task['warmup']<=p['call']<task['warmup']+task['scored']),key=lambda p:p['call'])
    if len(chosen)!=task['scored']:raise ValueError('incomplete isolated phases')
    rows=[]
    for phase in chosen:
        group=sorted((a for a in launches if a['caller']==phase['tid'] and
            phase['wall_start_ns']<=a['wall_start_ns']<=a['wall_end_ns']<=phase['wall_end_ns']),key=lambda a:a['wall_start_ns'])
        if len(group)!=12:raise ValueError('isolated kernel sequence incomplete')
        entry=body=cpu=tail=0
        for launch in group:
            ws=workers[launch['launch']]
            if sorted(w['task'] for w in ws)!=[0,1,2,3]:raise ValueError('isolated worker slots invalid')
            last=max(ws,key=lambda w:(w['wall_end_ns'],-w['task']))
            entry+=last['wall_start_ns']-launch['wall_start_ns']
            body+=last['wall_end_ns']-last['wall_start_ns']
            cpu+=last['cpu_end_ns']-last['cpu_start_ns']
            tail+=launch['wall_end_ns']-last['wall_end_ns']
        wall=phase['wall_end_ns']-phase['wall_start_ns'];outside=wall-entry-body-tail
        if outside<0:raise ValueError('isolated geometry double counted')
        rows.append(dict(frame=phase['call'],run_ms=wall/1e6,outside_ms=outside/1e6,
            last_entry_ms=entry/1e6,last_body_ms=body/1e6,last_body_CPU_ms=cpu/1e6,
            last_body_nonrunning_clock_difference_ms=(body-cpu)/1e6,return_ms=tail/1e6))
    return dict(rows=rows,means={k:st.mean(v[k] for v in rows) for k in rows[0] if k!='frame'},
        source={path:expected,str(index_path):sha256(index_path)},
        scope='historical observed isolated reference; no isolation scheduler timeline; last-worker path can change between contexts',
        whole_CPU_wait_sums_not_added=True)


def main():
    inputs, sources = verified_sources(); rows=[]; summaries=[]; partial=[]
    for source in inputs:
        a,b,c=extract(source); rows+=a; summaries+=b; partial+=c
    isolated=isolated_yolo_geometry()
    result = dict(summaries=summaries, kernels=rows, partial_frames=partial, sources=sources,
        isolated_yolo=isolated,
        new_native_sessions=0, new_DNN=0, exact_runtime_signal_observed=False,
        correction_promoted=False, formula_DP_Top20_unchanged=True,
        no_pipeline_durations_as_prediction_inputs=True,
        known_read_extension_is_not_total_probe_disturbance=True)
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    write(OUT/'RESULTS.json', result, replace=True); csv_write(OUT/'summary.csv', summaries)
    text=['完成通知观测边界：已有轨迹的离线核查', '',
        '没有新增推理、板端连接、运行库修改、评分修改或Top-20搜索。',
        '重新核验原始哈希、修复后的物理缓冲、线程映射、完整帧覆盖。ResNet原观察会话仍保持拒收；本次仅使用已经说明的离线诊断视图。', '',
        '记录的回调结束之后，探针仍会读取TASKSTATS并记账，之后运行库才递减num_pending。',
        '已有queue_read_end1只给出一个更晚、仍先于真正通知的时间点；它不能代替真正通知。',
        '以该点重算的返回尾部只是“真正通知之后到返回所用时间”的最大可能值；不能当实际CPU排队或可加到模型的惩罚。', '',
        '网络 / 阶段 / 完整帧数 / 原回调结束到返回ms每帧 / 已知后续读取扩展ms每帧 / 仍未分开的尾部ms每帧 / caller抢占离核ms每帧']
    for s in summaries:
        text.append(f"{s['network']} / {s['stage']} / {len(s['complete_frames'])} / "
            f"{s['original_marker_to_return_ms_per_frame']:.4f} / {s['known_pre_signal_read_extension_ms_per_frame']:.4f} / "
            f"{s['unresolved_to_return_ms_per_frame']:.4f} / {s['caller_post_marker_preempt_off_ms_per_frame']:.4f}")
    text += ['', 'YOLO最后完成worker路径的完整分解（不是所有线程等待之和）',
        '口径 / run墙钟 / 启动外 / 最后worker进入回调前 / 其回调执行墙钟 / 其中线程CPU钟 / 最后回调结束到返回，单位ms']
    iso=isolated['means']; yolo=next(s for s in summaries if s['network']=='yolov3_tiny416')
    text.append(f"隔离探针8帧 / {iso['run_ms']:.3f} / {iso['outside_ms']:.3f} / {iso['last_entry_ms']:.3f} / {iso['last_body_ms']:.3f} / {iso['last_body_CPU_ms']:.3f} / {iso['return_ms']:.3f}")
    text.append(f"流水探针完整12帧 / {yolo['stage_run_wall_ms']:.3f} / {yolo['outside_launch_wall_ms']:.3f} / {yolo['last_body_entry_wall_ms_per_frame']:.3f} / {yolo['last_body_wall_ms_per_frame']:.3f} / {yolo['last_body_CPU_clock_ms_per_frame']:.3f} / {yolo['original_marker_to_return_ms_per_frame']:.3f}")
    text += ['','关键worker离核时，同一核心上记录到的其他工作（时间重合，不是全部因果分摊）：']
    for part in ('entry','body'):
        roles=yolo['critical_path_coincident_actor_CPU_ms_per_frame'][part]
        text.append(part+' / '+', '.join(f'{role}: {value:.3f}ms' for role,value in sorted(roles.items(),key=lambda p:-p[1])))
        text.append(part+f" / 未观测竞争角色: {yolo['critical_path_unobserved_actor_ms_per_frame'][part]:.3f}ms")
    text += ['每次并行调用只取最后完成worker的一条路径，四段严格重建run墙钟，区间不重叠；不同上下文最后完成worker可能不同。',
        '流水最后worker回调内记录到约53.56ms离核；进入回调前约17.81ms离核。它们不是全部调度排队，真实唤醒时刻仍未直接观测。',
        '这些是旧探针会话及其完整帧子集，不是新一轮正式性能，不能与最新干净会话逐毫秒拼接。',
        '这个量级检查改变下一步优先级：先查worker进入与执行期间的共享调度，不能只盯着计算完成后的返回尾部。',
        '主机在关键worker回调离核期间的同核占用约34.14ms，CPU其他阶段约10.22ms，另约9.20ms未归入已记录角色。',
        '以上不是原模型尚缺的34.14ms；原模型已包含主机CPU需求及部分竞争，需要在相同区间定义下检查其时序预测，不能再次整段叠加。',
        '硬件指令事件零DNN自测已单独通过，但其DNN扰动未验证；本轮不因此自动补测DNN。', '',
        'TASKSTATS关闭时，没有更晚的读取时间戳；读取扩展填0只表示没有这项读取，绝不表示包装器开销为0。',
        '多个内核的返回区间顺序发生，因此可报告每阶段累计；多个worker的等待不相加到完成间隔。',
        'caller离核时同时运行的其他角色只是时间重合证据；不能证明所有等待都由该角色造成。',
        'ResNet精确残差归因的探针质量未通过，本报告保留限制。YOLO原协议支持较大隔离差距的诊断，不代表当前45ms残差的定量归因成立。',
        '真实通知仍缺。硬件指令事件能力依据：https://man7.org/linux/man-pages/man2/perf_event_open.2.html；本地自测artifacts/completion_breakpoint_capability_01/BOARD_RECEIPT.json。',
        '本报告不支持静态修正已解决误差或Top-20已改善的结论。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.2));bottom=[0,0]
    parts=[('Outside launches',[iso['outside_ms'],yolo['outside_launch_wall_ms']]),
           ('Launch to last-worker body',[iso['last_entry_ms'],yolo['last_body_entry_wall_ms_per_frame']]),
           ('Last-worker body',[iso['last_body_ms'],yolo['last_body_wall_ms_per_frame']]),
           ('After last body to return',[iso['return_ms'],yolo['original_marker_to_return_ms_per_frame']])]
    for label,values in parts:
        ax.bar([0,1],values,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,values)]
    ax.set_xticks([0,1],['Historical isolated probe\n8 frames','Historical pipeline probe\n12 fully covered frames'])
    ax.set_ylabel('CPU 0-2 run wall time (ms)');ax.set_title('YOLO: most added time precedes the final callback completion')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8)
    fig.text(.02,.01,'One non-overlapping last-worker path per launch. Probe observations, not new clean FPS or a deployable predictor.',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(REPORT/'critical_path.png',dpi=160);fig.savefig(REPORT/'critical_path.svg');plt.close(fig)
    print(json.dumps(dict(summaries=[{k:v for k,v in s.items() if k not in (
        'critical_path_coincident_actor_CPU_ms_per_frame',)} for s in summaries],
        isolated_means=isolated['means'],new_native_sessions=0,new_DNN=0),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
