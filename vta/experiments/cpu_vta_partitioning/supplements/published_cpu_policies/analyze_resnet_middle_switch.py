"""Attribute fully covered critical paths to recorded CPU actors, post-freeze.

Event coincidence is not proof of every wait's cause. In particular, switch-out
without PREEMPT does not distinguish sched_yield, sleep, and other blocking.
"""
from collections import defaultdict,Counter
from pathlib import Path
import json
import inspect
import shutil
import statistics as st

import run_resnet_middle_switch as campaign
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory,digest,inventory
from phase_model.extract import project_call
from yolo_scheduler_trace.events import clip,Attribution
from scheduler_gap_audit.curves import csv_write

ROOT=campaign.ROOT
OUT=campaign.POOL/"analysis"
REPORT=ROOT.parents[4]/"paper/my_paper/analysis/20261111_resnet_middle_switch"


def diagnostic_events(profile,doc,affinity,phases):
    """Keep the original rejection; allow only a proven, finished producer here.

    This separate offline view is not an original campaign acceptance receipt.
    An ended producer has no future work after its finite feed loop. Never infer
    a missing event interval's CPU state from this lifecycle exception.
    """
    roles=campaign.old_events.role_map(doc,affinity)
    total=sum(doc[k] for k in ('warmup','scored','tail'))
    retired=[]
    for stream in profile['streams']:
        if stream['alive_at_stop']:
            continue
        role=roles[stream['tid']]
        if role['role']!='producer':raise ValueError('critical thread exited during scheduler capture')
        feeds=sorted((p for p in phases if p['stage']=='_producer'),key=lambda p:p['call'])
        if (len(feeds)!=total or [p['call'] for p in feeds]!=list(range(total))
            or any(p['operation']!='feed' or p['tid']!=stream['tid'] for p in feeds)
            or any(a['wall_end_ns']>b['wall_start_ns'] for a,b in zip(feeds,feeds[1:]))
            or feeds[-1]['wall_end_ns']>profile['stop_start_ns']):
            raise ValueError('retired producer lacks complete finite feed evidence')
        retired.append(dict(tid=stream['tid'],role='producer',feeds=total,
            final_feed_end_ns=feeds[-1]['wall_end_ns'],alive_at_stop=False,
            unknown_intervals_remain_unknown=True))
    if (profile.get('ring_trace_revision')!=2 or profile.get('drain_policy')!='collector_completion'
        or not isinstance(profile.get('retention_end_ns'),int)
        or profile['retention_end_ns']<profile['stop_end_ns']
        or any(c['end_ns']>profile['retention_end_ns'] for s in profile['streams'] for c in s['drain_chunks'])):
        raise ValueError('repaired recorder extent invalid')
    text=inspect.getsource(campaign.old_events.validate_profile)
    before="or not r['alive_at_stop']"
    if text.count('len(roles)!=12')!=1 or text.count(before)!=1:
        raise ValueError('original lifecycle predicate changed')
    allowed={p['tid'] for p in retired}
    text=text.replace('len(roles)!=12','len(roles)!=13').replace(before,
        "or not (r['alive_at_stop'] or r['tid'] in retired_producers)")
    ns={**campaign.old_events.validate_profile.__globals__,
        'decode_stream':campaign.repaired_events.decode_stream,'retired_producers':allowed}
    exec(compile(text,'<offline completed-producer lifecycle audit>','exec'),ns)
    result=ns['validate_profile'](profile,doc,affinity)
    result.update(physical_drain_evidence_valid=True,expected_roles=13,
        offline_lifecycle_exception=retired,original_campaign_rejection_preserved=True)
    return result


def verified():
    state=read(campaign.RUN/"state.json")
    if (state["boot_id"]!=campaign.BOOT or state["status"]!='STOPPED_ON_ERROR'
        or state.get('error')!='scheduler target affinity/lifetime differs'):
        raise ValueError("unexpected original thread-switch campaign state")
    b=campaign.POOL/"resnet18/capsule";campaign.validate(b);result={};hashes={}
    for s in campaign.specs():
        task=campaign.make_task(b,s);p=campaign.RUN/"accepted"/campaign.BOOT/(task["task_id"]+".json")
        if s['phase']=='observed':
            if p.exists():raise ValueError('original rejected observation was relabeled as accepted')
            directory=campaign.RUN/'raw'/campaign.BOOT/task['task_id']
            candidates=list(directory.glob('transfers/*/collected/profile.json'))
            if len(candidates)!=1:raise ValueError('expected exactly one original native observation; do not repeat it')
            raw=candidates[0].parent
            if (read(directory/'task.json')!=task or (raw/'task_hash').read_text().strip()!=digest(task)
                or (raw/'boot_id').read_text().strip()!=campaign.BOOT or (raw/'exit_code').read_text().strip()!='0'
                or read(raw.parent/'capture.json')!={'incomplete':False,'transport_error':None}):
                raise ValueError('original rejected observation identity/transport invalid')
            evidence=dict(task=task,raw=str(raw),files=inventory(raw),
                archive_sha256=sha256(raw.parent/'result.tar'),capture_sha256=sha256(raw.parent/'capture.json'),
                original_state=state,original_campaign_acceptance=False)
            seal=OUT/'DIAGNOSTIC_RAW_FREEZE.json'
            campaign.immutable_freeze(seal,evidence)
            verify_inventory(raw,read(seal)['files'])
            from types import FunctionType
            f=campaign.phase.summary
            summary=FunctionType(f.__code__,{**f.__globals__,'phase_summary':campaign.phase_summary})(raw,task,b)
            queue=read(raw/'queue_status.json')
            if queue['enabled'] or queue['reads'] or queue['errors']:raise ValueError('unexpected queue-counter reads')
            trace=[json.loads(line) for line in (raw/'wait_trace.jsonl').read_text().splitlines()]
            campaign.disabled_queue_rows(trace,scope='worker')
            phases=[json.loads(line) for line in (raw/'phase_trace.jsonl').read_text().splitlines()]
            summary['scheduler']=diagnostic_events(read(raw/'profile.json'),task,read(raw/'affinity.json'),phases)
            summary['queue_status']=queue
            r=dict(task=task,boot_id=campaign.BOOT,raw=str(raw),files=evidence['files'],summary=summary,
                offline_diagnostic_only=True,original_campaign_acceptance=False)
            result[s['phase']]=r;hashes[str(seal)]=sha256(seal)
            continue
        r=read(p)
        if r["task"]!=task or r["boot_id"]!=campaign.BOOT:raise ValueError("thread-switch identity changed")
        verify_inventory(r["raw"],r["files"])
        if campaign.summary(r["raw"],task,b)!=r["summary"] or (Path(r["raw"])/"task_hash").read_text().strip()!=digest(task):
            raise ValueError("thread-switch raw evidence no longer replays")
        result[s["phase"]]=r;hashes[str(p)]=sha256(p)
    if any(r["summary"]["output_fingerprints"]!=result["qualify"]["summary"]["output_fingerprints"] for r in result.values()):
        raise ValueError("same-program outputs changed")
    return result,hashes


def interval(stream,which,start,end,attribution,core):
    c=clip(stream,start,end)
    if not c["fully_covered"]:raise ValueError("critical path not fully covered")
    actors=defaultdict(int);unknown=overlap=0
    for off in c["off"]:
        a=attribution.other_work(off,core)
        for role,cpu in a["by_role_ns"].items():actors[role]+=cpu
        unknown+=a["unobserved_ns"];overlap+=a["overlap_uncertainty_ns"]
    return dict(piece=which,wall_ns=end-start,on_CPU_ns=c["on_CPU_ns"],
        off_CPU_ns=c["off_CPU_ns"],preempt_off_ns=c["preempt_off_ns"],nonpreempt_off_ns=c["nonpreempt_off_ns"],
        actor_CPU_ns=dict(actors),unobserved_actor_ns=unknown,overlap_uncertainty_ns=overlap)


def analyze():
    records,source=verified();observed=records["observed"];raw=Path(observed["raw"]);task=observed["task"]
    stage=2;program=task["stages"][stage]["program"]
    phases=[json.loads(s) for s in (raw/"phase_trace.jsonl").read_text().splitlines()]
    trace=[json.loads(s) for s in (raw/"wait_trace.jsonl").read_text().splitlines()]
    launches=[r for r in trace if r["kind"]==0];callbacks=defaultdict(list);barriers=defaultdict(list)
    for r in trace:
        if r["kind"]==1:callbacks[r["launch"]].append(r)
        if r["kind"]==2:barriers[r["launch"]].append(r)
    affinity=read(raw/"affinity.json")["stages"][stage]
    tids={s["slot"]:s["tid"] for s in affinity["worker_slots"]}
    streams=observed["summary"]["scheduler"]["streams"]
    attribution=Attribution(streams)
    sequence=[k["function_offset"] for k in read(ROOT/"artifacts/compiled_gap_validation_01/INPUTS.json")["networks"]["resnet18"][campaign.CID]["stages"][stage]["profile"]["kernels"]]
    frames=[];kernels=[];partial=[];piece_rows=[];actor_rows=[];boundaries=[]
    wanted=[p for p in phases if p["stage"]==program and p["operation"]=="run" and 8<=p["call"]<24]
    if sorted(p["call"] for p in wanted)!=list(range(8,24)):raise ValueError("target frames incomplete")
    for phase in sorted(wanted,key=lambda p:p["call"]):
        a,b=phase["wall_start_ns"],phase["wall_end_ns"]
        coverage={str(slot):clip(streams[str(tid)],a,b) for slot,tid in tids.items()}
        if any(not value["fully_covered"] for value in coverage.values()):
            partial.append(dict(frame=phase["call"],unknown_ns_by_slot={s:x["unknown_ns"] for s,x in coverage.items()}));continue
        call=project_call(phase,launches,callbacks,barriers)
        if [k["function_offset"] for k in call["kernels"]]!=sequence:raise ValueError("target kernel pattern differs")
        group=sorted([r for r in launches if r["caller"]==phase["tid"] and a<=r["wall_start_ns"]<=r["wall_end_ns"]<=b],key=lambda r:r["wall_start_ns"])
        if len(group)!=10:raise ValueError("expected ten barrier-free target launches")
        segments=[];previous=a;run_kernels=[]
        for index,launch in enumerate(group):
            workers=sorted(callbacks[launch["launch"]],key=lambda r:r["task"])
            if len(workers)!=3 or any(w["tid"]!=tids[w["task"]] for w in workers):raise ValueError("target worker identity differs")
            last=max(workers,key=lambda r:(r["wall_end_ns"],-r["task"]))
            worker_core=task["stages"][stage]["worker_core_map"][last["task"]]
            caller_core=task["stages"][stage]["worker_core_map"][0]
            pieces=[interval(streams[str(tids[0])],"outside_launch",previous,launch["wall_start_ns"],attribution,caller_core),
                interval(streams[str(last["tid"])],"entry",launch["wall_start_ns"],last["wall_start_ns"],attribution,worker_core),
                interval(streams[str(last["tid"])],"callback",last["wall_start_ns"],last["wall_end_ns"],attribution,worker_core),
                interval(streams[str(tids[0])],"return",last["wall_end_ns"],launch["wall_end_ns"],attribution,caller_core)]
            segments+=pieces;previous=launch["wall_end_ns"]
            k=dict(frame=phase["call"],kernel=index,function_offset=launch["function_offset"],last_slot=last["task"],
                last_worker_CPU_clock_ns=last["cpu_end_ns"]-last["cpu_start_ns"],
                last_worker_callback_wall_ns=last["wall_end_ns"]-last["wall_start_ns"],
                worker_CPU_ns=[w["cpu_end_ns"]-w["cpu_start_ns"] for w in workers],pieces=pieces)
            run_kernels.append(k);kernels.append(k)
            if index+1<len(group):
                next_caller=next(w for w in callbacks[group[index+1]['launch']] if w['task']==0)
                span=interval(streams[str(tids[0])],'between_launches',last['wall_end_ns'],
                    next_caller['wall_start_ns'],attribution,caller_core)
                boundaries.append(dict(frame=phase['call'],after_kernel=index,**span))
        segments.append(interval(streams[str(tids[0])],"outside_launch",previous,b,attribution,task["stages"][stage]["worker_core_map"][0]))
        if sum(p["wall_ns"] for p in segments)!=b-a or any(p["on_CPU_ns"]+p["off_CPU_ns"]!=p["wall_ns"] for p in segments):
            raise ValueError("event critical path does not conserve run wall")
        row=dict(frame=phase["call"],run_wall_ms=(b-a)/1e6,
            caller_CPU_clock_ms=(phase["cpu_end_ns"]-phase["cpu_start_ns"])/1e6,
            caller_perf_on_ms=coverage['0']["on_CPU_ns"]/1e6,
            caller_preempt_off_ms=coverage['0']["preempt_off_ns"]/1e6,
            caller_nonpreempt_off_ms=coverage['0']["nonpreempt_off_ns"]/1e6,
            callback_CPU_by_slot_ms=[sum(k["worker_CPU_ns"][i] for k in run_kernels)/1e6 for i in range(3)])
        for name in ("entry","callback","return","outside_launch"):
            ps=[p for p in segments if p["piece"]==name];actors=defaultdict(int)
            for p in ps:
                for role,cpu in p["actor_CPU_ns"].items():actors[role]+=cpu
            piece_rows.append(dict(frame=phase["call"],piece=name,
                **{key.replace("_ns","_ms"):sum(p[key] for p in ps)/1e6 for key in
                    ("wall_ns","on_CPU_ns","off_CPU_ns","preempt_off_ns","nonpreempt_off_ns","unobserved_actor_ns","overlap_uncertainty_ns")}))
            actor_rows += [dict(frame=phase["call"],piece=name,role=role,coincident_CPU_ms=cpu/1e6) for role,cpu in actors.items()]
        frames.append(row)
    clean=records["clean"]["summary"];obs=observed["summary"]
    original=read(ROOT/"artifacts/resnet_middle_gap_01/analysis/RESULTS.json")
    model=next(r for r in original["models"] if r["method"]=="compiled_output_ring1")
    error=clean["stage_service_ms"][2]-model["stage_service_ms"]
    disturbance=obs["stage_service_ms"][2]-clean["stage_service_ms"][2]
    quality=dict(new_clean_stage_ms=clean["stage_service_ms"][2],new_observed_stage_ms=obs["stage_service_ms"][2],
        new_clean_FPS=clean["fps"],observed_FPS_not_formal=obs["fps"],probe_disturbance_ms=disturbance,
        frozen_model_stage_ms=model["stage_service_ms"],new_clean_model_error_ms=error,
        diagnostic_limit_ms=.2*abs(error),probe_quality_supported=abs(disturbance)<=.2*abs(error),
        old_program_clean_stage_ms=original["quality"]["clean_pipeline_ms"],
        old_program_isolated_stage_ms=original["quality"]["clean_isolated_ms"],
        exact_isolated_bridge_for_new_executor_not_collected=True)
    by_piece=[];by_actor=[]
    if frames:
        for name in ("entry","callback","return","outside_launch"):
            rs=[p for p in piece_rows if p["piece"]==name]
            by_piece.append(dict(piece=name,**{k:st.mean(r[k] for r in rs) for k in rs[0] if k not in ("piece","frame")}))
            roles=sorted({r["role"] for r in actor_rows if r["piece"]==name})
            for role in roles:
                by_actor.append(dict(piece=name,role=role,
                    coincident_CPU_ms_per_complete_frame=sum(r["coincident_CPU_ms"] for r in actor_rows if r["piece"]==name and r["role"]==role)/len(frames)))
    boundary_comparisons=[]
    seal=read(ROOT/'artifacts/compiled_gap_validation_01/SELECTION_FREEZE.json')
    for method in ('original','compiled','compiled_output_ring1'):
        src=seal['prediction_files']['resnet18/'+campaign.CID+'/'+method]
        if sha256(src['path'])!=src['sha256']:raise ValueError('frozen model trace changed')
        model=read(src['path'])['result']
        joins={(j['frame'],j['kernel']):j for j in model['kernel_joins'] if j['stage']==stage}
        callers={(t['frame'],t['kernel']):t for t in model['tasks'] if t['stage']==stage
            and t['operation']=='worker_compute' and t['slot']==0}
        for kernel in range(9):
            selected=[r for r in boundaries if r['after_kernel']==kernel]
            if not selected:continue
            delays=[callers[r['frame'],kernel+1]['first_service_ns']-joins[r['frame'],kernel]['join_ns'] for r in selected]
            if any(t<0 for t in delays):raise ValueError('modeled launch crossed an unfinished callback')
            boundary_comparisons.append(dict(method=method,after_kernel=kernel,frames=len(selected),
                native_wall_ms=st.mean(r['wall_ns'] for r in selected)/1e6,
                native_on_CPU_ms=st.mean(r['on_CPU_ns'] for r in selected)/1e6,
                native_preempt_off_ms=st.mean(r['preempt_off_ns'] for r in selected)/1e6,
                model_wall_ms=st.mean(delays)/1e6,model_trace_sha256=src['sha256'],
                probe_quality_supported=quality['probe_quality_supported']))
    result=dict(quality=quality,complete_frames=[r["frame"] for r in frames],partial_frames=partial,
        frame_rows=frames,pieces=by_piece,competing_actors=by_actor,
        last_worker_counts=dict(Counter(k["last_slot"] for k in kernels)),source_receipts=source,
        source_sha256=sha256(__file__),all_covered_originally_scored_frames_used=True,
        CPU_queue_not_reconstructed_from_nonpreempt_switches=True,no_TASKSTATS_reads=True,
        event_coincidence_does_not_prove_all_wait_causes=True,old_isolated_program_identity_preserved=True,
        no_new_prediction_rule_or_DNN_cost_projection=True,new_native_sessions=3,new_DNN=57,
        original_campaign_status='STOPPED_ON_ERROR',original_accepted_sessions=2,
        offline_diagnostic_exception=observed['summary']['scheduler']['offline_lifecycle_exception'],
        between_launch_comparison=boundary_comparisons,
        return_scope=dict(actual='last callback body end to next caller callback body start',
            modeled='fork completion join to next caller first CPU service',
            next_serial_and_launch_prefix_already_require_caller_CPU=True,
            separate_post_callback_bookkeeping_CPU_not_identified_from_isolation=True,
            neither_all_return_delay_omitted_nor_observed_delay_added_as_penalty=True))
    write(OUT/"RESULTS.json",result,replace=True);write(OUT/"KERNELS.json",kernels,replace=True)
    csv_write(OUT/"frames.csv",frames);csv_write(OUT/"pieces.csv",piece_rows);csv_write(OUT/"actors.csv",actor_rows)
    csv_write(OUT/'between_launches.csv',boundary_comparisons)
    render(result);return result


def render(r):
    q=r["quality"]
    lines=["ResNet CPU5–7：关闭逐次TASKSTATS查询的线程切换诊断", "",
        "复用已验收的修复版PERF_RECORD_SWITCH记录器，构建副本只将固定角色数12改为13；没有修改计算、绑定、等待策略、运行库或阶段模块。新程序的探针关/开共同测量单独冻结身份。",
        "原20ca程序、原始记录和已有13FPS成果保持原身份；本批不是全域Top20重测。", "",
        f"同一新程序关闭探针阶段{q['new_clean_stage_ms']:.3f} ms，开启后{q['new_observed_stage_ms']:.3f} ms，扰动{q['probe_disturbance_ms']:+.3f} ms。",
        f"冻结模型{q['frozen_model_stage_ms']:.3f} ms，相对本批干净阶段的误差{q['new_clean_model_error_ms']:.3f} ms；扰动是否低于该误差20%：{q['probe_quality_supported']}。",
        f"原程序干净阶段{q['old_program_clean_stage_ms']:.3f} ms另列为桥接。隔离CPU5–7的26.808ms来自原20ca程序，但模块/运行库相同；本轮没有新增该新程序的隔离测量，不冒称全部身份匹配。",
        "TASKSTATS读取数为0。输出中的零queue字段只表示该接口关闭，不作为零等待证据。所有角色的ABI、物理缓冲、丢失及实际核心检查仍执行。",
        "原控制器因生产线程在停止记录前退出而拒收观察会话，原STOPPED_ON_ERROR状态和两份已验收凭据不改写。另建离线诊断视图：确认生产线程已依次投递完0–27帧、正常结束；其他12个关键线程均存活，缺失边界仍保留未知。该视图不冒充原协议验收通过。", "",
        f"保留原始计入帧8–23；完整目标线程覆盖的帧：{r['complete_frames']}。其余帧只记覆盖不足，不选取对结果有利的帧。",
        "事件时钟与阶段时钟均为CLOCK_MONOTONIC。下表每次取最后完成worker一条路径，入口、回调、返回、启动外区间不重叠，严格重建整个run墙钟。", "",
        "区间 | 墙钟ms | 记录到的在核ms | 抢占离核ms | 非抢占离核ms | 未记录竞争角色ms"]
    for p in r["pieces"]:
        lines.append(f"{p['piece']} | {p['wall_ms']:.3f} | {p['on_CPU_ms']:.3f} | {p['preempt_off_ms']:.3f} | {p['nonpreempt_off_ms']:.3f} | {p['unobserved_actor_ms']:.3f}")
    lines += ["非抢占离核可能包含sched_yield、睡眠或其他阻塞，没有精确唤醒事件时不能统称CPU排队。CPU时间钟与调度切换区间的计量也不同，逐帧同时保留。", "",
        "在目标线程离核的同一核心上，同时记录到的其他CPU工作："]
    for p in sorted(r["competing_actors"],key=lambda p:(p["piece"],-p["coincident_CPU_ms_per_complete_frame"])):
        lines.append(f"{p['piece']} | {p['role']} | {p['coincident_CPU_ms_per_complete_frame']:.3f} ms/完整帧")
    lines += ["这些是同核同时段占用，不是所有等待的因果分摊。未记录系统线程、IRQ及边界不确定性单列，不填成0，也不相加多个并行worker等待。", "",
        "模型语义核对：最后worker完成后，模型立即关闭汇合，但下一段串行工作及下次launch前缀仍需由原调用线程取得CPU后执行。因此不能说模型完全没有返回后的调度，也不能把观察到的返回拖尾再整段叠加。未单独分离的是回调结束后计数更新、返回等固有CPU工作；隔离尾部还混合完成等待，不能全部当成固有开销。",
        "统一对照边界：最后worker回调结束→下一次调用线程回调开始；模型采用本次汇合→下次调用线程第一次取得CPU。该区间包含返回、串行及下一次启动，允许比较推进节拍，但不把它全部称作返回开销。",
        "内核之后 | 观察推进墙钟ms | 其中在核ms | 抢占离核ms | 冻结模型推进ms"]
    for p in r['between_launch_comparison']:
        if p['method']=='compiled_output_ring1':
            lines.append(f"{p['after_kernel']} | {p['native_wall_ms']:.3f} | {p['native_on_CPU_ms']:.3f} | {p['native_preempt_off_ms']:.3f} | {p['model_wall_ms']:.3f}")
    lines += ["以上比较仍受探针扰动限制，不用于拟合返回惩罚、调度参数或静态profile。", "",
        "本批只做可观测性与误差归因。未生成新的静态公式、DP名单或流水慢化系数。3会话、57次完整推理，一次启动，不自动加测。", ""]
    (OUT/"FINDINGS_zh.txt").write_text("\n".join(lines)+"\n")
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)


if __name__=="__main__":
    r=analyze();print(json.dumps({k:r[k] for k in ("quality","complete_frames","pieces","competing_actors")},ensure_ascii=False,indent=2))
