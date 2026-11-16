"""One critical completion path per CPU5-7 launch; never sum worker waits."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import shutil
import statistics as st

import run_resnet_middle_gap as campaign
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from phase_model.extract import project_call,average_calls
from phase_model.simulator import rational
from scheduler_gap_audit.analysis import call_details,queue_bounds
from scheduler_gap_audit.curves import csv_write
from analyze_three_thread_worker_gap import critical_path

ROOT=campaign.ROOT
OUT=campaign.POOL/"analysis"
REPORT=ROOT.parents[4]/"paper/my_paper/analysis/20261111_resnet_middle_gap"


def target_calls(receipt,program):
    """Keep physical module/TID identities and exact same-clock phase bounds."""
    raw=Path(receipt["raw"]);task=receipt["task"]
    phases=[json.loads(s) for s in (raw/"phase_trace.jsonl").read_text().splitlines()]
    events=[json.loads(s) for s in (raw/"wait_trace.jsonl").read_text().splitlines()]
    launches=[r for r in events if r["kind"]==0];callbacks=defaultdict(list);barriers=defaultdict(list)
    for r in events:
        if r["kind"]==1:callbacks[r["launch"]].append(r)
        if r["kind"]==2:barriers[r["launch"]].append(r)
    selected=[p for p in phases if p["stage"]==program and p["operation"]=="run"
        and task["warmup"]<=p["call"]<task["warmup"]+task["scored"]]
    if sorted(p["call"] for p in selected)!=list(range(task["warmup"],task["warmup"]+task["scored"])):
        raise ValueError("target phase frames incomplete")
    index=next(i for i,s in enumerate(task["stages"]) if s["program"]==program)
    affinity=read(raw/"affinity.json")["stages"][index]
    slots={s["slot"]:s["tid"] for s in affinity["worker_slots"]}
    if len(slots)!=3 or slots[0]!=affinity["caller_tid"]:
        raise ValueError("logical caller/worker ownership differs")
    projected=[];kernels=[];frames=[]
    for phase in sorted(selected,key=lambda p:p["call"]):
        call=project_call(phase,launches,callbacks,barriers);projected.append(call)
        group=sorted([r for r in launches if r["caller"]==phase["tid"]
            and phase["wall_start_ns"]<=r["wall_start_ns"]<=r["wall_end_ns"]<=phase["wall_end_ns"]],key=lambda r:r["wall_start_ns"])
        pieces=[]
        for i,launch in enumerate(group):
            workers=sorted(callbacks[launch["launch"]],key=lambda r:r["task"])
            if (launch["tasks"]!=3 or len(workers)!=3 or
                any(r["tid"]!=slots[r["task"]] for r in workers) or
                not launch["module"].endswith("/programs/"+program+"/module.so")):
                raise ValueError("module or worker group identity differs")
            vectors=dict(wall_start_ns=launch["wall_start_ns"],wall_end_ns=launch["wall_end_ns"],
                worker_wall_start_ns=[r["wall_start_ns"] for r in workers],
                worker_wall_end_ns=[r["wall_end_ns"] for r in workers],
                worker_cpu_ns=[r["cpu_end_ns"]-r["cpu_start_ns"] for r in workers])
            path=critical_path(vectors);last=workers[path["last_worker"]];lo,hi=queue_bounds(last)
            query_end=last["queue_read_end1_ns"]
            if not last["wall_end_ns"]<=query_end<=launch["wall_end_ns"]:
                raise ValueError("post-callback counter read escaped the launch")
            prefix=launch["worker_queue_baselines"][last["task"]]
            if prefix["tid"]!=last["tid"] or prefix["queue_ns"]>last["queue_before_ns"]:
                raise ValueError("last worker queue baseline missing or inconsistent")
            row=dict(frame=phase["call"],kernel=i,function_offset=launch["function_offset"],
                worker_CPU_ns=vectors["worker_cpu_ns"],launch_wall_ns=launch["wall_end_ns"]-launch["wall_start_ns"],
                queue_lower_ns=lo,queue_upper_ns=hi,pre_entry_queue_counter_ns=last["queue_before_ns"]-prefix["queue_ns"],
                caller_tail_CPU_ns=launch["cpu_end_ns"]-launch["caller_tail_cpu_start_ns"],
                caller_tail_wall_ns=launch["wall_end_ns"]-launch["caller_tail_wall_start_ns"],
                last_callback_post_read_envelope_ns=query_end-last["wall_end_ns"],
                remaining_return_after_last_query_ns=launch["wall_end_ns"]-query_end,**path)
            kernels.append(row);pieces.append(row)
        wall=phase["wall_end_ns"]-phase["wall_start_ns"]
        outside=wall-sum(p["launch_wall_ns"] for p in pieces)
        values={key:sum(p[key] for p in pieces) for key in
            ("entry_delay_ns","callback_CPU_ns","callback_wall_minus_CPU_ns","return_tail_ns")}
        if outside<0 or outside+sum(values.values())!=wall:
            raise ValueError("single critical path does not reconstruct run wall")
        frames.append(dict(frame=phase["call"],run_wall_ns=wall,outside_launch_wall_ns=outside,**values,
            **{k:sum(p[k] for p in pieces) for k in ("queue_lower_ns","queue_upper_ns","pre_entry_queue_counter_ns","caller_tail_CPU_ns","caller_tail_wall_ns",
                "last_callback_post_read_envelope_ns","remaining_return_after_last_query_ns")},
            callback_CPU_ns_by_slot=[sum(p["worker_CPU_ns"][s] for p in pieces) for s in range(3)]))
    avg=average_calls(projected)
    return dict(average=avg,kernels=kernels,frames=frames,
        sequence=[k["function_offset"] for k in avg["kernels"]],
        last_workers=dict(Counter(k["last_worker"] for k in kernels)),
        measured_kernel_count=len(kernels),source_module_identity=program)


def analyze():
    receipts=campaign.verified_receipts();selection=campaign.selection();bundle=campaign.POOL/"resnet18/capsule"
    cfg=selection["selected_configurations"]["resnet18"][0];program=cfg["stages"][2]["program"]
    intr=read(ROOT/"artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json")
    old=next(v for v in intr["CPU"].values() if v["identity"]["network"]=="resnet18"
        and v["identity"]["program"]==program and v["identity"]["threads"]==3)
    if sha256(old["provenance"]["receipt"])!=old["provenance"]["sha256"]:
        raise ValueError("old isolated trace provenance changed")
    expected=[k["function_offset"] for k in old["average"]["kernels"]]
    module=read(bundle/"plans/programs.json")[program]["files"]["module.so"]["sha256"]
    if module!=old["identity"]["module_sha256"]:raise ValueError("old/current stage modules differ")
    observed={};clean_iso=None;source_hashes={}
    for p,r in receipts:
        source_hashes[str(p)]=sha256(p);s=r["task"]["spec"]
        if s["scope"]=="component" and s["phase"]=="clean":clean_iso=r
        if s["phase"]!="observed":continue
        x=target_calls(r,program)
        if x["sequence"]!=expected:raise ValueError("target kernel sequence changed")
        observed[s["scope"]]=dict(receipt=r,data=x)
    refs={}
    for kind,item in selection["references"].items():
        p=Path(item["receipt"])
        if sha256(p)!=item["sha256"]:raise ValueError("clean reference changed")
        refs[kind]=read(p);source_hashes[str(p)]=sha256(p)
    iso=observed["component"];pipe=observed["pipeline"]
    clean_service=refs["clean"]["summary"]["stage_service_ms"][2]
    isolated_service=clean_iso["summary"]["cost"]["service_ms"]
    observed_iso=iso["receipt"]["summary"]["cost"]["service_ms"]
    observed_service=pipe["receipt"]["summary"]["stage_service_ms"][2]
    disturbance=max(abs(observed_service-clean_service),abs(observed_iso-isolated_service))
    whole_gap=clean_service-isolated_service;remaining=read(campaign.POOL/"PLAN.json")["remaining_gap_ms"]
    quality=dict(clean_isolated_ms=isolated_service,observed_isolated_ms=observed_iso,
        clean_pipeline_ms=clean_service,observed_pipeline_ms=observed_service,
        isolated_to_pipeline_ms=whole_gap,remaining_model_error_ms=remaining,
        maximum_probe_disturbance_ms=disturbance,
        broad_gap_attribution_allowed=disturbance<=.2*abs(whole_gap),
        precise_remaining_error_attribution_allowed=disturbance<=.2*abs(remaining),
        performance_not_rejected_and_no_automatic_repeat=True)
    rows=[];kernel_rows=[]
    for scope,item in observed.items():
        x=item["data"];a=x["average"];frames=x["frames"]
        details,_,_=call_details(item["receipt"]["raw"],item["receipt"]["task"])
        caller_details=[v for v in details if v["stage"]==program]
        if len(caller_details)!=len(frames):raise ValueError("caller run counter frames differ")
        row=dict(scope=scope,frames=len(frames),last_workers=x["last_workers"],
            kernel_launches_per_frame=len(x["sequence"]),
            callback_CPU_by_slot_ms=[st.mean(f["callback_CPU_ns_by_slot"][i] for f in frames)/1e6 for i in range(3)],
            all_callback_CPU_ms=sum(st.mean(f["callback_CPU_ns_by_slot"][i] for f in frames)/1e6 for i in range(3)),
            serial_caller_CPU_ms=float(sum(rational(v) for v in a["serial_cpu_segments"]))/1e6,
            serial_and_prefix_contain_counter_read_work_not_intrinsic_only=True)
        for key in ("caller_run_CPU_ms","caller_run_queue_lower_ms","caller_run_queue_upper_ms","caller_completion_queue_counter_ms"):
            row[key]=st.mean(v[key] for v in caller_details)
        for name in ("run_wall_ns","outside_launch_wall_ns","entry_delay_ns","callback_CPU_ns","callback_wall_minus_CPU_ns","return_tail_ns",
                     "queue_lower_ns","queue_upper_ns","pre_entry_queue_counter_ns","caller_tail_CPU_ns","caller_tail_wall_ns",
                     "last_callback_post_read_envelope_ns","remaining_return_after_last_query_ns"):
            row[name.replace("_ns","_ms")]=st.mean(f[name] for f in frames)/1e6
        rows.append(row)
        for i in range(len(expected)):
            ks=[k for k in x["kernels"] if k["kernel"]==i]
            kernel_rows.append(dict(scope=scope,kernel=i,function_offset=expected[i],
                worker_CPU_ms=[st.mean(k["worker_CPU_ns"][s] for k in ks)/1e6 for s in range(3)],
                last_workers=dict(Counter(k["last_worker"] for k in ks)),
                **{f.replace("_ns","_ms"):st.mean(k[f] for k in ks)/1e6 for f in
                   ("launch_wall_ns","entry_delay_ns","callback_wall_minus_CPU_ns","queue_lower_ns","queue_upper_ns","pre_entry_queue_counter_ns")}))
    seal=read(campaign.clean.POOL/"SELECTION_FREEZE.json");model_rows=[]
    for method in ("original","compiled","compiled_output_ring1"):
        p=seal["prediction_files"]["resnet18/"+cfg["id"]+"/"+method]
        if sha256(p["path"])!=p["sha256"]:raise ValueError("frozen model changed")
        m=read(p["path"])["result"]
        tasks={(t["frame"],t["kernel"],t["slot"]):t for t in m["tasks"] if t["stage"]==2 and t["operation"]=="worker_compute"}
        joins={(j["frame"],j["kernel"]):j for j in m["kernel_joins"] if j["stage"]==2}
        ks=pipe["data"]["kernels"];nframes=len(pipe["data"]["frames"])
        ready=sum(float(rational(tasks[k["frame"],k["kernel"],k["last_worker"]]["ready_not_running"])) for k in ks)/nframes/1e6
        agreement=st.mean(k["last_worker"] in joins[k["frame"],k["kernel"]]["last_slots"] for k in ks)
        model_rows.append(dict(method=method,source=p["path"],source_sha256=p["sha256"],
            stage_service_ms=m["replay"]["stage_service_ns"][2]/1e6,
            same_native_chosen_slot_ready_not_running_ms=ready,
            last_worker_identity_agreement=agreement,
            comparison_markers_not_identical_and_not_a_penalty=True))
        caller=[t for t in m["actor_totals"] if t["stage"]==2 and t["slot"]==0
            and t["operation"] not in ("set","get")]
        model_rows[-1].update(
            modeled_run_caller_CPU_ms=sum(float(rational(t["CPU"])) for t in caller)/nframes/1e6,
            modeled_run_caller_ready_ms=sum(float(rational(t["ready_not_running"])) for t in caller)/nframes/1e6,
            modeled_completion_wait_CPU_ms=sum(float(rational(t["CPU"])) for t in caller if t["operation"]=="completion_wait")/nframes/1e6,
            modeled_completion_wait_ready_ms=sum(float(rational(t["ready_not_running"])) for t in caller if t["operation"]=="completion_wait")/nframes/1e6)
    result=dict(quality=quality,rows=rows,kernels=kernel_rows,models=model_rows,
        source_receipts=source_hashes,module_sha256=module,source_sha256=sha256(__file__),
        original_isolated_source=old["provenance"],original_isolated_mask=old["identity"]["mask"],
        current_isolated_mask=[1,2,3],old_isolated_worker_CPU_ms=[float(sum(rational(w["compute_segments"][0])
            for k in old["average"]["kernels"] for w in k["workers"] if w["logical_worker"]==j))/1e6 for j in range(3)],
        old_isolated_observed_service_ms=old["observed_cost"]["service_ms"],
        new_DNN=28,new_CPU_stage_calls=21,new_native_sessions=4,
        no_new_model_no_FPS_fit_no_full_domain_search=True,
        one_critical_path_per_launch_not_sum_of_thread_waits=True,
        current_observed_serial_work_not_projected_as_static_cost=True)
    write(OUT/"RESULTS.json",result,replace=True);csv_write(OUT/"phase_paths.csv",rows)
    csv_write(OUT/"kernel_comparison.csv",kernel_rows);csv_write(OUT/"model_comparison.csv",model_rows)
    for scope,item in observed.items():
        write(OUT/(scope+"_CALLS.json"),item["data"],replace=True)
    render(result)
    return result


def render(r):
    a,b=r["rows"];q=r["quality"]
    growth=100*(b["all_callback_CPU_ms"]/a["all_callback_CPU_ms"]-1)
    lines=["ResNet CPU5–7剩余误差：匹配核组隔离与逐线程观察", "",
        "本轮只诊断已知局部布局，不改评分/DP，不开展新Top20，不重启。历史13FPS全域第一名不在本轮重测。",
        "完整切图：CPU0–2→VTA3–4→CPU5–7→VTA8→CPU9→VTA10–19→CPU20。CPU线程/核组3(012)/3(123)/1(0)/1(3)，主机与控制共享四核，门控关闭。",
        "复用已验收的同boot、同程序干净流水及完整输出资格；新增4会话、28次完整网络推理、21次CPU阶段调用。新组件实际caller核心1，匹配CPU5–7核组123；未继续依赖012→123的同构核近似。", "",
        "一、匹配隔离与探针质量", 
        f"旧隔离轨迹：{r['old_isolated_observed_service_ms']:.3f} ms，来自旧boot、旧观察器、核组012；相同模块/运行库身份保留，不能冒称本批无探针成本。",
        f"本批匹配核组123：无探针隔离{q['clean_isolated_ms']:.3f}，有探针隔离{q['observed_isolated_ms']:.3f} ms。",
        f"已有无探针流水{q['clean_pipeline_ms']:.3f}，本批有探针流水{q['observed_pipeline_ms']:.3f} ms；最大探针扰动{q['maximum_probe_disturbance_ms']:.3f} ms。",
        f"扰动不超过全部隔离→流水差{q['isolated_to_pipeline_ms']:.3f} ms的20%：{q['broad_gap_attribution_allowed']}；不超过剩余模型误差{q['remaining_model_error_ms']:.3f} ms的20%：{q['precise_remaining_error_attribution_allowed']}。",
        "所以不得用本批精确分摊11.35ms，不将观察模式更快当作执行优化；不拒收正式旧结果，也不自动重复测量。", "",
        "二、计算CPU量与非运行区间分开看", 
        f"三个worker回调CPU合计：隔离{a['all_callback_CPU_ms']:.3f}→流水{b['all_callback_CPU_ms']:.3f}核毫秒，变化{growth:+.2f}%。",
        f"沿每次最后完成worker路径，回调内CPU排队从隔离{a['queue_lower_ms']:.3f}–{a['queue_upper_ms']:.3f}增至流水{b['queue_lower_ms']:.3f}–{b['queue_upper_ms']:.3f} ms/帧。",
        "CPU时间包括实际在CPU上消耗的工作，不等于纯算术；队列计数来自同TID TASKSTATS，并保留读取包络。回调前的计数窗口更宽，不能再与回调内量相加。", "",
        "三、每次只取一条完成路径，严格重建run墙钟", 
        "范围 | 启动外 | 关键worker入口 | 关键worker CPU | 回调墙钟减CPU | 最后worker后的返回 | run总墙钟"]
    for x in r["rows"]:
        lines.append(" | ".join([x["scope"]]+[f"{x[k]:.3f}" for k in
            ("outside_launch_wall_ms","entry_delay_ms","callback_CPU_ms","callback_wall_minus_CPU_ms","return_tail_ms","run_wall_ms")]))
    lines += ["caller完成尾部与上面路径重叠，单列在phase_paths.csv，不能再次相加。观察到的serial/prefix CPU包含探针读数开销，不当作正常运行固有成本回填。", "",
        f"进一步核查返回尾部：隔离{a['return_tail_ms']:.3f} ms中，最后回调后的计数读取包络为{a['last_callback_post_read_envelope_ms']:.3f} ms；流水{b['return_tail_ms']:.3f} ms中该包络为{b['last_callback_post_read_envelope_ms']:.3f} ms。",
        "该包络在计算回调结束后、原始并行启动返回前发生，属于探针路径及期间可能的调度延迟，不能全部当作正常执行固有的完成等待。它与返回尾部重叠，不另外相加。", "",
        "四、与冻结回放的对应", 
        "版本 | 阶段服务预测ms | 同原生最后worker槽位的模型ready等待ms | 最后worker身份一致率"]
    for m in r["models"]:
        lines.append(f"{m['method']} | {m['stage_service_ms']:.3f} | {m['same_native_chosen_slot_ready_not_running_ms']:.3f} | {100*m['last_worker_identity_agreement']:.1f}%")
    lines += ["模型ready等待与原生回调CPU排队起止不同，不能以二者差额构造惩罚；逐内核见kernel_comparison.csv。", "",
        "再按同一调用线程的整个run区间核对，避免只看最后worker：",
        "版本 | caller执行CPU ms | caller模型ready等待或原生CPU排队ms | caller完成等待CPU ms"]
    for m in r["models"]:
        lines.append(f"{m['method']} | {m['modeled_run_caller_CPU_ms']:.3f} | {m['modeled_run_caller_ready_ms']:.3f} | {m['modeled_completion_wait_CPU_ms']:.3f}")
    lines.append(f"原生有探针隔离 | {a['caller_run_CPU_ms']:.3f} | {a['caller_run_queue_lower_ms']:.3f}–{a['caller_run_queue_upper_ms']:.3f} | {a['caller_tail_CPU_ms']:.3f}")
    lines.append(f"原生有探针流水 | {b['caller_run_CPU_ms']:.3f} | {b['caller_run_queue_lower_ms']:.3f}–{b['caller_run_queue_upper_ms']:.3f} | {b['caller_tail_CPU_ms']:.3f}")
    lines += ["调用线程执行CPU变化小，而run内排队大幅增加；当前模型已经包含完成等待CPU及部分ready等待，不能把实测等待再整体加一次。原生串行/前缀含探针CPU，模型和原生的细分标记也有差异，不能将本表差额直接解释成无探针11.35ms的精确归因。", "",
        "五、适用结论与剩余问题", 
        "本轮以CPU工作、关键线程排队和完成路径判断竞争机制，精确结论必须服从上述探针质量限制。没有建立新的通用修正，没有让流水耗时进入静态成本。",
        "若工作量变化较小而关键排队增加，只能支持该布局慢化具有CPU竞争/完成拖尾成分；不能据此排除访存、页故障或所有运行时开销，也不能宣称解释全部剩余误差。",
        "后续优先核对回放中的关键worker获得CPU与完成顺序，只有输入可提前取得、且经过未测验证的修正才考虑接回完整域。避免因本次诊断再增加无依据的CFS分支或全域标定。", ""]
    (OUT/"FINDINGS_zh.txt").write_text("\n".join(lines)+"\n")
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)


if __name__=="__main__":
    r=analyze();print(json.dumps({k:r[k] for k in ("quality","rows","models")},ensure_ascii=False,indent=2))
