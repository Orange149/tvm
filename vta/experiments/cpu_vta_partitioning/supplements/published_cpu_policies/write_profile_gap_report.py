"""Chinese evidence update for isolation-versus-pipeline phase error."""
from pathlib import Path
import json
import shutil
import statistics as st
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from cpu_vta_partitioning.common import read, write, sha256

ROOT=Path(__file__).resolve().parent
REPORT=ROOT.parents[4]/"paper/my_paper/analysis/20261111_isolated_profile_gap_update"


def clock_scope_audit():
    """Expose an observation-marker difference without changing any prediction."""
    p=ROOT/"artifacts/compiled_gap_validation_01";seal=read(p/"SELECTION_FREEZE.json")
    result=[]
    for network,configs in seal["selected_configurations"].items():
        for c in configs:
            item=seal["prediction_files"][network+"/"+c["id"]+"/compiled_output_ring1"]
            if sha256(item["path"])!=item["sha256"]:raise ValueError("frozen prediction changed")
            model=read(item["path"])["result"]
            for i,s in enumerate(c["stages"]):
                if s["device"]!="cpu":continue
                sets=[t for t in model["tasks"] if t["stage"]==i and t["operation"]=="set"]
                if len(sets)!=16 or any(t["first_service_ns"]<t["ready_ns"] for t in sets):
                    raise ValueError("CPU phase dispatch markers incomplete")
                result.append(dict(network=network,configuration_id=c["id"],stage=i,program=s["program"],
                    model_ready_to_first_set_CPU_ms=st.mean(t["first_service_ns"]-t["ready_ns"] for t in sets)/1e6,
                    recorded_model_stage_service_ms=model["replay"]["stage_service_ns"][i]/1e6))
    source=ROOT/"artifacts/scheduler_gap_audit_01/build/sources/vta_stage_pipeline_runner_phase.cc"
    body=source.read_text();model_source=ROOT/"model_semantics_audit/simulator.py"
    if 'timing->set_start_ms = NowMillis();' not in body or 'frame=self.queues[i].popleft();self.starts[i,frame]=self.time' not in model_source.read_text():
        raise ValueError("timing marker anchors changed")
    out=ROOT/"artifacts/phase_cycle_audit_01/CLOCK_SCOPE.json"
    write(out,dict(rows=result,source_sha256=sha256(__file__),
        native_start="executing caller in RunStage, immediately before input copy",
        model_start="queue/resource ready, before first input CPU service",
        native_queue_pop_marker="after Pop returned; not when blocked thread first became runnable",
        source_hashes={str(source):sha256(source),str(model_source):sha256(model_source)},
        no_score_or_timestamp_retiming=True,no_new_native_work=True),replace=True)
    return out


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    paths={
        "kernel_parameters":ROOT/"artifacts/compiled_sleeper_01/COMPILED_SLEEPER.json",
        "wake_sleep_development":ROOT/"artifacts/compiled_sleeper_01/RESULTS.json",
        "worker_alignment":ROOT/"artifacts/compiled_gap_alignment_01/RESULTS.json",
        "cross_network":ROOT/"artifacts/compiled_kernel_crosscheck_01/RESULTS.json",
        "independent_output_profile":ROOT/"artifacts/output_control_profile_02/RESULTS.json",
        "native_output_evidence":ROOT/"artifacts/output_control_profile_02/RAW_EVIDENCE.json",
        "output_replay":ROOT/"artifacts/isolated_output_replay_01/RESULTS.json",
    }
    data={k:read(p) for k,p in paths.items()}
    old=data["wake_sleep_development"];cross=data["cross_network"];out=data["output_replay"]
    lines=["隔离profile与流水内阶段耗时：本轮推进与剩余问题", "",
        "当前结论：已确认具体worker的CPU排队、两处回放参数语义不匹配及独立可测的输出处理遗漏；已知案例预测有改善。新增未测布局中有改善也有退化，因此尚不支持统一接入正式静态评分或宣称选点提升。",
        "本轮聚焦阶段误差，不重新整理基线，不重启开发板，不修改原执行器、运行库、阶段模块、正式评分或完整DP。",
        "已有ResNet18全域静态第一名13.001/13.043 FPS成果继续保留；下面ResNet局部布局不是其重测。",
        "独立输出标定部分新增4个CPU专用会话，DNN推理0次，FPGA调用0次；前六节的流水数值均引用已有验收数据。如附第七节，其中另列新布局验证的会话及推理数量。",
        "旧成本与轨迹保持原身份，新模型仅用于误差诊断，未宣布正式方法或Top20改善。", "",
        "一、已经确认的慢化机制", 
        "相同YOLO三线程前段，主机与重负载worker共用核心时，隔离197.534 ms，流水262.117 ms；三个worker回调CPU量只增加约1%。",
        "每次只保留最后完成worker，其回调期间CPU排队约从0.44–0.58 ms增至57.97–58.57 ms/帧。这支持竞争与完成拖尾，不支持凭空增加同等数量的计算工作。",
        "上述诊断探针对该流水前段扰动6.136 ms，支持约65 ms全慢化的机制判断，但不足以逐毫秒归因剩余23 ms。不能把所有线程等待相加。", "",
        "二、回放器的两项近似有独立的内核证据", 
        "从与运行内核notes/build ID匹配的启动Image核验实际编译指令，唤醒抢占阈值应按3 ms处理，休眠补偿裁剪应按9 ms处理。",
        "旧回放以标定服务粒度的一半、以及服务粒度×可运行任务数的一半代替这两者。标定服务粒度可用作近似服务预算，不能因此充当所有CFS参数。",
        "3/9 ms来自编译初值、已编译缩放路径和四核在线数量，不是用流水FPS拟合；没有直接读取运行内核变量，仍保留未记录内核内存变更等限制。",
        "只替换这两个已识别规则，不宣称复现完整Linux CFS，也没有新增网络名称相关公式。", "",
        "已知三个主机位置的开发回放：",
        "VTA主机位置 | 原回放前段ms | 只改唤醒 | 只改休眠补偿 | 两项都改 | 已有关闭探针流水ms"]
    labels=dict(host_on_heavier_worker="与重worker共核1",host_on_lighter_worker="与轻worker共核2",host_outside_front_group="前段核组外核心3")
    for role in labels:
        r={v["variant"]:v for v in old["rows"] if v["role"]==role}
        lines.append(f"{labels[role]} | {r['original']['predicted_front_ms']:.3f} | {r['wake_only']['predicted_front_ms']:.3f} | {r['sleep_only']['predicted_front_ms']:.3f} | {r['both']['predicted_front_ms']:.3f} | {r['both']['native_front_ms']:.3f}")
    lines += ["重worker共核案例误差从22.897 ms降到5.923 ms；三个配置的选择顺序未变，没有选点收益证据。", "",
        "三、总耗时接近，还必须检查等待出现的位置", 
        "修正后，沿原生最后worker槽位统计的模型ready等待从41.962增至56.499 ms/帧；其中首次获得CPU前约4.676 ms、首次获得CPU后约51.823 ms。",
        "它与原生回调CPU排队57.97–58.57 ms的计量起止不同，不能直接相减作为修正系数。",
        "部分主要卷积更接近，但第一个主要卷积仍少估约9 ms，另有内核等待偏多；不能仅以总阶段误差小就声称逐内核调度已正确。",
        "短内核的最后worker容易受微小时差改变；整体最后worker身份一致率80.2%降到64.1%，必须保留这一负向结果。", "",
        "四、独立profile确认了另一项遗漏：流水期间仍在处理输出", 
        "实际write_completed在流水运行期间执行ResultJSON，包含输出FNV哈希、数值统计和字符串生成；defer只延后写文件。",
        "将冻结源码的实际ResultJSON用于独立CPU程序，读取输入0的真实参考输出，不加载网络、不调用FPGA。两种缓冲方式在测前声明，全部报告：",
        "网络 | 输出张量数/字节 | 单缓冲CPU ms | 四缓冲CPU ms"]
    profiles=data["independent_output_profile"]["rows"]
    for network,zh in (("resnet18","ResNet18"),("yolov3_tiny416","YOLOv3-tiny416")):
        rr={r["ring_count"]:r for r in profiles if r["network"]==network}
        lines.append(f"{zh} | {rr[1]['output_count']} / {rr[1]['output_bytes']} | {rr[1]['CPU_ms']:.6f} | {rr[4]['CPU_ms']:.6f}")
    lines += ["这只测得输出序列化和延后字符串入队，不涵盖全部生产、收集、队列、帧分配与释放开销。缓冲复用方式不是原流水缓存状态的严格复现。",
        "独立成本在完整流水实测前可由真实参考输出取得，因此与直接使用流水内9.4 ms控制轨迹不同；本轮没有拿流水FPS或阶段耗时拟合该输入。", "",
        "五、把独立输出成本放回竞争模型，不能直接加到完成间隔", 
        "复用已有有界结果队列和收集线程，收集CPU工作通过核心竞争和后续帧反压发生作用；网络最后阶段完成时刻的定义不变。",
        "以下都是已知布局的开发回放：",
        "布局 | 原回放前段ms | 内核规则修正 | 再计单缓冲输出成本 | 再计四缓冲输出成本 | 已有流水ms"]
    for r in cross["rows"]:
        cid=r["configuration_id"]
        d={v["variant"]:v for v in out["rows"] if v["configuration_id"]==cid}
        label=("ResNet" if r["network"]=="resnet18" else "YOLO")+" CPU线程"+"/".join(map(str,r["CPU_threads"]))
        lines.append(f"{label} | {r['old_front_ms']:.3f} | {r['new_front_ms']:.3f} | {d['ring1']['predicted_front_ms']:.3f} | {d['ring4']['predicted_front_ms']:.3f} | {r['native_front_ms']:.3f}")
    lines += ["YOLO四/三/一线程布局的误差从26.872 ms降至约7.71–7.76 ms；四/四/一布局从20.911 ms降至约8.58 ms。",
        "ResNet两项局部布局并没有同步改善：输出工作较小，内核规则修正使前段误差相对原回放增加约0.8–1.5 ms。不能只展示YOLO的正向结果。",
        "主机固定在重/轻worker核心的三线程布局中，补入输出成本并未改变前段预测；核组外主机布局则由少估5.034 ms变为多估2.424 ms。输出成本不是通用加时系数。", "",
        "六、现在可以和不能下的结论", 
        "可以：同一子图进入流水后变慢，不仅是计算核时；已观察到具体worker的CPU排队与完成拖尾。原回放确有两处可由机器码独立核实的参数语义不匹配。",
        "可以：两个网络的输出规模相差约216倍，独立输出处理CPU时间相差约21倍，之前统一忽略控制工作对YOLO更不利。依据是工作规模，不是按网络名称设规则。",
        "不能：宣布剩余阶段误差已全部解释、所有共享布局都更准、真实FPS提高，或新Top20已优于既有方法。",
        "还缺：未测资源布局的冻结验证，进一步区分剩余启动时序、主机迁移与控制放置近似；完整域接入和成本另行评估。不要再重复基线差异整理，也不要自动展开全域标定。", "",
        "复现入口：", "python run_compiled_sleep_replay.py", "python analyze_compiled_gap.py",
        "python run_output_control_profile.py validate", "python run_output_control_profile.py analyze",
        "python run_isolated_output_replay.py", "",
        "来源：Linux5.4 fair.c 的place_entity与wakeup_preempt_entity；实际板端Image的同名编译路径单独核验。",
        "https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c", "",
        "本地验收包含：核心/逻辑线程身份、单条完成路径记账、隔离输入禁止流水字段、控制零成本回退、CPU工作守恒、队列反压、错误输出/哈希/样本计数拒绝、QEMU输出及原始文件哈希。",
        "初次QEMU校验误将虚拟CPU钟与原生墙钟的关系作为性能验收，准备批次01停止在本地；批次02对QEMU只核验加载、输出与结构，原生时钟检查保持，原批次未上板且未改写。", ""]
    if (ROOT/"artifacts/compiled_gap_validation_01/RESULTS.json").exists():
        p=ROOT/"artifacts/compiled_gap_validation_01/RESULTS.json";v=read(p);paths["prospective_validation"]=p
        lines += ["七、四个此前未测布局的短窗口验证（已先冻结预测）", 
            "每网络选仍有未测布局的最大两种前段线程数分层，原隔离评分各取最低；不依据流水FPS选点。YOLO四线程布局均已测，因此使用二/三线程；ResNet使用三/四线程。同一规则，不重复旧测量。一次单帧资格＋一次8/16/4性能窗口，共8会话、116次完整推理。",
            "所有配置均关闭门控、共享主机与控制线程，保持原执行器身份。它是阶段误差验证，不是全域Top20。",
            "网络/前段线程数 | 原回放ms | 内核规则修正ms | 加独立输出成本ms | 流水ms | 原误差ms → 新误差ms"]
        ids=list(dict.fromkeys((r["network"],r["configuration_id"]) for r in v["rows"]))
        for network,cid in ids:
            a={r["method"]:r for r in v["rows"] if r["network"]==network and r["configuration_id"]==cid}
            b=a["original"];n=a["compiled_output_ring1"]
            lines.append(f"{network}/{b['CPU_threads'][0]} | {b['predicted_front_ms']:.3f} | {a['compiled']['predicted_front_ms']:.3f} | {n['predicted_front_ms']:.3f} | {b['native_front_ms']:.3f} | {b['front_error_ms']:+.3f} → {n['front_error_ms']:+.3f}")
        lines += ["必须逐项看改善和退化，不能把局部阶段误差改善写成完整DP或真实吞吐改善。原始8会话凭据及逐阶段结果见prospective_validation_results.json。", ""]
        lines += ["新增布局的明确判定：ResNet前段三线程误差8.774→7.548 ms，四线程0.970→2.578 ms；YOLO二线程保持0.510 ms，三线程仅修正内核规则时1.019→0.399 ms，但再加独立输出成本后误差增至6.073 ms。",
            "所以不能因已知两个YOLO共享布局误差下降，就认定输出竞争回放已能泛化。独立输出CPU工作是真实存在的；它在模型中何时获得CPU、如何改变主机和计算相位，仍有近似。不要删掉这项真实工作来迎合结果，也不要挑更接近实测的核心位置回填。",
            "ResNet新三/三/一/一布局的CPU5–7，修正预测39.018 ms，流水50.364 ms，仍差11.346 ms。ResNet输出处理只有约0.434 ms，不能用统一输出加时解释该阶段误差。", ""]
        shutil.copy2(p,REPORT/"prospective_validation_results.json")
    cp=ROOT/"artifacts/phase_cycle_audit_01/RESULTS.json"
    if cp.exists():
        cycles=read(cp);paths["phase_cycles"]=cp
        lines += ["八、阶段内部慢化与有限窗口输出节拍必须分开", 
            "对同一帧定义阶段开始S、阶段结束E、网络完成C，严格核验：",
            "平均网络完成间隔＝同阶段平均服务＋相邻帧间空档＋(最后帧C−S减首帧C−S)/15。",
            "所有项均由同一原始时间戳、整数纳秒重建，没有模型拟合；最后一项是下游延迟在窗口内的变化，不是额外CPU工作。",
            "配置 | 同窗口15周期阶段服务ms | 帧间空档ms | 下游延迟变化/15 ms | 网络完成间隔ms"]
        for r in cycles["rows"]:
            if r["stage"]==0:
                label=("ResNet" if r["network"]=="resnet18" else "YOLO")+f"前段{r['threads']}线程"
                lines.append(f"{label} | {r['stage_service_ms_per_cycle']:.3f} | {r['between_frame_gap_ms']:.3f} | {r['downstream_latency_change_per_gap_ms']:+.3f} | {r['completion_interval_ms']:.3f}")
        lines += ["关键校正：YOLO新三线程布局并没有每帧约11 ms的前段空闲；空档实际仅0.030 ms。约11.075 ms来自C−S在保留窗口中增长166.123 ms后除以15。不能将它归给前段计算、收集CPU或调度等待。",
            "该窗口首末各阶段完成数量也已记录，见phase_cycles.json。短窗口的输出间隔与前段阶段均值不一定相等；本轮不自动延长、拒收或重测，也不据此宣布稳态吞吐上限。",
            "后续优先处理两个可区分的问题：一是有明确阶段误差的CPU线程组（例如ResNet CPU5–7）；二是服务预测接近、输出节拍仍有差距时的下游时序和窗口边界。不能继续用同一个慢化系数概括两者。", ""]
        shutil.copy2(cp,REPORT/"phase_cycles.json")
        paths["clock_scope"]=clock_scope_audit()
        lines += ["九、进一步核对了阶段计时起点，避免拿不同标记直接对比", 
            "原生set_start在调用线程已经执行到RunStage、即将复制输入时记录；回放stage_start在队列/资源已准备好、输入CPU任务尚未获得服务时记录。后者可能额外包含首次调度前等待。",
            "本批四个新布局前段的这项模型等待分别为0、0.150、0、0 ms，不能解释剩余主要前段误差。部分尾段则有约1.647或2.362 ms，需要与执行中的等待分开列出。",
            "原生queue_pop时间在Pop返回后才记录，不等于线程最初变为可运行的时刻，因此也不能据此恢复完整唤醒等待。clock_scope.json保留标记和逐项数据；没有为拉近误差而重写任何模型时间戳。", ""]
    (REPORT/"FINDINGS_zh.txt").write_text("\n".join(lines)+"\n")
    for key,path in paths.items():
        if key!="prospective_validation":shutil.copy2(path,REPORT/(key+".json"))
    write(REPORT/"EVIDENCE_INDEX.json",dict(source_files={str(p):sha256(p) for p in paths.values()},
        report_source_sha256=sha256(__file__), existing_13_FPS_result_retained=True,
        no_full_search_or_Top20_gain_claim=True),replace=True)
    return dict(report=str(REPORT/"FINDINGS_zh.txt"),evidence=len(paths))


if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
