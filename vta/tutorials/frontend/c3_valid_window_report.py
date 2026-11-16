#!/usr/bin/env python3
"""Render the completed fixed-pool evidence; never access or execute the board."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def render(root):
    analysis = read(root / "analysis.json")
    if analysis["status"] != "COMPLETE_SINGLE_BOOT_FIXED_POOL_RESULT" or list(root.rglob("invalid_attempt.json")):
        raise ValueError("A complete valid acquisition is required")
    for path, digest in analysis["inputs"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Analysis input changed: " + path)
    prepared = read(root / "board_prepared.json")["programs"]
    diagnostics = rows(root / "diagnostic/calls.jsonl")
    formal = rows(root / "board/calls.jsonl")
    signatures = {}
    request_fields = ("load_buffer_2d_inp_bytes", "load_buffer_2d_wgt_bytes", "load_buffer_2d_acc_bytes", "store_buffer_2d_out_bytes")
    names = ["Q1", "Q2", "Q4", "Q8", "TPS"]
    maxima = {}
    for name in names:
        calls = [row for row in diagnostics if row["program"] == name and row["call_kind"] == "warmed_diagnostic"]
        variants = {json.dumps(row["command_signature"], sort_keys=True) for row in calls}
        if len(calls) != 9 or len(variants) != 1:
            raise ValueError("Actual diagnostic commands are inconsistent: " + name)
        signatures[name] = calls[0]["command_signature"]
        for row in calls + [row for row in formal if row["program"] == name]:
            if any(row["runtime_profile"][key] != prepared[name]["request_totals"][key] for key in request_fields):
                raise ValueError("Actual tensor requests changed: " + name)
            profile = row["runtime_profile"]
            sig = signatures[name]
            if profile["push_gemm_op_calls"] != sig["mac"] + sig["reset"] or profile["driver_run_insns"] != sig["instructions"] or profile["driver_run_calls"] != sig["submissions"]:
                raise ValueError("Actual formal/diagnostic command profile changed: " + name)
        maxima[name] = read(root / ("tps_static_audit.json" if name == "TPS" else f"local_final02/{name}/full_local_certificate.json"))["max_kernel_uops"]
    first = rows(root / "qualification/first_cache_costs.jsonl")
    if len(first) != 15 or any(not row["excluded_from_formal_samples"] or not row["not_isolated_initializer_cost"] for row in first):
        raise ValueError("First-call costs must remain separate")
    primary = analysis["primary_Q8_versus_Q1"]
    candidate = analysis["Q_pool_minimum_T_program"]
    practical = analysis["comparisons"][candidate]["versus_TPS"]
    main = []
    requests = []
    for name in names:
        sig = signatures[name]
        low, high = analysis["round_range_ms"][name]
        segment = "封存单份" if name == "TPS" else name[1:]
        main.append(f"| {name} | {segment} | {analysis['T_ms'][name]:.6f} | {low:.6f}–{high:.6f} | {sig['mac']} / {sig['reset']} | {maxima[name]} | {sig['uop_loads']} / {sig['uop_payload_bytes']:,} | {sig['instructions']} | {sig['submissions']} |")
        totals = prepared[name]["request_totals"]
        requests.append("| " + name + " | " + " | ".join(f"{totals[key] / 1024:g}" for key in request_fields) + f" | {sig['wgt_loads']} |")
    paired = []
    for r in range(6):
        ms = [analysis["round_medians_ms"][name][r] for name in names]
        paired.append(f"| {r + 1} | " + " | ".join(f"{value:.6f}" for value in ms) + f" | {primary['paired_improvements_pct'][r]:.4f}% | {primary['paired_ms_saved'][r]:.6f} |")
    first_table = []
    for name in names:
        selected = [row for row in first if row["program"] == name]
        cold = [row["complete_call_ms"] for row in selected if row["first_cache_build"]]
        if len(selected) != 3 or len(cold) != 1:
            raise ValueError("Unexpected qualification cost positions")
        first_table.append(f"| {name} | {cold[0]:.6f} | " + " / ".join(f"{row['complete_call_ms']:.6f}" for row in selected[1:]) + " |")
    trend = "；".join(f"{name}相对Q1的G={analysis['comparisons'][name]['versus_Q1']['G_pct']:.4f}%，胜{analysis['comparisons'][name]['versus_Q1']['wins']}/6" for name in ["Q2", "Q4"])
    support = {
        "schema": "c3_valid_window_report_support_v1", "actual_warmed_command_signatures": signatures,
        "max_kernel_uops": maxima, "tensor_request_units": "requested payload bytes; not measured bus transactions",
        "first_cache_costs_excluded_from_formal": True, "no_board_access_or_target_calls": True,
        "input_sha256": {str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest() for name in [
            "analysis.json", "board_prepared.json", "diagnostic/calls.jsonl", "board/calls.jsonl", "qualification/first_cache_costs.jsonl",
            "tps_static_audit.json", "diagnostic/diagnostic_signatures.json",
            *[f"local_final02/{name}/full_local_certificate.json" for name in names if name != "TPS"]]},
        "render_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (root / "report_support.json").open("x") as stream:
        json.dump(support, stream, indent=2, sort_keys=True)
        stream.write("\n")
    report = f"""# C3有效窗口聚合：固定五程序单启动结果

本轮完成既定本地门禁、完整融合FSim、FPGA资格和六轮采样，范围仍为节点24、固定小Tile和13组常驻加单流式槽。没有新增候选、Tile、驻留比例、预取、selector、模型或硬件改造。

**能不能合？** 当前父程序中的八段MAC可以合。判断依据是实际地址上的正确数据版本、ACC初始化/归约/后处理阶段、逐输出乘积项及实际FIFO/token先后；不是八段形状或变量名字。已有LiftAttrScope已经能重组作用域。本方法的增量限定为明确支持域中的合法性检查、保留共同硬件循环的紧凑连接、保守分段及实际运行时表示认证。八段碎片部分来自本项目Node24WeightSlots显式展开。

**能合多少？** q为1/2/4/8上限时，实际每段分别连接1/2/4/8个原kernel，最大基址μop清单28/56/112/224条；MAC为896/448/224/112，RESET仍为112。真实容量是8192条μop，但容量不是唯一边界。未知关系拒绝，换数据、必要内部通知、RESET、后处理、STORE和不兼容表示均结束窗口。离线六段正例在q8下只合六段。封存TPS的七处相邻MAC边界均承担通知、版本或ACC阶段关系，本轮不可直接连接，也没有生成TPS新候选。

**合了值不值？** 本启动Q8对Q1的六轮配对改善中位数G为**{primary['G_pct']:.4f}%**，胜**{primary['wins']}/6**、平局{primary['ties']}，配对节省中位数**{primary['median_paired_ms_saved']:.6f} ms**。Q池T最小程序为**{candidate}**，相对本轮TPS的G为**{practical['G_pct']:.4f}%**，胜**{practical['wins']}/6**、平局{practical['ties']}。研究复验工程门槛{'通过' if analysis['research_replication_engineering_gate'] else '未通过'}，实用/整图投入工程门槛{'通过' if analysis['practical_graph_engineering_gate'] else '未通过'}。只描述当前完整算子调用，不换算整网FPS，不宣称统计显著性。

## 1 实际程序、时延和请求主表

T是六个轮次中位数的中位数；每轮中位数来自三个冻结seed。范围是六个轮次中位数的最小/最大，不是置信区间。命令列来自本启动独立诊断的预热后位置4/5/6，并逐正式调用匹配；三种seed轮转一致。μop最大清单来自冻结最终表示和实际内容认证。两张主表按相同程序对应。

| 程序 | 实际每段份数 | T / ms | 轮次范围 / ms | MAC / RESET | 最大kernel μop | UOP LOAD / 字节 | 总指令 | 提交 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main)}

| 程序 | INP请求 / KiB | WGT请求 / KiB | ADD→ACC请求 / KiB | OUT请求 / KiB | WGT有效LOAD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(requests)}

Q1至Q8逐描述符、条件路径、逻辑来源、重数和必要先后保持相同张量请求，不能只用612 KiB相同替代此证明。TPS的Tile和双缓冲不同，作为实用参照，不作为聚合单因素消融。表中请求量不是物理总线事务数；GEMM减少不等于乘加工作减少。当前共同runtime下四Q的UOP LOAD均1、载荷均14,408字节，提交均1；本轮收益不能归因于UOP LOAD减少或权重搬运减少，旧runtime的148→36及14,416字节不混入本轮。这是组织变化的完整调用净效果，未单独分解为纯CPU开销、器件空闲或某一种硬件等待周期；已有等待profile只作诊断保留。

## 2 六轮原始归纳与预定判据

| 轮次 | Q1 / ms | Q2 / ms | Q4 / ms | Q8 / ms | TPS / ms | Q8对Q1改善 | Q8节省 / ms |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired)}

固定口径：m[r,p]=当轮三seed时延中位数；T[p]=六个m的中位数；g[r,C;B]=100×(1−m[r,C]/m[r,B])；G=六个g的中位数。G不是用两个T直接相除，未四舍五入的平局不算胜。{trend}；二者只描述粒度趋势，不能替换预定Q8主对照。完整数值和全部配对差见[analysis.json](analysis.json)。

研究复验预定门槛为Q8对Q1的G≥2%、胜≥5/6；实用门槛为Q池最小T程序对Q1的G≥2%、胜≥5/6，且对TPS的G≥1%、胜≥5/6。这些是工程门槛。六轮与90次正式调用来自同一Linux启动，不是90个独立工作负载；旧runtime双启动不能补成本轮独立启动证据。当前不做第二启动或整图。

## 3 能不能合：证明和方法边界

同一Fold后、CPUAccessRewrite前父IR产生四Q，四个规范化执行表示不同，没有去重。每次调用从空张量版本表建立INP/WGT来源；同址重新LOAD建立新版本，实际FIFO证明LOAD完成先于读取、旧读取完成先于覆盖。ACC按RESET→累加→后处理→STORE检查，正常累加是允许的内容更新。所有有效地址、字段和独立SRAM容量通过检查，WGT实际最高1008元素即258,048字节=252 KiB。

逐3,136个16-lane packed输出核对72个输入向量×权重块归约项及重数，每标量输出1152项；共225,792个块乘加步骤，每步含256个标量乘积，全层57,802,752个标量乘积。四Q与封存TPS的逻辑乘积/后处理摘要相同。卷积任意部分和界18,874,368小于2^31；这个界不扩展为任意ADD不溢出保证。ADD来源/位宽、算术右移7、clip[-127,127]及int8转换保持原整数语义。

定向低幅输入在clip之前的int32归约逐元素相等，值域[-736,853]；输出有25,047负值、11,347正值、0个clip边界值。四Q共28次完整融合FSim通过三个seed、同句柄A→B→A和定向输入。12项离线正反例与独立补充反例均通过，包括重命名、非八段、同址异版本、内部通知、RESET/后处理、不兼容循环、真实容量和缓存身份。

缓存身份为独立静态LLVM handle与空/单int32签名，而非仅CO。四Q的GEMM不同handle数9/5/3/2；签名4字节且编码范围受限。同handle/同CO而内容不同会错误复用的反例已经保留。LLVM复制到互斥路径的callsite按同一handle/initializer/closure认证。μop内容认证用同session首次initializer定义、实际UOP LOAD及计算描述符和冻结runtime队列规则重建，按COMPUTE FIFO逐引用核对，不是FPGA UOP SRAM直接读回；ALU即时数不在原dump中完整显示，另以LLVM全字段证明补足。

通用greedy装载合并在回绕时提前覆盖在途引用的离线反例已重现，因此只看已装载区间不够。Q清单3602条、无SRAM回绕；TPS发生回绕，封存原模块同session及本启动实际引用均通过。未修改runtime，也不把这些程序通过外推为所有队列都安全。

支持域为冻结VTA规范ABI、静态noalias buffer、有限可解析循环/路径及已解释命令。未知别名、动态关系、外部调用或不能解释的阶段保守拒绝；这是局部后端执行优化，不是通用程序证明器、图算子融合、新ISA、内存分配或新的CPU/VTA共享DDR机制。

## 4 完整调用、首次成本与实验身份

完整融合ABI是X/W/ADD/Y：int32卷积→广播ADD→算术右移7→clip[-127,127]→int8。固定Q小Tile b1h2w14ci1co1、单线程、13组常驻加单流式槽，TPS保持封存输出通道双缓冲ci4。计时仅围住一次完整融合调用，包括命令组织、当次全部提交和原有完成等待；RPC、上传、分配、读回和校验均在外。没有隐藏预热或额外手工同步。

每个轮次×程序为fresh RPC和一致bitstream初始化，同单元使用长期参数句柄，三seed各显式预热一次后各单次计时一次。严格执行计划六轮顺序及seed轮换：90次预热＋90次计时。资格15次三seed＋12次Q原地权重A→B→A，以及独立诊断45次前缀＋45次预热后调用另记。板端完整目标调用合计**297次**，全部正确且输入参数不变；资格/诊断不拼入正式样本。诊断90次中，15次首次前缀和45次预热后调用共60份原dump被逐实际μop和FIFO/token重新认证；另外30次前缀保存输出和profile，不冒称直接μop观测。正式计时只有profile和输出核验，不在同一次计时中开启原生dump。

首次组合成本单独保存在[first_cache_costs.jsonl](qualification/first_cache_costs.jsonl)。以下均为资格完整调用；第1次含首次μop生成和缓存建立，其余两次仍为资格，不能作为正式热样本或纯initializer分解。

| 程序 | 第1次完整调用 / ms | 第2 / 第3次资格调用 / ms |
|---|---:|---:|
{chr(10).join(first_table)}

Linux boot为`{analysis['boot_id']}`。板端重新核验并恢复192 MiB udmabuf（201326592字节、物理起址0x68500000）、冻结bitstream和CPU3工具。只写RAM/tmpfs、未重启、未写SD、未调频或终止外部服务。每会话用/proc maps核对实际映射libvta/libtvm_runtime，并核对CPU3、空初始队列、replay disabled和bitstream；helper通过冻结本地源/二进制SHA、固定上传文件及明确加载路径绑定。程序二进制、源码和协议身份逐次检查。

共同runtime：libvta SHA `43aaf632…`，实际VALUE_UOPS=1、GREEDY_UOPS=1；bitstream SHA `7bf1ac95…`。完整身份见[board_protocol.json](board_protocol.json)、[board_prepared.json](board_prepared.json)。主机命令backing/提交阈值33554432字节、192 MiB DMA和32 KiB UOP SRAM是不同容量。

本轮没有失败的板端目标调用或补样本。早期本地开发失败、零板目标调用的交叉编译/封存准备错误及恢复前sandbox SSH阻止记录保留，不与最终固定池混用。恢复run02是此前没有板端写入的基础设施权限问题解决记录，不是失败性能样本重试。运行期错误停止门禁与独立账目审计记录可追溯。

## 5 交付与暂停

四个先行报告：[边界来源](BOUNDARY_ORIGIN.md)、[窗口规则](WINDOW_RULES.md)、[等价依据](EQUIVALENCE.md)、[TPS静态审计](TPS_STATIC_AUDIT.md)。它们保留编写时阶段状态，当前完成状态以本报告和各阶段summary为准。早期FSim builder summary的native proof pending也保留，最终本地验收是四个`local_final02/Q*/full_local_certificate.json`的ALL_LOCAL_WINDOW_GATES_PASSED，不覆盖早期快照。现有Lift/Fold同父IR七分支对照区分了scope重组已有、2016条无循环紧凑化表示、以及16次LoopBegin的非法表示；未为这些分支增加第六个板程序。

原始账目：[正式调用](board/calls.jsonl)、[会话及RPC日志](board/sessions.jsonl)、[调用开始/结束](board/dispatch.jsonl)、[单次helper计数](board/helper_counts.jsonl)、[诊断调用](diagnostic/calls.jsonl)、[实际μop证书](diagnostic/native_certificates.jsonl)。零新增目标调用独立审计见[资格审计](qualification/independent_audit.json)和[诊断审计](diagnostic/INDEPENDENT_AUDIT.md)。图形为[PNG](six_round_result.png)、[SVG](six_round_result.svg)、[PDF](six_round_result.pdf)，仅可视化本轮正式数据。

本轮已完成并暂停。工程门槛只决定后续申请资格；不自动开展独立启动复验、TPS新候选或整图实验。当前论文结论须限定于受限支持域、固定程序和本次单启动结果；未超过TPS时保留聚合净收益结论，停止整图投入。
"""
    with (root / "REPORT.md").open("x") as stream:
        stream.write(report)
    plot(root, analysis)
    print(json.dumps({"report": str(root / "REPORT.md"), "primary_G_pct": primary["G_pct"], "board_calls": 297}))


def plot(root, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Q1", "Q2", "Q4", "Q8", "TPS"]
    colors = ["#718096", "#4C78A8", "#72B7B2", "#D55E00", "#4E9A51"]
    with plt.rc_context({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
        for index, (name, color) in enumerate(zip(names, colors)):
            values = analysis["round_medians_ms"][name]
            axes[0].scatter([index] * 6, values, color=color, s=20, alpha=.75)
            axes[0].hlines(analysis["T_ms"][name], index-.23, index+.23, color=color, linewidth=2.5)
        axes[0].set(xticks=range(5), xticklabels=names, ylabel="Complete fused call (ms)", title="Six round medians; line = T")
        gains = analysis["primary_Q8_versus_Q1"]["paired_improvements_pct"]
        axes[1].plot(range(1, 7), gains, "o-", color=colors[3])
        axes[1].axhline(0, color="#777777", linewidth=.8)
        axes[1].axhline(2, color="#777777", linewidth=.8, linestyle=":", label="2% engineering threshold")
        axes[1].axhline(median(gains), color=colors[3], linewidth=1, linestyle="--", label="G (median paired improvement)")
        axes[1].set(xticks=range(1, 7), xlabel="Round", ylabel="Q8 vs Q1 improvement (%)", title="Frozen primary comparison")
        axes[1].legend(fontsize=7, loc="best")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
        fig.suptitle("Node24 | fixed five-program pool | one Linux boot", fontsize=10)
        for suffix in ["png", "svg", "pdf"]:
            target = root / ("six_round_result." + suffix)
            if target.exists():
                raise ValueError("Refuse to overwrite report figure")
            fig.savefig(target, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    render(parser.parse_args().root.resolve())
