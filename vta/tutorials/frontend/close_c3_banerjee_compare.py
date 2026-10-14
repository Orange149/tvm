#!/usr/bin/env python3
"""Publish the independently audited fixed comparison, then seal its artifacts."""
import argparse
import json
from pathlib import Path
from statistics import median

from c3_banerjee_compare_board import PROGRAMS, NEW, LOCAL, FIVE, execution_sources
from c3_residency_tradeoff import read, digest, write_new, round_medians, compare

LABELS = dict(zip(PROGRAMS, (
    "普通双缓冲适配控制", "Banerjee双缓冲输入复用适配版", "13组常驻加单槽",
    "12组常驻加单槽", "12组常驻加双槽预取", "7×7 Input实用程序")))


def close(root):
    root = Path(root).resolve()
    audit = read(root / "final_audit.json")
    if audit["status"] != "INDEPENDENT_FORMAL_COMPARISON_AUDIT_PASSED":
        raise ValueError("independent audit absent")
    for path, sha in audit["artifacts"].items():
        if digest(root / path) != sha:
            raise ValueError("audited artifact changed: " + path)
    if audit["frozen_sources"] != execution_sources():
        raise ValueError("frozen sources changed")
    rows = [json.loads(s) for s in (root / "board/calls.jsonl").read_text().splitlines()]
    rm = round_medians(rows, PROGRAMS)
    prepared = read(root / "prepared.json")
    signatures = read(root / "qualification/diagnostic_signatures.json")
    pairs = ((PROGRAMS[1], PROGRAMS[0]), ("R13-G", "R12-G"), (NEW, "R12-G"),
             (NEW, "R13-G"), (NEW, PROGRAMS[1]), (NEW, "Input-7x7"))
    comparisons = [compare(rm, c, b) for c, b in pairs]
    main = next(x for x in comparisons if x["candidate"] == NEW and x["baseline"] == PROGRAMS[1])
    programs = {}
    for name in PROGRAMS:
        part = [r for r in rows if r["program"] == name and r["call_kind"] == "timing"]
        sigs = [s for arr in signatures[name]["rotations"].values() for s in arr]
        programs[name] = dict(label=LABELS[name], median_ms=median(rm[r][name] for r in range(6)),
            round_ms=[rm[r][name] for r in range(6)],
            wait_ms_median_of_18_calls=median(r["runtime_profile"]["device_run_wait_us"] / 1000 for r in part),
            call_minus_wait_ms_median_of_18_calls=median(r["latency_ms"] - r["runtime_profile"]["device_run_wait_us"] / 1000 for r in part),
            compiled_requests=prepared["programs"][name]["request_totals"],
            warmed_commands={k: sorted({s[k] for s in sigs}) for k in (
                "reset", "mac", "wgt_loads", "wgt_payload_bytes", "uop_loads", "uop_payload_bytes", "instructions", "submissions")})
    data = dict(status="COMPLETE_PAUSED_AFTER_FIXED_COMPARISON", boot_id=audit["boot_id"],
        programs=programs, comparisons=comparisons,
        primary_gate_passed=main["median_improvement_pct"] >= 2 and main["wins"] >= 5,
        aggregate="median of six round medians, each over three input seeds",
        qualification_FPGA_calls=150, formal_warmups=108, formal_calls=108, total_FPGA_calls=366,
        CPU_helper_calls=3, local_tests_passed=19,
        old_rule_sha256=digest(FIVE / "rule_freeze.json"),
        prior_program_latency_exposure={n: n not in PROGRAMS[:2] for n in PROGRAMS},
        no_selection_or_model_refit=True, no_whole_graph_or_second_boot=True,
        not_author_TPS_best_or_full_paper_reproduction=True,
        report_generator_sha256=digest(__file__))
    write_new(root / "results.json", data)
    lines = ["# 节点24双缓冲与部分驻留比较结果", "",
        "六个固定程序在同一Linux启动完成150次FPGA资格调用、108次显式预热和108次正式计时，"
        "输出全部与完整融合参考逐元素一致，独立输入参数保持不变。另有3次CPU计时工具回归。"
        "本轮已完成并暂停，未进行新的整图或第二启动实验。", "",
        "## 单个完整融合算子的时延", "",
        "这些毫秒数是YOLOv3-tiny-224节点24的完整融合函数耗时，包含函数内部的命令组织、"
        "所有设备提交与完成等待。RPC往返、参数上传和输出检查在计时外。"
        "每轮先取三个输入seed的中位数，再取六轮中位数；不能换算成整网FPS。", "",
        "| 程序 | 完整融合时延 ms |", "|---|---:|"]
    for name in PROGRAMS:
        lines.append("| %s | %.6f |" % (LABELS[name], programs[name]["median_ms"]))
    lines += ["", "## 配对比较", "",
        "改善率按同轮时延计算，再取六轮中位数；它不必等于上表两个总中位数的比值。", "",
        "| 候选相对基准 | 配对改善率 | 获胜轮次 |", "|---|---:|---:|"]
    for x in comparisons:
        lines.append("| %s 相对 %s | %.4f%% | %d/6 |" % (
            LABELS[x["candidate"]], LABELS[x["baseline"]], x["median_improvement_pct"], x["wins"]))
    lines += ["", "12组双槽预取相对Banerjee适配实例的预定工程门槛（至少2%，至少5/6轮获胜）："
              + ("通过。" if data["primary_gate_passed"] else "未通过。")
              + "这是本轮固定配置、单启动下的程序比较，不是统计显著性或跨启动稳定性结论。", "",
        "## 请求与命令", "",
        "| 程序 | INP请求 KiB | WGT请求 KiB | RESET | 乘加GEMM | 预热后UOP载荷 bytes | 总指令 | 提交 |",
        "|---|---:|---:|---:|---:|---|---|---|"]
    fmt = lambda values: "/".join(str(v) for v in values)
    for name in PROGRAMS:
        rec, req = programs[name], programs[name]["compiled_requests"]
        sig = rec["warmed_commands"]
        lines.append("| %s | %.0f | %.0f | %s | %s | %s | %s | %s |" % (
            LABELS[name], req["load_buffer_2d_inp_bytes"] / 1024,
            req["load_buffer_2d_wgt_bytes"] / 1024, fmt(sig["reset"]), fmt(sig["mac"]),
            fmt(sig["uop_payload_bytes"]), fmt(sig["instructions"]), fmt(sig["submissions"])))
    lines += ["", "请求载荷来自指令合同，并经FPGA逐调用计数核验，不是实际AXI总线事务量。"
        "UOP等列列出匹配正式调用位置4/5/6的所有观察值；首次调用记录单独保存。", "",
        "| 程序 | 设备运行等待 ms | 调用时延减等待 ms |", "|---|---:|---:|"]
    for name in PROGRAMS:
        rec = programs[name]
        lines.append("| %s | %.6f | %.6f |" % (LABELS[name],
            rec["wait_ms_median_of_18_calls"], rec["call_minus_wait_ms_median_of_18_calls"]))
    lines += ["", "这两列各为18次正式调用中对应诊断量的中位数。差额按每次调用先相减再聚合，"
        "不命名为纯CPU时间、硬件空闲或流水等待周期。", "",
        "## 能够归因到什么", "",
        "普通双缓冲与Banerjee功能适配版具有相同Tile、线程、数学计算及共同硬件适配。"
        "两者比较检验重复输入LOAD删除、输入银行映射及连带变化。该适配仅覆盖输出通道双缓冲输入复用，"
        "没有复现作者的TPS搜索、空间线程和完整软硬件栈，所以不能代表论文最佳性能。", "",
        "本Tile下双缓冲权重槽为两份各2.25 KiB，合计4.5 KiB，"
        "并不是使用两个装满的128 KiB区域。13组单槽使用234 KiB常驻区加18 KiB流式槽；"
        "12组单槽使用216 KiB加18 KiB；12组双槽使用216 KiB加36 KiB。"
        "部分驻留程序还改变输入复用范围、线程和GEMM组织，因此与Banerjee实例的差距不能全部归为内存划分。", "",
        "12组单槽与双槽的同驻留程度对照用于观察预取及必要依赖变化；"
        "13组单槽与12组双槽的比较同时包含少驻留一组、多搬权重和允许预取的取舍。"
        "这些固定程序不能证明一般性的最优驻留数或最优缓冲比例。", "",
        "四个已有程序在本轮之前均已有性能数据；新比较没有重新选择赢家、拟合模型或改变历史结果。"
        "prepared.json内继承的候选历史标签属于原始冻结时刻，本轮暴露状态以results.json的"
        "prior_program_latency_exposure为准。", "",
        "## 证据与后续", "",
        "[全部六轮数据与配对改善](results.json)、[独立验收](qualification/independent_audit.json)、"
        "[正式会话审计](final_audit.json)、[逐调用记录](board/calls.jsonl)、[执行身份](protocol.json)。"
        "原生FPGA指令证明读取完成后覆盖的安全关系，不提供硬件重叠周期。19项本地测试通过。", "",
        "本轮仅为固定程序的前作对照。下一步应把本表与已有整图证据一起用于论文中的适用条件分析。"
        "如需宣称优于经过搜索的Banerjee方法，仍需另行规定相同调优预算的基线；本轮不自动追加搜索。"]
    with (root / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    manifest = dict(status="SEALED_COMPLETE_SINGLE_BOOT_COMPARISON", sources=execution_sources(),
        auxiliary_sources={str(Path(__file__).resolve()): digest(__file__),
            str(Path(__file__).with_name("audit_c3_banerjee_compare.py")): digest(Path(__file__).with_name("audit_c3_banerjee_compare.py"))},
        artifacts={str(x.relative_to(root)): digest(x) for x in root.rglob("*") if x.is_file()},
        protected_local_manifest_sha256=digest(LOCAL / "local_manifest.json"),
        protected_rule_sha256=digest(FIVE / "rule_freeze.json"))
    write_new(root / "closeout_manifest.json", manifest)
    print(json.dumps({"median_ms": {n: programs[n]["median_ms"] for n in PROGRAMS},
                      "comparisons": comparisons, "report": str(root / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    close(parser.parse_args().root)
