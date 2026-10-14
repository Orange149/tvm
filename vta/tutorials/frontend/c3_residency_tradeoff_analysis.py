"""Seal calibration before reading held-out journals or showing a ranking."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import c3_residency_tradeoff as p


def journal(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def calibration_seal(output):
    board = output / "board"
    completed = p.read(board / "summary.json")
    if completed.get("status") != "FORMAL_ACQUISITION_COMPLETE" or \
            completed.get("timed_calls") != 108 or completed.get("warmup_calls") != 108:
        raise ValueError("incomplete formal acquisition cannot be analyzed")
    # Only these three files are opened until model.json is committed.
    paths = {name: board / "programs" / (name + ".jsonl") for name in p.CALIBRATION}
    rows = [row for path in paths.values() for row in journal(path)]
    rm = p.round_medians(rows, p.CALIBRATION)
    targets = {name: median(rm[r][name] for r in range(6)) for name in p.CALIBRATION}
    features = {}
    for name in p.CALIBRATION:
        observed = [row["model_features"] for row in rows if row["program"] == name
                    and row["call_kind"] == "timing"]
        if not observed or any(item != observed[0] for item in observed):
            raise ValueError("model feature state not fixed across formal calls")
        features[name] = observed[0]
    model = p.fit_model(targets, features)
    model.update(calibration_journals_sha256={name: p.digest(path) for name, path in paths.items()},
                 analysis_source_sha256=p.digest(__file__), model_source_sha256=p.digest(p.__file__),
                 aggregate="median of six per-round three-seed medians", units="ms",
                 boot_id=completed["boot_id"], validation_read_before_seal=False)
    destination = output / "analysis"
    destination.mkdir(exist_ok=False)
    p.write_new(destination / "model.json", model)
    p.write_new(destination / "seal_receipt.json", {
        "model_sha256": p.digest(destination / "model.json"),
        "calibration_journals_sha256": model["calibration_journals_sha256"],
        "validation_read_before_commit": False,
    })
    return model


def analyze(output):
    output = Path(output)
    model = calibration_seal(output)
    paths = {name: output / "board/programs" / (name + ".jsonl") for name in p.PROGRAMS}
    rows = [row for path in paths.values() for row in journal(path)]
    result = p.summarize(rows)
    features = {}
    for name in p.VALIDATION:
        observed = [row["model_features"] for row in rows if row["program"] == name
                    and row["call_kind"] == "timing"]
        if any(item != observed[0] for item in observed):
            raise ValueError("held-out feature state not fixed")
        features[name] = observed[0]
    validation = p.validate_model(model, result["program_medians_ms"], features)
    p.write_new(output / "analysis/validation.json", validation)
    p.write_new(output / "analysis/results.json", result)
    timing = [r for r in rows if r["call_kind"] == "timing"]
    profile_table = {}
    for name in p.PROGRAMS:
        part = [r for r in timing if r["program"] == name]
        profile_table[name] = {
            "call_median_ms": result["program_medians_ms"][name],
            "device_run_wait_median_ms": median(r["runtime_profile"]["device_run_wait_us"] / 1000 for r in part),
            "call_minus_device_wait_median_ms": median(
                r["latency_ms"] - r["runtime_profile"]["device_run_wait_us"] / 1000 for r in part),
            "not_pure_cpu_or_hardware_idle": True,
        }
    p.write_new(output / "analysis/profile_diagnostics.json", profile_table)
    generate_plots(output, result, model, validation)
    lines = ["# 节点24驻留程度与命令组织单启动报告", "",
             "六个程序使用同一 Linux 启动，全部完成完整融合正确性和逐调用检查。"
             "以下为板端完整融合函数调用时延，不是纯硬件指令时间，也不换算整网FPS。", "",
             "| 程序 | 常驻权重 KiB | 名义WGT分配 KiB | WGT请求 KiB | 完整融合时延 ms |",
             "|---|---:|---:|---:|---:|"]
    for name in p.PROGRAMS:
        r = p.RESIDENT.get(name)
        resident, arena = (r * 18, (r + 1) * 18) if r is not None else ("不适用", "独立证书")
        lines.append("| %s | %s | %s | %.0f | %.6f |" %
                     (name, resident, arena, p.expectations(name)["wgt_payload_bytes"] / 1024,
                      result["program_medians_ms"][name]))
    lines += ["", "## 实测配对结果", "",
              "| 候选／基准 | 六轮配对中位改善 | 获胜轮次 |", "|---|---:|---:|"]
    for row in result["comparisons"].values():
        lines.append("| %s／%s | %.4f%% | %d/6 |" %
                     (row["candidate"], row["baseline"], row["median_improvement_pct"], row["wins"]))
    lines += ["", "驻留程度对照允许WGT地址与必要依赖变化，是固定Tile和紧凑组织下的程序级效果。"
              "Input使用不同Tile，只评价实际程序价值。F13-B→R13-G是GEMM及相关UOP组织的综合变化。",
              "", "## 局部插值模型", "",
              "模型状态：%s。留出验证通过：%s。" % (model["status"], validation["passed"]),
              "校准数据与系数已先封存，再读取R8/R12时延；校准零残差不作为验证证据。"
              "模型仅验证固定紧凑组织下的局部驻留插值，不验证组织系数泛化。"
              "10%误差通过不代表能区分2%的性能差异。"]
    for row in validation.get("rows", []):
        lines.append("\n%s：预测 %.6f ms，实测 %.6f ms，带符号误差 %.4f%%，绝对误差 %.6f ms。" %
                     (row["program"], row["predicted_ms"], row["observed_ms"],
                      row["signed_relative_error_pct"], row["absolute_error_ms"]))
    lines += ["", "## 证据边界和收尾", "",
              "后续确认候选：%s。" % (", ".join(result["future_confirmation_candidates"]) or "无"),
              "本轮没有第二启动、整图、双槽或预取。VTADeviceRun等待及调用与等待差额仅为诊断，"
              "不命名为纯CPU时间、硬件空闲或流水slack。未知动态UOP峰值保持unknown。",
              "历史F13-G整图为单启动，相对合理Input图增量很小，不用本轮算子时间替代整图结果。",
              "本轮完成后停止追加实验。"]
    report_path = output / "FINAL_REPORT.md"
    with report_path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    artifact_paths = [path for path in output.rglob("*") if path.is_file()]
    p.write_new(output / "artifact_hashes.json", {
        str(path.relative_to(output)): p.digest(path) for path in artifact_paths})
    print("Report complete: " + str(report_path))


def generate_plots(output, result, model, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest = output / "analysis"
    fig, ax = plt.subplots()
    names = ("R0-G", "R8-G", "R12-G", "R13-G")
    rs = [p.RESIDENT[name] for name in names]
    for name, r in zip(names, rs):
        ax.scatter([r] * 6, [result["round_medians_ms"][i][name] for i in range(6)], alpha=.4)
    ax.plot(rs, [result["program_medians_ms"][n] for n in names], marker="o", label="compact group GEMM")
    ax.scatter([13], [result["program_medians_ms"]["F13-B"]], marker="x", label="F13-B (unmerged)")
    ax.axhline(result["program_medians_ms"]["Input-7x7"], linestyle="--", label="Input (different Tile)")
    ax.set(xlabel="Resident groups", ylabel="Full fused call latency (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "residency_latency.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    names = ("F13-B", "R13-G")
    ax.bar(names, [result["program_medians_ms"][n] for n in names])
    ax.set(ylabel="Full fused call latency (ms)", title="Same residency, command organization")
    fig.tight_layout()
    fig.savefig(dest / "organization_latency.png")
    plt.close(fig)
    if model["status"] == "MODEL_SEALED":
        fig, ax = plt.subplots()
        for index, name in enumerate(p.CALIBRATION):
            observed = model["calibration_targets_ms"][index]
            predicted = observed + model["calibration_residual_ms"][index]
            ax.scatter([observed], [predicted], marker="x", color="gray")
            ax.annotate(name + " calibration", (observed, predicted), fontsize=8)
        for row in validation["rows"]:
            ax.scatter([row["observed_ms"]], [row["predicted_ms"]], marker="o", color="blue")
            ax.annotate("%s validation: %.4f ms / %.2f%%" %
                        (row["program"], row["absolute_error_ms"], row["signed_relative_error_pct"]),
                        (row["observed_ms"], row["predicted_ms"]), fontsize=8)
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], "--", color="black")
        ax.set(xlabel="Observed latency (ms)", ylabel="Predicted latency (ms)")
        fig.tight_layout()
        fig.savefig(dest / "model_validation.png")
        plt.close(fig)
