#!/usr/bin/env python3
"""Read-only C1 mechanism report. Never treats one boot as stable replication."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import statistics
import time

from run_vta_c3_reuse_supplement import read, write, verify, seal, file_hash, MODES
from analyze_vta_c3_reuse_local import table_row


def load_rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def family_results(tables, measurements):
    output = []
    for family in sorted({r["family_id"] for r in tables if not r["control"]}):
        modes = {r["mode"]: r for r in tables if r["family_id"] == family and not r["control"]}
        if not all(r["candidate_id"] in measurements for r in modes.values()):
            output.append({"family_id": family, "workload_id": next(iter(modes.values()))["workload_id"],
                           "four_fpga_measured": False, "reason": "at least one mode FPGA-invalid; no substitution"})
            continue
        values = {m: measurements[r["candidate_id"]] for m, r in modes.items()}
        if any(len(v) != 8 for v in values.values()):
            raise ValueError("incomplete balanced family")
        medians = {m: statistics.median(v) for m, v in values.items()}
        o = medians["original"]
        combined = "input_weight_resident_barrier"
        record = {"family_id": family, "workload_id": modes["original"]["workload_id"],
                  "four_fpga_measured": True, "median_ms": medians,
                  "fastest_mode": min(MODES, key=lambda m: medians[m]),
                  "combination_fastest": medians[combined] < min(medians[m] for m in MODES if m != combined),
                  "additive_interaction_ms": medians[combined] - medians["input_stationary"] - medians["weight_resident_barrier"] + o,
                  "effective_dual_reuse": modes[combined]["inp_bytes"] < modes["original"]["inp_bytes"]
                      and modes[combined]["wgt_bytes"] < modes["original"]["wgt_bytes"],
                  "stable_cross_boot_claim": False,
                  "improvement_vs_original_pct": {m: 100 * (1 - medians[m] / o) for m in MODES},
                  "wins_vs_original": {m: sum(a < b for a, b in zip(values[m], values["original"])) for m in MODES}}
        output.append(record)
    return output


def run(args):
    started = time.monotonic()
    board_dir = Path(args.board_dir).resolve()
    contract = verify(board_dir)
    if (board_dir / "invalid_session.json").exists():
        raise ValueError("interrupted/invalid session cannot supply formal timing")
    summary = read(board_dir / "summary.json")
    if not summary["status"].startswith(("C1_COMPLETE", "C2_COMPLETE")):
        raise ValueError("requires completed continuous session")
    parent = Path(contract["parent_directory"])
    verify(parent)
    if file_hash(parent / "artifact_hashes.json") != contract["parent_manifest_sha256"]:
        raise ValueError("parent ledger mismatch")
    candidates = read(parent / "candidates.json")
    controls = read(parent / "controls.json")
    tables = [table_row(r) for r in candidates + controls]
    profiles = {}
    measurements = defaultdict(list)
    results = load_rows(board_dir / "results.jsonl")
    attempts = sum(r["event"] == "dispatch_begin" for r in load_rows(board_dir / "timeline.jsonl"))
    if attempts != len(results) or any(r["complete_operator_calls"] != 1 or r["hidden_warmup_calls"] for r in results):
        raise ValueError("unaccounted actual invocation or hidden warmup")
    for result in results:
        if result["phase"] == "timing":
            if not result["correct"]:
                raise ValueError("wrong timed output")
            if result["block"] == "main":
                measurements[result["candidate_id"]].append(result["latency_ms"])
        if result["phase"] == "profile":
            profiles[result["candidate_id"]] = result["runtime_profile_complete"]
    by_id = {r["candidate_id"]: r for r in candidates + controls}
    for cid, profile in profiles.items():
        expected = by_id[cid]["local_qualification"]["semantics"]["totals"]
        for key, value in expected.items():
            if profile.get(key) != value:
                raise ValueError("FPGA/static logical DMA mismatch: " + cid + " " + key)
    classification = family_results(tables, measurements)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {"phase": "board_analysis", "input_directory": str(board_dir),
                                     "input_manifest_sha256": file_hash(board_dir / "artifact_hashes.json"),
                                     "source_sha256": file_hash(__file__), "cross_boot_confirmed": False})
    for table in tables:
        cid = table["candidate_id"]
        table["fpga_latency_ms"] = statistics.median(measurements[cid]) if cid in measurements else None
        table["driver_run_total_us_per_complete_operator"] = profiles[cid]["driver_run_total_us"] if cid in profiles else None
        table["FPGA_submissions_per_complete_operator"] = profiles[cid]["driver_run_calls"] if cid in profiles else None
        table["FPGA_synchronize_calls"] = profiles[cid]["synchronize_calls"] if cid in profiles else None
    with (output / "four_mode_fpga.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows({k: "unknown" if v is None else v for k, v in r.items()} for r in tables)
    write(output / "family_classification.json", classification)
    phase_costs = {}
    for phase in dict.fromkeys(r["phase"] for r in results):
        selected = [r for r in results if r["phase"] == phase]
        phase_costs[phase] = {
            "complete_operator_calls": len(selected), "wrong_outputs": sum(not r["correct"] for r in selected),
            "host_call_wall_seconds": sum(r["host_wall_seconds"] for r in selected),
            "logical_load_bytes": sum(r["runtime_profile_complete"]["load_buffer_2d_bytes"] for r in selected),
            "logical_store_bytes": sum(r["runtime_profile_complete"]["store_buffer_2d_bytes"] for r in selected),
            "logical_dma_calls": sum(r["runtime_profile_complete"]["load_buffer_2d_calls"] + r["runtime_profile_complete"]["store_buffer_2d_calls"] for r in selected),
            "FPGA_kernel_invocations": sum(r["runtime_profile_complete"]["driver_run_calls"] for r in selected),
        }
    measured = [r for r in classification if r["four_fpga_measured"]]
    best_space = {}
    for geometry in dict.fromkeys(r["workload_id"] for r in measured):
        families = [r for r in measured if r["workload_id"] == geometry]
        original = min(r["median_ms"]["original"] for r in families)
        extended = min(min(r["median_ms"].values()) for r in families)
        best_space[geometry] = {"best_original_ms": original, "best_extended_ms": extended,
                                "improvement_pct": 100 * (1 - extended / original),
                                "scope": "identical measured tile set; not complete ConfigSpace or TopHub"}
    report = {"status": "FIRST_BOOT_MECHANISM_EVIDENCE_PENDING_C2", "boot_id": summary["boot_id"],
              "four_measured": len(measured), "effective_dual_measured": sum(r["effective_dual_reuse"] for r in measured),
              "combination_fastest_families": sum(r["combination_fastest"] for r in measured),
              "fastest_mode_counts": dict(Counter(r["fastest_mode"] for r in measured)),
              "phase_costs": phase_costs, "best_original_vs_extended": best_space,
              "board_wall_seconds": summary["W0_to_T1_seconds"], "board_T0_wall_seconds": summary["T0_to_T1_seconds"],
              "static_FSim_FPGA_logical_DMA_verified_profiles": len(profiles),
              "all_actual_invocations_verified": attempts, "analysis_wall_seconds": time.monotonic() - started,
              "stable_cross_boot_claim": False, "physical_AXI_claim": False, "pure_compute_claim": False}
    write(output / "summary.json", report)
    lines = ["# C3四方案第一启动机制结果（待第二启动）", "",
             "| family | O ms | I ms | W ms | IW ms | 最快 | IW/O改善 |", "|---|---:|---:|---:|---:|---|---:|"]
    for r in measured:
        m = r["median_ms"]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {:.2f}% |".format(
            r["family_id"], *[m[mode] for mode in MODES], r["fastest_mode"], r["improvement_vs_original_pct"]["input_weight_resident_barrier"]))
    lines += ["", "仅一个boot，不声称跨启动稳定、整网FPS、物理AXI流量或纯计算时间。",
              "M2失败身份完整保留，不替换、不回退；profile按单次完整算子计账，不除以submission。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    # Four-mode traffic/latency plot, using numeric labels to avoid font ambiguity.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for index, mode in enumerate(MODES):
        xs = np.arange(len(measured)) + (index - 1.5) * 0.18
        rows = [next(t for t in tables if t["family_id"] == r["family_id"] and t["mode"] == mode and not t["control"]) for r in measured]
        for axis, key, scale in ((axes[0], "inp_bytes", 1024), (axes[1], "wgt_bytes", 1024), (axes[2], "fpga_latency_ms", 1)):
            axis.bar(xs, [t[key] / scale for t in rows], width=.18, label=mode)
    axes[0].set_ylabel("Input LOAD KiB")
    axes[1].set_ylabel("Weight LOAD KiB")
    axes[2].set_ylabel("Operator latency ms")
    axes[2].set_xticks(np.arange(len(measured)), [r["family_id"] for r in measured], rotation=30)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same-tile four-mode FPGA evidence: first boot only")
    fig.tight_layout()
    fig.savefig(output / "four_mode_dma_latency.svg")
    plt.close(fig)
    verify(board_dir)
    seal(output)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
