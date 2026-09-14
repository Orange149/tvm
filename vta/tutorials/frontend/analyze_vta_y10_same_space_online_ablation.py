#!/usr/bin/env python3
"""Aggregate the six independent Y10 same-space selector reexecutions.

The selector runs themselves are label blind.  This analyzer joins their immutable
candidate identities to the post-selection complete-pool labels only after all runs
have finished.  It reports both the ratios observed during each reexecution and the
fixed complete-pool regret used for algorithm attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


EXPECTED_STRATEGIES = ("dma_prior", "mode_aware_xgb")
EXPECTED_SEEDS = (44501, 44502, 44503)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(directory):
    directory = Path(directory)
    manifest_path = directory / "artifact_hashes.json"
    manifest = read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("invalid or empty artifact manifest: {}".format(manifest_path))
    for relative, expected in artifacts.items():
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError("artifact hash mismatch: {}".format(path))
    return len(artifacts)


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
        "q1": percentile(finite, 0.25),
        "q3": percentile(finite, 0.75),
    }


def invocation_cost(candidate_result):
    totals = {
        "graph_api_invocations": 0,
        "driver_kernel_invocations": 0,
        "logical_load_bytes": 0,
        "logical_store_bytes": 0,
        "logical_dma_bytes": 0,
        "logical_dma_calls": 0,
        "invocation_host_wall_ms": 0.0,
    }
    rows = list(candidate_result.get("correctness", [])) + list(
        candidate_result.get("timings", [])
    )
    for row in rows:
        profile = row.get("runtime_profile_complete")
        if not isinstance(profile, dict):
            raise ValueError("missing complete runtime profile for FPGA invocation")
        load_bytes = int(profile["load_buffer_2d_bytes"])
        store_bytes = int(profile["store_buffer_2d_bytes"])
        totals["graph_api_invocations"] += 1
        totals["driver_kernel_invocations"] += int(profile["driver_run_calls"])
        totals["logical_load_bytes"] += load_bytes
        totals["logical_store_bytes"] += store_bytes
        totals["logical_dma_bytes"] += load_bytes + store_bytes
        totals["logical_dma_calls"] += int(profile["load_buffer_2d_calls"])
        totals["logical_dma_calls"] += int(profile["store_buffer_2d_calls"])
        totals["invocation_host_wall_ms"] += float(row["host_wall_ms"])
    return totals


def empty_cost():
    return {
        "candidate_dispatches": 0,
        "complete_candidate_build_action_seconds": 0.0,
        "graph_api_invocations": 0,
        "driver_kernel_invocations": 0,
        "logical_load_bytes": 0,
        "logical_store_bytes": 0,
        "logical_dma_bytes": 0,
        "logical_dma_calls": 0,
        "invocation_host_wall_ms": 0.0,
    }


def add_cost(total, increment):
    for key, value in increment.items():
        total[key] += value


def candidate_cost(result, program):
    totals = empty_cost()
    totals["candidate_dispatches"] = 1
    totals["complete_candidate_build_action_seconds"] = float(
        program["complete_candidate_build_action_seconds"]
    )
    add_cost(totals, invocation_cost(result))
    return totals


def load_complete_pool(reference_online, oracle_completion, oracle_analysis):
    online = read_json(Path(reference_online) / "summary.json")
    completion = read_json(Path(oracle_completion) / "summary.json")
    analysis = read_json(Path(oracle_analysis) / "analysis.json")
    rows = list(online["candidate_results"]) + list(completion["candidate_results"])
    by_id = {row["candidate_id"]: row for row in rows}
    if len(by_id) != 12 or len(rows) != 12:
        raise ValueError("the Y10 complete label pool must contain 12 unique candidates")
    correct = [row for row in rows if row["status"] == "passed"]
    if len(correct) != 12:
        raise ValueError("the frozen Y10 attribution pool is expected to be 12/12 FPGA-correct")
    oracle = min(
        correct,
        key=lambda row: (float(row["median_paired_latency_ratio"]), row["candidate_id"]),
    )
    declared = analysis["oracle"]
    if oracle["candidate_id"] != declared["candidate_id"]:
        raise RuntimeError("reconstructed oracle identity differs from P7R441")
    if not math.isclose(
        float(oracle["median_paired_latency_ratio"]),
        float(declared["median_paired_latency_ratio"]),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("reconstructed oracle ratio differs from P7R441")
    return by_id, {
        "candidate_id": oracle["candidate_id"],
        "median_paired_latency_ratio": float(oracle["median_paired_latency_ratio"]),
        "candidate_median_latency_ms": float(declared["candidate_median_latency_ms"]),
    }


def summarize_run(path, pool_labels, oracle):
    path = Path(path)
    summary = read_json(path / "summary.json")
    contract = read_json(path / "contract.json")
    if summary.get("status") != "complete":
        raise RuntimeError("incomplete same-space run: {}".format(path))
    strategy = summary.get("strategy")
    seed = int(summary.get("seed"))
    if strategy not in EXPECTED_STRATEGIES or seed not in EXPECTED_SEEDS:
        raise ValueError("run is outside the frozen strategy/seed matrix: {}".format(path))
    if contract.get("target_label_files_read_by_selector") is not False:
        raise RuntimeError("selector label-isolation assertion is missing")
    if int(summary.get("candidate_budget")) != 6 or int(summary.get("dispatch_count")) != 6:
        raise ValueError("the frozen same-space budget is exactly six candidates")
    if summary.get("rejected_candidate_ids"):
        raise ValueError("P7R462--P7R467 are expected to have no rejected candidates")

    results = {row["candidate_id"]: row for row in summary["candidate_results"]}
    programs = {row["candidate_id"]: row for row in summary["built_programs"]}
    order = list(summary["dispatched_candidate_ids"])
    if len(set(order)) != 6 or set(order) != set(results) or set(order) != set(programs):
        raise RuntimeError("candidate order/result/program identity mismatch")
    if not set(order).issubset(pool_labels):
        raise RuntimeError("same-space run escaped the frozen complete pool")

    oracle_id = oracle["candidate_id"]
    oracle_ratio = oracle["median_paired_latency_ratio"]
    observed_best = math.inf
    pool_best = math.inf
    cumulative = empty_cost()
    curve = []
    first_exact = None
    first_plus2 = None
    for dispatch, candidate_id in enumerate(order, start=1):
        result = results[candidate_id]
        if result["status"] == "passed":
            observed_best = min(observed_best, float(result["median_paired_latency_ratio"]))
            pool_best = min(
                pool_best,
                float(pool_labels[candidate_id]["median_paired_latency_ratio"]),
            )
        add_cost(cumulative, candidate_cost(result, programs[candidate_id]))
        elapsed = float(result["outer_elapsed_seconds"])
        row = {
            "dispatch": dispatch,
            "candidate_id": candidate_id,
            "outer_elapsed_seconds": elapsed,
            "observed_best_paired_ratio": observed_best,
            "fixed_pool_best_paired_ratio": pool_best,
            "fixed_pool_regret_percent": 100.0 * (pool_best / oracle_ratio - 1.0),
            **dict(cumulative),
        }
        curve.append(row)
        if first_plus2 is None and pool_best <= 1.02 * oracle_ratio:
            first_plus2 = {
                "dispatch": dispatch,
                "outer_elapsed_seconds": elapsed,
                "candidate_id": min(
                    order[:dispatch],
                    key=lambda cid: float(pool_labels[cid]["median_paired_latency_ratio"]),
                ),
            }
        if first_exact is None and candidate_id == oracle_id:
            first_exact = {
                "dispatch": dispatch,
                "outer_elapsed_seconds": elapsed,
                "candidate_id": candidate_id,
            }

    observed_terminal = min(
        (results[candidate_id] for candidate_id in order),
        key=lambda row: (float(row["median_paired_latency_ratio"]), row["candidate_id"]),
    )
    pool_terminal_id = min(
        order,
        key=lambda cid: (float(pool_labels[cid]["median_paired_latency_ratio"]), cid),
    )
    return {
        "run_dir": str(path.resolve()),
        "artifact_manifest_sha256": sha256(path / "artifact_hashes.json"),
        "strategy": strategy,
        "seed": seed,
        "boot_id": summary["boot_id"],
        "candidate_budget": 6,
        "passed_candidates": len(summary["passed_candidate_ids"]),
        "rejected_candidates": len(summary["rejected_candidate_ids"]),
        "outer_process_seconds_before_summary_write": float(
            summary["outer_process_seconds_before_summary_write"]
        ),
        "total_cost": dict(cumulative),
        "first_oracle_plus_2pct": first_plus2,
        "first_exact_oracle": first_exact,
        "observed_terminal_best": {
            "candidate_id": observed_terminal["candidate_id"],
            "median_paired_latency_ratio": float(
                observed_terminal["median_paired_latency_ratio"]
            ),
        },
        "fixed_pool_terminal_best": {
            "candidate_id": pool_terminal_id,
            "median_paired_latency_ratio": float(
                pool_labels[pool_terminal_id]["median_paired_latency_ratio"]
            ),
            "regret_percent": 100.0
            * (
                float(pool_labels[pool_terminal_id]["median_paired_latency_ratio"])
                / oracle_ratio
                - 1.0
            ),
        },
        "curve": curve,
    }


def aggregate_strategy(runs):
    fields = {
        "outer_process_seconds": [
            row["outer_process_seconds_before_summary_write"] for row in runs
        ],
        "exact_oracle_dispatch": [
            row["first_exact_oracle"]["dispatch"]
            if row["first_exact_oracle"] is not None
            else None
            for row in runs
        ],
        "exact_oracle_seconds": [
            row["first_exact_oracle"]["outer_elapsed_seconds"]
            if row["first_exact_oracle"] is not None
            else None
            for row in runs
        ],
        "oracle_plus_2pct_dispatch": [
            row["first_oracle_plus_2pct"]["dispatch"]
            if row["first_oracle_plus_2pct"] is not None
            else None
            for row in runs
        ],
        "oracle_plus_2pct_seconds": [
            row["first_oracle_plus_2pct"]["outer_elapsed_seconds"]
            if row["first_oracle_plus_2pct"] is not None
            else None
            for row in runs
        ],
        "candidate_build_action_seconds": [
            row["total_cost"]["complete_candidate_build_action_seconds"] for row in runs
        ],
        "graph_api_invocations": [
            row["total_cost"]["graph_api_invocations"] for row in runs
        ],
        "driver_kernel_invocations": [
            row["total_cost"]["driver_kernel_invocations"] for row in runs
        ],
        "logical_dma_bytes": [row["total_cost"]["logical_dma_bytes"] for row in runs],
        "logical_dma_calls": [row["total_cost"]["logical_dma_calls"] for row in runs],
        "invocation_host_wall_ms": [
            row["total_cost"]["invocation_host_wall_ms"] for row in runs
        ],
        "observed_terminal_best_ratio": [
            row["observed_terminal_best"]["median_paired_latency_ratio"] for row in runs
        ],
        "fixed_pool_terminal_regret_percent": [
            row["fixed_pool_terminal_best"]["regret_percent"] for row in runs
        ],
    }
    return {
        "run_count": len(runs),
        "seeds": sorted(row["seed"] for row in runs),
        "all_runs_hit_oracle_plus_2pct": all(
            row["first_oracle_plus_2pct"] is not None for row in runs
        ),
        "all_runs_hit_exact_oracle": all(row["first_exact_oracle"] is not None for row in runs),
        "metrics": {key: numeric_summary(values) for key, values in fields.items()},
    }


def write_curves_csv(path, runs):
    fieldnames = [
        "strategy",
        "seed",
        "dispatch",
        "candidate_id",
        "outer_elapsed_seconds",
        "observed_best_paired_ratio",
        "fixed_pool_best_paired_ratio",
        "fixed_pool_regret_percent",
        "candidate_dispatches",
        "complete_candidate_build_action_seconds",
        "graph_api_invocations",
        "driver_kernel_invocations",
        "logical_load_bytes",
        "logical_store_bytes",
        "logical_dma_bytes",
        "logical_dma_calls",
        "invocation_host_wall_ms",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted(runs, key=lambda row: (row["strategy"], row["seed"])):
            for point in run["curve"]:
                writer.writerow(
                    {"strategy": run["strategy"], "seed": run["seed"], **point}
                )


def make_plots(output, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"dma_prior": "DMA prior", "mode_aware_xgb": "Mode-aware XGB"}
    colors = {"dma_prior": "#1f77b4", "mode_aware_xgb": "#d62728"}
    for x_key, x_label, filename in (
        ("dispatch", "FPGA candidate dispatch", "best_so_far_vs_dispatch.png"),
        ("outer_elapsed_seconds", "Elapsed wall time (s)", "best_so_far_vs_wall.png"),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        for strategy in EXPECTED_STRATEGIES:
            selected = sorted(
                (row for row in runs if row["strategy"] == strategy),
                key=lambda row: row["seed"],
            )
            for index, run in enumerate(selected):
                axis.step(
                    [point[x_key] for point in run["curve"]],
                    [point["fixed_pool_regret_percent"] for point in run["curve"]],
                    where="post",
                    color=colors[strategy],
                    alpha=0.42,
                    linewidth=1.4,
                    label=labels[strategy] if index == 0 else None,
                )
        axis.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0, label="exact oracle")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Best-so-far fixed-pool regret (%)")
        largest = max(
            point["fixed_pool_regret_percent"] for run in runs for point in run["curve"]
        )
        axis.set_ylim(-0.02, max(0.1, largest * 1.25))
        axis.set_title("All six runs enter the oracle +2% band at dispatch 1")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def fmt_metric(metric, digits=3):
    if metric is None:
        return "n/a"
    return "{:.{}f} ({:.{}f}--{:.{}f})".format(
        metric["median"], digits, metric["min"], digits, metric["max"], digits
    )


def write_report(path, aggregate):
    dma = aggregate["strategies"]["dma_prior"]["metrics"]
    xgb = aggregate["strategies"]["mode_aware_xgb"]["metrics"]
    exact_time_reduction = 100.0 * (
        1.0 - dma["exact_oracle_seconds"]["median"] / xgb["exact_oracle_seconds"]["median"]
    )
    exact_time_speedup = (
        xgb["exact_oracle_seconds"]["median"] / dma["exact_oracle_seconds"]["median"]
    )
    lines = [
        "# P7R468：Y10同空间在线选择器归因聚合",
        "",
        "状态：`COMPLETE; LABELS_PREEXISTED_ATTRIBUTION_ONLY`",
        "",
        "六次运行使用相同12点候选池、完整YOLOv3-tiny-320构建器、逐候选clean-start、",
        "三输入八输出正确性和七轮candidate/stock配对；每种策略三个独立seed、固定budget=6。",
        "选择期间不读取旧标签，完成后才按不可变候选身份连接P7R435/P7R440完整池。",
        "",
        "| 指标，中位数（min--max） | DMA prior | mode-aware XGB |",
        "|---|---:|---:|",
        "| 完整外层墙钟/s | {} | {} |".format(
            fmt_metric(dma["outer_process_seconds"]), fmt_metric(xgb["outer_process_seconds"])
        ),
        "| exact oracle首次dispatch | {} | {} |".format(
            fmt_metric(dma["exact_oracle_dispatch"], 1),
            fmt_metric(xgb["exact_oracle_dispatch"], 1),
        ),
        "| exact oracle首次墙钟/s | {} | {} |".format(
            fmt_metric(dma["exact_oracle_seconds"]), fmt_metric(xgb["exact_oracle_seconds"])
        ),
        "| oracle+2%首次dispatch | {} | {} |".format(
            fmt_metric(dma["oracle_plus_2pct_dispatch"], 1),
            fmt_metric(xgb["oracle_plus_2pct_dispatch"], 1),
        ),
        "| oracle+2%首次墙钟/s | {} | {} |".format(
            fmt_metric(dma["oracle_plus_2pct_seconds"]),
            fmt_metric(xgb["oracle_plus_2pct_seconds"]),
        ),
        "| 候选构建action/s | {} | {} |".format(
            fmt_metric(dma["candidate_build_action_seconds"]),
            fmt_metric(xgb["candidate_build_action_seconds"]),
        ),
        "| graph API调用 | {} | {} |".format(
            fmt_metric(dma["graph_api_invocations"], 1),
            fmt_metric(xgb["graph_api_invocations"], 1),
        ),
        "| driver kernel调用 | {} | {} |".format(
            fmt_metric(dma["driver_kernel_invocations"], 1),
            fmt_metric(xgb["driver_kernel_invocations"], 1),
        ),
        "| 逻辑DMA字节 | {} | {} |".format(
            fmt_metric(dma["logical_dma_bytes"], 1), fmt_metric(xgb["logical_dma_bytes"], 1)
        ),
        "| 逻辑DMA calls | {} | {} |".format(
            fmt_metric(dma["logical_dma_calls"], 1), fmt_metric(xgb["logical_dma_calls"], 1)
        ),
        "",
        "完整池oracle为`{}`，历史paired ratio为`{:.9f}`。".format(
            aggregate["oracle"]["candidate_id"],
            aggregate["oracle"]["median_paired_latency_ratio"],
        ),
        "固定DMA顺序三次均在第3次dispatch遇到exact oracle；mode-aware XGB的exact位置随",
        "seed变化，为4/6/6次。exact time-to-quality中位数由XGB的`{:.3f} s`降至DMA prior的".format(
            xgb["exact_oracle_seconds"]["median"]
        ),
        "`{:.3f} s`，减少`{:.2f}%`、提前`{:.2f}×`。两种方法固定执行相同六个候选，因此完整".format(
            dma["exact_oracle_seconds"]["median"], exact_time_reduction, exact_time_speedup
        ),
        "墙钟和总执行成本主要反映板端漂移，",
        "真正的算法差异应看time-to-quality，而不能用budget结束时的总成本强行分胜负。",
        "",
        "本实验发生在Y10完整池标签已经存在之后，只能作为选择器同空间、同执行路径的成本重演",
        "与归因消融；不能重新表述为prospective holdout。逻辑LOAD/STORE是runtime语义计数，",
        "不是物理AXI burst。图中的质量来自运行结束后的固定完整池身份连接；本次重测ratio另存于",
        "`analysis.json`，没有反向影响选择过程。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--oracle-analysis", required=True)
    parser.add_argument("--reference-online", required=True)
    parser.add_argument("--oracle-completion", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run(args):
    if len(args.run) != 6:
        raise ValueError("the frozen attribution matrix contains exactly six runs")
    inputs = [Path(path).resolve() for path in args.run]
    oracle_analysis = Path(args.oracle_analysis).resolve()
    reference_online = Path(args.reference_online).resolve()
    oracle_completion = Path(args.oracle_completion).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))

    verified = {
        str(path): verify_manifest(path)
        for path in inputs + [oracle_analysis, reference_online, oracle_completion]
    }
    pool_labels, oracle = load_complete_pool(
        reference_online, oracle_completion, oracle_analysis
    )
    runs = [summarize_run(path, pool_labels, oracle) for path in inputs]
    matrix = {(row["strategy"], row["seed"]) for row in runs}
    expected = {
        (strategy, seed) for strategy in EXPECTED_STRATEGIES for seed in EXPECTED_SEEDS
    }
    if matrix != expected:
        raise ValueError("run matrix is incomplete or duplicated")
    candidate_universes = {
        tuple(sorted(read_json(path / "contract.json")["candidate_ids"])) for path in inputs
    }
    if len(candidate_universes) != 1 or set(next(iter(candidate_universes))) != set(pool_labels):
        raise RuntimeError("the six runs do not bind the same complete candidate universe")

    aggregate = {
        "schema": "c3_y10_same_space_online_ablation_aggregate_v1",
        "status": "complete_labels_preexisted_attribution_only",
        "workload_id": "Y10",
        "candidate_universe_size": 12,
        "candidate_budget_per_run": 6,
        "oracle": oracle,
        "strategies": {
            strategy: aggregate_strategy([row for row in runs if row["strategy"] == strategy])
            for strategy in EXPECTED_STRATEGIES
        },
        "runs": sorted(runs, key=lambda row: (row["strategy"], row["seed"])),
        "verified_artifact_counts": verified,
        "bound_artifact_manifests": {
            str(path): sha256(path / "artifact_hashes.json")
            for path in inputs + [oracle_analysis, reference_online, oracle_completion]
        },
        "analyzer_source_sha256": sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Y10 labels existed before these six independent cost reexecutions. Selectors did not "
            "read them; the complete-pool labels are joined only after execution for fixed-quality "
            "attribution. This is not a prospective holdout. Runtime DMA counters are logical VTA "
            "operations, not physical AXI traffic."
        ),
    }

    output.mkdir(parents=True)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "analysis.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_curves_csv(output / "curves.csv", runs)
    make_plots(output, runs)
    write_report(output / "REPORT.md", aggregate)
    artifacts = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": aggregate["status"],
        "output_dir": str(output),
        "oracle": oracle,
        "strategies": aggregate["strategies"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
