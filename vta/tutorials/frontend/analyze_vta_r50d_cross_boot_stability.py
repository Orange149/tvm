#!/usr/bin/env python3
"""Compare two R50D full-pool audits and expose invalid-dominator instability."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = read(directory / "artifact_hashes.json")["artifacts"]
    for relative, expected in artifacts.items():
        if sha(directory / relative) != expected:
            raise ValueError("artifact mismatch: " + relative)
    return sha(directory / "artifact_hashes.json")


def axes(program):
    traffic = program["traffic"]
    return (
        traffic["load_buffer_2d_bytes"] + traffic["store_buffer_2d_bytes"],
        traffic["load_buffer_2d_calls"] + traffic["store_buffer_2d_calls"],
        program["extra_submissions"],
    )


def dominates(left, right):
    return all(x <= y for x, y in zip(left, right)) and any(x < y for x, y in zip(left, right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-front", type=Path, required=True)
    parser.add_argument("--new-front", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bindings = {
        "pool": verify(args.pool), "old": verify(args.old), "new": verify(args.new),
        "old_front": verify(args.old_front), "new_front": verify(args.new_front),
    }
    pool = read(args.pool / "pool.json")
    audits = [read(args.old / "analysis.json"), read(args.new / "analysis.json")]
    if any(len(item["candidate_table"]) != 18 for item in audits):
        raise ValueError("not the same completed 18-program pool")
    if len({item["bound_inputs"]["pool_artifact_manifest_sha256"] for item in audits}) != 1:
        raise ValueError("audits bind different pools")

    front_summaries = [read(args.old_front / "summary.json"), read(args.new_front / "summary.json")]
    boot_rows = []
    for audit, front_summary in zip(audits, front_summaries):
        if audit["bound_inputs"]["front_artifact_manifest_sha256"] != sha(
            (args.old_front if audit is audits[0] else args.new_front) / "artifact_hashes.json"
        ):
            raise ValueError("audit does not bind supplied front result")
        by_id = {row["candidate_id"]: row for row in audit["candidate_table"]}
        passed_front = [row for row in audit["candidate_table"] if row["pareto_front"] and row["fpga_status"] == "passed"]
        front_best = min(passed_front, key=lambda row: row["median_paired_latency_ratio"])
        oracle = by_id[audit["oracle_candidate_id"]]
        boot_rows.append({
            "boot_id": front_summary["boot_id"],
            "correct_candidates": sum(row["fpga_status"] == "passed" for row in audit["candidate_table"]),
            "oracle_candidate_id": oracle["candidate_id"],
            "oracle_latency_ms": oracle["candidate_median_latency_ms"],
            "oracle_paired_ratio": oracle["median_paired_latency_ratio"],
            "front_best_candidate_id": front_best["candidate_id"],
            "front_best_latency_ms": front_best["candidate_median_latency_ms"],
            "front_best_paired_ratio": front_best["median_paired_latency_ratio"],
            "front_retains_exact_oracle": audit["front_retains_full_fpga_correct_pool_oracle"],
            "front_absolute_latency_regret_percent": 100.0 * (front_best["candidate_median_latency_ms"] / oracle["candidate_median_latency_ms"] - 1.0),
            "front_paired_ratio_regret_percent": 100.0 * (front_best["median_paired_latency_ratio"] / oracle["median_paired_latency_ratio"] - 1.0),
            "front_within_oracle_2pct": front_best["median_paired_latency_ratio"] <= oracle["median_paired_latency_ratio"] * 1.02,
        })

    latest = audits[1]
    latest_status = {row["candidate_id"]: row["fpga_status"] for row in latest["candidate_table"]}
    rejected_front = {cid for cid in pool["pareto_front_candidate_ids"] if latest_status[cid] != "passed"}
    remaining = [row for row in pool["programs"] if row["candidate_id"] not in rejected_front]
    peeled_front = [
        row for row in remaining
        if not any(dominates(axes(other), axes(row)) for other in remaining if other is not row)
    ]
    newly_exposed = sorted(set(row["candidate_id"] for row in peeled_front) - set(pool["pareto_front_candidate_ids"]))
    new_oracle = latest["oracle_candidate_id"]
    dominators = [
        row["candidate_id"] for row in pool["programs"]
        if dominates(axes(row), axes(next(item for item in pool["programs"] if item["candidate_id"] == new_oracle)))
    ]
    result = {
        "schema": "c3_r50d_cross_boot_stability_v1",
        "status": "correctness_stable_quality_band_stable_exact_oracle_not_stable",
        "bindings": bindings,
        "boot_rows": boot_rows,
        "same_correct_candidate_count": boot_rows[0]["correct_candidates"] == boot_rows[1]["correct_candidates"] == 15,
        "front_exact_oracle_boots": sum(row["front_retains_exact_oracle"] for row in boot_rows),
        "front_oracle_2pct_boots": sum(row["front_within_oracle_2pct"] for row in boot_rows),
        "current_oracle_static_dominators": dominators,
        "current_oracle_dominated_only_by_rejected_front": bool(dominators) and set(dominators).issubset(rejected_front),
        "rejected_front_candidate_ids": sorted(rejected_front),
        "invalid_dominator_peeled_front_candidate_ids": sorted(row["candidate_id"] for row in peeled_front),
        "new_candidates_after_invalid_dominator_peeling": newly_exposed,
        "peeling_would_admit_current_oracle": new_oracle in newly_exposed,
        "claim_boundary": "Post-hoc cross-boot audit. Invalid-dominator peeling is a development repair suggested by exposed current-boot correctness/latency and requires a new frozen holdout before becoming a method claim.",
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(f"""# P7R384：R50D Pareto 跨启动稳定性审计

同一冻结 18 点最终融合程序池在两个独立 boot 上均为 15/18 FPGA-correct，说明三项 weight-barrier
失败分类稳定。旧 boot 的四点静态前沿包含精确 pool oracle；当前 boot 的精确 oracle 改为前沿外
R50DF04 weight-barrier (`{new_oracle[:12]}`)，因此固定前沿只在 1/2 个 boot 保留精确 oracle。

当前 boot 的前沿最佳仍是 `{boot_rows[1]['front_best_candidate_id'][:12]}`，绝对中位 latency 仅比
新 oracle 慢 {boot_rows[1]['front_absolute_latency_regret_percent']:.4f}%，按预注册主指标
candidate/stock paired ratio 的 regret 为 {boot_rows[1]['front_paired_ratio_regret_percent']:.4f}%，
所以两个 boot 都保留 oracle+2% 质量带。结论应从“跨启动保留精确 oracle”降为“跨启动保持 2%
等价质量”，不能隐藏这次 exact-oracle miss。

根因不是随机哈希：当前 oracle 只被前沿中的 FPGA-invalid 候选 `{dominators[0][:12]}` 在
`(bytes,calls,extra submissions)` 上支配。若在第一波 correctness 后移除两个 invalid front
候选并重新计算前沿，唯一新增点就是当前 oracle；这提示“invalid-dominator peeling”作为下一版
多保真动作。但该规则是看到本次标签后得到的开发修正，必须在新 workload 上先冻结再验证。
""")
    artifacts = {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}
    (args.output / "artifact_hashes.json").write_text(json.dumps({"artifacts": artifacts, "source_sha256": sha(__file__)}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
