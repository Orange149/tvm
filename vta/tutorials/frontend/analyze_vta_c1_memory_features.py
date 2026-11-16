#!/usr/bin/env python3
"""Test existing C1 memory ranking features and replay archived board evidence.

Current-DP static decisions and the historical, exactly matched board cohort are
separate experiments.  No new board calls, compilation, or fitted penalties.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import statistics
from pathlib import Path

import analyze_vta_shared_memory_topk_board as board
import freeze_vta_c1_unified_structured_order as unified
from shared_memory_topk import (
    copy_realization, order_within_m1_band, policy_feature_row, read_json,
    sha256_file, shared_realization, topology_diverse, validate_topology,
)


AUDIT = unified.REPORT / "compile_context_audit"
BOOT_PATHS = [AUDIT / (name + "/boot.json") for name in (
    "c1_shared_memory_topk_board_boot0_run6", "c1_shared_memory_topk_board_boot1_run9",
    "c1_shared_memory_topk_board_boot2_run1",
)]
DEPLOYMENT = AUDIT / "c1_shared_memory_topk_board_prepare_run8/deployment_manifest.json"
BOUNDARY = ("framework_materialization_bytes", "additional_slot_bytes")
BYTES = ("vta_load_inp_bytes", "vta_load_wgt_bytes", "vta_load_acc_bytes", "vta_store_out_bytes")
REQUESTS = tuple(d for d in unified.UNIFIED_DIMENSIONS if (
    d.startswith("vta_") and ("calls" in d or "density" in d)
))
GROUPS = {
    "no_memory": (),
    "boundary": BOUNDARY,
    "bytes": BOUNDARY + BYTES,
    "requests_only": REQUESTS,
    "full": unified.UNIFIED_DIMENSIONS,
    "full_no_requests": tuple(d for d in unified.UNIFIED_DIMENSIONS if d not in REQUESTS),
    "full_no_boundary": tuple(d for d in unified.UNIFIED_DIMENSIONS if d not in BOUNDARY),
}
GROUP_LABELS = {
    "no_memory": "仅M1", "boundary": "边界物化/slot", "bytes": "边界+DMA字节",
    "requests_only": "仅DMA请求形态", "full": "完整内存特征",
    "full_no_requests": "完整特征去请求项", "full_no_boundary": "完整特征去边界项",
}


def snapshot(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def freeze_orders(rows, budget=5):
    if not rows or any(not r["unified_feature_exact"] for r in rows):
        raise ValueError("exact DMA/fusion features required; unknown is not zero")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate candidate identity")
    for row in rows:
        for dimension in unified.UNIFIED_DIMENSIONS:
            value = row[dimension]
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError("invalid memory feature: " + dimension)
    result = {}
    for name, dimensions in GROUPS.items():
        ordered = order_within_m1_band(rows, 0.02, dimensions)
        ids = [unified.configuration_id(r) for r in ordered]
        result[name] = {
            "dimensions": list(dimensions), "full_order_sha256": unified.seal({"ids": ids})["artifact_sha256"],
            "selected_configuration_ids": ids[:budget], "selected_rows": ordered[:budget],
        }
    return result


def prefix_metrics(order, labels, oracle):
    missing = [key for key in order if key not in labels]
    if missing:
        return {"status": "missing_exact_labels", "missing": missing, "curves": None}
    if oracle is None or not math.isfinite(oracle) or oracle <= 0 or any(
        not math.isfinite(labels[key]) or labels[key] <= 0 for key in order
    ):
        raise ValueError("positive finite timing labels and oracle required")
    curves, first = [], None
    for k in range(1, len(order) + 1):
        best = min(labels[key] for key in order[:k])
        hit = best <= oracle * 1.02
        if hit and first is None:
            first = k
        curves.append({"k": k, "best_ii_ms": best, "hit": hit, "regret": best / oracle - 1})
    return {"status": "complete", "first_oracle_plus_2pct_index": first, "curves": curves}


def exact_fixed_first_random(pool, first, labels, oracle):
    if first not in pool or len(pool) > 8:
        return {"status": "unsupported_pool"}
    hits = []
    for rest in itertools.permutations(sorted(set(pool) - {first})):
        hit = prefix_metrics([first] + list(rest), labels, oracle)["first_oracle_plus_2pct_index"]
        hits.append(hit)
    return {"status": "complete", "permutations": len(hits),
            "mean_first_hit": statistics.mean(hits) if all(x is not None for x in hits) else None,
            "first_hit_distribution": {str(x): hits.count(x) for x in sorted(set(hits), key=str)}}


def current_dp_rows(topologies, estimates, scoring_mode="current"):
    # Reuse the existing exhaustive traversal, retaining the best thread vector
    # per topology under the *current* score. VTA/edge byte features depend on the
    # same topology; old policy certificates are never promoted to new threads.
    import solve_cpu_vta_pipeline_v1_p3 as p3

    paths = [p3.DEFAULT_OUTPUT / name for name in (
        "v1_unit_and_boundary_schema.json", "v1_profile_manifest.json", "v1_local_cost_table.json",
    )]
    sources = [snapshot(path) for path in paths] + [snapshot(p3.__file__)]
    schema, profile, costs = [read_json(path) for path in paths]
    for payload in (schema, profile, costs):
        p3.validate_artifact_sha256(payload)
    context = p3.build_context(profile, costs)
    best, count = {}, 0
    for scheme in p3.enumerate_reachable(schema)[0]:
        cpu_count = sum(stage["device"] == "cpu" for stage in scheme)
        for allocation in p3.positive_allocations(cpu_count):
            path = p3.path_from_scheme(scheme, allocation)
            label = p3.label_from_path(path, context, scoring_mode=scoring_mode)
            tid = p3.topology_id(path)
            if tid not in best or p3.rank_key(label) < p3.rank_key(best[tid]):
                best[tid] = label
            count += 1
    if set(best) != {t["topology_id"] for t in topologies}:
        raise ValueError("current DP and compiled-feature topology domains differ")
    rows = []
    for topology in topologies:
        label = best[topology["topology_id"]]
        current = copy.deepcopy(topology)
        current.update(candidate_id=p3.candidate_id(label.path), path=label.path, m1_score_ms=label.score)
        row = make_row(current, [], estimates)
        row["compiled_feature_source_candidate_id"] = topology["candidate_id"]
        row["resource_load_ms"] = p3.candidate_record(label, 0, context)["resource_load_ms"]
        rows.append(row)
    return rows, {"execution_configurations": count, "topologies": len(rows),
                  "scoring_mode": scoring_mode,
                  "policy": "ordinary_copy_only", "measured_labels_joined": False,
                  "sources": sources}


def make_row(topology, certificates, estimates):
    realization = (shared_realization(topology, unified.select_certificate(topology["candidate_id"], certificates))
                   if certificates else copy_realization(topology))
    row = policy_feature_row(topology, realization)
    fusion = unified.fusion_summary(topology, estimates)
    if row["cpu_helper_logical_bytes"] != fusion["cpu_read_bytes"] + fusion["cpu_write_bytes"]:
        raise ValueError("compiler helper-byte provenance mismatch")
    row.update(convert_calls=fusion["convert_calls"], cpu_helper_calls=fusion["cpu_helper_calls"],
               unified_feature_exact=fusion["exact_reuse"] and row["detailed_dma_complete"])
    return row


def replay_board():
    contract = read_json(unified.DEFAULT_LEGACY_CONTRACT)
    contract_hash = sha256_file(unified.DEFAULT_LEGACY_CONTRACT)
    deployment = read_json(DEPLOYMENT)
    if deployment["selection_contract"]["sha256"] != contract_hash:
        raise ValueError("deployment contract binding differs")
    boots = [board.validate_boot_artifacts(p, deployment, sha256_file(DEPLOYMENT)) for p in BOOT_PATHS]
    sources, replayed = [snapshot(DEPLOYMENT)] + [snapshot(p) for p in BOOT_PATHS], []
    protocol = deployment["performance_protocol"]
    for path, boot in zip(BOOT_PATHS, boots):
        indexed = {row["configuration_id"]: row for row in boot["rows"]}
        for position, key in enumerate(boot["configuration_order"]):
            row = indexed[key]
            token = "cfg{:02d}_{}".format(position, board.sha256_text(key)[:12])
            perf = path.parent / (token + "_performance.jsonl")
            profile_path = path.parent / (token + "_profile.json")
            frames = [json.loads(x) for x in perf.read_text().splitlines() if x.strip()]
            if len(frames) != protocol["runs"]:
                raise ValueError("performance frame coverage mismatch")
            frames.sort(key=lambda f: int(f["completion_index"]))
            if [int(f["completion_index"]) for f in frames] != list(range(len(frames))):
                raise ValueError("completion indices incomplete")
            scored = frames[protocol["discard_first_scored_rows"]:]
            ii = (scored[-1]["completion_ms"] - scored[0]["completion_ms"]) / (len(scored) - 1)
            profile = read_json(profile_path)
            dma_bytes = (profile["load_buffer_2d_bytes"] + profile["store_buffer_2d_bytes"]) / len(frames)
            dma_calls = (profile["load_buffer_2d_calls"] + profile["store_buffer_2d_calls"]) / len(frames)
            stage_count = len(row["candidate_id"].split("__"))
            api = statistics.median(sum(f["stage{}_get_ms".format(i)] + f["stage{}_set_ms".format(i + 1)]
                                        for i in range(stage_count - 1)) for f in frames)
            for observed, expected in ((ii, row["ii_ms"]), (api, row["boundary_api_service_ms"]),
                                       (dma_bytes, row["vta_internal_dma_bytes"]), (dma_calls, row["vta_dma_calls"])):
                if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("raw metric differs from boot record")
            replayed.append({"boot_id": boot["boot_id"], "configuration_id": key,
                             "ii_ms": ii, "boundary_api_service_ms": api,
                             "dma_bytes": dma_bytes, "dma_calls": dma_calls})
            sources.extend([snapshot(perf), snapshot(profile_path)])
    boot_ids, aggregate = board.aggregate_boots(boots, contract, contract_hash)
    return aggregate, {"boot_ids": boot_ids, "raw_windows_recomputed": len(replayed),
                       "points": replayed, "sources": sources,
                       "raw_archives_and_serial_output_bytes_verified": True,
                       "current_board_or_runtime_requalified": False}


def evaluate_cohort(rows, orders, measurements, baseline_group="no_memory"):
    keys = {unified.configuration_id(row) for row in rows}
    labels = {key: row["ii_ms_median"] for key, row in measurements.items() if key in keys and row["correct"]}
    oracle = min(labels.values()) if labels else None
    for row in rows:
        key = unified.configuration_id(row)
        if key not in measurements:
            continue
        measured = measurements[key]
        for field, expected in (("vta_internal_dma_bytes", row["vta_internal_dma_bytes"]),
                                ("vta_dma_calls", row["vta_dma_calls"]),
                                ("framework_materialization_bytes", row["framework_materialization_bytes"]),
                                ("additional_slot_bytes", row["additional_slot_bytes"])):
            if any(value != expected for value in measured[field]):
                raise ValueError("static/board feature mismatch: " + field)
    metrics = {name: prefix_metrics(order["selected_configuration_ids"], labels, oracle)
               for name, order in orders.items()} if labels else {}
    per_boot = []
    if labels:
        for index in range(len(next(iter(measurements.values()))["ii_ms_by_boot"])):
            boot_labels = {key: measurements[key]["ii_ms_by_boot"][index] for key in labels}
            boot_oracle = min(boot_labels.values())
            per_boot.append({"boot_index": index, "oracle_ii_ms": boot_oracle,
                             "metrics": {name: prefix_metrics(order["selected_configuration_ids"], boot_labels, boot_oracle)
                                         for name, order in orders.items()}})
    first = orders[baseline_group]["selected_configuration_ids"][0]
    return {"oracle_scope": "historical exactly matched measured configurations in this fixed-policy cohort",
            "measured_pool_size": len(labels), "oracle_ii_ms": oracle, "metrics": metrics,
            "random_fixed_first": exact_fixed_first_random(list(labels), first, labels, oracle),
            "matched_configuration_ids": sorted(labels), "per_boot": per_boot}


def run(output_dir, include_current=True):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite " + str(output_dir))
    static_paths = [unified.DEFAULT_TOPOLOGIES, unified.DEFAULT_EDGE_CONTEXTS,
                    unified.DEFAULT_CERTIFICATES, unified.DEFAULT_FUSION_ESTIMATES,
                    unified.DEFAULT_LEGACY_CONTRACT, Path(__file__), Path(unified.__file__),
                    Path(board.__file__), Path(unified.__file__).with_name("shared_memory_topk.py")]
    sources = [snapshot(path) for path in static_paths]
    topologies = topology_diverse(read_json(unified.DEFAULT_TOPOLOGIES))
    edges = read_json(unified.DEFAULT_EDGE_CONTEXTS)
    for topology in topologies:
        validate_topology(topology, edges)
    estimates = {r["segment_id"]: r for r in read_json(unified.DEFAULT_FUSION_ESTIMATES)["rows"]}
    certificates = read_json(unified.DEFAULT_CERTIFICATES)["certificates"]
    cohorts = {"copy": [make_row(t, [], estimates) for t in topologies],
               "certified": [make_row(t, certificates, estimates) for t in topologies]}
    orders = {name: freeze_orders(rows) for name, rows in cohorts.items()}
    current = None
    if include_current:
        print("Enumerating current DP thread optima; no board labels read", flush=True)
        current_rows, current = current_dp_rows(topologies, estimates)
        current["orders"] = freeze_orders(current_rows)
        sources.extend(current["sources"])
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = unified.seal({"groups": GROUPS, "band_fraction": 0.02, "budget": 5,
                           "legacy_orders": orders, "current_static": current,
                           "source_hashes": sources, "status": "post_hoc_orders_before_replay"})
    (output_dir / "orders.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print("Orders saved; verifying archived raw board evidence", flush=True)
    measurements, replay = replay_board()
    evaluations = {name: evaluate_cohort(rows, orders[name], measurements) for name, rows in cohorts.items()}
    pairs = board.same_candidate_policy_invariants(measurements)
    for pair in pairs:
        cid = pair["candidate_id"]
        ordinary = measurements[cid + "::ordinary_copy"]
        shared = measurements[pair["shared_configuration_id"]]
        pair.update(copy_api_ms=ordinary["boundary_api_service_ms_median"],
                    shared_api_ms=shared["boundary_api_service_ms_median"],
                    copy_ii_ms=ordinary["ii_ms_median"], shared_ii_ms=shared["ii_ms_median"])
    capacity_checks = []
    by_candidate = {t["candidate_id"]: t for t in topologies}
    eligible = [c for c in certificates if c["candidate_id"] in by_candidate and c["additional_slot_bytes"] > 0]
    for certificate in (min(eligible, key=lambda c: c["additional_slot_bytes"]),
                        max(eligible, key=lambda c: c["additional_slot_bytes"])):
        topology = by_candidate[certificate["candidate_id"]]
        demand = int(certificate["additional_slot_bytes"])
        enough = shared_realization(topology, certificate, demand)
        short = shared_realization(topology, certificate, demand - 1)
        if enough["fallback_to_copy"] or not short["fallback_to_copy"]:
            raise ValueError("capacity boundary fallback failed")
        capacity_checks.append({"candidate_id": topology["candidate_id"], "policy": certificate["policy_id"],
                                "demand_bytes": demand, "at_demand": enough["realized_policy"],
                                "one_byte_short": short["realized_policy"], "board_test": False})
    sources.extend(replay["sources"])
    for source in sources:
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("source changed during analysis: " + source["path"])
    result = unified.seal({"schema": "c1_memory_features_retrospective_v1", "new_board_calls": 0,
        "orders_sha256": sha256_file(output_dir / "orders.json"), "domain_topologies": len(topologies),
        "evaluations": evaluations, "board_replay": replay, "same_candidate_policy_pairs": pairs,
        "capacity_software_checks": capacity_checks, "sources": sources,
        "limitations": ["这是事后复核，不是新采集的前瞻选型证据。",
            "历史M1输入不同于当前CPU核池评分；没有给当前新候选套用历史性能标签。",
            "历史候选各自匹配自身串行参考，尚未证明它们来自同一份一次量化图，因此跨候选性能比较仅作描述性线索。",
            "每个固定策略组只有5个实测拓扑；池最优不是全域最优，位次也不是实测在线TTQ。",
            "当前静态特征按相同拓扑复用，策略统一copy；没有把旧shared证书转给新线程配置。"]})
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(output_dir / "REPORT.md", frozen, result)
    return result


def write_report(path, frozen, result):
    lines = ["# C1 内存特征针对性测试", "", "## 先看结论", "",
             "本次同时检查当前 DP 的静态选择，以及历史冻结候选的三启动真实记录。两套数据分开评价。",
             "内存排序沿用 2% 近优带，固定 M1 第一名；因此主要检验后续测量顺序。", ""]
    metrics = result["evaluations"]["copy"]["metrics"]
    if all(metrics[name]["status"] == "complete" for name in ("no_memory", "full", "requests_only")):
        lines.extend(["历史普通复制组找到好方案的位次：仅M1={}，完整内存特征={}，仅请求形态={}。".format(
            *(metrics[name]["first_oracle_plus_2pct_index"] for name in ("no_memory", "full", "requests_only"))),
            "好方案指该5拓扑实测池最优II的+2%范围，不是全空间最优。三次启动逐次结果见下表。", ""])
    if frozen["current_static"]:
        current = frozen["current_static"]
        base = current["orders"]["no_memory"]["selected_configuration_ids"]
        lines.extend(["## 当前 DP：加入内存特征是否改变选择", "",
                      "重新枚举 {} 个配置，按当前评分为 {} 个拓扑各取最优线程；实际策略统一 copy。".format(
                          current["execution_configurations"], current["topologies"]), "",
                      "| 内存项 | Top-5 顺序变化 | 与原 Top-5 重合数 |", "|---|---|---:|"])
        for name, order in current["orders"].items():
            ids = order["selected_configuration_ids"]
            lines.append("| {} | {} | {} |".format(GROUP_LABELS[name], "改变" if ids != base else "不变", len(set(ids) & set(base))))
        lines.extend(["", "当前新候选只有静态结果，没有匹配的板端标签；下面的历史时延不移用到这些新线程配置。", ""])
    lines.extend(["## 历史实测：保持每个候选的实际策略不变，只改变测量顺序", "",
                  "no_memory=仅M1；boundary=边界物化/slot；bytes=边界+分存储域字节；requests_only=仅DMA请求形态；",
                  "full=现有完整结构化特征；其余为去项消融。各组都保留第一名，未改带宽或拟合新权重。", "",
                  "| 固定策略组 | 内存项 | 首次进入池最优+2%的位次 | Regret@3 | Regret@5 |",
                  "|---|---|---:|---:|---:|"])
    for cohort, evaluation in result["evaluations"].items():
        for name, metric in evaluation["metrics"].items():
            if metric["status"] == "complete":
                lines.append("| {} | {} | {} | {:.2%} | {:.2%} |".format(
                    "普通复制" if cohort == "copy" else "固定认证策略", GROUP_LABELS[name], metric["first_oracle_plus_2pct_index"],
                    metric["curves"][2]["regret"], metric["curves"][4]["regret"]))
            else:
                lines.append("| {} | {} | 缺精确标签 | — | — |".format(cohort, GROUP_LABELS[name]))
    for cohort, evaluation in result["evaluations"].items():
        lines.extend(["", "{}：匹配首点的随机排列首次命中平均 {}；穷举 {} 种排列。".format(
            cohort, evaluation["random_fixed_first"].get("mean_first_hit"),
            evaluation["random_fixed_first"].get("permutations")), ""])
    lines.extend(["## 逐启动复核（首次命中位次）", "",
                  "| 策略组 | 启动 | 仅M1 | 仅请求形态 | 完整内存特征 |", "|---|---:|---:|---:|---:|"])
    for cohort, evaluation in result["evaluations"].items():
        for point in evaluation["per_boot"]:
            hits = [point["metrics"][name].get("first_oracle_plus_2pct_index")
                    for name in ("no_memory", "requests_only", "full")]
            lines.append("| {} | {} | {} | {} | {} |".format(cohort, point["boot_index"], *hits))
    lines.extend(["## 同一程序 copy/shared 的真实机制对照", "",
                  "| 候选 | copy边界API/ms | shared边界API/ms | copy II/ms | shared II/ms | 内部DMA不变 |",
                  "|---|---:|---:|---:|---:|---|"])
    for row in result["same_candidate_policy_pairs"]:
        lines.append("| `{}` | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
            row["candidate_id"], row["copy_api_ms"], row["shared_api_ms"], row["copy_ii_ms"],
            row["shared_ii_ms"], row["vta_internal_dma_equal"] and row["vta_dma_calls_equal"]))
    lines.extend(["", "共享物化均为0；局部API服务下降不自动等于流水吞吐提升。以上为三boot中位数描述。",
                  "", "## 校验与边界", "",
                  "重新核对27个窗口的原始JSONL、DMA profile、完整输出归档和串行参考，复算II与边界API服务。",
                  "容量软件检查：精确需求时允许shared，少1字节回退copy；不是新增板端容量实验。",
                  "新增板端调用0；只读复用旧记录，没有访问或改动当前开发板。", ""])
    lines.extend("- " + text for text in result["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-only", action="store_true", help="skip current DP enumeration")
    args = parser.parse_args()
    result = run(args.output_dir, not args.legacy_only)
    print(json.dumps({name: {method: metric.get("first_oracle_plus_2pct_index")
                            for method, metric in row["metrics"].items()}
                      for name, row in result["evaluations"].items()}, indent=2))


if __name__ == "__main__":
    main()
