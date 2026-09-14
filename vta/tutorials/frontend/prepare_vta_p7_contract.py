#!/usr/bin/env python3
"""Freeze the P7 grouped-holdout pool and label-free search policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from c3_candidate_identity import canonical_json_bytes


SCHEMA = "c3_p7_grouped_holdout_contract_v1"
HOLDOUTS = ("W01", "W04", "W07", "W08")
EXPECTED_COUNTS = {"W01": 19, "W04": 19, "W07": 21, "W08": 21}
SEEDS = (0, 20250901, 20260910)
RANDOM_SEED = 20260911
XGB_SEED = 20250901

DMA_OBJECTIVES = (
    "dma_total_bytes",
    "dma_total_calls",
    "input_dma_bytes",
    "input_dma_calls",
    "weight_dma_bytes",
    "weight_dma_calls",
    "output_dma_bytes",
    "output_dma_calls",
    "input_reload",
    "weight_reload",
    "output_reload",
    "small_dma_calls",
    "strided_dma_calls",
    "padded_dma_calls",
)
COMMAND_OBJECTIVES = (
    "submissions",
    "total_insn_bytes",
    "peak_insn_bytes",
    "total_uop_bytes",
    "peak_uop_bytes",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ledger_files(value):
    return value.get("output_sha256", value.get("artifacts", value))


def verify_run(run_dir, required):
    run_dir = Path(run_dir).resolve()
    hashes = ledger_files(load_json(run_dir / "artifact_hashes.json"))
    for name in required:
        if hashes.get(name) != file_sha256(run_dir / name):
            raise ValueError("frozen hash mismatch: {} {}".format(run_dir, name))
    return run_dir


def by_id(rows, label):
    result = {row["candidate_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate candidate IDs in {}".format(label))
    return result


def distinct_rank(values):
    ordered = sorted(set(values))
    if len(ordered) == 1:
        return {ordered[0]: 0.0}
    return {value: index / float(len(ordered) - 1) for index, value in enumerate(ordered)}


def mean_rank(rows, names):
    ranks = {name: distinct_rank([row["metrics"][name] for row in rows]) for name in names}
    return {
        row["candidate_id"]: sum(ranks[name][row["metrics"][name]] for name in names) / len(names)
        for row in rows
    }


def dominates(left, right, names):
    lhs = [left["metrics"][name] for name in names]
    rhs = [right["metrics"][name] for name in names]
    return all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs))


def pareto_order(rows, names):
    remaining = list(rows)
    ordered = []
    while remaining:
        front = [
            row
            for row in remaining
            if not any(
                other["candidate_id"] != row["candidate_id"]
                and dominates(other, row, names)
                for other in remaining
            )
        ]
        scores = mean_rank(front, names)
        front.sort(key=lambda row: (scores[row["candidate_id"]], row["candidate_id"]))
        ordered.extend(front)
        selected = {row["candidate_id"] for row in front}
        remaining = [row for row in remaining if row["candidate_id"] not in selected]
    return [row["candidate_id"] for row in ordered]


def stratify_modes(candidate_ids, index):
    groups = defaultdict(list)
    for candidate_id in candidate_ids:
        groups[index[candidate_id]["residence_mode"]].append(candidate_id)
    modes = sorted(groups, key=lambda mode: (candidate_ids.index(groups[mode][0]), mode))
    result = []
    offset = 0
    while len(result) < len(candidate_ids):
        for mode in modes:
            if offset < len(groups[mode]):
                result.append(groups[mode][offset])
        offset += 1
    return result


def split_values(record):
    entities = {name: value for name, _, value in record["identity"]["complete_config_entity"]["entity"]}
    result = {}
    for name in ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co"):
        result[name] = int(entities[name][-1])
    result["oc_nthread"] = int(entities["oc_nthread"])
    result["h_nthread"] = int(entities["h_nthread"])
    return result


def output_extents(record):
    workload = record["identity"]["workload"]
    data_shape = workload[1][1]
    weight_shape = workload[2][1]
    strides, padding = workload[3], workload[4]
    height, width = int(data_shape[2]), int(data_shape[3])
    kernel_h, kernel_w = int(weight_shape[2]), int(weight_shape[3])
    out_h = (height + int(padding[0]) + int(padding[2]) - kernel_h) // int(strides[0]) + 1
    out_w = (width + int(padding[1]) + int(padding[3]) - kernel_w) // int(strides[1]) + 1
    return out_h, out_w, int(weight_shape[0]), int(weight_shape[1])


def compute_features(record):
    split = split_values(record)
    out_h, out_w, co_blocks, ci_blocks = output_extents(record)
    waves = (
        math.ceil(out_h / split["tile_h"])
        * math.ceil(out_w / split["tile_w"])
        * math.ceil(co_blocks / split["tile_co"])
        * math.ceil(ci_blocks / split["tile_ci"])
    )
    tile_work = (
        split["tile_h"] * split["tile_w"] * split["tile_co"] * split["tile_ci"]
    )
    vthreads = split["oc_nthread"] * split["h_nthread"]
    return {
        "outer_tile_waves": int(waves),
        "negative_inner_tile_work": -int(tile_work),
        "negative_virtual_threads": -int(vthreads),
        "split_knobs": split,
    }


def joint_b8_order(rows, source_index, incumbent_id):
    dma = mean_rank(rows, DMA_OBJECTIVES)
    command = mean_rank(rows, COMMAND_OBJECTIVES)
    compute = {row["candidate_id"]: compute_features(source_index[row["candidate_id"]]) for row in rows}
    compute_names = ("outer_tile_waves", "negative_inner_tile_work", "negative_virtual_threads")
    compute_ranks = {
        name: distinct_rank([compute[row["candidate_id"]][name] for row in rows])
        for name in compute_names
    }
    for row in rows:
        cid = row["candidate_id"]
        compute[cid]["group_rank"] = sum(
            compute_ranks[name][compute[cid][name]] for name in compute_names
        ) / len(compute_names)
        compute[cid]["dma_group_rank"] = dma[cid]
        compute[cid]["command_group_rank"] = command[cid]
        compute[cid]["joint_score"] = (dma[cid] + command[cid] + compute[cid]["group_rank"]) / 3.0
    order = sorted(
        (row["candidate_id"] for row in rows),
        key=lambda cid: (compute[cid]["joint_score"], cid),
    )
    order = [incumbent_id] + [cid for cid in order if cid != incumbent_id]
    return order, compute


def prepare(p5c_dir, boot_id, dmesg_after):
    p5c_dir = verify_run(
        p5c_dir, ("shortlist.json", "results.jsonl", "manifest.json", "summary.json")
    )
    shortlist = load_json(p5c_dir / "shortlist.json")
    ranked = by_id(load_jsonl(p5c_dir / "results.jsonl"), "P5c")
    p5c_manifest = load_json(p5c_dir / "manifest.json")
    source_runs = {}
    for name in ("p4b", "p4e", "p4f", "p4g", "p4i"):
        source = p5c_manifest["inputs"][name]
        run_dir = verify_run(source["run_dir"], ("results.jsonl", "manifest.json", "summary.json"))
        if file_sha256(run_dir / "results.jsonl") != source["sha256"]:
            raise ValueError("P5c source binding mismatch: {}".format(name))
        source_runs[name] = {
            "run_dir": str(run_dir),
            "results_sha256": source["sha256"],
            "manifest_sha256": file_sha256(run_dir / "manifest.json"),
        }
    p4b = load_jsonl(Path(source_runs["p4b"]["run_dir"]) / "results.jsonl")
    p4e = by_id(load_jsonl(Path(source_runs["p4e"]["run_dir"]) / "results.jsonl"), "P4e")
    p4f = by_id(load_jsonl(Path(source_runs["p4f"]["run_dir"]) / "results.jsonl"), "P4f")
    p4g = load_jsonl(Path(source_runs["p4g"]["run_dir"]) / "results.jsonl")
    p4i = by_id(load_jsonl(Path(source_runs["p4i"]["run_dir"]) / "results.jsonl"), "P4i")
    source_index = by_id(p4b + p4g, "P4b+P4g")
    base_guards = load_json(Path(source_runs["p4f"]["run_dir"]) / "manifest.json")["source_guards_after"]
    barrier_guards = load_json(Path(source_runs["p4i"]["run_dir"]) / "manifest.json")["source_guards_after"]

    workloads = {}
    total = 0
    for workload_id in HOLDOUTS:
        frozen_order = shortlist["workloads"][workload_id]["ranking_candidate_ids"]
        if len(frozen_order) != EXPECTED_COUNTS[workload_id]:
            raise ValueError("unexpected P7 pool size for {}".format(workload_id))
        rows = [ranked[cid] for cid in frozen_order]
        incumbent = shortlist["workloads"][workload_id]["protected_original_incumbent"]
        entries = []
        for candidate_id in frozen_order:
            static = source_index[candidate_id]
            ranking = ranked[candidate_id]
            if static.get("status") != "ok" or static["workload_id"] != workload_id:
                raise ValueError("selected static candidate is not valid")
            if hashlib.sha256(canonical_json_bytes(static["identity"])).hexdigest() != candidate_id:
                raise ValueError("stable candidate ID mismatch")
            if static["tir_sha256"] is None or static["residence_mode"] != ranking["residence_mode"]:
                raise ValueError("candidate metadata mismatch")
            if static["residence_mode"] == "weight_stationary_barrier":
                fsim = static["fsim_qualification"]
                cross = p4i.get(candidate_id)
                if (
                    fsim.get("overall_status") != "passed"
                    or [item["seed"] for item in fsim.get("seeds", [])] != list(SEEDS)
                    or not cross
                    or cross.get("overall_status") != "passed"
                    or cross.get("p4g_tir_sha256") != static["tir_sha256"]
                ):
                    raise ValueError("barrier qualification mismatch")
                guards = barrier_guards
                qualification = "P4g three-seed FSim + P4i AXU cross-compile"
            else:
                local, cross = p4e.get(candidate_id), p4f.get(candidate_id)
                if (
                    not local
                    or local.get("local_status") != "fsim_passed"
                    or local.get("correctness_seeds") != list(SEEDS)
                    or local.get("tir_sha256") != static["tir_sha256"]
                    or not cross
                    or cross.get("overall_status") != "passed"
                    or cross.get("tir_sha256") != static["tir_sha256"]
                ):
                    raise ValueError("base qualification mismatch")
                guards = base_guards
                qualification = "P4e three-seed FSim + P4f AXU cross-compile"
            config = dict(static["identity"]["complete_config_entity"])
            config["index"] = int(static["debug"]["config_index"])
            entries.append(
                {
                    "candidate_id": candidate_id,
                    "workload_id": workload_id,
                    "p5c_position": ranking["ranking_position"],
                    "candidate_role": static["candidate_role"],
                    "residence_mode": static["residence_mode"],
                    "template_name": static["identity"]["template_name"],
                    "config_index": int(static["debug"]["config_index"]),
                    "complete_config_entity": config,
                    "tir_sha256": static["tir_sha256"],
                    "workload": static["identity"]["workload"],
                    "qualification": qualification,
                    "source_guards_sha256": guards,
                    "static_metrics": ranking["metrics"],
                }
            )
        b4 = [row["candidate_id"] for row in p4b + p4g if row.get("candidate_id") in set(frozen_order)]
        b5 = sorted(frozen_order, key=lambda cid: (ranked[cid]["metrics"]["dma_total_bytes"], cid))
        b6 = pareto_order(rows, DMA_OBJECTIVES)
        b7 = [incumbent] + [cid for cid in stratify_modes(b6, ranked) if cid != incumbent]
        b8, compute = joint_b8_order(rows, source_index, incumbent)
        b1_candidates = [cid for cid in b8 if ranked[cid]["residence_mode"] == "paper_inspired_hybrid"]
        policies = {
            "B0": [incumbent],
            "B1": b1_candidates[:1],
            "B4": b4,
            "B5": b5,
            "B6": b6,
            "B7": b7,
            "B8": b8,
            "P5c_diagnostic": frozen_order,
        }
        if any(len(order) != len(frozen_order) for name, order in policies.items() if name not in ("B0", "B1")):
            raise ValueError("incomplete policy order")
        correctness_order = sorted(
            frozen_order,
            key=lambda cid: hashlib.sha256(
                "{}:{}:{}".format(RANDOM_SEED, workload_id, cid).encode()
            ).hexdigest(),
        )
        workloads[workload_id] = {
            "candidate_count": len(entries),
            "protected_original_incumbent": incumbent,
            "candidates": entries,
            "correctness_order": correctness_order,
            "policy_orders": policies,
            "b8_hardware_features": compute,
        }
        total += len(entries)
    if total != 80 or len({e["candidate_id"] for w in workloads.values() for e in w["candidates"]}) != 80:
        raise ValueError("P7 pool must contain exactly 80 unique candidates")
    return {
        "schema": SCHEMA,
        "status": "frozen_before_any_P7_board_label",
        "date": "2026-09-11",
        "board_contract": {
            "host": "192.168.1.247",
            "boot_id": boot_id,
            "fpga_state": "operating",
            "udmabuf_bytes": 201326592,
            "rpc_cwd_prefix": "/var/volatile/",
            "sd_p2_min_free_percent": 20,
            "storage_error_dmesg_after_seconds": float(dmesg_after),
        },
        "pool": {
            "holdouts": list(HOLDOUTS),
            "candidate_count": total,
            "counts": EXPECTED_COUNTS,
            "geometric_holdouts": "E00-E02 excluded from this run because no frozen lower/build qualification exists; no replacement geometry is selected after labels",
        },
        "measurement": {
            "correctness_seeds": list(SEEDS),
            "correctness": "exact NumPy equality for every seed before any timing",
            "correctness_batches": "one workload per immutable run; stop on first failure and do not replace candidates",
            "future_timing": "only after all 80 pass: warmup=3, five complete randomized blocks per workload, timer number=1, median; incumbent drift sentinels recorded separately",
            "budgets": [4, 8, 16, 24],
            "budget_accounting": "failed positions consume gross dispatch budget; duplicate candidates across policies share one physical measurement",
            "oracle": "pool oracle is claimable only after every candidate in the 80-candidate pool has a valid timing label",
        },
        "baselines": {
            "B0": "protected original TopHub incumbent",
            "B1": "paper-inspired hybrid selected by frozen B8 static score; not an exact 2026 reproduction",
            "B2": "1000 deterministic uniform random permutations per workload; permutation key SHA256(20260911:workload:replicate:candidate_id)",
            "B3": "sequential XGB pool tuner over ConfigEntity knobs only; four deterministic random warmup trials, then refit after every label using seed 20250901; implementation must be frozen before timing",
            "B4": "compile/FSim/AXU-valid pool in frozen source enumeration order",
            "B5": "B4 ordered by total DMA bytes then candidate_id",
            "B6": "B4 ordered by request-shape-only Pareto fronts and equal distinct-value ranks",
            "B7": "B6 plus residence-mode round robin and protected B0 first",
            "B8": "B7 concept extended with equal one-third DMA, command, and compute-structure group ranks; protected B0 first",
        },
        "b8_compute_group": {
            "outer_tile_waves": "ceil(OH/tile_h)*ceil(OW/tile_w)*ceil(CO_blocks/tile_co)*ceil(CI_blocks/tile_ci), minimize",
            "inner_tile_work": "tile_h*tile_w*tile_co*tile_ci, maximize",
            "virtual_threads": "oc_nthread*h_nthread, maximize",
            "group_weighting": "mean normalized distinct-value rank inside each group, then equal mean of DMA/command/compute groups",
        },
        "random_policy": {"replicates": 1000, "seed": RANDOM_SEED, "algorithm": "SHA256 key sort"},
        "xgb_policy": {"seed": XGB_SEED, "warmup_unique_trials": 4, "feature_scope": "ConfigEntity only"},
        "mode4_policy": "weight_stationary_barrier remains an experimental separately identified mode; it is admitted to the P7 measurement pool but cannot be promoted to production without board correctness and performance gates",
        "source_runs": {
            "P5c": {
                "run_dir": str(p5c_dir),
                "shortlist_sha256": file_sha256(p5c_dir / "shortlist.json"),
                "results_sha256": file_sha256(p5c_dir / "results.jsonl"),
            },
            **source_runs,
        },
        "workloads": workloads,
        "claim_boundary": "contract and future correctness labels only; no latency, search-efficiency, G7, stage, FPS, or cross-boot claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p5c-run-dir", required=True)
    parser.add_argument("--board-boot-id", required=True)
    parser.add_argument("--dmesg-after", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty P7 contract directory")
    contract = prepare(args.p5c_run_dir, args.board_boot_id, args.dmesg_after)
    (output / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "command.txt").write_text(" ".join([sys.executable] + sys.argv) + "\n")
    (output / "STATUS.md").write_text(
        "# P7 grouped-holdout contract\n\nStatus: **frozen before P7 board labels**. "
        "The common pool contains W01/W04/W07/W08 with 19/19/21/21 candidates "
        "(80 unique total). Policy definitions, correctness rules, failure accounting, "
        "mode-4 boundary, and future timing protocol are immutable. The first permitted "
        "board batch is W01 correctness only; no P7 latency may be collected until all "
        "four correctness batches pass.\n"
    )
    artifacts = {
        path.name: file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "output_sha256": artifacts,
                "source_sha256": {str(Path(__file__).resolve()): file_sha256(Path(__file__))},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("frozen P7 contract: workloads=4 candidates=80 board_labels=false")


if __name__ == "__main__":
    main()
