#!/usr/bin/env python3
"""Run M0/M1/M2 search ablation with exact incumbent-tile VTA DMA."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from analyze_stage_memory_additivity import canonical, occurrence_count
from build_cpu_vta_pipeline_v1_p1 import enumerate_reachable
from freeze_cpu_vta_pipeline_v1 import load_sealed_artifact
from iterate_cpu_vta_pipeline_v1_p5b_iteration2 import (
    apply_atomic_cpu_costs,
    summarize_atomic_profiles,
)
from solve_cpu_vta_pipeline_v1_p3 import (
    CPU_THREAD_CHOICES,
    PHYSICAL_CPU_CORES,
    build_context,
    candidate_id,
    label_from_path,
    path_from_scheme,
    stage_ddr_demand,
    topology_id,
)


TOP_K = 20
DMA_KEYS = (
    "load_buffer_2d_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_small_calls",
    "load_buffer_2d_strided_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "store_buffer_2d_calls",
    "store_buffer_2d_bytes",
)
REPRESENTATIVE_RANK_TO_TOPOLOGY = {1: "A", 5: "B", 7: "C", 13: "D"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average_ranks(values):
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[index][1], rel_tol=0.0, abs_tol=1.0e-12
        ):
            end += 1
        rank = (index + 1 + end) / 2.0
        for key, _ in ordered[index:end]:
            result[key] = rank
        index = end
    return result


def spearman(left, right):
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    keys = sorted(left)
    x = np.asarray([left_rank[key] for key in keys], dtype="float64")
    y = np.asarray([right_rank[key] for key in keys], dtype="float64")
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def static_segment_dma(segment, selected, static_by_workload):
    totals = {key: 0 for key in DMA_KEYS}
    for item in selected:
        count = occurrence_count(item["workload"], segment["unit_names"])
        if not count:
            continue
        source = static_by_workload[canonical(item["workload"])]["static_dma"]["totals"]
        for key in DMA_KEYS:
            totals[key] += count * int(source[key])
    return totals


def add_dma(left, right):
    return {key: left[key] + right[key] for key in DMA_KEYS}


def model_metrics(label, path, context, vta_dma_by_segment):
    vta_dma = {key: 0 for key in DMA_KEYS}
    cpu_ddr_service_ms = 0.0
    for segment_id, threads in path:
        segment = context["profile_segments"][segment_id]
        if segment["device"] == "vta":
            vta_dma = add_dma(vta_dma, vta_dma_by_segment[segment_id])
        else:
            cpu_ddr_service_ms += stage_ddr_demand(segment, threads, context)["service_ms"]

    cpu_compute_pool_ms = max(
        label.cpu_stage_core_work_sum_ms / PHYSICAL_CPU_CORES,
        label.cpu_prefix_mask_lower_bound_ms,
    )
    m0_components = {
        "max_cpu_stage_run_only_ms": label.max_cpu_stage_run_only_ms,
        "single_vta_service_ms": label.vta_service_sum_ms,
        "cpu_compute_pool_lower_bound_ms": cpu_compute_pool_ms,
    }
    m0_score = max(m0_components.values())
    m1_components = {
        "max_cpu_stage_with_owned_boundary_ms": label.max_cpu_stage_ms,
        "single_vta_plus_mutex_boundary_ms": (
            label.vta_service_sum_ms + label.vta_mutex_boundary_sum_ms
        ),
        "cpu_pool_with_boundary_lower_bound_ms": label.cpu_core_pool_lower_bound_ms,
    }
    m1_score = max(m1_components.values())

    dma = context["vta_dma_qualification"]
    vta_ddr_service_ms = (
        vta_dma["load_buffer_2d_bytes"] / (dma["load_bandwidth_GBps"] * 1.0e6)
        + vta_dma["store_buffer_2d_bytes"] / (dma["store_bandwidth_GBps"] * 1.0e6)
    )
    exact_shared_ddr_service_ms = cpu_ddr_service_ms + vta_ddr_service_ms
    m2_score = max(m1_score, exact_shared_ddr_service_ms)
    return {
        "m0_score_ms": m0_score,
        "m1_score_ms": m1_score,
        "m2_score_ms": m2_score,
        "m0_components": m0_components,
        "m1_components": m1_components,
        "m2_exact_shared_ddr_service_ms": exact_shared_ddr_service_ms,
        "m2_cpu_ddr_service_ms": cpu_ddr_service_ms,
        "m2_vta_ddr_service_ms": vta_ddr_service_ms,
        "m2_memory_is_bottleneck": exact_shared_ddr_service_ms > m1_score + 1.0e-12,
        "vta_dma": vta_dma,
        "boundary_bytes": label.boundary_ddr_bytes,
        "vta_islands": label.vta_islands,
    }


def rank_key(record, model):
    if model != "m2":
        return (record[model + "_score_ms"], record["candidate_id"])
    dma = record["vta_dma"]
    return (
        record["m2_score_ms"],
        dma["load_buffer_2d_small_calls"],
        dma["load_buffer_2d_calls"],
        dma["load_buffer_2d_bytes"] + dma["store_buffer_2d_bytes"],
        record["boundary_bytes"],
        record["vta_islands"],
        record["candidate_id"],
    )


def retain_top(items, record, model, limit=TOP_K):
    key = rank_key(record, model)
    if len(items) < limit or key < rank_key(items[-1], model):
        items.append(record)
        items.sort(key=lambda item: rank_key(item, model))
        del items[limit:]


def compact_record(label, metrics):
    path = label.path
    return {
        "candidate_id": candidate_id(path),
        "topology_id": topology_id(path),
        "path": [[segment, threads] for segment, threads in path],
        "cpu_thread_parameters": [
            threads for segment, threads in path if segment.startswith("cpu:")
        ],
        **metrics,
    }


def model_topology_summary(rows, model):
    best = {}
    for row in rows:
        topology = row["topology_id"]
        if topology not in best or rank_key(row, model) < rank_key(best[topology], model):
            best[topology] = row
    return {
        "distinct_topologies": len(best),
        "ordered_topologies": [
            topology for topology, _ in sorted(
                best.items(), key=lambda item: rank_key(item[1], model)
            )
        ],
    }


def evaluate_measured(models, frozen_rows, boot3):
    actual_cycle = {
        topology: float(boot3["topology_summary"][topology]["cycle_ms_median"])
        for topology in "ABCD"
    }
    actual_fps = {
        topology: float(boot3["topology_summary"][topology]["fps_median"])
        for topology in "ABCD"
    }
    rank_to_row = {int(row["rank"]): row for row in frozen_rows}
    representatives = {
        topology: rank_to_row[rank]["candidate_id"]
        for rank, topology in REPRESENTATIVE_RANK_TO_TOPOLOGY.items()
    }
    all_records = {
        row["candidate_id"]: row
        for rows in models.values() for row in rows
    }
    missing = [cid for cid in representatives.values() if cid not in all_records]
    if missing:
        raise RuntimeError("measured representatives missing from full enumeration: {}".format(missing))
    oracle_fps = max(actual_fps.values())
    evaluation = {}
    for model in ("m0", "m1", "m2"):
        selected = {topology: all_records[cid] for topology, cid in representatives.items()}
        predicted = {topology: row[model + "_score_ms"] for topology, row in selected.items()}
        ordering = sorted("ABCD", key=lambda topology: (predicted[topology], topology))
        regret = {}
        for k in range(1, 5):
            best = max(actual_fps[topology] for topology in ordering[:k])
            regret[str(k)] = 1.0 - best / oracle_fps
        evaluations_to_95 = next(
            k for k in range(1, 5)
            if max(actual_fps[topology] for topology in ordering[:k]) >= 0.95 * oracle_fps
        )
        evaluation[model] = {
            "predicted_cycle_ms": predicted,
            "predicted_ordering": ordering,
            "actual_cycle_ms": actual_cycle,
            "actual_fps": actual_fps,
            "actual_ordering": sorted("ABCD", key=lambda topology: (-actual_fps[topology], topology)),
            "cycle_spearman": spearman(predicted, actual_cycle),
            "top1_hits_actual_best": ordering[0] == max(actual_fps, key=actual_fps.get),
            "throughput_regret_at_k": regret,
            "evaluations_to_95pct_oracle": evaluations_to_95,
        }
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    resource = Path("vta/tutorials/frontend/report_out/resource_aware_maxplus")
    stage_tile = Path("vta/tutorials/frontend/report_out/stage_tile_cotuning")
    parser.add_argument("--resource-dir", default=str(resource))
    parser.add_argument("--selected-tasks", default=str(stage_tile / "iteration6_safe_overlay_final/selected_tasks.json"))
    parser.add_argument("--static-dma", default=str(stage_tile / "stage_memory_experiments/static_workload_dma.json"))
    parser.add_argument("--experiment-ab", default=str(stage_tile / "stage_memory_experiments/experiment_ab.json"))
    parser.add_argument("--tile-dma-analysis", default=str(stage_tile / "iteration6_safe_overlay_final/tile_dma_analysis.json"))
    parser.add_argument("--boot3", default=str(stage_tile / "stage_memory_experiments/experiment_d_third_boot_latin.json"))
    parser.add_argument("--frozen-top20", default=str(resource / "v1_p5b_iteration2_ranked_candidates.json"))
    parser.add_argument("--output-dir", default=str(stage_tile / "stage_memory_experiments"))
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    unit_schema = load_sealed_artifact(
        resource_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    profile = load_sealed_artifact(
        resource_dir / "v1_profile_manifest.json", "cpu_vta_pipeline_v1_profile_manifest"
    )
    local_cost = load_sealed_artifact(
        resource_dir / "v1_local_cost_table.json", "cpu_vta_pipeline_v1_local_cost_table"
    )
    base_context = build_context(profile, local_cost)
    cpu_wall, cpu_core, _, _ = summarize_atomic_profiles(resource_dir)
    context, updated_cpu_segments = apply_atomic_cpu_costs(
        base_context, cpu_wall, cpu_core
    )
    selected = json.loads(Path(args.selected_tasks).read_text())
    static_rows = json.loads(Path(args.static_dma).read_text())["rows"]
    static_by_workload = {canonical(row["workload"]): row for row in static_rows}
    vta_dma_by_segment = {
        segment_id: static_segment_dma(segment, selected, static_by_workload)
        for segment_id, segment in context["profile_segments"].items()
        if segment["device"] == "vta"
    }

    schemes, _, _ = enumerate_reachable(unit_schema)
    top = {model: [] for model in ("m0", "m1", "m2")}
    execution_count = 0
    topology_ids = set()
    memory_bottleneck_count = 0
    memory_changes_m1_score_count = 0
    maximum_ddr_fraction = {"fraction": 0.0, "candidate_id": None}
    representative_records = {}
    frozen = json.loads(Path(args.frozen_top20).read_text())["rows"]
    representative_ids = {
        next(row for row in frozen if int(row["rank"]) == rank)["candidate_id"]
        for rank in REPRESENTATIVE_RANK_TO_TOPOLOGY
    }

    for scheme in schemes:
        cpu_stages = sum(stage["device"] == "cpu" for stage in scheme)
        for allocation in itertools.product(CPU_THREAD_CHOICES, repeat=cpu_stages):
            path = path_from_scheme(scheme, allocation)
            label = label_from_path(path, context)
            metrics = model_metrics(label, path, context, vta_dma_by_segment)
            record = compact_record(label, metrics)
            execution_count += 1
            topology_ids.add(record["topology_id"])
            memory_bottleneck_count += int(metrics["m2_memory_is_bottleneck"])
            memory_changes_m1_score_count += int(
                metrics["m2_score_ms"] > metrics["m1_score_ms"] + 1.0e-12
            )
            ddr_fraction = metrics["m2_exact_shared_ddr_service_ms"] / metrics["m1_score_ms"]
            if ddr_fraction > maximum_ddr_fraction["fraction"]:
                maximum_ddr_fraction = {
                    "fraction": ddr_fraction,
                    "candidate_id": record["candidate_id"],
                    "shared_ddr_service_ms": metrics["m2_exact_shared_ddr_service_ms"],
                    "m1_score_ms": metrics["m1_score_ms"],
                }
            if record["candidate_id"] in representative_ids:
                representative_records[record["candidate_id"]] = record
            for model in top:
                retain_top(top[model], record, model)

    if execution_count != 972528 or len(topology_ids) != 4623:
        raise RuntimeError(
            "candidate-space mismatch: configurations={} topologies={}".format(
                execution_count, len(topology_ids)
            )
        )
    for candidate in representative_records.values():
        for model in top:
            if not any(row["candidate_id"] == candidate["candidate_id"] for row in top[model]):
                top[model].append(candidate)

    boot3 = json.loads(Path(args.boot3).read_text())
    measured = evaluate_measured(top, frozen, boot3)
    experiment_ab = json.loads(Path(args.experiment_ab).read_text())
    tile_dma_analysis = json.loads(Path(args.tile_dma_analysis).read_text())
    stage_workload_placements = len(experiment_ab["experiment_a"]["placements"])
    unique_workloads = len(experiment_ab["experiment_a"]["invariance"])
    trials_per_workload = int(tile_dma_analysis["measurement_count"] / unique_workloads)
    hypothetical_stage_specific_trials = stage_workload_placements * trials_per_workload
    actual_unique_workload_trials = int(tile_dma_analysis["measurement_count"])
    model_summaries = {}
    for model, rows in top.items():
        ranked = sorted(rows[:TOP_K], key=lambda row: rank_key(row, model))
        model_summaries[model] = {
            "top20": [
                {
                    "rank": index + 1,
                    **row,
                }
                for index, row in enumerate(ranked)
            ],
            **model_topology_summary(ranked, model),
        }

    m1_ids = [row["candidate_id"] for row in model_summaries["m1"]["top20"]]
    m2_ids = [row["candidate_id"] for row in model_summaries["m2"]["top20"]]
    payload = {
        "schema_version": 1,
        "kind": "stage_memory_model_ablation",
        "candidate_space": {
            "execution_configurations": execution_count,
            "topologies": len(topology_ids),
            "unique_incumbent_workloads": len(selected),
            "vta_segments_with_static_dma": len(vta_dma_by_segment),
            "cpu_segments_with_atomic_costs": updated_cpu_segments,
        },
        "model_definitions": {
            "m0": "compute resources only: CPU stage, serialized VTA, and four-core CPU lower bounds",
            "m1": "M0 plus owned CPU-VTA boundary service, boundary host work, and VTA mutex boundary service",
            "m2": (
                "M1 plus shared-DDR lower bound using CPU component bandwidth and exact incumbent-tile "
                "TIR LOAD/STORE bytes; DMA calls/small calls are deterministic tie-break and pruning features"
            ),
        },
        "parameter_policy": {
            "topology_fps_fitted_coefficients": False,
            "cpu_cost_source": "P5B 21-unit atomic profiles with grouped-holdout gate",
            "vta_dma_bandwidth_source": context["vta_dma_qualification"],
            "double_count_prevention": (
                "exact DMA service is a max-resource lower bound, never added to measured VTA service"
            ),
            "boundary_bytes_in_ddr_sum": False,
            "boundary_reason": "adjacent CPU/VTA stage traffic already owns the physical tensor demand",
        },
        "search_statistics": {
            "memory_bottleneck_configurations": memory_bottleneck_count,
            "memory_changes_m1_score_configurations": memory_changes_m1_score_count,
            "m1_m2_top20_identical": m1_ids == m2_ids,
            "m1_m2_top20_overlap": len(set(m1_ids) & set(m2_ids)),
            "maximum_shared_ddr_fraction_of_m1": maximum_ddr_fraction,
        },
        "measurement_budget": {
            "stage_workload_placements_in_frozen_topologies": stage_workload_placements,
            "unique_workloads": unique_workloads,
            "trials_per_workload": trials_per_workload,
            "hypothetical_stage_specific_trials": hypothetical_stage_specific_trials,
            "actual_unique_workload_trials": actual_unique_workload_trials,
            "avoided_trials": hypothetical_stage_specific_trials - actual_unique_workload_trials,
            "reduction_fraction": 1.0 - actual_unique_workload_trials / hypothetical_stage_specific_trials,
            "outcomes": tile_dma_analysis["outcomes"],
        },
        "models": model_summaries,
        "measured_four_topology_evaluation": measured,
        "conclusion": {
            "m2_improves_four_topology_spearman": (
                measured["m2"]["cycle_spearman"] is not None
                and measured["m1"]["cycle_spearman"] is not None
                and measured["m2"]["cycle_spearman"] > measured["m1"]["cycle_spearman"]
            ),
            "m2_changes_natural_top20": m1_ids != m2_ids,
            "interpretation": (
                "Report a negative ranking result if M2 is off the critical resource in the natural Top-20; "
                "retain exact DMA for endpoint deltas, dominance checks, and rejecting fragmented tile candidates."
            ),
        },
        "source_sha256": {
            "selected_tasks": sha256(args.selected_tasks),
            "static_dma": sha256(args.static_dma),
            "boot3": sha256(args.boot3),
            "frozen_top20": sha256(args.frozen_top20),
            "experiment_ab": sha256(args.experiment_ab),
            "tile_dma_analysis": sha256(args.tile_dma_analysis),
        },
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "memory_model_ablation.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# M0/M1/M2 stage-memory search ablation",
        "",
        "The complete search covers {:,} execution configurations and {:,} topologies. No".format(
            execution_count, len(topology_ids)
        ),
        "complete-topology FPS is used to fit a coefficient. M2 reuses the qualified component",
        "bandwidth and replaces the older GOP traffic proxy with exact incumbent-tile TIR DMA.",
        "",
        "| model | measured A/B/C/D predicted order | Spearman | regret@1 | evals to 95% oracle | Top-20 topologies |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in ("m0", "m1", "m2"):
        item = measured[model]
        rho = "--" if item["cycle_spearman"] is None else "{:.3f}".format(item["cycle_spearman"])
        lines.append(
            "| {} | `{}` | {} | {:.2%} | {} | {} |".format(
                model.upper(), ">".join(item["predicted_ordering"]), rho,
                item["throughput_regret_at_k"]["1"], item["evaluations_to_95pct_oracle"],
                model_summaries[model]["distinct_topologies"],
            )
        )
    lines += [
        "",
        "Actual third-boot median order is `{}`. M1/M2 Top-20 overlap is {}/20; identical: `{}`.".format(
            ">".join(measured["m2"]["actual_ordering"]),
            payload["search_statistics"]["m1_m2_top20_overlap"],
            payload["search_statistics"]["m1_m2_top20_identical"],
        ),
        "M2 becomes the predicted bottleneck for {:,}/{:,} configurations.".format(
            memory_bottleneck_count, execution_count
        ),
        "The closest case reaches {:.2%} of its M1 resource bound, so the qualified DDR lower".format(
            maximum_ddr_fraction["fraction"]
        ),
        "bound is not merely absent from Top-20; it is non-critical throughout this search space.",
        "Per-workload reuse reduces the same 8-trial neighborhood from a hypothetical {} stage-".format(
            hypothetical_stage_specific_trials
        ),
        "placement trials to {} unique-workload trials ({:.2%} reduction; {} avoided).".format(
            actual_unique_workload_trials,
            1.0 - actual_unique_workload_trials / hypothetical_stage_specific_trials,
            hypothetical_stage_specific_trials - actual_unique_workload_trials,
        ),
        "",
        "## Decision",
        "",
        "- M2 does not receive a fitted DDR coefficient and does not double-count VTA LOAD/STORE.",
        "- If M2 leaves the natural Top-20 unchanged, this is a valid negative ranking result:",
        "  those candidates are compute/single-VTA limited, so memory demand is off the critical path.",
        "- Exact DMA remains useful before Top-20: it exposes endpoint tradeoffs, supplies a shared",
        "  resource lower bound, and rejects tile candidates with more fragmented/redundant transfers.",
        "- The measured validation contains only four frozen topology representatives. Spearman and",
        "  regret are reported descriptively, not as a fitted generalization claim.",
    ]
    report_path = output / "MEMORY_MODEL_ABLATION.md"
    report_path.write_text("\n".join(lines) + "\n")

    hashes_path = output / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    for path in (Path(__file__), json_path, report_path):
        hashes[path.name] = sha256(path)
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    print("wrote", json_path)
    print("wrote", report_path)


if __name__ == "__main__":
    main()
