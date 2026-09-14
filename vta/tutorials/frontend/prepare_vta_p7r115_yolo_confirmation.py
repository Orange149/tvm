#!/usr/bin/env python3
"""Freeze a label-isolated YOLO confirmation pool for C3 residency schedules.

This program is deliberately local-only.  It never opens TopHub, RPC, or a board.
It enumerates each geometry's complete AutoTVM ConfigSpace, applies only declared
hardware/structural predicates, and deterministically samples three ConfigEntity
families.  Each family is paired across original/input/weight/hybrid modes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import (
    candidate_identity_record,
    canonical_json_bytes,
    normalize_config_entity,
)


SCHEMA = "c3_p7r115_yolo_confirmation_contract_v1"
DOMAIN_SCHEMA = "c3_p7r115_complete_config_domain_v1"
TEMPLATE = "conv2d_packed_residency.vta"
SELECTION_SEED = "c3-p7r115-yolo-label-isolated-v1"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}
SELECTED_CONVS = {
    "Y00": "conv2",
    "Y01": "conv13",
    "Y02": "conv21",
}

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
YOLO_SOURCE = HERE / "search_yolov3_tiny_stage_splits.py"
EXPOSURE_SOURCES = {
    "W00_W09": C3
    / "04_dma_command_signatures"
    / "20260910_p4b_local_residency_pool_run01"
    / "results.jsonl",
    "P7Q": C3
    / "07_grouped_holdout"
    / "20260911_p7q_qualified_timing_contract_run01"
    / "contract.json",
    "E00": C3
    / "07_grouped_holdout"
    / "20260911_p7r16_unseen_board_contracts_run01"
    / "e00_contract.json",
    "E01": C3
    / "07_grouped_holdout"
    / "20260911_p7r16_unseen_board_contracts_run01"
    / "e01_contract.json",
    "E02": C3
    / "07_grouped_holdout"
    / "20260911_p7r16_unseen_board_contracts_run01"
    / "e02_contract.json",
    "E03": C3
    / "07_grouped_holdout"
    / "20260911_p7r46_e03_board_contract_run01"
    / "e03_contract.json",
}
SCHEDULE_SOURCES = (
    REPO / "vta" / "python" / "vta" / "top" / "vta_conv2d.py",
    REPO / "vta" / "python" / "vta" / "top" / "vta_conv2d_residency.py",
)
DEFAULT_OUTPUT = (
    C3
    / "07_grouped_holdout"
    / "20260911_p7r115_yolo_confirmation_protocol_run01"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_yolo_convs(path):
    """Read the literal YOLO_CONVS table without importing/downloading the model."""

    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "YOLO_CONVS" for target in node.targets
        ):
            rows = ast.literal_eval(node.value)
            return {
                str(row[1]): {
                    "model_index": int(row[0]),
                    "name": str(row[1]),
                    "ci": int(row[2]),
                    "co": int(row[3]),
                    "height": int(row[4]),
                    "width": int(row[5]),
                    "kernel": int(row[6]),
                }
                for row in rows
            }
    raise ValueError("YOLO_CONVS literal not found in {}".format(path))


def make_workload(spec, env):
    if spec["ci"] % env.BLOCK_IN or spec["co"] % env.BLOCK_OUT:
        raise ValueError("{} is not exactly representable in packed VTA layout".format(spec["name"]))
    kernel = int(spec["kernel"])
    pad = kernel // 2
    return [
        "conv2d_packed.vta",
        [
            "TENSOR",
            [env.BATCH, spec["ci"] // env.BLOCK_IN, spec["height"], spec["width"], 1, env.BLOCK_IN],
            "int8",
        ],
        [
            "TENSOR",
            [spec["co"] // env.BLOCK_OUT, spec["ci"] // env.BLOCK_IN, kernel, kernel, env.BLOCK_OUT, env.BLOCK_IN],
            "int8",
        ],
        [1, 1],
        [pad, pad, pad, pad],
        [1, 1],
        "NCHW1n16c",
        "int32",
    ]


def workload_signature(workload):
    data, weight = workload[1][1], workload[2][1]
    return [
        int(data[0] * data[4]),
        int(data[1] * data[5]),
        int(data[2]),
        int(data[3]),
        int(weight[0] * weight[4]),
        int(weight[2]),
        int(weight[3]),
        *[int(item) for item in workload[3]],
        *[int(item) for item in workload[4]],
    ]


def _workloads_in(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "workload" and isinstance(item, list) and item and "conv2d" in str(item[0]):
                found.append(item)
            found.extend(_workloads_in(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_workloads_in(item))
    return found


def exposure_audit(sources=EXPOSURE_SOURCES):
    rows = []
    for label, path in sources.items():
        if not Path(path).is_file():
            raise FileNotFoundError("required exposure evidence is missing: {}".format(path))
        if Path(path).suffix == ".jsonl":
            payloads = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        else:
            payloads = [json.loads(Path(path).read_text())]
        signatures = sorted(
            {tuple(workload_signature(workload)) for payload in payloads for workload in _workloads_in(payload)}
        )
        rows.append(
            {
                "label_namespace": label,
                "source": str(Path(path).relative_to(REPO)),
                "source_sha256": sha256_file(path),
                "unique_geometry_count": len(signatures),
                "geometry_signatures": [list(item) for item in signatures],
            }
        )
    return rows


def config_knobs(config):
    knobs = {}
    for name, kind, value in config["entity"]:
        knobs[name] = int(value[-1] if kind == "sp" else value)
    return knobs


def hardware_predicate(workload, config, env):
    """Conservative, label-free capacity and reuse-opportunity certificate."""

    knobs = config_knobs(config)
    data, weight = workload[1][1], workload[2][1]
    stride_h, stride_w = workload[3]
    kh, kw = int(weight[2]), int(weight[3])
    input_h = (knobs["tile_h"] - 1) * int(stride_h) + kh
    input_w = (knobs["tile_w"] - 1) * int(stride_w) + kw
    required = {
        "input_vectors": knobs["tile_ci"] * input_h * input_w,
        "weight_vectors": knobs["tile_co"] * knobs["tile_ci"] * kh * kw,
        "accumulator_vectors": knobs["tile_co"] * knobs["tile_h"] * knobs["tile_w"],
    }
    limits = {
        "input_vectors": env.INP_BUFF_SIZE // env.INP_ELEM_BYTES,
        "weight_vectors": env.WGT_BUFF_SIZE // env.WGT_ELEM_BYTES,
        "accumulator_vectors": env.ACC_BUFF_SIZE // env.ACC_ELEM_BYTES,
    }
    opportunities = {
        "output_channel_outer_tiles": int(weight[0]) // knobs["tile_co"],
        "output_width_outer_tiles": int(data[3]) // knobs["tile_w"],
    }
    reasons = []
    if knobs["oc_nthread"] != 1 or knobs["h_nthread"] != 1:
        reasons.append("shared_four_mode_domain_requires_single_vthread")
    for key in sorted(required):
        if required[key] > limits[key]:
            reasons.append("{}_capacity".format(key))
    if opportunities["output_channel_outer_tiles"] < 2:
        reasons.append("no_input_residency_reuse_opportunity")
    if opportunities["output_width_outer_tiles"] < 2:
        reasons.append("no_weight_grouping_opportunity")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "required": required,
        "limits": limits,
        "opportunities": opportunities,
        "formula_scope": "necessary static bound; not a lowering, correctness, or speed certificate",
    }


def create_tasks(workload, env):
    return {
        name: autotvm.task.create(
            TEMPLATE,
            args=tuple(workload[1:]) + (number,),
            target=env.target,
            target_host=env.target_host,
        )
        for name, number in MODE_NUMBERS.items()
    }


def enumerate_shared_domain(workload_id, workload, tasks, env, families_per_geometry):
    per_mode = {}
    mode_entities = {}
    for mode, task in tasks.items():
        entities = [
            normalize_config_entity(task.config_space.get(index).to_json_dict())
            for index in range(len(task.config_space))
        ]
        mode_entities[mode] = entities
        per_mode[mode] = {
            "config_count": len(entities),
            "ordered_domain_sha256": canonical_sha256(entities),
            "set_domain_sha256": canonical_sha256(sorted(canonical_sha256(item) for item in entities)),
        }
    reference = mode_entities["original"]
    if any(mode_entities[mode] != reference for mode in MODE_NUMBERS):
        raise ValueError("{} mode ConfigSpaces are not exactly aligned".format(workload_id))
    rows = []
    for index, entity in enumerate(reference):
        certificate = hardware_predicate(workload, entity, env)
        semantic_hash = canonical_sha256(entity)
        rank_hash = hashlib.sha256(
            "{}|{}|{}".format(SELECTION_SEED, workload_id, semantic_hash).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "schema": DOMAIN_SCHEMA,
                "workload_id": workload_id,
                "semantic_config_sha256": semantic_hash,
                "sampling_rank_sha256": rank_hash,
                "debug": {"config_index": index},
                "complete_config_entity": entity,
                "hardware_predicate": certificate,
            }
        )
    eligible = sorted(
        (row for row in rows if row["hardware_predicate"]["passed"]),
        key=lambda row: (row["sampling_rank_sha256"], row["semantic_config_sha256"]),
    )
    if len(eligible) < families_per_geometry:
        raise ValueError(
            "{} has only {} label-free eligible ConfigEntities; {} requested".format(
                workload_id, len(eligible), families_per_geometry
            )
        )
    return rows, eligible[:families_per_geometry], per_mode


def hardware_fingerprint(env):
    return {
        "target": env.TARGET,
        "batch": env.BATCH,
        "block_in": env.BLOCK_IN,
        "block_out": env.BLOCK_OUT,
        "inp_buff_size": env.INP_BUFF_SIZE,
        "wgt_buff_size": env.WGT_BUFF_SIZE,
        "acc_buff_size": env.ACC_BUFF_SIZE,
        "uop_buff_size": env.UOP_BUFF_SIZE,
    }


def exact_tophub_workload_audit(workloads, pool_commitment_sha256):
    """Return only exact-hit metadata after selection is committed.

    ConfigEntity, index, result cost and ordering are intentionally never returned.
    A missing local package is a closed audit, not a network-download request.
    """

    from tvm.autotvm.record import load_from_file  # pylint: disable=import-outside-toplevel
    from tvm.autotvm.tophub import (  # pylint: disable=import-outside-toplevel
        AUTOTVM_TOPHUB_ROOT_PATH,
        PACKAGE_VERSION,
    )

    path = Path(AUTOTVM_TOPHUB_ROOT_PATH) / "vta_{}.log".format(PACKAGE_VERSION["vta"])
    if not path.is_file():
        return {
            "audit_phase": "after_candidate_pool_commitment",
            "pool_commitment_sha256": pool_commitment_sha256,
            "status": "closed_no_local_package",
            "package_path": str(path),
            "network_download_attempted": False,
            "workloads": {
                workload_id: {"exact_hit": False, "exact_record_count": 0}
                for workload_id in workloads
            },
            "all_exact_hits": False,
        }
    targets = {workload_id: json.loads(json.dumps(workload)) for workload_id, workload in workloads.items()}
    counts = {workload_id: 0 for workload_id in targets}
    models = {workload_id: set() for workload_id in targets}
    for measure_input, _ in load_from_file(path):
        actual = json.loads(json.dumps(measure_input.task.workload))
        for workload_id, expected in targets.items():
            if actual == expected:
                counts[workload_id] += 1
                models[workload_id].add(str(measure_input.target.model))
    return {
        "audit_phase": "after_candidate_pool_commitment",
        "pool_commitment_sha256": pool_commitment_sha256,
        "status": "complete",
        "package_path": str(path),
        "package_sha256": sha256_file(path),
        "network_download_attempted": False,
        "disclosed_fields": "exact-hit count and target-model name only; no ConfigEntity/index/cost",
        "workloads": {
            workload_id: {
                "exact_hit": counts[workload_id] > 0,
                "exact_record_count": counts[workload_id],
                "target_models": sorted(models[workload_id]),
            }
            for workload_id in sorted(targets)
        },
        "all_exact_hits": all(count > 0 for count in counts.values()),
    }


def workload_cost_proxy(workload):
    data, weight = workload[1][1], workload[2][1]
    stride_h, stride_w = workload[3]
    pad = workload[4]
    output_h = (data[2] + pad[0] + pad[2] - weight[2]) // stride_h + 1
    output_w = (data[3] + pad[1] + pad[3] - weight[3]) // stride_w + 1
    ci, co = data[1] * data[5], weight[0] * weight[4]
    macs = data[0] * data[4] * output_h * output_w * ci * co * weight[2] * weight[3]
    return {
        "macs": int(macs),
        "operations_if_mac_is_two_ops": int(2 * macs),
        "input_bytes": int(data[0] * data[1] * data[2] * data[3] * data[4] * data[5]),
        "weight_bytes": int(
            weight[0] * weight[1] * weight[2] * weight[3] * weight[4] * weight[5]
        ),
        "latency_estimate_ms": None,
        "latency_note": "withheld: no runtime label is permitted before pool freeze",
    }


def build_contract(families_per_geometry=3):
    if families_per_geometry < 3:
        raise ValueError("families_per_geometry must be at least three")
    env = vta.get_env()
    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in SCHEDULE_SOURCES}
    schedule_version = canonical_sha256(source_hashes)
    exposed = exposure_audit()
    exposed_signatures = {
        tuple(signature)
        for source in exposed
        for signature in source["geometry_signatures"]
    }
    yolo = parse_yolo_convs(YOLO_SOURCE)
    fingerprint = hardware_fingerprint(env)
    domain_rows = []
    candidates = []
    geometries = []
    for workload_id, conv_name in SELECTED_CONVS.items():
        spec = yolo[conv_name]
        workload = make_workload(spec, env)
        signature = workload_signature(workload)
        if tuple(signature) in exposed_signatures:
            raise ValueError("{} collides with an exposed performance geometry".format(workload_id))
        tasks = create_tasks(workload, env)
        rows, selected, mode_domains = enumerate_shared_domain(
            workload_id, workload, tasks, env, families_per_geometry
        )
        domain_rows.extend(rows)
        family_ids = []
        for family_ordinal, row in enumerate(selected):
            family_id = "{}F{:02d}".format(workload_id, family_ordinal)
            family_ids.append(family_id)
            config_index = row["debug"]["config_index"]
            for mode in MODE_NUMBERS:
                identity = candidate_identity_record(
                    fingerprint,
                    TEMPLATE,
                    schedule_version,
                    workload,
                    mode,
                    row["complete_config_entity"],
                    config_index=config_index,
                )
                candidates.append(
                    {
                        "candidate_id": identity["candidate_id"],
                        "identity": identity["identity"],
                        "debug": identity["debug"],
                        "workload_id": workload_id,
                        "family_id": family_id,
                        "family_ordinal": family_ordinal,
                        "residence_mode": mode,
                        "mode_number": MODE_NUMBERS[mode],
                        "selection_rank_sha256": row["sampling_rank_sha256"],
                        "hardware_predicate": row["hardware_predicate"],
                        "local_lower_status": "not_run_by_contract_generator",
                        "board_status": "not_dispatched",
                    }
                )
        geometries.append(
            {
                "workload_id": workload_id,
                "source_model": "yolov3-tiny",
                "source_conv": spec,
                "selection_rationale": {
                    "Y00": "early packed convolution; large spatial map and low channels",
                    "Y01": "late 1x1 channel bottleneck; high input-channel pressure",
                    "Y02": "route-concatenation convolution; irregular 384 input channels",
                }[workload_id],
                "workload": workload,
                "label_free_cost_proxy": workload_cost_proxy(workload),
                "bounded_stage_timeouts_seconds": {
                    "compile_or_lower_per_candidate": 300,
                    "correctness_execution_per_candidate": 120,
                },
                "geometry_signature": signature,
                "exposed_performance_label_collision": False,
                "config_domain": mode_domains,
                "complete_shared_domain_count": len(rows),
                "label_free_eligible_count": sum(
                    row["hardware_predicate"]["passed"] for row in rows
                ),
                "selected_family_ids": family_ids,
            }
        )
    expected_count = len(SELECTED_CONVS) * families_per_geometry * len(MODE_NUMBERS)
    if len(candidates) != expected_count:
        raise AssertionError("confirmation pool cardinality mismatch")
    pool_commitment_sha256 = canonical_sha256(candidates)
    tophub_audit = exact_tophub_workload_audit(
        {item["workload_id"]: item["workload"] for item in geometries},
        pool_commitment_sha256,
    )
    if tophub_audit["all_exact_hits"]:
        sealed_reference = {
            "kind": "exact-workload TopHub reference",
            "allowed_roles": ["hidden final acceptance threshold", "final deployment fallback"],
            "eligibility_basis": "all three original workloads have exact local TopHub log hits",
        }
    else:
        sealed_reference = {
            "kind": "hidden full-pool oracle or independent long-budget stock-XGB reference",
            "allowed_roles": ["hidden final acceptance threshold", "final deployment fallback"],
            "eligibility_basis": "TopHub exact-hit audit failed closed for at least one workload",
        }
    study_tier = "p7r115_pilot" if families_per_geometry == 3 else "expanded_confirmation"
    contract = {
        "schema": SCHEMA,
        "status": "frozen_local_pilot_not_board_evidence" if families_per_geometry == 3 else "frozen_local_expanded_contract_not_board_evidence",
        "study_tier": study_tier,
        "sufficient_for_final_ccf_b_search_claim": False,
        "purpose": "label-isolated multi-geometry confirmation of hardware-led residency scheduling",
        "hardware_fingerprint": fingerprint,
        "template": TEMPLATE,
        "schedule_version": schedule_version,
        "schedule_source_sha256": source_hashes,
        "network_geometry_source": {
            "path": str(YOLO_SOURCE.relative_to(REPO)),
            "sha256": sha256_file(YOLO_SOURCE),
            "read_method": "AST literal only; model module is not imported",
        },
        "prior_performance_label_exposure": exposed,
        "geometries": geometries,
        "selection": {
            "seed": SELECTION_SEED,
            "candidate_cap": expected_count,
            "families_per_geometry": families_per_geometry,
            "rule": "per geometry enumerate complete shared ConfigEntity domain; retain declared hardware predicate; ascending SHA256(seed|workload_id|semantic_config_sha256); first N; expand each over four modes",
            "mode_balance": {mode: len(SELECTED_CONVS) * families_per_geometry for mode in MODE_NUMBERS},
            "per_geometry_balance": {
                item: families_per_geometry * len(MODE_NUMBERS) for item in SELECTED_CONVS
            },
            "uses_runtime_or_board_measurements": False,
            "uses_static_dma_as_ranking_label": False,
            "uses_tophub": False,
            "config_index_semantics": "audit/debug only; never part of identity or ranking",
        },
        "development_confirmation_isolation": {
            "development_inputs": "YOLO literal geometry, complete ConfigSpace entities, hardware fingerprint and capacity/reuse predicates only",
            "forbidden_before_freeze": [
                "TopHub ConfigEntity or index",
                "TopHub latency threshold",
                "prior board timing for Y00/Y01/Y02",
                "candidate runtime labels",
            ],
            "confirmation_pool_is_immutable": True,
            "confirmation_failures_remain_in_gross_budget": True,
        },
        "candidate_pool_commitment_sha256": pool_commitment_sha256,
        "post_commit_exact_tophub_audit": tophub_audit,
        "sealed_reference_role": {
            **sealed_reference,
            "included_in_confirmation_pool": False,
            "exact_hit_metadata_read_after_commitment": True,
            "reference_config_index_entity_or_cost_disclosed": False,
            "may_be_revealed": "only after this candidate contract and all candidate measurements are frozen",
            "prohibited_role": "first dispatch, candidate generation, pruning, ranking, or adaptation",
        },
        "scale_up_contract": {
            "pilot_limit": "N=3 gives 12 candidates/geometry and 36 total; method-debug only",
            "minimum_next_confirmation": "N=8 gives 32 candidates/geometry and 96 total",
            "command": "prepare_vta_p7r115_yolo_confirmation.py --families-per-geometry 8 --output-dir <new-immutable-directory>",
            "prefix_stability": "same seed and ascending-hash rule; N=3 pilot is an exact prefix of N=8",
            "final_ccf_b_claim_requires": "expanded confirmation plus preregistered statistics; pilot alone is insufficient",
        },
        "confirmation_execution": {
            "stage_0": "verify artifact hashes, hardware fingerprint and exact gross candidate count",
            "stage_1": "one bounded compile/lower call per gross candidate; failures stay charged",
            "stage_2": "correctness on every buildable candidate before any timing; wrong answers never timed",
            "stage_3": "time only correctness-passed candidates using preregistered balanced interleaving",
            "equal_gross_dispatch_budget": "three ConfigEntities per mode per geometry; no replacement after any failure",
            "timing_order": "for each repeat and geometry rotate mode order by repeat, then rotate family order by geometry+repeat; each surviving candidate exactly once per repeat",
            "minimum_timing_repeats": 7,
            "primary_comparison": "within-family mechanism mode versus same-family original",
            "incumbent_protection": "replace sealed TopHub fallback only if correctness passes and preregistered threshold is beaten; otherwise retain fallback",
        },
        "claim_limits": [
            "N=3 is a P7R115 pilot and cannot support a final CCF-B search conclusion",
            "hardware predicate is necessary-only and is not lowering proof",
            "weight_stationary is the repository's bounded grouping feasibility mode; realized weight DMA reuse is not assumed",
            "paper_inspired_hybrid is not claimed as an exact reproduction until its implementation/evidence says so",
            "this local artifact contains no FPGA correctness, latency, FPS, or deployment result",
        ],
        "candidate_count": len(candidates),
        "complete_domain_entity_count": len(domain_rows),
    }
    return contract, domain_rows, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families-per-geometry", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    if args.families_per_geometry != 3 and args.output_dir == DEFAULT_OUTPUT:
        raise ValueError("expanded contracts require an explicit new --output-dir")
    contract, domain_rows, candidates = build_contract(args.families_per_geometry)
    write_json(output / "contract.json", contract)
    with (output / "complete_domain.jsonl").open("x", encoding="utf-8") as stream:
        for row in domain_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "candidates.jsonl").open("x", encoding="utf-8") as stream:
        for row in candidates:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    status = [
        "# P7R115 YOLO label-isolated confirmation contract",
        "",
        "- Status: `{}`".format(contract["status"]),
        "- Tier: `{}`; this artifact alone is insufficient for a final CCF-B search claim".format(contract["study_tier"]),
        "- Geometries: Y00 conv2, Y01 conv13, Y02 conv21",
        "- Complete ConfigEntity rows: {}".format(len(domain_rows)),
        "- Confirmation pool: {} ({} families × 4 modes × 3 geometries)".format(len(candidates), args.families_per_geometry),
        "- TopHub selection leakage: none; exact-hit boolean audit occurs only after pool commitment",
        "- Sealed reference: {}".format(contract["sealed_reference_role"]["kind"]),
        "- Next gate: equal-budget compile/lower, then correctness for every buildable candidate",
    ]
    (output / "STATUS.md").write_text("\n".join(status) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(
        "{} --families-per-geometry {} --output-dir {}\n".format(
            Path(__file__).resolve(), args.families_per_geometry, output.resolve()
        ),
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps({"output": str(output), "candidates": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
