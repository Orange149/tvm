#!/usr/bin/env python3
"""Freeze the next label-blind max-min batch after a Node-C expansion trigger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import vta

from prepare_vta_c3_resnet18_literature_candidates import (
    GEOMETRIES,
    MODES,
    candidate_row,
    digest_value,
    max_min_select,
    sha256_file,
    write_json,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_artifacts(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in ledger.items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return {name: value for name, value in sorted(ledger.items())}


def first_batch_entities(pool):
    originals = [row for row in pool["candidates"] if row["residence_mode"] == "original"]
    originals.sort(key=lambda row: int(row["family_id"].rsplit("F", 1)[1]))
    return [row["complete_config_entity"] for row in originals]


def build_extension(workload_id, original_space, prior_pools, prior_count, batch_size, env):
    domain_rows = original_space["candidates"]
    entities = [row["complete_config_entity"] for row in domain_rows]
    ordered = max_min_select(entities, prior_count + batch_size, workload_id)
    prefix_rows = []
    for pool in prior_pools:
        prefix_rows.extend(first_batch_entities(pool))
    prefix_rows.sort(
        key=lambda entity: next(
            int(row["family_id"].rsplit("F", 1)[1])
            for pool in prior_pools
            for row in pool["candidates"]
            if row["residence_mode"] == "original"
            and digest_value(row["complete_config_entity"]) == digest_value(entity)
        )
    )
    expected_prefix = [digest_value(value) for value in prefix_rows]
    if len(expected_prefix) != prior_count or len(set(expected_prefix)) != prior_count:
        raise ValueError("prior pool prefix is not contiguous/unique for {}".format(workload_id))
    observed_prefix = [digest_value(value) for value in ordered[:prior_count]]
    if observed_prefix != expected_prefix:
        raise ValueError("recomputed max-min prefix differs from frozen P7R470 for {}".format(workload_id))
    row_by_entity = {digest_value(row["complete_config_entity"]): row for row in domain_rows}
    first_pool = prior_pools[0]
    schedule_version = first_pool["source_hashes"]
    fingerprint = first_pool["hardware_fingerprint"]
    spec = first_pool["geometry"]
    workload = first_pool["workload"]
    candidates = []
    for ordinal, entity in enumerate(ordered[prior_count:], prior_count):
        domain = row_by_entity[digest_value(entity)]
        family_id = "{}F{:02d}".format(workload_id, ordinal)
        for mode in MODES:
            candidates.append(
                candidate_row(
                    workload_id,
                    spec,
                    workload,
                    entity,
                    int(domain["debug"]["config_index"]),
                    family_id,
                    mode,
                    schedule_version,
                    fingerprint,
                    env,
                )
            )
    return {
        "schema": "c3_literature_workload_contract_v1",
        "status": "preregistered_expansion_before_real_lowering_fsim_or_board",
        "role": "four_mode_candidate_pool",
        "workload_id": workload_id,
        "geometry": spec,
        "workload": workload,
        "hardware_fingerprint": fingerprint,
        "selection": {
            "method": "deterministic normalized-knob max-min continuation",
            "prior_tile_count": prior_count,
            "batch_tile_count": batch_size,
            "ordinal_start_inclusive": prior_count,
            "ordinal_end_exclusive": prior_count + batch_size,
            "trigger": "prior local FSim legal identities < 12",
            "performance_labels_used": False,
            "board_labels_used": False,
        },
        "source_hashes": schedule_version,
        "board_contacted": False,
        "candidates": candidates,
    }


def run(args):
    source = Path(args.preregistered_dir).resolve()
    qualifications = [Path(value).resolve() for value in args.qualification_dir]
    prior_pool_dirs = [Path(value).resolve() for value in args.prior_pool_dir]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    source_hashes = verify_artifacts(source)
    qualification_hashes = {str(path): verify_artifacts(path) for path in qualifications}
    prior_pool_hashes = {str(path): verify_artifacts(path) for path in prior_pool_dirs}
    cumulative_legal = {workload_id: 0 for workload_id in GEOMETRIES}
    for qualification in qualifications:
        summary = read_json(qualification / "summary.json")
        for workload_id, item in summary["geometries"].items():
            cumulative_legal[workload_id] += int(item["local_fsim_legal_identities"])
    selected_workloads = [workload_id for workload_id in GEOMETRIES if cumulative_legal[workload_id] < 12]
    if args.workload_id:
        selected_workloads = [
            workload_id for workload_id in selected_workloads
            if workload_id == args.workload_id
        ]
    if not selected_workloads:
        raise ValueError("no geometry satisfies the preregistered expansion trigger")
    output.mkdir(parents=True)
    env = vta.get_env()
    summary = {
        "schema": "c3_resnet18_candidate_expansion_v1",
        "status": "frozen_before_expansion_lowering_or_fsim",
        "trigger_evidence": [str(path) for path in qualifications],
        "source_evidence": str(source),
        "performance_labels_used": False,
        "board_contacted": False,
        "geometries": {},
    }
    all_rows = []
    for workload_id in selected_workloads:
        stem = workload_id.lower()
        original = read_json(source / "{}_complete_original_space.json".format(stem))
        prior_pools = [
            read_json(directory / "{}_four_mode_96.json".format(stem))
            for directory in prior_pool_dirs
            if (directory / "{}_four_mode_96.json".format(stem)).is_file()
        ]
        extension = build_extension(workload_id, original, prior_pools, args.prior_tiles, args.batch_tiles, env)
        path = output / "{}_four_mode_96.json".format(stem)
        write_json(path, extension)
        all_rows.extend(extension["candidates"])
        summary["geometries"][workload_id] = {
            "prior_local_fsim_legal_identities": cumulative_legal[workload_id],
            "new_tile_count": args.batch_tiles,
            "new_identity_count": len(extension["candidates"]),
            "family_ordinals": [args.prior_tiles, args.prior_tiles + args.batch_tiles - 1],
            "pool_sha256": sha256_file(path),
        }
    if len(all_rows) != len(selected_workloads) * args.batch_tiles * 4 or len({row["candidate_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("expansion identity cardinality/uniqueness failure")
    write_json(output / "summary.json", summary)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "inputs": {"preregistration": source_hashes, "qualifications": qualification_hashes, "prior_pools": prior_pool_hashes},
        "source_hashes": {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())},
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistered-dir", required=True)
    parser.add_argument("--qualification-dir", action="append", required=True)
    parser.add_argument("--prior-pool-dir", action="append", required=True)
    parser.add_argument("--prior-tiles", type=int, default=24)
    parser.add_argument("--batch-tiles", type=int, default=24)
    parser.add_argument("--workload-id", choices=tuple(GEOMETRIES))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
