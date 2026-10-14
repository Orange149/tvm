#!/usr/bin/env python3
"""Expand the real-task reference pools after label-free local legality audit.

The first compact proposal intentionally remains immutable.  This revision
removes its all-mode lowering-negative max-min family and adds fixed full-width
single-vthread families.  The family list is frozen before FSim qualification
of the new identities and before any current-board latency is collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import vta

from c3_candidate_identity import normalize_config_entity
from prepare_vta_c3_real_task_reference_pools import knobs
from prepare_vta_c3_resnet18_literature_candidates import candidate_row, create_task


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_INPUT = GROUPED / "20260924_c3_real_task_reference_prepare_run01"
DEFAULT_LOCAL = GROUPED / "20260924_c3_real_task_reference_local_run03"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_reference_prepare_run02"
SCHEMA = "c3_real_task_reference_pool_expanded_v1"
MODES = (
    "original",
    "input_stationary",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
)
EXPANSION_TILES = {
    "R18-U128C128H28K3S1": (
        (7, 28, 1, 2),
        (4, 28, 1, 2),
    ),
    "R18-U64C64H56K1S1": (
        (7, 56, 1, 2),
        (7, 56, 2, 2),
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_directory(directory):
    directory = Path(directory).resolve()
    ledger = json.loads((directory / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(name))
    return sha256_file(directory / "artifact_hashes.json")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def retained_rows(pool, local_static):
    status = {row["candidate_id"]: row["status"] for row in local_static}
    families = {}
    for row in pool["candidates"]:
        families.setdefault(row["family_id"], []).append(row)
    incumbent_family = next(
        family
        for family, rows in families.items()
        if any(row["candidate_id"] == pool["incumbent"]["candidate_id"] for row in rows)
    )
    other = [
        family
        for family, rows in families.items()
        if family != incumbent_family and any(status.get(row["candidate_id"]) == "ok" for row in rows)
    ]
    if len(other) != 1:
        raise ValueError("expected exactly one non-incumbent lowering-positive family")
    retained = [
        row
        for row in pool["candidates"]
        if row["family_id"] in (incumbent_family, other[0])
    ]
    if len(retained) != 8:
        raise ValueError("retained pool must contain two complete four-mode families")
    return retained


def find_entity(pool, desired, env):
    task = create_task(pool["workload"], 0, env)
    matches = []
    for index in range(len(task.config_space)):
        entity = normalize_config_entity(task.config_space.get(index).to_json_dict())
        if knobs(entity) == desired:
            matches.append((index, entity))
    if len(matches) != 1:
        raise ValueError("expanded tile missing or ambiguous: {}".format(desired))
    return matches[0]


def expanded_rows(pool, env):
    schedule_versions = {
        row["candidate_identity_fields"]["schedule_version"] for row in pool["candidates"]
    }
    if len(schedule_versions) != 1:
        raise ValueError("input pool has mixed schedule versions")
    schedule_version = next(iter(schedule_versions))
    rows = []
    for ordinal, (tile_h, tile_w, tile_ci, tile_co) in enumerate(
        EXPANSION_TILES[pool["workload_id"]], 3
    ):
        desired = {
            "tile_b": 1,
            "tile_h": tile_h,
            "tile_w": tile_w,
            "tile_ci": tile_ci,
            "tile_co": tile_co,
            "oc_nthread": 1,
            "h_nthread": 1,
        }
        index, entity = find_entity(pool, desired, env)
        family_id = "{}F{:02d}".format(pool["workload_id"], ordinal)
        for mode in MODES:
            row = candidate_row(
                pool["workload_id"],
                pool["geometry"],
                pool["workload"],
                entity,
                index,
                family_id,
                mode,
                schedule_version,
                pool["hardware_fingerprint"],
                env,
            )
            row["selection_role"] = "preregistered_full_width_local_legality_expansion"
            row["dispatch_scope_id"] = pool["dispatch_scope_id"]
            row["candidate_identity_fields"] = {
                "tile": row["visible_features"],
                "residence_mode": mode,
                "implementation_mode": row["implementation_mode"],
                "weight_load_policy": (
                    "barrier_full_or_group_residency"
                    if mode in ("weight_resident_barrier", "input_weight_resident_barrier")
                    else "template_default"
                ),
                "schedule_version": schedule_version,
            }
            rows.append(row)
    return rows


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    input_manifest = verify_directory(args.input_dir)
    local_manifest = verify_directory(args.local_dir)
    local_summary = json.loads((Path(args.local_dir) / "summary.json").read_text(encoding="utf-8"))
    if local_summary.get("board_contacted") is not False:
        raise ValueError("local audit contacted board")
    if local_summary.get("current_board_performance_labels_collected") is not False:
        raise ValueError("local audit collected current-board performance")
    local_static = load_jsonl(Path(args.local_dir) / "static_results.jsonl")
    pools = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(args.input_dir).glob("r18-h*_four_mode_96.json"))
    ]
    if {pool["workload_id"] for pool in pools} != set(EXPANSION_TILES):
        raise ValueError("unexpected reference workload set")
    env = vta.get_env()
    output.mkdir(parents=True)
    pool_files = []
    summaries = {}
    for ordinal, pool in enumerate(sorted(pools, key=lambda row: row["workload_id"]), 4):
        retained = retained_rows(
            pool, [row for row in local_static if row["workload_id"] == pool["workload_id"]]
        )
        added = expanded_rows(pool, env)
        revised = json.loads(json.dumps(pool))
        revised.update(
            schema=SCHEMA,
            status="expanded_and_frozen_before_new_fsim_or_current_board_latency",
            role="full_width_local_legality_expanded_reference_candidate_pool",
            candidates=retained + added,
            claim_boundary=(
                "candidate identities after label-free local legality audit; "
                "not new-candidate FSim correctness, board latency, or oracle"
            ),
        )
        revised["selection"] = {
            "method": (
                "retain incumbent and sole lowering-positive original alternative; add two "
                "predeclared full-width single-vthread families"
            ),
            "retained_identity_count": len(retained),
            "added_identity_count": len(added),
            "gross_candidate_count": len(revised["candidates"]),
            "local_static_legality_used": True,
            "new_candidate_fsim_used": False,
            "current_board_performance_labels_used": False,
            "invalid_modes_remain_explicit": True,
        }
        name = "r18-h{}_four_mode_96.json".format(ordinal)
        write_json(output / name, revised)
        pool_files.append(name)
        summaries[pool["workload_id"]] = revised["selection"]
    contract = {
        "schema": SCHEMA + "_contract",
        "status": "expanded_pools_frozen_before_new_fsim_or_current_board_latency",
        "pool_files": pool_files,
        "workloads": summaries,
        "input_prepare_manifest_sha256": input_manifest,
        "local_legality_manifest_sha256": local_manifest,
        "expansion_tiles": {key: list(value) for key, value in EXPANSION_TILES.items()},
        "board_contacted": False,
        "current_board_performance_labels_read": False,
        "external_tophub_source_cost_read_for_incumbent_only": True,
        "new_candidate_fsim_correctness_read": False,
    }
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "workloads": summaries,
            "board_contacted": False,
        },
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
