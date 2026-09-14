#!/usr/bin/env python3
"""Preregister label-blind ResNet18 candidate spaces for the C3 literature study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import candidate_identity_record, normalize_config_entity


TEMPLATE = "conv2d_packed_residency.vta"
SELECTION_SALT = "c3-resnet18-literature-maxmin-v1"
MODES = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
    "input_weight_resident_barrier": 5,
}
KNOBS = (
    "tile_b",
    "tile_h",
    "tile_w",
    "tile_ci",
    "tile_co",
    "oc_nthread",
    "h_nthread",
)
GEOMETRIES = {
    "R18-H1": {"ci": 64, "co": 64, "height": 56, "width": 56, "kernel": 3, "stride": 1},
    "R18-H2": {"ci": 256, "co": 256, "height": 14, "width": 14, "kernel": 3, "stride": 1},
    "R18-H3": {"ci": 256, "co": 512, "height": 14, "width": 14, "kernel": 1, "stride": 2},
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_workload(spec, env):
    pad = spec["kernel"] // 2
    return [
        "conv2d_packed.vta",
        [
            "TENSOR",
            [env.BATCH, spec["ci"] // env.BLOCK_IN, spec["height"], spec["width"], 1, env.BLOCK_IN],
            "int8",
        ],
        [
            "TENSOR",
            [
                spec["co"] // env.BLOCK_OUT,
                spec["ci"] // env.BLOCK_IN,
                spec["kernel"],
                spec["kernel"],
                env.BLOCK_OUT,
                env.BLOCK_IN,
            ],
            "int8",
        ],
        [spec["stride"], spec["stride"]],
        [pad, pad, pad, pad],
        [1, 1],
        "NCHW1n16c",
        "int32",
    ]


def knobs(entity):
    result = {}
    for name, kind, value in entity["entity"]:
        result[name] = int(value[-1] if kind == "sp" else value)
    if set(result) != set(KNOBS):
        raise ValueError("unexpected ConfigEntity knobs")
    return result


def normalized_coordinates(entities):
    values = {name: sorted({knobs(entity)[name] for entity in entities}) for name in KNOBS}
    return {
        digest_value(entity): tuple(
            values[name].index(knobs(entity)[name]) / max(1, len(values[name]) - 1)
            for name in KNOBS
        )
        for entity in entities
    }


def max_min_select(entities, count, workload_id):
    if count > len(entities):
        raise ValueError("max-min request exceeds ConfigSpace")
    coordinates = normalized_coordinates(entities)

    def rank(entity):
        return hashlib.sha256(
            "{}:{}:{}".format(SELECTION_SALT, workload_id, digest_value(entity)).encode()
        ).hexdigest()

    remaining = list(entities)
    first = min(remaining, key=rank)
    selected = [first]
    remaining.remove(first)
    while len(selected) < count:
        def score(entity):
            coordinate = coordinates[digest_value(entity)]
            distances = []
            for other in selected:
                other_coordinate = coordinates[digest_value(other)]
                distances.append(sum(abs(a - b) for a, b in zip(coordinate, other_coordinate)))
            return min(distances), sum(distances), rank(entity)

        choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return selected


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


def analytic_applicability(spec, knob_values, env):
    output_h = (spec["height"] + 2 * (spec["kernel"] // 2) - spec["kernel"]) // spec["stride"] + 1
    output_w = (spec["width"] + 2 * (spec["kernel"] // 2) - spec["kernel"]) // spec["stride"] + 1
    full_weight_bytes = spec["ci"] * spec["co"] * spec["kernel"] * spec["kernel"]
    acc_tile_bytes = (
        knob_values["tile_co"]
        * knob_values["tile_h"]
        * knob_values["tile_w"]
        * env.BLOCK_OUT
        * env.ACC_ELEM_BYTES
    )
    return {
        "full_weight_bytes": full_weight_bytes,
        "full_weight_fits_sram": full_weight_bytes <= env.WGT_BUFF_SIZE,
        "accumulator_tile_bytes_necessary_bound": acc_tile_bytes,
        "accumulator_fits_sram": acc_tile_bytes <= env.ACC_BUFF_SIZE,
        "output_height": output_h,
        "output_width": output_w,
        "scope": "label-free necessary analytic bounds; real lowering remains authoritative",
    }


def create_task(workload, mode_number, env):
    return autotvm.task.create(
        TEMPLATE,
        args=tuple(workload[1:]) + (mode_number,),
        target=env.target,
        target_host=env.target_host,
    )


def candidate_row(
    workload_id,
    spec,
    workload,
    entity,
    index,
    family_id,
    mode,
    schedule_version,
    fingerprint,
    env,
):
    identity = candidate_identity_record(
        fingerprint,
        TEMPLATE,
        schedule_version,
        workload,
        mode,
        entity,
        config_index=index,
    )
    visible = knobs(entity)
    return {
        "candidate_id": identity["candidate_id"],
        "workload_id": workload_id,
        "family_id": family_id,
        "model_hash": digest_value({"model": "relay.testing.resnet18", "geometry": spec}),
        "hardware_fingerprint": fingerprint,
        "complete_config_entity": entity,
        "debug": {"config_index": index},
        "residence_mode": mode,
        "implementation_mode": MODES[mode],
        "visible_features": visible,
        "workload_features": dict(spec),
        "static_features": {"status": "pending_real_lowering"},
        "applicability": analytic_applicability(spec, visible, env),
        "lowered_tir_sha256": None,
        "implementation_identity_status": "pending_real_lowering",
    }


def build_geometry(workload_id, spec, selected_count, env, schedule_version, fingerprint):
    workload = make_workload(spec, env)
    full_weight_bytes = spec["ci"] * spec["co"] * spec["kernel"] * spec["kernel"]
    task_modes = {
        mode: number
        for mode, number in MODES.items()
        if mode != "input_weight_resident_barrier" or full_weight_bytes <= env.WGT_BUFF_SIZE
    }
    tasks = {mode: create_task(workload, number, env) for mode, number in task_modes.items()}
    domains = {}
    for mode, task in tasks.items():
        domains[mode] = [
            normalize_config_entity(task.config_space.get(index).to_json_dict())
            for index in range(len(task.config_space))
        ]
    reference = domains["original"]
    if any(domain != reference for domain in domains.values()):
        raise ValueError("four mode ConfigSpaces differ")
    selected = max_min_select(reference, selected_count, workload_id)
    index_by_hash = {digest_value(entity): index for index, entity in enumerate(reference)}
    original_rows = []
    for index, entity in enumerate(reference):
        original_rows.append(
            candidate_row(
                workload_id,
                spec,
                workload,
                entity,
                index,
                "{}D{:04d}".format(workload_id, index),
                "original",
                schedule_version,
                fingerprint,
                env,
            )
        )
    pool = []
    for ordinal, entity in enumerate(selected):
        index = index_by_hash[digest_value(entity)]
        family_id = "{}F{:02d}".format(workload_id, ordinal)
        for mode in MODES:
            pool.append(
                candidate_row(
                    workload_id,
                    spec,
                    workload,
                    entity,
                    index,
                    family_id,
                    mode,
                    schedule_version,
                    fingerprint,
                    env,
                )
            )
    common = {
        "schema": "c3_literature_workload_contract_v1",
        "status": "preregistered_before_real_lowering_fsim_or_board",
        "workload_id": workload_id,
        "geometry": spec,
        "workload": workload,
        "hardware_fingerprint": fingerprint,
        "selection": {
            "method": "deterministic normalized-knob max-min",
            "salt": SELECTION_SALT,
            "selected_tile_count": selected_count,
            "performance_labels_used": False,
            "lowering_labels_used": False,
        },
        "source_hashes": schedule_version,
        "board_contacted": False,
    }
    return (
        {
            **common,
            "role": "complete_original_space_for_hw_aware",
            "complete_config_space_count": len(original_rows),
            "candidates": original_rows,
        },
        {**common, "role": "four_mode_candidate_pool", "candidates": pool},
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    if args.selected_tiles != 24:
        raise ValueError("frozen protocol requires 24 max-min tiles per geometry")
    output.mkdir(parents=True)
    env = vta.get_env()
    source_paths = (
        Path(__file__).resolve(),
        Path(vta.__file__).resolve().parent / "top" / "vta_conv2d.py",
        Path(vta.__file__).resolve().parent / "top" / "vta_conv2d_residency.py",
        Path(__file__).resolve().parent / "c3_candidate_identity.py",
    )
    sources = {str(path): sha256_file(path) for path in source_paths}
    schedule_version = digest_value(sources)
    fingerprint = hardware_fingerprint(env)
    summary = {"status": "complete_local_preregistration", "geometries": {}, "board_contacted": False}
    for workload_id, spec in GEOMETRIES.items():
        complete, selected = build_geometry(
            workload_id, spec, args.selected_tiles, env, schedule_version, fingerprint
        )
        complete_name = "{}_complete_original_space.json".format(workload_id.lower())
        selected_name = "{}_four_mode_96.json".format(workload_id.lower())
        write_json(output / complete_name, complete)
        write_json(output / selected_name, selected)
        summary["geometries"][workload_id] = {
            "complete_original_count": len(complete["candidates"]),
            "selected_tile_count": args.selected_tiles,
            "four_mode_identity_count": len(selected["candidates"]),
            "complete_original_sha256": sha256_file(output / complete_name),
            "four_mode_pool_sha256": sha256_file(output / selected_name),
            "combined_applicable_count": sum(
                row["residence_mode"] == "input_weight_resident_barrier"
                and row["applicability"]["full_weight_fits_sram"]
                for row in selected["candidates"]
            ),
        }
    write_json(output / "summary.json", summary)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts, "source_hashes": sources})
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-tiles", type=int, default=24)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
