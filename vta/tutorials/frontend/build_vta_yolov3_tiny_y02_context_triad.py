#!/usr/bin/env python3
"""Build a YOLO triad separating Y02 residency benefit from tile quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import vta

from build_vta_yolov3_tiny_multi_residency_pair import build_variant
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    load_candidate,
    make_relay_program,
    sha256,
    write_json,
)


VARIANT_STOCK = "y00_input_y02_stock"
VARIANT_ORIGINAL = "y00_input_y02_same_tile_original"
VARIANT_WEIGHT = "y00_input_y02_same_tile_weight"


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    for asset in (args.cfg, args.weights, args.darknet_lib):
        if not Path(asset).is_file():
            raise FileNotFoundError(asset)

    y00 = load_candidate(args.y00_contract, args.y00_input_id)
    y02_original = load_candidate(args.y02_contract, args.y02_original_id)
    y02_weight = load_candidate(args.y02_contract, args.y02_weight_id)
    if y00["public_mode"] != "input_stationary" or int(y00["implementation_mode"]) != 1:
        raise ValueError("Y00 route must be input_stationary/mode1")
    if y02_original["public_mode"] != "original" or int(y02_original["implementation_mode"]) != 0:
        raise ValueError("Y02 original route must be original/mode0")
    if y02_weight["public_mode"] != "weight_resident_barrier" or int(y02_weight["implementation_mode"]) != 4:
        raise ValueError("Y02 weight route must be weight_resident_barrier/mode4")
    if y02_original["identity"]["workload"] != y02_weight["identity"]["workload"]:
        raise ValueError("Y02 same-tile pair workload mismatch")
    if (
        y02_original["identity"]["complete_config_entity"]
        != y02_weight["identity"]["complete_config_entity"]
    ):
        raise ValueError("Y02 same-tile pair ConfigEntity mismatch")
    if y00["identity"]["workload"] == y02_original["identity"]["workload"]:
        raise ValueError("Y00 and Y02 workloads must be distinct")

    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    builds = [
        build_variant(VARIANT_STOCK, [y00], relay_program, params, output, env),
        build_variant(
            VARIANT_ORIGINAL, [y00, y02_original], relay_program, params, output, env
        ),
        build_variant(
            VARIANT_WEIGHT, [y00, y02_weight], relay_program, params, output, env
        ),
    ]
    if len({row["graph_sha256"] for row in builds}) != 1:
        raise RuntimeError("triad variants produced different graph JSON")
    summary = {
        "schema": "c3_vta_yolov3_tiny_y02_context_triad_build_v1",
        "status": "yolov3_tiny_y02_context_triad_cross_build_verified",
        "model": "yolov3-tiny",
        "input_shape": list(shape),
        "variants": builds,
        "controlled_factors": {
            "all_variants_share_y00_input_candidate": y00["candidate_id"],
            "stock_variant_delegates_y02_to_tophub": True,
            "same_tile_y02_original_candidate": y02_original["candidate_id"],
            "same_tile_y02_weight_candidate": y02_weight["candidate_id"],
            "y02_workload_equal": True,
            "y02_complete_config_entity_equal": True,
        },
        "assets": {
            "cfg": {"path": str(Path(args.cfg).resolve()), "sha256": sha256(args.cfg)},
            "weights": {"path": str(Path(args.weights).resolve()), "sha256": sha256(args.weights)},
            "darknet_lib": {
                "path": str(Path(args.darknet_lib).resolve()),
                "sha256": sha256(args.darknet_lib),
            },
        },
        "board_contacted": False,
        "claim_boundary": (
            "The triad separates same-tile Y02 residency effect from the absolute "
            "quality of that tile relative to the enclosing TopHub schedule"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"],
        "input_shape": summary["input_shape"],
        "controlled_factors": summary["controlled_factors"],
        "variants": [
            {
                "deployment_variant": row["deployment_variant"],
                "candidate_ids": row["candidate_ids"],
                "build_seconds": row["build_seconds"],
                "schedule_hits": len(row["dispatch_audit"]["schedule_hits"]),
                "graph_sha256": row["graph_sha256"],
                "graphlib_sha256": row["graphlib_sha256"],
            }
            for row in builds
        ],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y00-contract", required=True)
    parser.add_argument("--y00-input-id", required=True)
    parser.add_argument("--y02-contract", required=True)
    parser.add_argument("--y02-original-id", required=True)
    parser.add_argument("--y02-weight-id", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--darknet-lib", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
