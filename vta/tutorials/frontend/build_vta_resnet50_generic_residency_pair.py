#!/usr/bin/env python3
"""Cross-build pretrained ResNet50 for an exact original/residency candidate pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import vta

from build_vta_resnet50_relay_residency_dispatch_pair import (
    build_one,
    load_candidate,
    make_relay_program,
    sha256,
    write_json,
)


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    rows = [
        load_candidate(args.candidates, args.original_id),
        load_candidate(args.candidates, args.residency_id),
    ]
    modes = [row["public_mode"] for row in rows]
    if modes != ["original", args.residency_mode]:
        raise ValueError("unexpected exact candidate modes: " + repr(modes))
    if rows[0]["identity"]["workload"] != rows[1]["identity"]["workload"]:
        raise ValueError("workload mismatch")
    if (rows[0]["identity"]["complete_config_entity"]
            != rows[1]["identity"]["complete_config_entity"]):
        raise ValueError("ConfigEntity mismatch")

    env = vta.get_env()
    relay_program, params = make_relay_program(env, pretrained=True)
    builds = [build_one(row, relay_program, params, output, env) for row in rows]
    hit_counts = [len(row["dispatch_audit"]["config_hits"]) for row in builds]
    schedule_hit_counts = [len(row["dispatch_audit"]["schedule_hits"]) for row in builds]
    if len(set(hit_counts)) != 1 or len(set(schedule_hit_counts)) != 1:
        raise RuntimeError("A/B dispatch multiplicity differs")
    summary = {
        "schema": "c3_vta_resnet50_generic_exact_residency_build_v1",
        "status": "resnet50_generic_pair_cross_build_dispatch_verified",
        "model": "resnet50_v2",
        "pretrained": True,
        "input_shape": [env.BATCH, 3, 224, 224],
        "same_workload_and_config": True,
        "public_modes": modes,
        "config_hit_count": hit_counts[0],
        "schedule_hit_count": schedule_hit_counts[0],
        "builder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_source_sha256": sha256(args.candidates),
        "builds": builds,
        "board_contacted": False,
        "claim_boundary": (
            "Whole-model compilation of one exact candidate pair; graph-node multiplicity, "
            "board correctness, and latency remain unmeasured"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join(__import__("sys").argv) + "\n", encoding="utf-8"
    )
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--original-id", required=True)
    parser.add_argument("--residency-id", required=True)
    parser.add_argument("--residency-mode", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
