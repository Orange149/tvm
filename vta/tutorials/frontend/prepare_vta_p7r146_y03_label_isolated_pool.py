#!/usr/bin/env python3
"""Freeze a second label-isolated YOLO geometry before local or board labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import vta

import prepare_vta_p7r115_yolo_confirmation as base


SCHEMA = "c3_p7r146_y03_label_isolated_contract_v1"
SELECTION_SEED = "c3-p7r146-y03-label-isolated-v1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
P7 = (
    HERE
    / "report_out"
    / "stage_tile_cotuning"
    / "c3_dma_residency_autotune"
    / "07_grouped_holdout"
)
DEFAULT_OUTPUT = P7 / "20260912_p7r146_y03_conv8_label_isolated_contract_run01"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable contract {}".format(output))
    if args.families != 8:
        raise ValueError("the confirmation protocol is frozen at exactly eight families")

    env = vta.get_env()
    yolo = base.parse_yolo_convs(base.YOLO_SOURCE)
    if args.conv not in yolo:
        raise ValueError("unknown YOLO convolution {}".format(args.conv))
    workload = base.make_workload(yolo[args.conv], env)
    signature = base.workload_signature(workload)
    exposure = base.exposure_audit()
    exposed = {
        tuple(item)
        for source in exposure
        for item in source["geometry_signatures"]
    }
    if tuple(signature) in exposed:
        raise ValueError("{} collides with an exposed performance geometry".format(args.conv))

    source_hashes = {
        str(path.relative_to(REPO)): base.sha256_file(path)
        for path in base.SCHEDULE_SOURCES
    }
    schedule_version = canonical_sha256(source_hashes)
    fingerprint = base.hardware_fingerprint(env)
    tasks = base.create_tasks(workload, env)
    schema = args.schema or "c3_{}_label_isolated_contract_v1".format(
        args.workload_id.lower()
    )
    selection_seed = args.selection_seed or SELECTION_SEED
    original_seed = base.SELECTION_SEED
    try:
        base.SELECTION_SEED = selection_seed
        domain, selected, mode_domains = base.enumerate_shared_domain(
            args.workload_id, workload, tasks, env, args.families
        )
    finally:
        base.SELECTION_SEED = original_seed

    candidates = []
    for ordinal, row in enumerate(selected):
        family_id = "{}F{:02d}".format(args.workload_id, ordinal)
        for mode, mode_number in base.MODE_NUMBERS.items():
            identity = base.candidate_identity_record(
                fingerprint,
                base.TEMPLATE,
                schedule_version,
                workload,
                mode,
                row["complete_config_entity"],
                config_index=row["debug"]["config_index"],
            )
            candidates.append(
                {
                    "candidate_id": identity["candidate_id"],
                    "identity": identity["identity"],
                    "debug": identity["debug"],
                    "workload_id": args.workload_id,
                    "family_id": family_id,
                    "family_ordinal": ordinal,
                    "residence_mode": mode,
                    "mode_number": mode_number,
                    "selection_rank_sha256": row["sampling_rank_sha256"],
                    "hardware_predicate": row["hardware_predicate"],
                    "local_lower_status": "not_run_by_contract_generator",
                    "board_status": "not_dispatched",
                    "performance_label": None,
                }
            )
    if len(candidates) != 32:
        raise AssertionError("expected eight families by four legacy modes")

    contract = {
        "schema": schema,
        "status": "frozen_before_static_fsim_fpga_or_performance_observation",
        "workload_id": args.workload_id,
        "source_model": "yolov3-tiny",
        "source_conv": yolo[args.conv],
        "selection_rationale": args.selection_rationale,
        "workload": workload,
        "geometry_signature": signature,
        "exposed_performance_label_collision": False,
        "prior_performance_label_exposure": exposure,
        "hardware_fingerprint": fingerprint,
        "template": base.TEMPLATE,
        "schedule_source_sha256": source_hashes,
        "schedule_version": schedule_version,
        "complete_config_domain": mode_domains,
        "complete_original_domain_count": len(domain),
        "label_free_eligible_count": sum(row["hardware_predicate"]["passed"] for row in domain),
        "selection": {
            "seed": selection_seed,
            "families": args.families,
            "modes": base.MODE_NUMBERS,
            "rule": "capacity/reuse predicate, then ascending SHA256(seed|workload|semantic ConfigEntity)",
            "uses_static_dma_label": False,
            "uses_fsim": False,
            "uses_board_or_runtime_measurement": False,
            "uses_tophub": False,
            "candidate_commitment_sha256": canonical_sha256(candidates),
        },
        "future_protocol": {
            "local": "derive v2 original/input/weight-barrier identities; lower all; three-seed FSim all buildable",
            "board": "freeze the FSim-pass pool before RPC; correctness all; balanced timing all correct",
            "search": "same five-phase equal-budget protocol as P7R132/P7R144",
            "failure_policy": "all failures retained; no label-driven replacement",
        },
        "claim_boundary": "candidate identity freeze only; no legality, correctness, latency, DMA benefit, or search claim",
    }

    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    (output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates), encoding="utf-8"
    )
    (output / "complete_domain.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in domain), encoding="utf-8"
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": artifacts, "source_sha256": {str(Path(__file__).resolve()): sha256(__file__)}},
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-id", default="Y03")
    parser.add_argument("--conv", default="conv8")
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--schema")
    parser.add_argument("--selection-seed")
    parser.add_argument(
        "--selection-rationale",
        default="middle-spatial 3x3 convolution between prior Y00 and Y02 geometries",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
