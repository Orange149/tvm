#!/usr/bin/env python3
"""Refreeze Y03 families using original-schedule lowering as a validity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from run_vta_p7r119_yolo_barrier_pilot import candidate_identity, config_knobs, static_result


SCHEMA = "c3_p7r148_y03_lowerable_refreeze_v1"
HERE = Path(__file__).resolve().parent
P7 = (
    HERE
    / "report_out"
    / "stage_tile_cotuning"
    / "c3_dma_residency_autotune"
    / "07_grouped_holdout"
)
DEFAULT_INPUT = P7 / "20260912_p7r146_y03_conv8_label_isolated_contract_run01"
DEFAULT_OUTPUT = P7 / "20260912_p7r148_y03_original_lowerable_refreeze_run01"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(directory):
    ledger = read_json(Path(directory) / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError("input artifact hash mismatch: {}".format(name))
    return sha256(Path(directory) / "artifact_hashes.json")


def original_candidate(contract, domain_row):
    identity = {
        "schema": "c3_candidate_id_v1",
        "hardware_fingerprint": contract["hardware_fingerprint"],
        "template_name": contract["template"],
        "schedule_version": contract["schedule_version"],
        "workload": contract["workload"],
        "residence_mode": "original",
        "complete_config_entity": domain_row["complete_config_entity"],
    }
    base = {
        "identity": identity,
        "complete_config_entity": domain_row["complete_config_entity"],
        "debug": domain_row["debug"],
    }
    semantic = candidate_identity(base, "original", 0)
    return {
        **semantic,
        "schema": SCHEMA,
        "workload_id": contract["workload_id"],
        "family_id": "{}-scan-{}".format(
            contract["workload_id"], domain_row["semantic_config_sha256"][:12]
        ),
        "family_ordinal": -1,
        "public_mode": "original",
        "implementation_mode": 0,
        "knobs": config_knobs(domain_row["complete_config_entity"]),
        "residence_mode": "original",
        "mode_number": 0,
        "hardware_predicate": domain_row["hardware_predicate"],
        "selection_rank_sha256": domain_row["sampling_rank_sha256"],
        "local_lower_status": "pending_refreeze_scan",
        "board_status": "not_dispatched",
        "performance_label": None,
    }


def run(args):
    input_dir = Path(args.input_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable refreeze {}".format(output))
    input_ledger = verify(input_dir)
    contract = read_json(input_dir / "contract.json")
    if contract["status"] != "frozen_before_static_fsim_fpga_or_performance_observation":
        raise ValueError("input contract is not a pre-observation freeze")
    domain = [
        json.loads(line)
        for line in (input_dir / "complete_domain.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = sorted(
        (row for row in domain if row["hardware_predicate"]["passed"]),
        key=lambda row: (row["sampling_rank_sha256"], row["semantic_config_sha256"]),
    )
    if len(eligible) != contract["label_free_eligible_count"]:
        raise ValueError("eligible-domain count changed")

    output.mkdir(parents=True)
    scan_rows = []
    selected = []
    started = time.perf_counter()
    with (output / "original_lowering_scan.jsonl").open("x", encoding="utf-8") as stream:
        for gross_index, domain_row in enumerate(eligible, start=1):
            candidate = original_candidate(contract, domain_row)
            begin = time.perf_counter()
            result = static_result(candidate)
            wall_ms = (time.perf_counter() - begin) * 1000.0
            record = {
                "schema": SCHEMA,
                "gross_scan_index": gross_index,
                "candidate_id": candidate["candidate_id"],
                "selection_rank_sha256": candidate["selection_rank_sha256"],
                "status": result["status"],
                "wall_ms": wall_ms,
                "failure": result.get("failure"),
            }
            scan_rows.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print("scan {}/{} {}".format(gross_index, len(eligible), result["status"]), flush=True)
            if result["status"] == "ok":
                candidate["family_ordinal"] = len(selected)
                candidate["family_id"] = "{}F{:02d}".format(contract["workload_id"], len(selected))
                candidate["local_lower_status"] = "passed_original_refreeze_gate"
                selected.append(candidate)
                if len(selected) == args.families:
                    break
    if len(selected) != args.families:
        raise RuntimeError("only {} original schedules lower successfully".format(len(selected)))

    selection_contract = {
        "schema": SCHEMA,
        "status": "refrozen_after_original_lowering_before_mode_lowering_fsim_fpga_or_performance",
        "input": {"path": str(input_dir.resolve()), "artifact_ledger_sha256": input_ledger},
        "workload_id": contract["workload_id"],
        "workload": contract["workload"],
        "geometry_signature": contract["geometry_signature"],
        "hardware_fingerprint": contract["hardware_fingerprint"],
        "selection": {
            "rule": "frozen hash order; accept only original schedule with successful actual lowering; stop after eight",
            "gross_hardware_eligible_considered": len(scan_rows),
            "original_lowering_pass": len(selected),
            "original_lowering_fail": sum(row["status"] != "ok" for row in scan_rows),
            "scan_wall_ms": (time.perf_counter() - started) * 1000.0,
            "uses_dma_magnitude_or_shape_for_ordering": False,
            "uses_fsim": False,
            "uses_board_or_performance_label": False,
            "uses_tophub": False,
            "failure_policy": "all scanned failures retained; P7R146/P7R147 remain immutable negative evidence",
        },
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "claim_boundary": "original lowering-valid identity refreeze only; no mode legality, FSim, FPGA, latency, or search claim",
    }
    write_json(output / "contract.json", selection_contract)
    (output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8"
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
    print(json.dumps(selection_contract, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
