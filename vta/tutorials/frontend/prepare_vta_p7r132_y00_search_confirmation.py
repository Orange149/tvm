#!/usr/bin/env python3
"""Freeze the Y00 prospective shared-memory search confirmation pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from replay_vta_equal_budget_search import POOL_SCHEMA, validate_pool


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_LOCAL = P7 / "20260912_p7r127_y00_v2_local_pool_run01"
DEFAULT_FACTOR = P7 / "20260912_p7r130_tile_dma_factor_map_run02"
DEFAULT_DEVELOPMENT = P7 / "20260912_p7r131_search_rule_ablation_p7q_development_run02"
DEFAULT_HEALTH = P7 / "20260912_p7r126_y01_remaining_correctness_run03"
DEFAULT_OUTPUT = P7 / "20260912_p7r132_y00_search_confirmation_contract_run02"
KNOBS = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread")
BASE_DMA = (
    "input_dma_bytes", "weight_dma_bytes", "output_dma_bytes",
    "input_dma_calls", "weight_dma_calls", "output_dma_calls",
)
REQUEST = BASE_DMA + (
    "small_dma_calls", "strided_dma_calls", "padded_dma_calls",
    "input_reload", "weight_reload", "output_reload",
)
COMMAND = REQUEST + ("insn_peak_bytes", "uop_peak_bytes", "submissions", "residency_drains")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(directory):
    directory = Path(directory)
    ledger_path = directory / "artifact_hashes.json"
    ledger = read_json(ledger_path)
    entries = ledger.get("artifacts", ledger.get("files"))
    if not isinstance(entries, dict):
        raise ValueError("unknown artifact ledger schema: " + str(ledger_path))
    for name, expected in entries.items():
        if sha256(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256(ledger_path)


def command_values(fsim):
    values = fsim["command_evidence"]["values"]
    return {
        "insn_peak_bytes": float(values["peaks"]["insn_bytes"]),
        "uop_peak_bytes": float(values["peaks"]["uop_bytes"]),
        "submissions": float(values["submissions"]),
    }


def feature_values(static, fsim):
    values = {name: float(static["static_feature_vector"][name]) for name in REQUEST}
    values.update(command_values(fsim))
    values["residency_drains"] = float((static.get("sync") or {}).get("residency_drains", 0))
    values["dma_total_bytes"] = (
        values["input_dma_bytes"] + values["weight_dma_bytes"] + values["output_dma_bytes"]
    )
    values["dma_total_calls"] = (
        values["input_dma_calls"] + values["weight_dma_calls"] + values["output_dma_calls"]
    )
    sram = static["sram_features"]
    for memory, name in (
        ("input", "input_vectors"), ("weight", "weight_vectors"),
        ("accumulator", "accumulator_vectors"),
    ):
        values[f"{memory}_sram_required_vectors"] = float(sram["required_vectors"][name])
        values[f"{memory}_sram_headroom_vectors"] = float(sram["headroom_vectors"][name])
    return values


def delta(values, control):
    return {name: float(values[name] - control[name]) for name in COMMAND}


def build_pool(candidates, static_rows, fsim_rows, workload_id="Y00"):
    by_static = {row["candidate_id"]: row for row in static_rows}
    by_fsim = {row["candidate_id"]: row for row in fsim_rows}
    by_candidate = {row["candidate_id"]: row for row in candidates}
    eligible_ids = sorted(
        candidate_id for candidate_id, fsim in by_fsim.items()
        if fsim.get("status") == "passed" and by_static[candidate_id].get("status") == "ok"
    )
    features = {
        candidate_id: feature_values(by_static[candidate_id], by_fsim[candidate_id])
        for candidate_id in eligible_ids
    }
    family_original = {
        row["family_id"]: row["candidate_id"] for row in candidates if row["public_mode"] == "original"
    }
    public = []
    for candidate_id in eligible_ids:
        source = by_candidate[candidate_id]
        control_id = family_original[source["family_id"]]
        control_values = None
        if control_id in by_static and by_static[control_id].get("status") == "ok":
            control_fsim = by_fsim.get(control_id)
            if control_fsim and control_fsim.get("status") == "passed":
                control_values = features[control_id]
        if control_values is None:
            control_values = features[candidate_id]
        values = features[candidate_id]
        knobs = {name: float(source["knobs"][name]) for name in KNOBS}
        validity = dict(knobs)
        validity.update({"static_" + name: float(value) for name, value in values.items()})
        public.append({
            "candidate_id": candidate_id,
            "workload_id": workload_id,
            "residence_mode": source["public_mode"],
            "same_tile_control_id": control_id,
            "predispatch": {
                "feature_provenance": "frozen_before_target_labels",
                "rule_score": values["dma_total_bytes"],
                "knobs": knobs,
                "validity": validity,
                "delta_t": delta(values, control_values),
            },
        })
    pool = {
        "schema": POOL_SCHEMA,
        "pool_id": "{}_full_fsim_pass_pool".format(workload_id.lower()),
        "frozen_before_target_labels": True,
        "claim_status": "prospective_{}_board_labels_unobserved".format(workload_id.lower()),
        "delta_feature_sets": {
            "bytes_calls": list(BASE_DMA),
            "request": list(REQUEST),
            "command": list(COMMAND),
        },
        "workloads": {
            workload_id: {
                "sealed_reference": {
                    "reference_kind": "full_pool_oracle",
                    "reference_id": None,
                    "exact_identity": False,
                    "latency_ms": None,
                    "role": "derived_only_after_all_correct_candidates_are_timed",
                },
                "candidates": public,
            }
        },
    }
    validate_pool(pool, require_oracle=False)
    return pool, eligible_ids


def lexicographic_order(pool, workload_id="Y00"):
    candidates = pool["workloads"][workload_id]["candidates"]
    return [row["candidate_id"] for row in sorted(candidates, key=lambda row: (
        row["predispatch"]["validity"]["static_dma_total_bytes"],
        row["predispatch"]["validity"]["static_dma_total_calls"],
        row["predispatch"]["validity"]["static_padded_dma_calls"],
        row["predispatch"]["validity"]["static_submissions"],
        row["candidate_id"],
    ))]


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable contract " + str(output))
    local = Path(args.local_dir)
    factor = Path(args.factor_dir)
    development = Path(args.development_dir)
    health_dir = Path(args.health_dir)
    bindings = {
        "local_v2": {"path": str(local.resolve()), "ledger_sha256": verify_run(local)},
        "factor_map": {"path": str(factor.resolve()), "ledger_sha256": verify_run(factor)},
        "development_ablation": {
            "path": str(development.resolve()), "ledger_sha256": verify_run(development)
        },
        "health_identity": {"path": str(health_dir.resolve()), "ledger_sha256": verify_run(health_dir)},
    }
    exposure_runs = tuple(Path(path) for path in getattr(args, "exposure_run", ()))
    if exposure_runs:
        bindings["historical_exposure_audit"] = [
            {"path": str(path.resolve()), "ledger_sha256": verify_run(path)}
            for path in exposure_runs
        ]
    candidates = read_jsonl(local / "candidates_v2.jsonl")
    static_rows = read_jsonl(local / "static_results.jsonl")
    fsim_rows = read_jsonl(local / "fsim_results.jsonl")
    workload_id = args.workload_id
    recovery_latency_unseen = bool(getattr(args, "recovery_latency_unseen", False))
    if any(row.get("performance_label") is not None for row in candidates + static_rows + fsim_rows):
        raise ValueError("{} target performance label was exposed before freeze".format(workload_id))
    pool, eligible_ids = build_pool(candidates, static_rows, fsim_rows, workload_id)
    if recovery_latency_unseen:
        pool["claim_status"] = "recovery_{}_latency_unseen_labels_unobserved".format(
            workload_id.lower()
        )
    by_id = {row["candidate_id"]: row for row in candidates}
    selected_families = tuple(
        item.strip() for item in str(getattr(args, "selected_families", "")).split(",")
        if item.strip()
    )
    if selected_families:
        unknown = set(selected_families).difference(row["family_id"] for row in candidates)
        if unknown:
            raise ValueError("unknown selected families: " + ",".join(sorted(unknown)))
        eligible_ids = [
            candidate_id for candidate_id in eligible_ids
            if by_id[candidate_id]["family_id"] in selected_families
        ]
        if not eligible_ids:
            raise ValueError("selected families have no eligible candidates")
        pool["pool_id"] += "_selected_" + "_".join(selected_families).lower()
        pool["workloads"][workload_id]["candidates"] = [
            row for row in pool["workloads"][workload_id]["candidates"]
            if row["candidate_id"] in set(eligible_ids)
        ]
        validate_pool(pool, require_oracle=False)
    search_order = lexicographic_order(pool, workload_id)
    board_order = sorted(eligible_ids, key=lambda candidate_id: hashlib.sha256(
        ("prospective-board-collection:" + workload_id + ":" + candidate_id).encode()
    ).hexdigest())
    health_canary = read_json(health_dir / "contract.json")["health_canary"]
    health_entry = health_canary.get("entry", health_canary)
    board_contract = {
        "schema": "c3_prospective_search_confirmation_contract_v2",
        "status": (
            "frozen_before_{}_latency_recovery_confirmation".format(workload_id.lower())
            if recovery_latency_unseen else
            "frozen_before_{}_fpga_correctness_or_latency".format(workload_id.lower())
        ),
        "workload_id": workload_id,
        "gross_config_identities": len(candidates),
        "static_pass": sum(row.get("status") == "ok" for row in static_rows),
        "fsim_pass_board_pool": len(eligible_ids),
        "selected_families": list(selected_families),
        "selection_rationale": getattr(args, "selection_rationale", None),
        "candidate_order_for_unbiased_full_label_collection": board_order,
        "shared_memory_lexicographic_order": search_order,
        "candidates": [by_id[candidate_id] for candidate_id in board_order],
        "qualification_bindings": {
            candidate_id: {
                "tir_sha256": next(
                    row["tir_sha256"] for row in static_rows if row["candidate_id"] == candidate_id
                ),
                "fsim_command_sha256": next(
                    row["command_evidence"]["per_seed_sha256"]
                    for row in fsim_rows if row["candidate_id"] == candidate_id
                ),
                "fsim_seed_correctness": next(
                    [{"seed": item["seed"], "correct": item["correct"]} for item in row["seeds"]]
                    for row in fsim_rows if row["candidate_id"] == candidate_id
                ),
            }
            for candidate_id in board_order
        },
        "health_canary": health_entry,
        "protocol": {
            "precondition": (
                "stop default RPC, reload exact frozen bitstream, start a fresh default RPC, "
                "then require W05 config575 three-seed health pass before target allocation"
            ),
            "correctness": "all {} candidates, frozen order, first-error stop per identity, pass requires three seeds".format(len(eligible_ids)),
            "timing": "all correctness-passed candidates, seven balanced rotated rounds",
            "pool_oracle": "minimum median latency among all correctness-passed candidates",
            "search_replay": "20 seeds; budgets 1,2,4,8; target pool-oracle +2%/+5%",
            "baselines": [
                "random", "stock_knob_xgb", "paper_minimum_access",
                "rules_only_bytes", "shared_memory_lexicographic",
                "shared_memory_service_proxy_64KiB_per_request_128KiB_per_extra_submission",
            ],
            "tophub": "not used by selector and not required for the primary endpoint",
            "current_boot_result": "not observed at contract freeze",
        },
        "claim_boundary": (
            "latency-unseen recovery confirmation only: historical FPGA correctness outcomes were "
            "already exposed, but no target latency label existed; candidate identities and service-proxy "
            "weights remain frozen"
            if recovery_latency_unseen else
            "prospective performance-ranking confirmation only; local static/FSim filtering was "
            "already observed and is reported separately"
        ),
        "historical_exposure_policy": (
            "historical correctness may be reported, but no runtime profile proxy or candidate latency "
            "was used to choose the selected families or fit the frozen service proxy"
            if recovery_latency_unseen else None
        ),
    }
    output.mkdir(parents=True)
    files = {
        "input_bindings.json": json.dumps(bindings, indent=2, sort_keys=True) + "\n",
        "prospective_pool.json": json.dumps(pool, indent=2, sort_keys=True) + "\n",
        "board_collection_contract.json": json.dumps(board_contract, indent=2, sort_keys=True) + "\n",
        "STATUS.md": (
            "# {} prospective search confirmation\n\n".format(workload_id)
            + (
                "- Frozen after historical correctness exposure but before any {} latency label; this is a latency-unseen recovery confirmation.\n".format(workload_id)
                if recovery_latency_unseen else
                "- Frozen before any {} FPGA correctness or latency label.\n".format(workload_id)
            )
            + f"- Original identities: {len(candidates)}; static pass: {board_contract['static_pass']}; "
            + f"FSim-pass board pool: {len(eligible_ids)}.\n"
            + "- Primary endpoint: board dispatches needed to reach the completed pool oracle +2%/+5%.\n"
            + "- TopHub is not a search target; execution awaits the frozen board protocol.\n"
        ),
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    hashes = {name: sha256(output / name) for name in sorted(files)}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes, "source_sha256": sha256(__file__)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return board_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--workload-id", default="Y00")
    parser.add_argument("--factor-dir", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--development-dir", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--health-dir", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recovery-latency-unseen", action="store_true",
        help="attest that prior correctness was exposed but target latency remains unseen",
    )
    parser.add_argument(
        "--selected-families", default="",
        help="optional comma-separated pre-registered family subset",
    )
    parser.add_argument("--selection-rationale")
    parser.add_argument(
        "--exposure-run", action="append", type=Path, default=[],
        help="immutable historical result run included only for exposure auditing",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
