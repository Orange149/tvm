#!/usr/bin/env python3
"""P7R126: exhaust the frozen, previously undispatched Y01 correctness domain.

This is a correctness-only RPC experiment.  It takes the exact complement of
the six P7R125 identities in the 24-identity P7R124 pool, gates dispatch on the
independent W05 health canary, and then runs all 18 identities at three seeds.
It deliberately contains no timing path and performs no replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes
from prepare_vta_p7r120_yolo_rpc_contract import (
    build_and_export,
    load_json,
    sha256_file,
    write_json,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r121_tophub_adapter_diagnostic import result_detail
from run_vta_p7r122_y02_health_gated_correctness import (
    DEFAULT_OUTPUT as DEFAULT_P7R122,
    verify_run,
)
from run_vta_p7r124_y01_v2_local_pool import DEFAULT_OUTPUT as DEFAULT_P7R124
from run_vta_p7r125_y01_complete_board_pool import DEFAULT_OUTPUT as DEFAULT_P7R125


SCHEMA = "c3_p7r126_y01_remaining_correctness_v1"
SEEDS = (0, 20250901, 20260910)
MODE_ORDER = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 2,
}
EXPECTED_FAMILIES = {"Y01F00", "Y01F01", "Y01F03", "Y01F04", "Y01F05", "Y01F06"}
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
DEFAULT_OUTPUT = P7 / "20260912_p7r126_y01_remaining_correctness_run01"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def finalize(output):
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": hashes,
            "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
        },
    )


def select_complement(p7r124_dir, p7r125_dir):
    """Return the exact P7R124 \ P7R125 identity complement in fixed order."""
    all_candidates = load_jsonl(Path(p7r124_dir) / "candidates_v2.jsonl")
    prior_contract = load_json(Path(p7r125_dir) / "contract.json")
    prior_candidates = prior_contract["candidate_pool"]["candidates"]
    all_ids = {row["candidate_id"] for row in all_candidates}
    prior_ids = {row["candidate_id"] for row in prior_candidates}
    if len(all_candidates) != 24 or len(all_ids) != 24:
        raise ValueError("P7R124 must contain 24 unique identities")
    if len(prior_candidates) != 6 or len(prior_ids) != 6:
        raise ValueError("P7R125 must contain six unique identities")
    if not prior_ids.issubset(all_ids):
        raise ValueError("P7R125 is not a subset of P7R124")
    selected = [row for row in all_candidates if row["candidate_id"] not in prior_ids]
    if len(selected) != 18 or {row["family_id"] for row in selected} != EXPECTED_FAMILIES:
        raise ValueError("unexpected remaining P7R124 family complement")
    for family in EXPECTED_FAMILIES:
        modes = {row["public_mode"] for row in selected if row["family_id"] == family}
        if modes != set(MODE_ORDER):
            raise ValueError("remaining family is not a complete three-mode family")
    if {row["candidate_id"] for row in selected} | prior_ids != all_ids:
        raise ValueError("P7R125 and P7R126 do not exhaust P7R124")
    return sorted(selected, key=lambda row: (row["family_id"], MODE_ORDER[row["public_mode"]]))


def qualification_bindings(p7r124_dir, candidates):
    static = {
        row["candidate_id"]: row
        for row in load_jsonl(Path(p7r124_dir) / "static_results.jsonl")
    }
    fsim = {
        row["candidate_id"]: row
        for row in load_jsonl(Path(p7r124_dir) / "fsim_results.jsonl")
    }
    bindings = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        srow = static[candidate_id]
        frow = fsim[candidate_id]
        evidence = frow.get("command_evidence", {})
        if srow["status"] != "ok":
            raise ValueError("P7R126 candidate lacks P7R124 static pass")
        if (
            frow["status"] != "passed"
            or len(frow.get("seeds", [])) != 3
            or not all(seed.get("correct") for seed in frow["seeds"])
            or evidence.get("status") != "available_three_seed_consistent"
        ):
            raise ValueError("P7R126 candidate lacks P7R124 three-seed FSim qualification")
        values = evidence["values"]
        bindings[candidate_id] = {
            "tir_sha256": srow["tir_sha256"],
            "static_feature_vector_sha256": canonical_sha256(srow["static_feature_vector"]),
            "command_signature_sha256": canonical_sha256(values),
            "command_signature": {
                "submissions": values["submissions"],
                "insn_peak_bytes": values["peaks"]["insn_bytes"],
                "uop_peak_bytes": values["peaks"]["uop_bytes"],
                "insn_total_bytes": values["totals"]["insn_bytes"],
                "uop_total_bytes": values["totals"]["uop_bytes"],
                "load_total_bytes": values["totals"]["load_bytes"],
                "store_total_bytes": values["totals"]["store_bytes"],
                "residency_drains": values["drain_check"]["observed_residency_drain_count"],
            },
        }
    return bindings, static


def full_profile_seed(function, device, workload, seed, functions):
    """One untimed, fresh-buffer exact check plus the complete runtime profile."""
    clear = functions.get("vta.runtime.profiler_clear")
    if clear is None:
        raise RuntimeError("profiler_clear is required by the frozen P7R126 contract")
    clear()
    try:
        row = result_detail(function, device, workload, seed)
        row["status"] = "passed" if row["correct"] else "wrong_answer"
    except Exception as error:  # Preserve an identity/seed failure and continue the exhaustion.
        row = {
            "seed": int(seed),
            "correct": False,
            "status": "execution_failed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
        }
    try:
        row["runtime_profile_complete"] = json.loads(
            functions["vta.runtime.profiler_status"]()
        )
    except Exception as error:
        row["runtime_profile_error"] = (str(error) or type(error).__name__)[:2000]
    return row


def grouped_association(rows, candidates, bindings):
    """Descriptive summaries inside this frozen domain; never a causal model."""
    by_id = {row["candidate_id"]: row for row in rows}
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    identity_rows = []
    for candidate_id in sorted(by_id):
        result = by_id[candidate_id]
        candidate = candidate_by_id[candidate_id]
        mismatches = [seed.get("mismatch_count") for seed in result["seeds"]]
        numeric_mismatches = [value for value in mismatches if value is not None]
        identity_rows.append(
            {
                "candidate_id": candidate_id,
                "family_id": candidate["family_id"],
                "public_mode": candidate["public_mode"],
                "knobs": candidate["knobs"],
                "command_signature": bindings[candidate_id]["command_signature"],
                "identity_pass_3_of_3": result["status"] == "passed",
                "seed_pass_count": sum(seed.get("correct", False) for seed in result["seeds"]),
                "mismatch_counts": mismatches,
                "median_mismatch_count": (
                    statistics.median(numeric_mismatches) if numeric_mismatches else None
                ),
            }
        )

    def group(field, getter):
        groups = defaultdict(list)
        for row in identity_rows:
            groups[str(getter(row))].append(row)
        return {
            field: {
                value: {
                    "identities": len(items),
                    "identity_pass_3_of_3": sum(item["identity_pass_3_of_3"] for item in items),
                    "seed_passes": sum(item["seed_pass_count"] for item in items),
                    "seed_checks": 3 * len(items),
                    "candidate_ids": [item["candidate_id"] for item in items],
                }
                for value, items in sorted(groups.items())
            }
        }

    grouped = {}
    for knob in ("tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread"):
        grouped.update(group("knob." + knob, lambda row, name=knob: row["knobs"][name]))
    grouped.update(group("public_mode", lambda row: row["public_mode"]))
    for feature in (
        "submissions", "insn_peak_bytes", "uop_peak_bytes", "load_total_bytes",
        "residency_drains",
    ):
        grouped.update(
            group("command." + feature, lambda row, name=feature: row["command_signature"][name])
        )
    return {
        "scope": "descriptive association inside the frozen P7R126 18-identity domain only",
        "causal_or_generalization_claim": False,
        "identity_rows": identity_rows,
        "grouped_counts": grouped,
    }


def summarize(rows, candidates, bindings):
    by_family = {}
    for family in sorted(EXPECTED_FAMILIES):
        family_rows = [row for row in rows if row["family_id"] == family]
        mode_rows = {row["public_mode"]: row for row in family_rows}
        all_three = len(mode_rows) == 3 and all(row["status"] == "passed" for row in mode_rows.values())
        by_family[family] = {
            "all_three_modes_correct_3_of_3": all_three,
            "passed_modes": sorted(
                mode for mode, row in mode_rows.items() if row["status"] == "passed"
            ),
            "modes": {
                mode: {
                    "candidate_id": row["candidate_id"],
                    "status": row["status"],
                    "seed_pass_count": sum(seed.get("correct", False) for seed in row["seeds"]),
                    "mismatch_counts": [seed.get("mismatch_count") for seed in row["seeds"]],
                }
                for mode, row in sorted(mode_rows.items(), key=lambda item: MODE_ORDER[item[0]])
            },
        }
    return {
        "families": by_family,
        "complete_three_mode_correct_family_count": sum(
            row["all_three_modes_correct_3_of_3"] for row in by_family.values()
        ),
        "association": grouped_association(rows, candidates, bindings),
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable P7R126 output {}".format(output))

    p7r124_ledger = verify_run(args.p7r124_dir)
    p7r125_ledger = verify_run(args.p7r125_dir)
    p7r122_ledger = verify_run(args.p7r122_dir)
    p7r124_contract = load_json(Path(args.p7r124_dir) / "contract.json")
    p7r125_contract = load_json(Path(args.p7r125_dir) / "contract.json")
    if p7r124_contract["local_protocol"]["board_contact"] != "forbidden":
        raise ValueError("unexpected P7R124 local protocol")
    if p7r125_contract["rpc"] != {
        "host": args.host, "port": args.port, "boot_id": "unknown_not_exposed_by_rpc"
    }:
        raise ValueError("RPC endpoint differs from the bound P7R125 contract")
    observed_guards = {
        name: sha256_file(REPO / name) for name in p7r124_contract["source_guards_sha256"]
    }
    if observed_guards != p7r124_contract["source_guards_sha256"]:
        raise ValueError("P7R124 source/hardware guards changed")

    candidates = select_complement(args.p7r124_dir, args.p7r125_dir)
    bindings, static = qualification_bindings(args.p7r124_dir, candidates)
    p7r122_contract = load_json(Path(args.p7r122_dir) / "contract.json")
    health = p7r122_contract["health_canary"]
    candidate_ids = [row["candidate_id"] for row in candidates]
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_cross_compile_or_rpc_observations",
        "inputs": {
            "p7r124_dir": str(Path(args.p7r124_dir).resolve()),
            "p7r124_artifact_ledger_sha256": p7r124_ledger,
            "p7r125_dir": str(Path(args.p7r125_dir).resolve()),
            "p7r125_artifact_ledger_sha256": p7r125_ledger,
            "p7r122_health_identity_ledger_sha256": p7r122_ledger,
        },
        "source_guards_sha256": p7r124_contract["source_guards_sha256"],
        "health_canary": {
            "entry": health,
            "role": "independent current-session health gate only",
            "candidate_or_training_role": False,
            "performance_baseline_role": False,
        },
        "candidate_pool": {
            "construction": "exact P7R124 24-identity pool minus frozen P7R125 six identities",
            "families": sorted(EXPECTED_FAMILIES),
            "modes": list(MODE_ORDER),
            "gross_identities": 18,
            "frozen_order": candidate_ids,
            "commitment_sha256": canonical_sha256(candidates),
            "candidates": candidates,
            "qualification_bindings": bindings,
            "no_replacement": True,
            "one_time_exhaustion_not_post_failure_selection": True,
        },
        "partition_proof": {
            "p7r124_gross": 24,
            "p7r125_gross": 6,
            "p7r126_gross": 18,
            "p7r125_p7r126_disjoint": True,
            "union_equals_p7r124": True,
        },
        "correctness": {
            "seeds": list(SEEDS),
            "gross_checks": 54,
            "dispatch": "all 18 identities x all three seeds after W05 3/3 health pass",
            "each_call": "profiler_clear; fresh input, weight, and output buffers; exact output check",
        },
        "prohibitions": {
            "timing_or_time_evaluator": True,
            "Y01_TopHub_query_reference_or_selection": True,
            "replacement_after_failure": True,
            "ssh_restart_reconfiguration_persistent_write": True,
        },
        "analysis_boundary": (
            "knob and command-signature associations are descriptive within this frozen domain only"
        ),
        "rpc": {"host": args.host, "port": args.port, "boot_id": "unknown_not_exposed_by_rpc"},
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "pre_cross_hashes.json",
        {"artifacts": {"contract.json": sha256_file(output / "contract.json")}},
    )

    temporary = None
    correctness_rows = []
    dispatches = 0
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R126 requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r126_cross_")
        directory = Path(temporary.name)
        certificates = {"health_canary": build_and_export(health, env, directory), "candidates": {}}
        for entry in candidates:
            certificates["candidates"][entry["candidate_id"]] = build_and_export(
                entry, env, directory, static[entry["candidate_id"]]["tir_sha256"]
            )
        write_json(output / "fresh_cross_certificates.json", certificates)
        write_json(
            output / "pre_rpc_hashes.json",
            {"artifacts": {
                "contract.json": sha256_file(output / "contract.json"),
                "fresh_cross_certificates.json": sha256_file(output / "fresh_cross_certificates.json"),
            }},
        )

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        if "vta.runtime.profiler_clear" not in functions:
            raise RuntimeError("profiler_clear is required")
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)

        health_module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
        health_rows = []
        for seed in SEEDS:
            row = full_profile_seed(
                health_module["main"], device, health["identity"]["workload"], seed, functions
            )
            health_rows.append(row)
            if not row.get("correct"):
                break
        health_pass = len(health_rows) == 3 and all(row.get("correct") for row in health_rows)
        write_json(
            output / "health_gate.json",
            {
                "status": "passed" if health_pass else "failed_stop_before_y01",
                "seeds": health_rows,
                "performance_measurement": "not_collected",
            },
        )
        if not health_pass:
            raise RuntimeError("W05 health canary failed; stop before P7R126 candidate dispatch")

        with (output / "correctness.jsonl").open("x", encoding="utf-8") as stream:
            for entry in candidates:
                module = load_ephemeral(remote, directory / (entry["candidate_id"] + ".so"))
                seeds = []
                for seed in SEEDS:
                    seeds.append(
                        full_profile_seed(
                            module["main"], device, entry["identity"]["workload"], seed, functions
                        )
                    )
                    dispatches += 1
                row = {
                    "schema": SCHEMA,
                    "candidate_id": entry["candidate_id"],
                    "family_id": entry["family_id"],
                    "public_mode": entry["public_mode"],
                    "knobs": entry["knobs"],
                    "qualification_binding": bindings[entry["candidate_id"]],
                    "status": "passed" if all(seed.get("correct") for seed in seeds) else "failed",
                    "seeds": seeds,
                    "performance_measurement": "not_collected",
                }
                correctness_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print(
                    "correctness {} {} {}".format(
                        entry["family_id"], entry["public_mode"], row["status"]
                    ),
                    flush=True,
                )

        if len(correctness_rows) != 18 or dispatches != 54:
            raise RuntimeError("P7R126 did not complete the frozen 18x3 exhaustion")
        analysis = summarize(correctness_rows, candidates, bindings)
        write_json(output / "within_domain_association.json", analysis)
        summary = {
            "schema": SCHEMA,
            "status": "y01_remaining_domain_correctness_complete",
            "w05_health_gate": {"passed": True, "seed_checks": 3},
            "y01_remaining_domain": {
                "families": 6,
                "identities": 18,
                "passed_identities": sum(row["status"] == "passed" for row in correctness_rows),
                "seed_checks": dispatches,
                "complete_three_mode_correct_family_count": analysis[
                    "complete_three_mode_correct_family_count"
                ],
            },
            "timing_samples": 0,
            "y01_tophub_used": False,
            "replacement_used": False,
            "ssh_used": False,
            "board_restart_or_reconfiguration": False,
            "persistent_board_write": False,
            "boot_id": "unknown_not_exposed_by_rpc",
            "claim_boundary": (
                "complete correctness scan and descriptive association for the frozen P7R126 domain only; "
                "no performance, causality, or cross-domain generalization claim"
            ),
        }
        write_json(output / "summary.json", summary)
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        finalize(output)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed_closed",
                "exception_type": type(error).__name__,
                "message": (str(error) or type(error).__name__)[:4000],
                "traceback": traceback.format_exc()[-12000:],
                "candidate_identities_completed": len(correctness_rows),
                "candidate_dispatches": dispatches,
                "timing_samples": 0,
                "ssh_used": False,
            },
        )
        finalize(output)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7r124-dir", type=Path, default=DEFAULT_P7R124)
    parser.add_argument("--p7r125-dir", type=Path, default=DEFAULT_P7R125)
    parser.add_argument("--p7r122-dir", type=Path, default=DEFAULT_P7R122)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
