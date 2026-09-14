#!/usr/bin/env python3
"""Recover only the interrupted final identity of P7R126 run01.

The first immutable attempt completed 17/18 identities before its RPC stream
was interrupted.  This script binds that failure, repeats no completed
identity, reruns the W05 health gate, and records each of the three missing
identity checks immediately before producing a combined 18-identity report.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export, load_json, sha256_file, write_json
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r126_y01_remaining_correctness import (
    DEFAULT_OUTPUT as DEFAULT_INTERRUPTED,
    REPO,
    SCHEMA as PARENT_SCHEMA,
    SEEDS,
    full_profile_seed,
    summarize,
)
from run_vta_p7r122_y02_health_gated_correctness import verify_run


SCHEMA = "c3_p7r126_y01_remaining_correctness_recovery_v1"
HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
DEFAULT_OUTPUT = P7 / "20260912_p7r126_y01_remaining_correctness_run02"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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


def recovery_plan(interrupted_dir):
    interrupted_dir = Path(interrupted_dir)
    contract = load_json(interrupted_dir / "contract.json")
    failure = load_json(interrupted_dir / "failure.json")
    rows = load_jsonl(interrupted_dir / "correctness.jsonl")
    if contract.get("schema") != PARENT_SCHEMA or failure.get("status") != "failed_closed":
        raise ValueError("P7R126 recovery requires the frozen failed run01")
    completed = {row["candidate_id"] for row in rows}
    candidates = contract["candidate_pool"]["candidates"]
    remaining = [row for row in candidates if row["candidate_id"] not in completed]
    if len(rows) != 17 or any(len(row["seeds"]) != 3 for row in rows):
        raise ValueError("expected exactly 17 complete identities in interrupted run")
    if len(remaining) != 1:
        raise ValueError("recovery must contain exactly one remaining identity")
    candidate = remaining[0]
    if candidate["family_id"] != "Y01F06" or candidate["public_mode"] != "weight_resident_barrier":
        raise ValueError("unexpected interrupted identity")
    if failure.get("candidate_dispatches") != 52:
        raise ValueError("unexpected interrupted dispatch count")
    return contract, failure, rows, candidate


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable recovery {}".format(output))
    interrupted_ledger = verify_run(args.interrupted_dir)
    parent, failure, prior_rows, candidate = recovery_plan(args.interrupted_dir)
    observed_guards = {
        name: sha256_file(REPO / name) for name in parent["source_guards_sha256"]
    }
    if observed_guards != parent["source_guards_sha256"]:
        raise ValueError("source/hardware guards changed since interrupted contract")
    if parent["rpc"]["host"] != args.host or parent["rpc"]["port"] != args.port:
        raise ValueError("RPC endpoint changed")
    health = parent["health_canary"]["entry"]
    binding = parent["candidate_pool"]["qualification_bindings"][candidate["candidate_id"]]
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_recovery_cross_compile_or_rpc_observations",
        "interrupted_run": {
            "path": str(Path(args.interrupted_dir).resolve()),
            "artifact_ledger_sha256": interrupted_ledger,
            "contract_sha256": sha256_file(Path(args.interrupted_dir) / "contract.json"),
            "correctness_sha256": sha256_file(Path(args.interrupted_dir) / "correctness.jsonl"),
            "failure_sha256": sha256_file(Path(args.interrupted_dir) / "failure.json"),
            "complete_identities": 17,
            "complete_seed_checks": 51,
            "reported_candidate_dispatch_attempts": failure["candidate_dispatches"],
        },
        "health_canary": parent["health_canary"],
        "recovery_identity": candidate,
        "qualification_binding": binding,
        "protocol": {
            "health_gate": "W05 must pass all three seeds before recovery identity",
            "recovery_checks": "Y01F06 weight_resident_barrier at all three frozen seeds",
            "per_seed_durable_record": True,
            "completed_identity_repeated": False,
            "replacement": False,
            "timing": "forbidden",
            "Y01_TopHub": "forbidden",
            "transport": "direct RPC ephemeral upload/load/remove only",
            "ssh_restart_reconfiguration_persistent_write": "forbidden",
        },
        "deviation_disclosure": (
            "run01 reported 52 candidate attempts but durably recorded 51 checks; this recovery "
            "repeats the interrupted identity's full three-seed set, so the combined complete "
            "dataset has 54 auditable checks while gross reported attempts are 55"
        ),
        "rpc": {"host": args.host, "port": args.port, "boot_id": "unknown_not_exposed_by_rpc"},
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(output / "pre_cross_hashes.json", {"artifacts": {"contract.json": sha256_file(output / "contract.json")}})

    temporary = None
    recovery_seeds = []
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R126 recovery requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r126_recovery_cross_")
        directory = Path(temporary.name)
        certificates = {
            "health_canary": build_and_export(health, env, directory),
            "recovery_identity": build_and_export(candidate, env, directory, binding["tir_sha256"]),
        }
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
            {"status": "passed" if health_pass else "failed_stop", "seeds": health_rows},
        )
        if not health_pass:
            raise RuntimeError("W05 recovery health gate failed")

        module = load_ephemeral(remote, directory / (candidate["candidate_id"] + ".so"))
        function = module["main"]
        with (output / "recovery_seed_checks.jsonl").open("x", encoding="utf-8") as stream:
            for seed in SEEDS:
                row = full_profile_seed(
                    function, device, candidate["identity"]["workload"], seed, functions
                )
                recovery_seeds.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
        recovered_row = {
            "schema": PARENT_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "family_id": candidate["family_id"],
            "public_mode": candidate["public_mode"],
            "knobs": candidate["knobs"],
            "qualification_binding": binding,
            "status": "passed" if all(row.get("correct") for row in recovery_seeds) else "failed",
            "seeds": recovery_seeds,
            "performance_measurement": "not_collected",
            "recovery_provenance": SCHEMA,
        }
        write_json(output / "recovered_identity.json", recovered_row)
        combined = prior_rows + [recovered_row]
        if len(combined) != 18 or sum(len(row["seeds"]) for row in combined) != 54:
            raise RuntimeError("combined P7R126 audit is not 18 identities x 3 seeds")
        (output / "combined_correctness.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined),
            encoding="utf-8",
        )
        bindings = parent["candidate_pool"]["qualification_bindings"]
        candidates = parent["candidate_pool"]["candidates"]
        analysis = summarize(combined, candidates, bindings)
        write_json(output / "within_domain_association.json", analysis)
        summary = {
            "schema": SCHEMA,
            "status": "recovered_y01_remaining_domain_correctness_complete",
            "w05_recovery_health_gate": {"passed": True, "seed_checks": 3},
            "combined": {
                "families": 6,
                "identities": 18,
                "passed_identities": sum(row["status"] == "passed" for row in combined),
                "auditable_seed_checks": 54,
                "complete_three_mode_correct_family_count": analysis[
                    "complete_three_mode_correct_family_count"
                ],
            },
            "recovery_candidate_checks": 3,
            "gross_candidate_attempts_including_interrupted_retry": 55,
            "timing_samples": 0,
            "y01_tophub_used": False,
            "replacement_used": False,
            "completed_identity_repeated": False,
            "ssh_used": False,
            "board_restart_or_reconfiguration": False,
            "persistent_board_write": False,
            "boot_id": "unknown_not_exposed_by_rpc",
            "claim_boundary": "frozen Y01 P7R126 correctness domain only; no performance or generalization claim",
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
                "recovery_seed_checks_completed": len(recovery_seeds),
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
    parser.add_argument("--interrupted-dir", type=Path, default=DEFAULT_INTERRUPTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
