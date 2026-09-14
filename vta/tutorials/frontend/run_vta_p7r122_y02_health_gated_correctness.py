#!/usr/bin/env python3
"""Health-gated, RPC-only correctness diagnostic for six P7R119 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import traceback
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
from run_vta_p7r121_tophub_adapter_diagnostic import board_seed, finalize


SCHEMA = "c3_p7r122_y02_health_gated_correctness_v1"
SEEDS = (0, 20250901, 20260910)
HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = C3 / "07_grouped_holdout"
DEFAULT_P7R120 = P7 / "20260911_p7r120_y02_rpc_contract_v2_run01" / "contract.json"
DEFAULT_P7R121 = P7 / "20260912_p7r121_tophub_adapter_diagnostic_run02"
DEFAULT_W05_EVIDENCE = P7 / "20260911_p7r80_w05_t2_hybrid_board_correctness_run01"
DEFAULT_W05_MANIFEST = (
    P7 / "20260911_p7r106_w05_ready_deployment_manifest_run02" / "deployment_manifest.json"
)
DEFAULT_W05_CONTRACT = (
    P7 / "20260911_p7r57_support_transfer_board_contracts_run01" / "w05_contract.json"
)
DEFAULT_OUTPUT = P7 / "20260912_p7r122_y02_health_gated_correctness_run01"


def verify_artifact(path):
    path = Path(path).resolve()
    ledger_path = path.parent / "artifact_hashes.json"
    ledger = load_json(ledger_path)
    if ledger["artifacts"].get(path.name) != sha256_file(path):
        raise ValueError("artifact hash mismatch: {}".format(path))
    return sha256_file(path), sha256_file(ledger_path)


def verify_run(run_dir):
    run_dir = Path(run_dir).resolve()
    ledger_path = run_dir / "artifact_hashes.json"
    ledger = load_json(ledger_path)["artifacts"]
    for name, expected in ledger.items():
        if sha256_file(run_dir / name) != expected:
            raise ValueError("run artifact hash mismatch: {}".format(run_dir / name))
    return sha256_file(ledger_path)


def knob_map(entry):
    return {
        name: int(value[-1] if kind == "sp" else value)
        for name, kind, value in entry["identity"]["complete_config_entity"]["entity"]
    }


def audit_knobs(candidates, p7r121_summary):
    rows = []
    for entry in candidates:
        rows.append(
            {
                "candidate_id": entry["candidate_id"],
                "family_id": entry["family_id"],
                "public_mode": entry["public_mode"],
                "knobs": knob_map(entry),
            }
        )
    if len(rows) != 6 or {row["family_id"] for row in rows} != {"Y02B00", "Y02B01"}:
        raise ValueError("P7R122 requires exactly the frozen six Y02 candidates")
    oc = {row["knobs"]["oc_nthread"] for row in rows}
    ht = {row["knobs"]["h_nthread"] for row in rows}
    if oc != {1} or ht != {1}:
        raise ValueError("unexpected frozen thread-knob stratum")
    return {
        "rows": rows,
        "six_candidate_common": {"oc_nthread": 1, "h_nthread": 1},
        "p7r121_failed_tophub": {
            "oc_nthread": 2,
            "h_nthread": 1,
            "status": p7r121_summary["status"],
        },
        "pre_board_inference": (
            "P7R121 failure is not evidence against oc_nthread=1: it tested and failed "
            "an oc_nthread=2 identity. P7R122 independently checks the oc_nthread=1 stratum."
        ),
    }


def w05_health_spec(manifest, workload):
    primary = next(
        row
        for row in manifest["deployment_identities"]
        if row["role"] == "primary" and row["residence_mode"] == "original"
    )
    if primary["config_index"] != 575:
        raise ValueError("expected historical W05 original config575")
    entity = dict(primary["complete_config_entity"])
    entity.pop("index", None)
    identity = {
        "schema": "c3_p7r122_w05_health_canary_identity_v1",
        "workload_id": "W05",
        "workload": workload,
        "implementation": "conv2d_packed_residency.vta mode0 original branch",
        "complete_config_entity": entity,
        "historical_candidate_id": primary["candidate_id"],
        "historical_tir_sha256": primary["tir_sha256"],
    }
    return {
        "candidate_id": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        "identity": identity,
        "debug": {"config_index": 575},
        "implementation_mode": 0,
        "public_mode": "original",
        "role": "health_gate_only",
        "performance_measurement": "not_collected",
    }


def prior_w05_pass(evidence_dir):
    rows = [
        json.loads(line)
        for line in (Path(evidence_dir) / "correctness.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matched = [row for row in rows if row["config_index"] == 575 and row["mode"] == "original"]
    if len(matched) != 2 or not all(
        len(row["seeds"]) == 3 and all(seed["correct"] for seed in row["seeds"])
        for row in matched
    ):
        raise ValueError("W05 original config575 lacks historical 3-seed before/after pass")
    return {
        "records": len(matched),
        "seed_checks": sum(len(row["seeds"]) for row in matched),
        "binary_sha256": sorted({row["binary_sha256"] for row in matched}),
        "tir_sha256": sorted({row["tir_sha256"] for row in matched}),
    }


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable diagnostic {}".format(output))
    p7r120_sha, p7r120_ledger = verify_artifact(args.p7r120_contract)
    p7r120 = load_json(args.p7r120_contract)
    p7r121_ledger = verify_run(args.p7r121_dir)
    p7r121_summary = load_json(Path(args.p7r121_dir) / "summary.json")
    if p7r121_summary["status"] != "both_fail_p7r120_remains_blocked":
        raise ValueError("P7R122 expects the frozen P7R121 both-fail result")
    w05_ledger = verify_run(args.w05_evidence_dir)
    w05_history = prior_w05_pass(args.w05_evidence_dir)
    manifest_sha, manifest_ledger = verify_artifact(args.w05_manifest)
    workload = load_json(args.w05_contract)["workload"]
    candidates = p7r120["candidate_pool"]["candidates"]
    by_id = {row["candidate_id"]: row for row in candidates}
    candidates = [by_id[candidate_id] for candidate_id in p7r120["correctness"]["candidate_order"]]
    audit = audit_knobs(candidates, p7r121_summary)
    health = w05_health_spec(load_json(args.w05_manifest), workload)
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_current_rpc_observations",
        "inputs": {
            "p7r120_contract": {"sha256": p7r120_sha, "ledger_sha256": p7r120_ledger},
            "p7r121_run": {"ledger_sha256": p7r121_ledger},
            "w05_prior_evidence": {"ledger_sha256": w05_ledger, **w05_history},
            "w05_manifest": {"sha256": manifest_sha, "ledger_sha256": manifest_ledger},
        },
        "knob_audit": audit,
        "health_canary": health,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "candidates": candidates,
        "seeds": list(SEEDS),
        "protocol": {
            "order": "W05 original health gate first; six frozen Y02 candidates only after 3/3 pass",
            "health_failure": "stop immediately without any Y02 candidate upload or dispatch",
            "each_invocation": "profiler_clear then fresh input/weight/output NDArrays",
            "candidate_correctness": "three seeds per identity; no replacement",
            "latency_or_performance_measurement": "forbidden",
            "ssh_restart_reconfiguration_persistent_write": "forbidden",
            "transport": "direct RPC ephemeral upload/load/remove only",
            "boot_id": "unknown_not_exposed_by_rpc",
        },
        "rpc_endpoint": {"host": args.host, "port": args.port},
    }
    if args.host != p7r120["rpc_contract"]["host"] or args.port != p7r120["rpc_contract"]["port"]:
        raise ValueError("RPC endpoint differs from frozen P7R120 endpoint")
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "pre_observation_hashes.json",
        {"artifacts": {"contract.json": sha256_file(output / "contract.json")}},
    )

    temporary = None
    candidate_dispatches = 0
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("P7R122 requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_p7r122_cross_")
        directory = Path(temporary.name)
        health_cert = build_and_export(health, env, directory)
        candidate_certs = {}
        for entry in candidates:
            expected = entry["fresh_axu_cross_certificate"]
            cert = build_and_export(entry, env, directory, expected["tir_sha256"])
            if cert["binary_sha256"] != expected["binary_sha256"]:
                raise RuntimeError("candidate cross binary changed")
            candidate_certs[entry["candidate_id"]] = cert
        write_json(
            output / "fresh_cross_certificates.json",
            {"health_canary": health_cert, "candidates": candidate_certs},
        )

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)

        health_module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
        health_rows = []
        for seed in SEEDS:
            row = board_seed(health_module["main"], device, workload, seed, functions)
            health_rows.append(row)
            if not row.get("correct"):
                break
        health_pass = len(health_rows) == 3 and all(row["correct"] for row in health_rows)
        write_json(
            output / "health_canary.json",
            {
                "identity": health,
                "status": "passed" if health_pass else "failed_stop_before_y02",
                "seeds": health_rows,
                "performance_measurement": "not_collected",
            },
        )

        candidate_rows = []
        if health_pass:
            for entry in candidates:
                module = load_ephemeral(remote, directory / (entry["candidate_id"] + ".so"))
                seeds = []
                for seed in SEEDS:
                    seeds.append(
                        board_seed(
                            module["main"], device, entry["identity"]["workload"], seed, functions
                        )
                    )
                    candidate_dispatches += 1
                candidate_rows.append(
                    {
                        "candidate_id": entry["candidate_id"],
                        "family_id": entry["family_id"],
                        "public_mode": entry["public_mode"],
                        "knobs": knob_map(entry),
                        "status": "passed" if all(seed.get("correct") for seed in seeds) else "failed",
                        "seeds": seeds,
                        "performance_measurement": "not_collected",
                    }
                )
                print(
                    "correctness {} {} {}".format(
                        entry["family_id"], entry["public_mode"], candidate_rows[-1]["status"]
                    ),
                    flush=True,
                )
        (output / "candidate_correctness.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows),
            encoding="utf-8",
        )
        summary = {
            "schema": SCHEMA,
            "status": (
                "health_gate_failed_stop_before_y02"
                if not health_pass
                else (
                    "y02_correctness_complete"
                    if len(candidate_rows) == 6
                    else "incomplete_fail_closed"
                )
            ),
            "w05_health_gate": {"passed": health_pass, "seed_checks": len(health_rows)},
            "y02": {
                "identities": len(candidate_rows),
                "passed": sum(row["status"] == "passed" for row in candidate_rows),
                "seed_checks": sum(len(row["seeds"]) for row in candidate_rows),
            },
            "candidate_dispatches": candidate_dispatches,
            "latency_samples": 0,
            "ssh_used": False,
            "board_restart_or_reconfiguration": False,
            "persistent_board_write": False,
            "boot_id": "unknown_not_exposed_by_rpc",
            "claim_boundary": "correctness/health diagnostic only; no latency or performance claim",
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
                "candidate_dispatches": candidate_dispatches,
                "latency_samples": 0,
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
    parser.add_argument("--p7r120-contract", type=Path, default=DEFAULT_P7R120)
    parser.add_argument("--p7r121-dir", type=Path, default=DEFAULT_P7R121)
    parser.add_argument("--w05-evidence-dir", type=Path, default=DEFAULT_W05_EVIDENCE)
    parser.add_argument("--w05-manifest", type=Path, default=DEFAULT_W05_MANIFEST)
    parser.add_argument("--w05-contract", type=Path, default=DEFAULT_W05_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
