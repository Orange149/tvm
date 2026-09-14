#!/usr/bin/env python3
"""Build an exact, hardware-scoped VTA FPGA correctness certificate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c3_candidate_identity import canonical_json_bytes, normalize_config_entity


HARDWARE_CERTIFICATE_LEDGER_SCHEMA = "c3_vta_hardware_certificate_ledger_v2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hardware_fingerprint(contract):
    board = contract["board"]
    value = {
        "boot_id": board["boot_id"],
        "fpga_state": board["fpga_state"],
        "udmabuf_bytes": board["udmabuf_bytes"],
        "source_guards_sha256": contract["source_guards_sha256"],
    }
    return {"key": canonical_hash(value), "value": value}


def normalized_complete_config_entity(row):
    """Return the semantic ConfigEntity or fail closed on legacy index-only rows."""

    if "complete_config_entity" in row:
        config_entity = row["complete_config_entity"]
    elif isinstance(row.get("identity"), dict) and "complete_config_entity" in row["identity"]:
        config_entity = row["identity"]["complete_config_entity"]
    else:
        raise ValueError(
            "missing complete_config_entity; refusing legacy config_index-only identity"
        )
    normalized = normalize_config_entity(config_entity)
    if not isinstance(normalized, dict) or "entity" not in normalized:
        raise ValueError("complete_config_entity must be a mapping containing entity")
    return normalized


def certificate_identity(contract, row, fingerprint_key):
    value = {
        "hardware_fingerprint_key": fingerprint_key,
        "workload": contract["workload"],
        "mode": row["mode"],
        "complete_config_entity": normalized_complete_config_entity(row),
        "tir_sha256": row["tir_sha256"],
    }
    return canonical_hash(value), value


def route(status):
    if status == "passed":
        return "allow_timing"
    if status == "failed":
        return "reject_candidate"
    return "correctness_canary_only"


def ingest(contract, rows, ledger):
    fingerprint = hardware_fingerprint(contract)
    for row in rows:
        key, identity = certificate_identity(contract, row, fingerprint["key"])
        config_index = int(row["config_index"])
        seeds = row["seeds"]
        if not seeds:
            raise ValueError("hardware certificate requires non-empty seed evidence")
        correct = bool(row["correct"])
        if correct != all(bool(seed["correct"]) for seed in seeds):
            raise ValueError("row correctness contradicts per-seed evidence")
        status = "passed" if correct else "failed"
        certificate = {
            "certificate_key": key,
            "identity": identity,
            "audit": {"observed_config_indices": [config_index]},
            "candidate_id": row.get("candidate_id"),
            "workload_id": contract["workload_id"],
            "status": status,
            "route": route(status),
            "seed_count": len(seeds),
            "passed_seed_count": sum(seed["correct"] for seed in seeds),
            "mismatch_counts": [seed["mismatch_count"] for seed in seeds],
            "binary_sha256": row["binary_sha256"],
            "performance_measurement": row["performance_measurement"],
        }
        previous = ledger.get(key)
        if previous and previous["status"] != status:
            raise ValueError("contradictory hardware certificate {}".format(key))
        if previous:
            previous["observations"] += 1
            previous["roles"] = sorted(set(previous["roles"] + [row["role"]]))
            previous["audit"]["observed_config_indices"] = sorted(
                set(previous["audit"]["observed_config_indices"] + [config_index])
            )
        else:
            certificate["observations"] = 1
            certificate["roles"] = [row["role"]]
            ledger[key] = certificate
    return fingerprint


def parse_source(value):
    try:
        contract, correctness = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source must be CONTRACT=CORRECTNESS") from error
    return Path(contract).resolve(), Path(correctness).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    ledger, fingerprints, inputs = {}, {}, {}
    for contract_path, correctness_path in args.source:
        contract = json.loads(contract_path.read_text())
        rows = [json.loads(line) for line in correctness_path.read_text().splitlines()
                if line.strip()]
        fingerprint = ingest(contract, rows, ledger)
        fingerprints[fingerprint["key"]] = fingerprint["value"]
        inputs[str(contract_path)] = sha256_file(contract_path)
        inputs[str(correctness_path)] = sha256_file(correctness_path)
    certificates = sorted(ledger.values(), key=lambda row: row["certificate_key"])
    result = {
        "schema": HARDWARE_CERTIFICATE_LEDGER_SCHEMA,
        "status": "completed",
        "scope": (
            "exact hardware fingerprint, normalized complete ConfigEntity, and exact lowered-TIR "
            "identity; config_index is audit-only; no shape generalization"
        ),
        "routing": {
            "passed": "allow_timing",
            "failed": "reject_candidate",
            "unknown": "correctness_canary_only",
        },
        "fingerprints": fingerprints,
        "summary": {
            "certificate_count": len(certificates),
            "passed": sum(row["status"] == "passed" for row in certificates),
            "failed": sum(row["status"] == "failed" for row in certificates),
            "timing_allowed": sum(row["route"] == "allow_timing" for row in certificates),
            "timing_rejected": sum(row["route"] == "reject_candidate" for row in certificates),
        },
        "certificates": certificates,
        "inputs": inputs,
    }
    ledger_path = output / "ledger.json"
    ledger_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# VTA hardware certificate ledger\n\n"
        "- Schema: `{}`\n"
        "- Identity: hardware + workload + mode + normalized complete ConfigEntity + TIR\n"
        "- Config indices: audit-only\n"
        "- Exact certificates: {}\n- Passed/allow timing: {}\n"
        "- Failed/reject before timing: {}\n"
        "- Unknown identity policy: correctness canary only\n".format(
            HARDWARE_CERTIFICATE_LEDGER_SCHEMA,
            result["summary"]["certificate_count"], result["summary"]["passed"],
            result["summary"]["failed"]
        )
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
