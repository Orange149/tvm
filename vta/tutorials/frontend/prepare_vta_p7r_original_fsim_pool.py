#!/usr/bin/env python3
"""Create an original-schedule FSim pool paired with P7R residency candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="append", required=True)
    parser.add_argument(
        "--select",
        required=True,
        help="Comma-separated WORKLOAD:CONFIG pairs",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    lookup = {}
    for scan in args.scan:
        for row in load_jsonl(scan):
            lookup[(row["workload_id"], int(row["config_index"]))] = row
    selected_keys = []
    for text in args.select.split(","):
        workload, index = text.strip().split(":", 1)
        selected_keys.append((workload.upper(), int(index)))
    rows = []
    for key in selected_keys:
        source = lookup[key]
        original = source["original"]
        if original["status"] != "ok":
            raise ValueError("original did not lower for {}".format(key))
        identity = {
            "hardware": source.get("hardware", "axu5evb-configured-fsim"),
            "workload": source["identity"]["workload"],
            "complete_config_entity": source["identity"]["complete_config_entity"],
            "schedule_sha256": source["identity"]["schedule_sha256"],
            "residence_mode": "original",
        }
        residency_mode = source["residence_mode"]
        residency = source[residency_mode]
        rows.append(
            {
                "schema": "c3_p7r_paired_original_fsim_candidate_v1",
                "candidate_id": canonical_hash(identity),
                "candidate_role": "p7r_original_pair",
                "status": "ok",
                "workload_id": key[0],
                "residence_mode": "original",
                "tir_sha256": original["tir_sha256"],
                "tir_hash_encoding": "tvm.ir.save_json",
                "debug": {"config_index": key[1]},
                "identity": {
                    "workload": identity["workload"],
                    "complete_config_entity": identity["complete_config_entity"],
                    "schedule_sha256": identity["schedule_sha256"],
                },
                "pair": {
                    "residency_candidate_id": source["candidate_id"],
                    "residency_mode": residency_mode,
                    "residency_tir_sha256": residency["tir_sha256"],
                },
            }
        )
    candidates = output / "candidates.jsonl"
    candidates.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    protocol = {
        "schema": "c3_p7r_paired_original_fsim_pool_v1",
        "purpose": "close the missing same-tile original FSim gate; no performance timing",
        "selection": ["{}:{}".format(*key) for key in selected_keys],
        "candidate_count": len(rows),
        "inputs": {str(path): sha256_file(path) for path in args.scan},
        "output_sha256": sha256_file(candidates),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("prepared {} original-schedule candidates".format(len(rows)))


if __name__ == "__main__":
    main()
