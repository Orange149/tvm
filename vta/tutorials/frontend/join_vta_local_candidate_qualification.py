#!/usr/bin/env python3
"""Join P2/P4b/P4c/P4d into one local-only C3 candidate certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path


SCHEMA = "c3_local_candidate_qualification_v1"
SEEDS = (0, 20250901, 20260910)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_by(rows, key, source):
    answer = {}
    for row in rows:
        value = row[key]
        if value in answer:
            raise ValueError("duplicate {} in {}: {}".format(key, source, value))
        answer[value] = row
    return answer


def _check_candidate_certificate(candidate, certificate, source):
    if certificate.get("overall_status") != "passed":
        raise ValueError("non-passing local certificate for {}".format(candidate["candidate_id"]))
    if certificate.get("p4b_tir_sha256") != candidate.get("tir_sha256"):
        raise ValueError("P4b TIR hash mismatch for {}".format(candidate["candidate_id"]))
    if certificate.get("relowered_tir_sha256") != candidate.get("tir_sha256"):
        raise ValueError("re-lowered TIR hash mismatch for {}".format(candidate["candidate_id"]))
    seeds = certificate.get("seeds", [])
    if [row.get("seed") for row in seeds] != list(SEEDS) or not all(
        row.get("status") == "passed" and row.get("correct") is True for row in seeds
    ):
        raise ValueError("incomplete three-seed certificate for {}".format(candidate["candidate_id"]))
    return {
        "local_status": "fsim_passed",
        "correctness_seeds": list(SEEDS),
        "tir_sha256": candidate["tir_sha256"],
        "evidence_source": source,
    }


def join_qualifications(p4b_rows, p4c_rows, p4d_rows, p2_fsim, p2_tir):
    p4b = _unique_by(p4b_rows, "candidate_id", "P4b")
    experimental = _unique_by(p4c_rows, "candidate_id", "P4c")
    controls = _unique_by(p4d_rows, "candidate_id", "P4d")
    baseline = _unique_by(p2_fsim, "workload_id", "P2 FSim")
    baseline_tir = _unique_by(p2_tir, "workload_id", "P2 TIR")
    records = []
    consumed_experimental, consumed_controls, consumed_baseline = set(), set(), set()

    for candidate_id in sorted(p4b):
        candidate = p4b[candidate_id]
        record = {
            "schema": SCHEMA,
            "candidate_id": candidate_id,
            "workload_id": candidate["workload_id"],
            "candidate_role": candidate["candidate_role"],
            "residence_mode": candidate["residence_mode"],
            "lower_status": candidate["status"],
        }
        if candidate["status"] != "ok":
            record.update(
                local_status="lower_failed",
                correctness_seeds=[],
                tir_sha256=None,
                evidence_source="P4b static lowering",
                failure=candidate.get("failure"),
            )
        elif candidate["candidate_role"] == "protected_original_incumbent":
            workload_id = candidate["workload_id"]
            if workload_id not in baseline or workload_id not in baseline_tir:
                raise ValueError("missing P2 baseline evidence for {}".format(workload_id))
            fsim, tir = baseline[workload_id], baseline_tir[workload_id]
            if int(fsim["config_index"]) != int(candidate["debug"]["config_index"]):
                raise ValueError("P2 config mismatch for {}".format(workload_id))
            if fsim.get("correct_seeds") != list(SEEDS):
                raise ValueError("incomplete P2 seeds for {}".format(workload_id))
            if tir.get("tir_ir_json_sha256") != candidate.get("tir_sha256"):
                raise ValueError("P2/P4b TIR mismatch for {}".format(workload_id))
            record.update(
                local_status="fsim_passed",
                correctness_seeds=list(SEEDS),
                tir_sha256=candidate["tir_sha256"],
                evidence_source="P2 local baseline run02",
            )
            consumed_baseline.add(workload_id)
        elif candidate["candidate_role"] == "same_tile_original_control":
            if candidate_id not in controls:
                raise ValueError("missing P4d certificate for {}".format(candidate_id))
            record.update(_check_candidate_certificate(candidate, controls[candidate_id], "P4d run02"))
            consumed_controls.add(candidate_id)
        elif candidate["candidate_role"] == "residency_experiment":
            if candidate_id not in experimental:
                raise ValueError("missing P4c certificate for {}".format(candidate_id))
            record.update(
                _check_candidate_certificate(candidate, experimental[candidate_id], "P4c run01")
            )
            consumed_experimental.add(candidate_id)
        else:
            raise ValueError("unknown candidate role: {}".format(candidate["candidate_role"]))
        records.append(record)

    if consumed_experimental != set(experimental):
        raise ValueError("unjoined P4c candidate certificate(s)")
    if consumed_controls != set(controls):
        raise ValueError("unjoined P4d candidate certificate(s)")
    if consumed_baseline != set(baseline):
        raise ValueError("unjoined P2 baseline certificate(s)")

    statuses = Counter(row["local_status"] for row in records)
    roles = Counter((row["candidate_role"], row["local_status"]) for row in records)
    summary = {
        "schema": "c3_local_candidate_qualification_summary_v1",
        "candidate_records": len(records),
        "unique_candidate_ids": len({row["candidate_id"] for row in records}),
        "lower_failed": statuses["lower_failed"],
        "fsim_passed_all_three_seeds": statuses["fsim_passed"],
        "unqualified_lower_success": sum(
            row["lower_status"] == "ok" and row["local_status"] != "fsim_passed"
            for row in records
        ),
        "role_status_counts": {
            "{}:{}".format(role, status): count
            for (role, status), count in sorted(roles.items())
        },
        "board_access": False,
        "performance_measurement": "not_collected",
    }
    return records, summary


def _write_new(path, text):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen artifact: {}".format(path))
    path.write_text(text, encoding="utf-8")


def _write_json(path, value):
    _write_new(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4b-results", required=True)
    parser.add_argument("--p4c-results", required=True)
    parser.add_argument("--p4d-results", required=True)
    parser.add_argument("--p2-fsim", required=True)
    parser.add_argument("--p2-tir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    inputs = {
        "p4b_results": args.p4b_results,
        "p4c_results": args.p4c_results,
        "p4d_results": args.p4d_results,
        "p2_fsim": args.p2_fsim,
        "p2_tir": args.p2_tir,
    }
    records, summary = join_qualifications(
        load_jsonl(args.p4b_results),
        load_jsonl(args.p4c_results),
        load_jsonl(args.p4d_results),
        load_json(args.p2_fsim),
        load_json(args.p2_tir),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "results.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "manifest.json",
        {
            "schema": "c3_local_candidate_qualification_manifest_v1",
            "python": sys.executable,
            "python_version": platform.python_version(),
            "inputs": {name: {"path": path, "sha256": sha256_file(path)} for name, path in inputs.items()},
            "source": {"path": str(Path(__file__)), "sha256": sha256_file(__file__)},
            "board_access": False,
        },
    )
    _write_json(
        output / "preregistered.json",
        {
            "schema": "c3_local_candidate_qualification_protocol_v1",
            "expected_candidate_count": 250,
            "required_seeds": list(SEEDS),
            "join_key": "candidate_id; protected incumbents additionally join by workload/config/TIR",
            "acceptance": "every P4b lower-success candidate has exact FSim evidence for all seeds",
            "performance_use": "forbidden",
        },
    )
    command = " ".join([sys.executable] + sys.argv) + "\n"
    _write_new(output / "command.txt", command)
    _write_new(output / "stdout.log", json.dumps(summary, sort_keys=True) + "\n")
    _write_new(output / "stderr.log", "")
    _write_new(
        output / "STATUS.md",
        "# C3-P4e unified local qualification\n\n"
        "Status: **completed_local_only**. All {} P4b lower-success candidates have exact "
        "three-seed FSim evidence; {} lower failures remain filtered. No board or performance "
        "measurement was used.\n".format(summary["fsim_passed_all_three_seeds"], summary["lower_failed"]),
    )
    _write_new(
        output / "HANDOFF.md",
        "# HANDOFF\n\nUse `results.jsonl` as the local correctness certificate joined by stable "
        "candidate ID. `fsim_passed` is not board correctness and contains no latency signal. "
        "The 85 `lower_failed` records remain part of screening cost but are not dispatchable.\n",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", artifacts)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
