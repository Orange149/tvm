#!/usr/bin/env python3
"""Read-only inventory and integrity audit for thesis experiment run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = ("contract.json", "artifact_hashes.json", "timeline.jsonl", "summary.json", "REPORT.md")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_entries(payload):
    source = payload.get("artifacts", payload) if isinstance(payload, dict) else {}
    for relative, record in source.items():
        if isinstance(record, str):
            yield relative, record, None
        elif isinstance(record, dict) and record.get("sha256"):
            yield relative, record["sha256"], record.get("bytes")


def audit_run(run_dir, results_root):
    relative_run = str(run_dir.relative_to(results_root))
    issues, warnings = [], []
    present = {name: (run_dir / name).is_file() for name in REQUIRED}
    for name, exists in present.items():
        if not exists:
            issues.append({"kind": "missing_required_file", "path": name})
    parsed = {}
    for name in ("contract.json", "artifact_hashes.json", "summary.json", "invalid_session.json"):
        path = run_dir / name
        if path.is_file():
            try:
                parsed[name] = _json(path)
            except (OSError, json.JSONDecodeError) as error:
                issues.append({"kind": "invalid_json", "path": name, "detail": str(error)})
    timeline = run_dir / "timeline.jsonl"
    event_ids, event_count = set(), 0
    if timeline.is_file():
        for line_number, line in enumerate(timeline.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append({"kind": "invalid_timeline_json", "path": "timeline.jsonl", "line": line_number, "detail": str(error)})
                continue
            event_count += 1
            event_id = event.get("event_id")
            if event_id is not None:
                if event_id in event_ids:
                    issues.append({"kind": "duplicate_event_id", "event_id": event_id})
                event_ids.add(event_id)
    checked_hashes = 0
    ledger = parsed.get("artifact_hashes.json")
    if ledger is not None:
        for relative, expected, expected_bytes in _ledger_entries(ledger):
            target = run_dir / relative
            if not target.is_file():
                issues.append({"kind": "ledger_target_missing", "path": relative})
                continue
            checked_hashes += 1
            actual = sha256_file(target)
            if actual != expected:
                issues.append({"kind": "sha256_mismatch", "path": relative, "expected": expected, "actual": actual})
            if expected_bytes is not None and target.stat().st_size != int(expected_bytes):
                issues.append({"kind": "byte_size_mismatch", "path": relative})
    contract = parsed.get("contract.json", {})
    if contract and str(contract.get("stage", run_dir.parent.name)).zfill(2) != run_dir.parent.name:
        issues.append({"kind": "stage_directory_mismatch", "contract_stage": contract.get("stage")})
    if contract and contract.get("run_id") not in (None, run_dir.name):
        issues.append({"kind": "run_id_directory_mismatch", "contract_run_id": contract.get("run_id")})
    if event_count and not event_ids:
        warnings.append({"kind": "timeline_has_no_event_ids", "events": event_count})
    return {
        "run": relative_run,
        "stage": run_dir.parent.name,
        "required_files_present": present,
        "invalid_session_present": (run_dir / "invalid_session.json").is_file(),
        "timeline_events": event_count,
        "artifact_hashes_checked": checked_hashes,
        "issues": issues,
        "warnings": warnings,
        "integrity_status": "passed" if not issues else "issues_found",
    }


def audit_results(results_root):
    root = Path(results_root).resolve()
    runs = []
    for stage in sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit()):
        for run in sorted(path for path in stage.iterdir() if path.is_dir()):
            runs.append(audit_run(run, root))
    counts_by_stage = {}
    for row in runs:
        counts_by_stage[row["stage"]] = counts_by_stage.get(row["stage"], 0) + 1
    issues = sum(len(row["issues"]) for row in runs)
    return {
        "schema": "thesis_result_inventory_audit_v1",
        "status": "complete_with_issues" if issues else "passed",
        "read_only": True,
        "run_count": len(runs),
        "counts_by_stage": counts_by_stage,
        "invalid_session_run_count": sum(row["invalid_session_present"] for row in runs),
        "issue_count": issues,
        "warning_count": sum(len(row["warnings"]) for row in runs),
        "runs": runs,
        "claim_boundary": "filesystem inventory/integrity only; no experimental claim is certified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_results(args.results_root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "run_count", "invalid_session_run_count", "issue_count", "warning_count")}, indent=2))


if __name__ == "__main__":
    main()

