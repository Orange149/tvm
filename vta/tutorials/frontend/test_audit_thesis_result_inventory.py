import hashlib
import json

from audit_thesis_result_inventory import audit_results


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_inventory_keeps_failed_runs_and_detects_missing_and_hash_drift(tmp_path):
    first = tmp_path / "00" / "run-good"
    first.mkdir(parents=True)
    write(first / "contract.json", json.dumps({"stage": "00", "run_id": "run-good"}))
    write(first / "summary.json", "{}")
    write(first / "timeline.jsonl", '{"event_id":"e1","event":"done"}\n')
    write(first / "REPORT.md", "report\n")
    digest = hashlib.sha256((first / "contract.json").read_bytes()).hexdigest()
    write(first / "artifact_hashes.json", json.dumps({"artifacts": {"contract.json": digest}}))

    failed = tmp_path / "01" / "run-failed"
    failed.mkdir(parents=True)
    write(failed / "contract.json", json.dumps({"stage": "01", "run_id": "run-failed"}))
    write(failed / "summary.json", "{}")
    write(failed / "timeline.jsonl", '{"event_id":"same"}\n{"event_id":"same"}\n')
    write(failed / "REPORT.md", "report\n")
    write(failed / "invalid_session.json", "{}")
    write(failed / "artifact_hashes.json", json.dumps({"REPORT.md": {"sha256": "0" * 64, "bytes": 7}}))

    missing = tmp_path / "02" / "run-missing"
    missing.mkdir(parents=True)
    write(missing / "contract.json", json.dumps({"stage": "02", "run_id": "run-missing"}))

    result = audit_results(tmp_path)
    assert result["run_count"] == 3
    assert result["invalid_session_run_count"] == 1
    assert result["status"] == "complete_with_issues"
    failures = next(row for row in result["runs"] if row["run"] == "01/run-failed")
    assert {issue["kind"] for issue in failures["issues"]} == {"duplicate_event_id", "sha256_mismatch"}
    gaps = next(row for row in result["runs"] if row["run"] == "02/run-missing")
    assert sum(issue["kind"] == "missing_required_file" for issue in gaps["issues"]) == 4


def test_inventory_accepts_both_hash_ledger_shapes(tmp_path):
    run = tmp_path / "03" / "run"
    run.mkdir(parents=True)
    for name, content in {
        "contract.json": '{"stage":"03","run_id":"run"}',
        "summary.json": "{}",
        "timeline.jsonl": '{"event":"done"}\n',
        "REPORT.md": "ok\n",
    }.items():
        write(run / name, content)
    digest = hashlib.sha256((run / "REPORT.md").read_bytes()).hexdigest()
    write(run / "artifact_hashes.json", json.dumps({"REPORT.md": {"sha256": digest, "bytes": 3}}))
    result = audit_results(tmp_path)
    assert result["status"] == "passed"
    assert result["runs"][0]["artifact_hashes_checked"] == 1

