#!/usr/bin/env python3
"""Seal local reproduction provenance, protected-history checks and costs."""

import argparse
import json
from pathlib import Path
import subprocess

import tvm

from c3_banerjee_local import COMMIT, AUTHOR_HASHES, SYMBOL
from c3_residency_tradeoff import read, write_new, digest
from c3_residency_tradeoff_evidence import check_runtime
from audit_c3_banerjee_local import audit


def canonical_payload(value):
    """Compare precisely the serialized artifact values, not tuple/list types."""
    value = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    for record in value.get("programs", {}).values():
        if "directory" in record:
            record["directory"] = str(Path(record["directory"]).resolve())
    return value


def seal(root):
    root = Path(root).resolve()
    old_root = root.parent / "20261012_c3_residency_instruction_tradeoff_run01"
    old = read(old_root / "closeout_manifest.json")
    for path, sha in old["source_files"].items():
        if digest(path) != sha:
            raise ValueError("protected historical source changed: " + path)
    for path, sha in old["artifacts"].items():
        if digest(old_root / path) != sha:
            raise ValueError("protected historical artifact changed: " + path)
    recent = root.parent / "20261012_c3_five_tradeoff_selection_run01"
    for name in ("rule_freeze.json", "execution_protocol.json"):
        data = read(recent / name)
        for path, sha in data.get("source_bindings", data.get("source_sha256", {})).items():
            if digest(path) != sha:
                raise ValueError("sealed prefetch execution source changed: " + path)
    verified = audit(root)
    proof = read(root / "local_qualification.json")
    if canonical_payload(verified) != canonical_payload(proof):
        raise ValueError("local qualification no longer reproduces exactly")
    author = {path.name: {"sha256": digest(path)} for path in sorted((root / "author_sources").iterdir())}
    for name, sha in AUTHOR_HASHES.items():
        if author[name]["sha256"] != sha:
            raise ValueError("author commit source drift")
    arms = {}
    for name in ("ordinary", "reuse"):
        directory = root / "binaries" / (name + "01")
        data = read(directory / "result.json")
        local = Path(proof["programs"][name]["directory"])
        if data["status"] != "ARM_BINARY_MATCHES_LOCAL_FSIM_TARGET_FPGA_PENDING" or \
                data["binary_sha256"] != digest(directory / "graphlib.so") or \
                data["local_qualification_sha256"] != digest(root / "local_qualification.json"):
            raise ValueError("ARM identity missing or changed")
        arm = tvm.ir.load_json((directory / "node24_final.tir.json").read_text())
        reference = tvm.ir.load_json((local / "lowered.json").read_text())[SYMBOL]
        if not tvm.ir.structural_equal(arm, reference, map_free_vars=True):
            raise ValueError("ARM/native actual target is not structurally identical")
        arms[name] = {"directory": str(directory), "binary_sha256": data["binary_sha256"],
                      "TIR_sha256": digest(directory / "node24_final.tir.json")}
    tests = subprocess.run([__import__("sys").executable, "-m", "pytest", "-q",
                           str(Path(__file__).with_name("test_c3_banerjee_local.py"))],
                          capture_output=True, text=True, timeout=120)
    write_new(root / "tests.json", {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr})
    if tests.returncode:
        raise ValueError("local reproduction regression failed")
    costs = []
    for section in ("local", "binaries"):
        for folder in sorted((root / section).iterdir()):
            path = folder / "result.json" if (folder / "result.json").is_file() else folder / "failure.json"
            if not path.is_file():
                continue
            record = read(path)
            costs.append({"attempt": str(folder.relative_to(root)), "status": record["status"],
                          "elapsed_seconds": record["seconds"], "board_calls": 0,
                          "successfully_recorded_fsim_calls": len(record.get("fsim_calls", [])),
                          "uncompleted_target_attempts": None if path.name == "failure.json" else 0,
                          "performance_samples": 0})
    write_new(root / "cost_ledger.json", {"attempts": costs, "FPGA_calls": 0, "timing_calls": 0,
        "parallel_attempt_seconds_not_total_wall_clock": True, "total_round_wall_clock_seconds": None,
        "failed_call_counts_unknown_not_zero": True,
        "final_qualified_fsim_calls": 12, "all_successfully_recorded_fsim_calls": sum(x["successfully_recorded_fsim_calls"] for x in costs)})
    sources = {str(Path(__file__).with_name(name)): digest(Path(__file__).with_name(name)) for name in (
        "c3_banerjee_local.py", "run_c3_banerjee_local.py", "run_c3_banerjee_cross_compile.py",
        "audit_c3_banerjee_local.py", "test_c3_banerjee_local.py", "seal_c3_banerjee_local.py")}
    manifest = {"status": "LOCAL_REPRODUCTION_SEALED_FPGA_QUALIFICATION_PENDING", "author_commit": COMMIT,
        "author_sources": author, "paper": "https://arxiv.org/html/2111.15024#S4.SS4.SSS2",
        "repository": "https://github.com/pasqoc/incubator-tvm/tree/" + COMMIT,
        "scope": "functional adaptation of output-channel double-buffer input reuse, not complete paper/TPS reproduction",
        "sources": sources, "runtime": check_runtime(), "ARM_programs": arms,
        "protected_historical_sources_verified": len(old["source_files"]),
        "protected_historical_artifacts_verified": len(old["artifacts"]),
        "prefetch_frozen_rule_and_sources_unchanged": True,
        "FPGA_calls": 0, "latency_ms": None,
        "artifacts": {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*")) if path.is_file()},
        "manifest_self_excluded": True}
    write_new(root / "local_manifest.json", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    seal(args.root)
