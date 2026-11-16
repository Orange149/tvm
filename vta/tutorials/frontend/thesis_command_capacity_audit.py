#!/usr/bin/env python3
"""Independently recompute a host P-CAP preflight qualification result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align(value):
    return ((max(4096, 2 * int(value)) + 255) // 256) * 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    cases = load(run / "raw/cases.json")
    certificate_path = run / "raw/certificate/command_capacity_certificate.json"
    certificate = load(certificate_path)
    profile_path = run / "raw/certificate/default_capacity_profile.json"
    profile = load(profile_path)
    launcher = load(run / "raw/certificate/launcher_environment.json")
    derivation = certificate["derivation"]
    expected = {
        "insn": align(profile["insn_peak_bytes"]),
        "uop": align(profile["uop_peak_bytes"]),
    }
    failures = []
    if certificate["profile"]["sha256"] != sha256(profile_path):
        failures.append("profile hash")
    if int(derivation["insn_capacity_bytes"]) != expected["insn"]:
        failures.append("instruction capacity")
    if int(derivation["uop_capacity_bytes"]) != expected["uop"]:
        failures.append("UOP capacity")
    if launcher["VTA_COMMAND_MANIFEST_ID"] != sha256(certificate_path):
        failures.append("certificate manifest ID")
    if launcher["VTA_INSN_BUFFER_BYTES"] != str(expected["insn"]):
        failures.append("launcher instruction capacity")
    if launcher["VTA_UOP_BUFFER_BYTES"] != str(expected["uop"]):
        failures.append("launcher UOP capacity")
    if launcher["VTA_REPLAY_POLICY"] != "disabled":
        failures.append("launcher replay policy")
    positives = [row for row in cases if row["expected_success"]]
    negatives = [row for row in cases if not row["expected_success"]]
    if len(positives) != 1 or not positives[0]["observed_success"]:
        failures.append("positive case")
    if len(negatives) != 9 or any(row["observed_success"] for row in negatives):
        failures.append("negative cases")
    if any(not row["passed"] for row in cases):
        failures.append("case verdict")
    source_root = Path(__file__).resolve().parents[3]
    runner_source = (source_root / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc").read_text(
        encoding="utf-8"
    )
    header_source = (source_root / "vta/apps/native_deploy/vta_command_capacity_certificate.h").read_text(
        encoding="utf-8"
    )
    main_offset = runner_source.index("int main(int argc, char** argv)")
    call_offset = runner_source.index("ValidateCommandCapacityCertificate(args)", main_offset)
    plan_offset = runner_source.index("ValidateSharedBufferPlan(args)", main_offset)
    snapshot_offset = runner_source.index('WriteAllocationSnapshot(args, "before_graph_load")', main_offset)
    load_offset = runner_source.index("LoadStage(stage_arg.name", main_offset)
    preallocation_order_verified = call_offset < plan_offset < snapshot_offset < load_offset
    hardware_tokens = [token for token in ("VTAMemAlloc", "VTADeviceRun", "VTADeviceAlloc") if token in header_source]
    if not preallocation_order_verified:
        failures.append("native preallocation call order")
    if hardware_tokens:
        failures.append("certificate header hardware calls")
    result = {
        "schema": "thesis_command_capacity_host_audit_v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "case_count": len(cases),
        "positive_accepted": len(positives),
        "negative_rejected": len(negatives),
        "recomputed_capacity_bytes": expected,
        "certificate_sha256": sha256(certificate_path),
        "profile_sha256": sha256(profile_path),
        "hardware_calls": 0,
        "formal_q1_certificate_issued": False,
        "native_preallocation_order_verified": preallocation_order_verified,
        "certificate_header_hardware_call_tokens": hardware_tokens,
    }
    output = run / "analysis/recalculation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
