#!/usr/bin/env python3
"""Create a fixed-K2 binding plan from a hash-checked storage audit (C2 enhancement)."""
import argparse
import copy
import json
from pathlib import Path

from audit_shared_buffer_storage import finalize_bundles, sha, verify_hashes


def make_plan(audit, build):
    package = Path(audit["package"])
    verify_hashes(package, audit["package_sha256"])
    auditor = Path(__file__).with_name("audit_shared_buffer_storage.py")
    if sha(auditor) != audit["audit_source_sha256"]:
        raise ValueError("audit source changed; rerun the static audit")
    manifest = json.loads((package / "manifest.json").read_text())
    edges = finalize_bundles(copy.deepcopy(audit["edges"]))
    return {"schema": "shared-buffer-plan-v1", "scope": "C2 fixed-K2 cross-executor input-pool reuse",
            "candidate_id": manifest["candidate_id"], "package_sha256": audit["package_sha256"],
            "runtime_sha256": audit["package_sha256"]["libtvm_runtime.so"],
            "driver_sha256": build["artifacts"]["libvta.so"],
            "stages": [{"device": s["device"], "input_names": s["input_names"],
                        **s["artifact_sha256"]} for s in manifest["stages"]],
            "edges": edges,
            "predicted_saved_bytes": sum(e["predicted_saved_bytes"] for e in edges),
            "startup_policy": "hash mismatch aborts; graph/physical eligibility failure falls back whole edge"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(json.loads(args.audit.read_text()), json.loads(args.build_manifest.read_text()))
    plan["audit_sha256"] = sha(args.audit)
    plan["planner_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(plan, out, indent=2, sort_keys=True)
        out.write("\n")


if __name__ == "__main__":
    main()
