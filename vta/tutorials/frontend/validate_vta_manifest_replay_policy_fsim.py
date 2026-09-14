#!/usr/bin/env python3
"""Prove that a manifest-disabled replay policy fails closed in actual FSim."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tvm
import vta

from build_vta_command_resource_certificate import load_json, sha256_file
from build_vta_deployment_manifest import validate_manifest
from collect_vta_full_pool_fsim_commands import loaded_library_identity


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def worker(manifest_id, output):
    from vta.testing import simulator

    if not simulator.enabled():
        raise RuntimeError("VTA FSim runtime is not enabled")
    runtime = loaded_library_identity("libvta_fsim.so")
    status = json.loads(tvm.get_global_func("vta.runtime.queue_capacity_status")())
    rejected = {}
    for name, function_name in (
        ("capture", "vta.runtime.replay_begin_capture"),
        ("replay", "vta.runtime.replay_begin_replay"),
    ):
        try:
            tvm.get_global_func(function_name)("manifest-negative-control")
        except Exception as error:  # expected runtime CHECK propagated through FFI
            message = str(error)
            rejected[name] = {
                "rejected": "disabled by the deployment manifest" in message,
                "message_contains_policy_reason": "disabled by the deployment manifest" in message,
            }
        else:
            rejected[name] = {"rejected": False, "message_contains_policy_reason": False}
    write_json(output, {
        "manifest_id": manifest_id,
        "execution_runtime": runtime,
        "queue_status": status,
        "calls": rejected,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--worker-manifest-id")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker_manifest_id:
        worker(args.worker_manifest_id, args.worker_output)
        return
    if not args.ready_manifest or not args.output_dir:
        raise ValueError("ready manifest and output directory are required")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    manifest_path = Path(args.ready_manifest).resolve()
    manifest_ledger = load_json(manifest_path.parent / "artifact_hashes.json")["artifacts"]
    if manifest_ledger.get(manifest_path.name) != sha256_file(manifest_path):
        raise ValueError("ready manifest hash mismatch")
    manifest = load_json(manifest_path)
    certificate_path = Path(
        manifest["provenance"]["qualified_resource_certificate"]["path"]
    )
    certificate = load_json(certificate_path)
    validate_manifest(manifest, certificate)
    with tempfile.TemporaryDirectory(prefix="c3_manifest_replay_policy_") as temporary:
        worker_output = Path(temporary) / "result.json"
        environment = os.environ.copy()
        environment.update(manifest["launcher_environment"])
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-manifest-id", manifest["manifest_id"],
                "--worker-output", str(worker_output),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        if completed.returncode != 0 or not worker_output.is_file():
            raise RuntimeError("replay-policy worker failed: {}".format(completed.stderr[-1000:]))
        result = load_json(worker_output)
    if result["execution_runtime"]["sha256"] != certificate["fsim_execution_runtime"]["sha256"]:
        raise ValueError("negative control loaded a different FSim runtime")
    if (
        result["queue_status"].get("command_manifest_id") != manifest["manifest_id"]
        or result["queue_status"].get("replay_policy") != "disabled"
        or not all(row["rejected"] for row in result["calls"].values())
    ):
        raise ValueError("manifest replay policy did not fail closed")
    result.update(
        schema="c3_vta_manifest_replay_policy_fsim_v1",
        status="passed",
        performance_measurement="not_collected",
        board_contacted=False,
    )
    result_path = output / "replay_policy_negative_control.json"
    write_json(result_path, result)
    (output / "STATUS.md").write_text(
        "# VTA manifest replay-policy negative control\n\n"
        "- Status: `passed`\n"
        "- Actual FSim runtime hash matched the resource certificate\n"
        "- Capture rejected: yes\n"
        "- Replay rejected: yes\n"
        "- Board contacted: no\n"
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
