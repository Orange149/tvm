#!/usr/bin/env python3
"""Build the whole-program P-CAP profile and fail-closed native certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROFILE_SCHEMA = "thesis_vta_default_capacity_profile_v1"
CERTIFICATE_SCHEMA = "thesis_vta_command_capacity_certificate_v1"
DEFAULT_MAX_XFER = 1 << 25


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def align_up(value, alignment):
    value, alignment = int(value), int(alignment)
    if value <= 0 or alignment <= 0:
        raise ValueError("positive value and alignment required")
    return ((value + alignment - 1) // alignment) * alignment


def graph_input_metadata(graph_path, name):
    graph = load_json(graph_path)
    matches = [index for index, node in enumerate(graph["nodes"]) if node["name"] == name]
    if len(matches) != 1:
        raise ValueError("graph input must resolve exactly once: {}".format(name))
    index = matches[0]
    rows = graph["node_row_ptr"]
    entry = int(rows[index])
    if int(rows[index + 1]) != entry + 1:
        raise ValueError("graph input must own exactly one entry: {}".format(name))
    return {
        "name": name,
        "dtype": graph["attrs"]["dltype"][1][entry],
        "shape": [int(value) for value in graph["attrs"]["shape"][1][entry]],
    }


def resolve_package_artifact(package_dir, value):
    path = Path(value)
    return path if path.is_absolute() else package_dir / path


def build_program_binding(args):
    package_dir = Path(args.package_dir).resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = load_json(manifest_path)
    stages = []
    for index, record in enumerate(manifest.get("stages") or []):
        if int(record.get("index", index)) != index:
            raise ValueError("manifest stages must be contiguous")
        paths = {
            key: resolve_package_artifact(package_dir, record[key]).resolve()
            for key in ("graph", "lib", "params")
        }
        hashes = {key + "_sha256": sha256_file(path) for key, path in paths.items()}
        declared = record.get("artifact_sha256") or {}
        for key, observed in hashes.items():
            if declared and declared.get(key) != observed:
                raise ValueError("manifest stage artifact hash mismatch: stage{} {}".format(index, key))
        names = list(record.get("input_names") or [])
        if not names:
            raise ValueError("manifest stage has no declared inputs")
        stages.append(
            {
                "index": index,
                "name": record.get("name") or "stage{}".format(index),
                "device": record["device"],
                "input_names": names,
                "inputs": [graph_input_metadata(paths["graph"], name) for name in names],
                **hashes,
            }
        )
    if not stages:
        raise ValueError("package manifest has no stages")
    artifact_paths = {
        "runner": Path(args.runner).resolve(),
        "runtime": Path(args.runtime).resolve(),
        "driver": Path(args.driver).resolve(),
        "bitstream": Path(args.bitstream).resolve(),
    }
    for label, path in artifact_paths.items():
        if not path.is_file():
            raise ValueError("missing {} artifact: {}".format(label, path))
    shared_plan = Path(args.shared_buffer_plan).resolve() if args.shared_buffer_plan else None
    if shared_plan is not None and not shared_plan.is_file():
        raise ValueError("missing shared-buffer plan: {}".format(shared_plan))
    artifacts = {
        label + "_sha256": sha256_file(path) for label, path in artifact_paths.items()
    }
    artifacts["shared_buffer_plan_sha256"] = (
        sha256_file(shared_plan) if shared_plan is not None else "none"
    )
    binding = {
        "schema": "thesis_vta_program_binding_v1",
        "package_manifest_sha256": sha256_file(manifest_path),
        "stages": stages,
        "artifacts": artifacts,
        "vta_max_xfer_bytes": int(args.vta_max_xfer_bytes),
    }
    return binding, canonical_hash(binding)


def validate_queue_status(status, max_xfer):
    if status.get("schema") != "vta_queue_capacity_status_v1":
        raise ValueError("unexpected queue status schema")
    if (
        status.get("aggregation_schema") != "thesis_vta_worker_queue_aggregation_v1"
        or status.get("finish_included") is not True
        or int(status.get("queue_instances", 0)) <= 0
    ):
        raise ValueError("peak source must aggregate all worker queues after FINISH")
    if int(status.get("insn_capacity_bytes", -1)) != max_xfer or int(
        status.get("uop_capacity_bytes", -1)
    ) != max_xfer:
        raise ValueError("peak profile must use both default VTA_MAX_XFER capacities")
    if status.get("replay_policy") != "disabled":
        raise ValueError("peak profile must disable replay")
    if status.get("insn_submit_threshold_configured") is not False:
        raise ValueError("peak profile must leave early submit threshold unset")
    if min(
        int(status.get("insn_peak_bytes", 0)),
        int(status.get("uop_peak_bytes", 0)),
        int(status.get("submissions", 0)),
    ) <= 0:
        raise ValueError("peak profile must contain positive peaks and submissions")


def create_profile(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    binding, binding_sha = build_program_binding(args)
    status_path = Path(args.queue_status).resolve()
    status = load_json(status_path)
    max_xfer = int(args.vta_max_xfer_bytes)
    validate_queue_status(status, max_xfer)
    if args.evidence_scope == "board_profiled_program" and not args.boot_id:
        raise ValueError("board profile requires --boot-id")
    output.mkdir(parents=True)
    profile = {
        "schema": PROFILE_SCHEMA,
        "evidence_scope": args.evidence_scope,
        "program_binding_sha256": binding_sha,
        "program_id": binding_sha,
        "boot_id": args.boot_id or None,
        "queue_status_source": {
            "path": str(status_path),
            "sha256": sha256_file(status_path),
        },
        "finish_included": True,
        "finish_evidence": (
            "vta.runtime.queue_capacity_status peaks are updated after Synchronize appends FINISH "
            "and passes both queue capacity checks"
        ),
        "insn_capacity_bytes": int(status["insn_capacity_bytes"]),
        "uop_capacity_bytes": int(status["uop_capacity_bytes"]),
        "insn_peak_bytes": int(status["insn_peak_bytes"]),
        "uop_peak_bytes": int(status["uop_peak_bytes"]),
        "submissions": int(status["submissions"]),
        "queue_instances": int(status["queue_instances"]),
        "aggregation_schema": status["aggregation_schema"],
        "replay_policy": status["replay_policy"],
        "submit_threshold_configured": bool(status["insn_submit_threshold_configured"]),
        "command_manifest_id_during_default_profile": status.get("command_manifest_id", ""),
        "program_binding": binding,
    }
    profile_path = output / "default_capacity_profile.json"
    write_json(profile_path, profile)
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": {profile_path.name: sha256_file(profile_path)}},
    )
    return profile_path


def create_certificate(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    binding, binding_sha = build_program_binding(args)
    source_profile = Path(args.peak_profile).resolve()
    profile = load_json(source_profile)
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("unexpected peak profile schema")
    if profile.get("program_binding_sha256") != binding_sha:
        raise ValueError("peak profile belongs to a different exact program")
    scope = profile.get("evidence_scope")
    if scope not in {"board_profiled_program", "synthetic_host_fixture"}:
        raise ValueError("unsupported peak profile evidence scope")
    max_xfer = int(args.vta_max_xfer_bytes)
    validate_queue_status(
        {
            "schema": "vta_queue_capacity_status_v1",
            "aggregation_schema": profile["aggregation_schema"],
            "finish_included": profile["finish_included"],
            "queue_instances": profile["queue_instances"],
            "insn_capacity_bytes": profile["insn_capacity_bytes"],
            "uop_capacity_bytes": profile["uop_capacity_bytes"],
            "insn_peak_bytes": profile["insn_peak_bytes"],
            "uop_peak_bytes": profile["uop_peak_bytes"],
            "submissions": profile["submissions"],
            "replay_policy": profile["replay_policy"],
            "insn_submit_threshold_configured": profile["submit_threshold_configured"],
        },
        max_xfer,
    )
    multiplier = {"manual1": 1, "manual2": 2, "q1": 2}[args.mode]
    insn_peak = int(profile["insn_peak_bytes"])
    uop_peak = int(profile["uop_peak_bytes"])
    insn_capacity = align_up(max(4096, multiplier * insn_peak), 256)
    uop_capacity = align_up(max(4096, multiplier * uop_peak), 256)
    if max(insn_capacity, uop_capacity) > max_xfer:
        raise ValueError("derived P-CAP capacity exceeds VTA_MAX_XFER")
    output.mkdir(parents=True)
    copied_profile = output / "default_capacity_profile.json"
    shutil.copyfile(source_profile, copied_profile)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "evidence_scope": scope,
        "capacity_mode": args.mode,
        "program_id": binding_sha,
        "program_binding_sha256": binding_sha,
        "vta_max_xfer_bytes": max_xfer,
        "stages": binding["stages"],
        "artifacts": binding["artifacts"],
        "profile": {
            "file": copied_profile.name,
            "sha256": sha256_file(copied_profile),
        },
        "derivation": {
            "formula": "align256(max(4096, multiplier * exact_default_capacity_peak))",
            "alignment_bytes": 256,
            "minimum_bytes": 4096,
            "multiplier": multiplier,
            "insn_peak_bytes": insn_peak,
            "uop_peak_bytes": uop_peak,
            "insn_capacity_bytes": insn_capacity,
            "uop_capacity_bytes": uop_capacity,
        },
        "runtime_contract": {
            "environment": {
                "VTA_INSN_BUFFER_BYTES": str(insn_capacity),
                "VTA_UOP_BUFFER_BYTES": str(uop_capacity),
            },
            "VTA_COMMAND_MANIFEST_ID": "sha256(command_capacity_certificate.json)",
            "VTA_REPLAY_POLICY": "disabled",
            "VTA_INSN_SUBMIT_THRESHOLD_BYTES": "unset",
        },
        "claim_boundary": (
            "exact stage graph/lib/params, input dtype/shape, runner/runtime/driver, bitstream "
            "artifact, shared-buffer plan, and frozen default-capacity profile"
        ),
    }
    certificate_path = output / "command_capacity_certificate.json"
    write_json(certificate_path, certificate)
    certificate_sha = sha256_file(certificate_path)
    write_json(
        output / "launcher_environment.json",
        {
            **certificate["runtime_contract"]["environment"],
            "VTA_COMMAND_MANIFEST_ID": certificate_sha,
            "VTA_REPLAY_POLICY": "disabled",
        },
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: sha256_file(path)
                for path in output.iterdir()
                if path.is_file() and path.name != "artifact_hashes.json"
            }
        },
    )
    return certificate_path


def add_program_arguments(parser):
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--shared-buffer-plan", default="")
    parser.add_argument("--vta-max-xfer-bytes", type=int, default=DEFAULT_MAX_XFER)
    parser.add_argument("--output-dir", required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="freeze an isolated default-capacity profile")
    add_program_arguments(profile)
    profile.add_argument("--queue-status", required=True)
    profile.add_argument(
        "--evidence-scope",
        choices=("board_profiled_program", "synthetic_host_fixture"),
        required=True,
    )
    profile.add_argument("--boot-id", default="")
    certificate = commands.add_parser("certificate", help="derive a P-CAP certificate")
    add_program_arguments(certificate)
    certificate.add_argument("--peak-profile", required=True)
    certificate.add_argument("--mode", choices=("manual1", "manual2", "q1"), required=True)
    args = parser.parse_args()
    result = create_profile(args) if args.command == "profile" else create_certificate(args)
    print(json.dumps({"status": "created", "path": str(result), "sha256": sha256_file(result)}, indent=2))


if __name__ == "__main__":
    main()
