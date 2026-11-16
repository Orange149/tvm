#!/usr/bin/env python3
"""Host-only positive/negative qualification for the whole-program P-CAP preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from build_thesis_command_capacity_certificate import (
    create_certificate,
    create_profile,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def graph_fixture():
    return {
        "nodes": [
            {"op": "null", "name": "data0", "inputs": []},
            {"op": "tvm_op", "name": "copy", "inputs": [[0, 0, 0]], "attrs": {"func_name": "copy"}},
        ],
        "arg_nodes": [0],
        "node_row_ptr": [0, 1, 2],
        "heads": [[1, 0, 0]],
        "attrs": {
            "dltype": ["list_str", ["float32", "float32"]],
            "shape": ["list_shape", [[1, 3, 4, 4], [1, 3, 4, 4]]],
            "storage_id": ["list_int", [0, 1]],
        },
    }


def compile_probe(path):
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            str(ROOT / "vta/tutorials/frontend/thesis_command_capacity_probe.cc"),
            "-o",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def common_args(fixture, output_dir):
    return dict(
        package_dir=str(fixture / "package"),
        runner=str(fixture / "capacity_probe"),
        runtime=str(fixture / "libtvm_runtime.fixture.so"),
        driver=str(fixture / "libvta.fixture.so"),
        bitstream=str(fixture / "vta_hpc.fixture.bit"),
        shared_buffer_plan=str(fixture / "shared_buffer_plan.json"),
        vta_max_xfer_bytes=1 << 25,
        output_dir=str(output_dir),
    )


def prepare_fixture(root):
    package = root / "package"
    stage = package / "stages/stage0"
    stage.mkdir(parents=True)
    graph = stage / "graph.json"
    lib = stage / "graphlib.so"
    params = stage / "params.params"
    write_json(graph, graph_fixture())
    lib.write_bytes(b"synthetic graph library fixture\n")
    params.write_bytes(b"synthetic parameter fixture\n")
    for name, data in (
        ("libtvm_runtime.fixture.so", b"synthetic runtime fixture\n"),
        ("libvta.fixture.so", b"synthetic driver fixture\n"),
        ("vta_hpc.fixture.bit", b"synthetic bitstream fixture\n"),
    ):
        (root / name).write_bytes(data)
    write_json(root / "shared_buffer_plan.json", {"schema": "synthetic-shared-plan-v1"})
    hashes = {
        "graph_sha256": sha256_file(graph),
        "lib_sha256": sha256_file(lib),
        "params_sha256": sha256_file(params),
    }
    write_json(
        package / "manifest.json",
        {
            "kind": "synthetic_command_capacity_host_fixture",
            "stages": [
                {
                    "index": 0,
                    "name": "stage0",
                    "device": "vta",
                    "graph": "stages/stage0/graph.json",
                    "lib": "stages/stage0/graphlib.so",
                    "params": "stages/stage0/params.params",
                    "input_names": ["data0"],
                    "artifact_sha256": hashes,
                }
            ],
        },
    )
    compile_probe(root / "capacity_probe")
    write_json(
        root / "queue_status.json",
        {
            "schema": "vta_queue_capacity_status_v1",
            "aggregation_schema": "thesis_vta_worker_queue_aggregation_v1",
            "finish_included": True,
            "queue_instances": 2,
            "insn_capacity_bytes": 1 << 25,
            "uop_capacity_bytes": 1 << 25,
            "insn_peak_bytes": 5000,
            "uop_peak_bytes": 2000,
            "submissions": 7,
            "insn_submit_threshold_bytes": 1 << 25,
            "insn_submit_threshold_configured": False,
            "command_manifest_id": "",
            "replay_policy": "disabled",
        },
    )


def run_probe(fixture, certificate_dir, overrides=None, actual_mutator=None):
    certificate = certificate_dir / "command_capacity_certificate.json"
    launcher = json.loads((certificate_dir / "launcher_environment.json").read_text())
    environment = os.environ.copy()
    environment.update(launcher)
    environment["VTA_COMMAND_CERTIFICATE_ALLOW_SYNTHETIC"] = "1"
    environment.pop("VTA_INSN_SUBMIT_THRESHOLD_BYTES", None)
    if overrides:
        for key, value in overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
    actual = {
        "stages": [
            {
                "name": "stage0",
                "device": "vta",
                "graph": str(fixture / "package/stages/stage0/graph.json"),
                "lib": str(fixture / "package/stages/stage0/graphlib.so"),
                "params": str(fixture / "package/stages/stage0/params.params"),
                "input_names": ["data0"],
            }
        ],
        "runtime": str(fixture / "libtvm_runtime.fixture.so"),
        "driver": str(fixture / "libvta.fixture.so"),
        "bitstream": str(fixture / "vta_hpc.fixture.bit"),
        "shared_buffer_plan": str(fixture / "shared_buffer_plan.json"),
        "vta_max_xfer_bytes": 1 << 25,
    }
    if actual_mutator:
        actual_mutator(actual)
    actual_path = fixture / "actual_program.json"
    write_json(actual_path, actual)
    return subprocess.run(
        [str(fixture / "capacity_probe"), str(certificate), str(actual_path)],
        text=True,
        capture_output=True,
        env=environment,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    raw = output / "raw"
    fixture = raw / "fixture"
    fixture.mkdir(parents=True)
    prepare_fixture(fixture)

    profile_dir = raw / "profile"
    profile_args = SimpleNamespace(
        **common_args(fixture, profile_dir),
        queue_status=str(fixture / "queue_status.json"),
        evidence_scope="synthetic_host_fixture",
        boot_id="",
    )
    profile_path = create_profile(profile_args)
    certificate_dir = raw / "certificate"
    certificate_args = SimpleNamespace(
        **common_args(fixture, certificate_dir),
        peak_profile=str(profile_path),
        mode="q1",
    )
    create_certificate(certificate_args)

    cases = []

    def record(name, expected_success, completed):
        observed_success = completed.returncode == 0
        cases.append(
            {
                "name": name,
                "expected_success": expected_success,
                "observed_success": observed_success,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "passed": observed_success == expected_success,
            }
        )

    record("exact_certificate_accepts", True, run_probe(fixture, certificate_dir))
    record(
        "wrong_instruction_capacity_rejected",
        False,
        run_probe(fixture, certificate_dir, {"VTA_INSN_BUFFER_BYTES": "4096"}),
    )
    record(
        "missing_manifest_id_rejected",
        False,
        run_probe(fixture, certificate_dir, {"VTA_COMMAND_MANIFEST_ID": None}),
    )
    record(
        "replay_enabled_rejected",
        False,
        run_probe(fixture, certificate_dir, {"VTA_REPLAY_POLICY": "enabled"}),
    )
    record(
        "early_submit_rejected",
        False,
        run_probe(fixture, certificate_dir, {"VTA_INSN_SUBMIT_THRESHOLD_BYTES": "4096"}),
    )
    record(
        "wrong_input_shape_rejected",
        False,
        run_probe(
            fixture,
            certificate_dir,
            actual_mutator=lambda value: value["stages"][0].update(input_names=["wrong"]),
        ),
    )
    original_graph = fixture / "package/stages/stage0/graph.json"
    saved_graph = original_graph.read_bytes()
    original_graph.write_bytes(saved_graph + b"\n")
    record("graph_hash_drift_rejected", False, run_probe(fixture, certificate_dir))
    original_graph.write_bytes(saved_graph)

    copied = raw / "certificate_profile_drift"
    shutil.copytree(certificate_dir, copied)
    profile = copied / "default_capacity_profile.json"
    profile.write_bytes(profile.read_bytes() + b"\n")
    record("profile_hash_drift_rejected", False, run_probe(fixture, copied))

    record(
        "synthetic_gate_missing_rejected",
        False,
        run_probe(fixture, certificate_dir, {"VTA_COMMAND_CERTIFICATE_ALLOW_SYNTHETIC": None}),
    )
    record(
        "wrong_runtime_rejected",
        False,
        run_probe(
            fixture,
            certificate_dir,
            actual_mutator=lambda value: value.update(runtime=str(fixture / "libvta.fixture.so")),
        ),
    )

    output.mkdir(exist_ok=True)
    results_path = raw / "cases.json"
    write_json(results_path, cases)
    passed = sum(case["passed"] for case in cases)
    summary = {
        "schema": "thesis_command_capacity_host_summary_v1",
        "status": "passed" if passed == len(cases) else "failed",
        "passed": passed,
        "selected": len(cases),
        "positive_cases": sum(case["expected_success"] for case in cases),
        "negative_cases": sum(not case["expected_success"] for case in cases),
        "hardware_calls": 0,
        "evidence_scope": "synthetic_host_fixture",
        "formal_q1_certificate_issued": False,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in output.rglob("*")
                if path.is_file() and path.name != "artifact_hashes.json"
            }
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
