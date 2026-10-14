#!/usr/bin/env python3
"""Cross-compile and seal the correctness-only transition boundary helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "c3_transition_boundary_helper.cc"
SDK = Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    binary = output / "c3_transition_boundary_helper.so"
    compile_args = [
        "-std=c++17", "-shared", "-fPIC", str(SOURCE),
        "-I" + str(ROOT / "include"),
        "-I" + str(ROOT / "3rdparty/dlpack/include"),
        "-I" + str(ROOT / "3rdparty/dmlc-core/include"),
        "-L" + str(ROOT / "build_axu_aarch64"),
        "-ltvm_runtime", "-ldl", "-Wl,-rpath,$ORIGIN", "-o", str(binary),
    ]
    command = [
        "bash", "-c", 'unset LD_LIBRARY_PATH && source "$1" >/dev/null && shift && exec $CXX "$@"',
        "transition-boundary-helper", str(SDK), *compile_args,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    write(output / "build.json", {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "source_sha256": sha256_file(SOURCE),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
    })
    if completed.returncode != 0 or not binary.is_file():
        raise RuntimeError("transition boundary helper cross-compile failed")
    contract = {
        "schema": "c3_transition_boundary_helper_v1",
        "purpose": "correctness-only empty-sync and CommandQueue shutdown controls",
        "latency_collection_forbidden": True,
        "registered_functions": ["c3.transition.empty_sync", "c3.transition.shutdown_queue"],
        "source_sha256": sha256_file(SOURCE),
        "binary_sha256": sha256_file(binary),
    }
    write(output / "contract.json", contract)
    write(output / "artifact_hashes.json", {
        "artifacts": {
            path.name: sha256_file(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        }
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
