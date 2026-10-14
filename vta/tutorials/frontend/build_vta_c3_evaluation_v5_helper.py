#!/usr/bin/env python3
"""Cross-compile and seal the board-side C3 v5 timing helper."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from c3_evaluation_v5 import seal_flat, sha256_file, write_json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "c3_evaluation_v5_helper.cc"
SDK = Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_OUTPUT = GROUPED / "20260926_c3_evaluation_v5_helper_build_run02"


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    binary = output / "c3_evaluation_v5_helper.so"
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
        "c3-v5-helper", str(args.sdk_environment), *compile_args,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    write_json(output / "build.json", {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "source_sha256": sha256_file(SOURCE),
    })
    if completed.returncode != 0 or not binary.is_file():
        raise RuntimeError("C3 v5 helper cross-compile failed")
    contract = {
        "schema": "c3_evaluation_v5_helper_v1",
        "status": "CROSS_COMPILED_BEFORE_BOARD_AUDIT",
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "source_sha256": sha256_file(SOURCE),
        "runtime_linkage": "libtvm_runtime only; VTA symbols resolved with RTLD_DEFAULT",
        "registered_functions": [
            "c3.v5.runtime_symbols", "c3.v5.reset_counts", "c3.v5.counts",
            "c3.v5.empty_sync", "c3.v5.operator_timed_once", "c3.v5.legacy_timed_once",
            "c3.v5.legacy_audit_once",
            "c3.v5.graph_run_only_timed_once", "c3.v5.graph_run_sync_timed_once",
        ],
    }
    write_json(output / "contract.json", contract)
    seal_flat(output, (Path(__file__).resolve(), SOURCE))
    print(json.dumps(contract, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-environment", type=Path, default=SDK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
