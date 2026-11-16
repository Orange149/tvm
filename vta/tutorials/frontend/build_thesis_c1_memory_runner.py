#!/usr/bin/env python3
"""Build a source-snapshotted AArch64 runner, without deploying or invoking it."""

import argparse
from pathlib import Path
import shlex
import shutil

from thesis_c1_memory_ablation import file_hash, seal, write_new
from thesis_supplement_run import RunStore


FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parents[2]
OLD_BUILD = FRONTEND / "report_out/stage_tile_cotuning/thesis_supplement_experiments/results/01/20260916T011355Z_completion_0670cfb1/raw/aarch64_build"


def build(output, sdk, runtime, driver):
    for path in (sdk, runtime, driver):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = ROOT / "vta/apps/native_deploy"
    sources = [source / "vta_stage_pipeline_runner.cc", *sorted(source.glob("*.h"))]
    store = RunStore.create(output, {"run_id": output.name, "stage": "C1_MEMORY_RUNNER_BUILD",
                            "campaign_id": "c1_memory_ablation_implementation",
                            "sdk": {"path": str(sdk.resolve()), "sha256": file_hash(sdk)},
                            "runtime": {"path": str(runtime.resolve()), "sha256": file_hash(runtime)},
                            "driver": {"path": str(driver.resolve()), "sha256": file_hash(driver)},
                            "sources": {str(p): file_hash(p) for p in sources}})
    snapshot = output / "raw/source"
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    runner = output / "raw/vta_stage_pipeline_runner_c1_memory"
    script = output / "raw/build.sh"
    includes = " ".join("-I " + shlex.quote(str(ROOT / name)) for name in
                        ("include", "3rdparty/dlpack/include", "3rdparty/dmlc-core/include", "3rdparty/vta-hw/include", "vta/include"))
    script.write_text("\n".join([
        "set -e", "unset LD_LIBRARY_PATH", "source " + shlex.quote(str(sdk.resolve())), "set -u",
        "aarch64-xilinx-linux-g++ -std=c++17 -O2 --sysroot=\"$SDKTARGETSYSROOT\" " + includes +
        " " + shlex.quote(str(snapshot / "vta_stage_pipeline_runner.cc")) +
        " " + shlex.quote(str(runtime.resolve())) + " " + shlex.quote(str(driver.resolve())) +
        " -Wl,-rpath-link,\"$SDKTARGETSYSROOT/lib\" -Wl,-rpath-link,\"$SDKTARGETSYSROOT/usr/lib\"" +
        " -ldl -pthread -o " + shlex.quote(str(runner)), ""]))
    action = store.execute(["bash", str(script)], "compile_aarch64_runner", "host_preparation", 300)
    if action["status"] != "ok":
        write_new(output / "invalid_session.json", {"reason": "AArch64 build failed", "action": action})
        raise RuntimeError("AArch64 build failed: " + action["stderr"])
    header = runner.read_bytes()[:20]
    if header[:4] != b"\x7fELF" or int.from_bytes(header[18:20], "little") != 183:
        raise ValueError("compiler did not produce an AArch64 ELF")
    manifest = seal({"schema": "c1_memory_runner_build_v1", "status": "built_not_board_qualified",
                     "runner": {"path": str(runner.resolve()), "sha256": file_hash(runner)},
                     "runtime": store.contract["runtime"], "driver": store.contract["driver"],
                     "sources": {p.name: file_hash(snapshot / p.name) for p in sources},
                     "compile_wall_seconds": action["wall_seconds"], "board_contacted": False,
                     "hardware_calls": 0, "multi_island_qualified": False})
    write_new(output / "build_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sdk", type=Path, default=Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"))
    parser.add_argument("--runtime", type=Path, default=OLD_BUILD / "libtvm_runtime.so")
    parser.add_argument("--driver", type=Path, default=OLD_BUILD / "libvta.so")
    args = parser.parse_args()
    result = build(args.output_dir.resolve(), args.sdk, args.runtime, args.driver)
    print(result["status"] + ": " + result["runner"]["path"])


if __name__ == "__main__":
    main()
