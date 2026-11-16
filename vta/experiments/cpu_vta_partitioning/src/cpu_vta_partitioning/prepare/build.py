"""Prepare immutable packages locally, with resumable compilation workspaces."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import PROJECT, REPO, read, write, digest, sha256, inventory, verify_inventory, locked
from cpu_vta_partitioning.adapters import legacy_domain
from cpu_vta_partitioning.prepare.toolchain import inspect_toolchain, require_prepared_toolchain


def source_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO / p


def copy_checked(source, target, expected=None):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = sha256(source)
    if expected is not None and actual != expected:
        raise ValueError("source identity mismatch: " + str(source))
    if target.exists():
        if sha256(target) != actual:
            raise ValueError("preparation cache identity mismatch: " + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return actual


def sdk_command(local, argv, output, *, cwd=None):
    """Only explicit local tool argv. This module cannot run SSH/SCP."""
    if Path(argv[0]).name in ("ssh", "scp", "sftp"):
        raise ValueError("board transport forbidden in offline preparation")
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join(map(str, [PROJECT / "src", REPO / "python", REPO / "vta/python"]))
    env["TVM_LIBRARY_PATH"] = str(source_path(local["host_tvm_library"]).parent)
    env["OMP_NUM_THREADS"] = "1"
    env["TVM_NUM_THREADS"] = "1"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        result = subprocess.run(["bash", "-c", 'source "$1" || exit $?; shift; exec "$@"',
                                 "sdk", str(source_path(local["sdk_setup"])), *map(str, argv)],
                                cwd=cwd or REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"local tool exited {result.returncode}; see {output}")


@contextmanager
def initialize(config_path, local_path):
    """Hold the workspace lock from its first write through the caller's sealing."""
    config, local = read(config_path), read(local_path)
    from cpu_vta_partitioning.domain import validate_protocol
    validate_protocol(config)
    inputs = read(source_path(local["inputs_manifest"]))
    graph = source_path(local["canonical_graph"])
    if sha256(graph) != inputs["canonical_qir_sha256"]:
        raise ValueError("inputs and graph are not from the same quantized DAG")
    identity = {"protocol": config, "graph": sha256(graph),
                "history": sha256(source_path(local["tuning_history"])),
                "inputs": sha256(source_path(local["inputs_manifest"])),
                "domain": sha256(source_path(local["segment_manifest"])),
                "vta_config": sha256(REPO / "3rdparty/vta-hw/config/vta_config.json")}
    workspace = PROJECT / "artifacts/builds" / ("resnet18-" + digest(identity)[:16])
    with locked(workspace):
        if (workspace / "preparation_identity.json").exists():
            if read(workspace / "preparation_identity.json") != identity:
                raise ValueError("preparation identity drift")
        else:
            write(workspace / "preparation_identity.json", identity)
        write(workspace / "plans/protocol.json", config, replace=True)
        copy_checked(Path(config_path).parent / config["sensitivity_config"], workspace / "plans/sensitivity.json")
        copy_checked(graph, workspace / "sources/canonical_qir.json")
        copy_checked(source_path(local["tuning_history"]), workspace / "sources/tuning_history.log")
        copy_checked(REPO / "3rdparty/vta-hw/config/vta_config.json", workspace / "sources/vta_config.json")
        specs = legacy_domain(read(source_path(local["segment_manifest"])))
        write(workspace / "plans/segments.json", specs, replace=True)
        compiler_identity = {"host_tvm": sha256(source_path(local["host_tvm_library"])),
            "compiler_worker": sha256(PROJECT / "src/cpu_vta_partitioning/prepare/compiler.py"),
            "python_sources": {str(p.relative_to(REPO)): sha256(p) for folder in (REPO / "python/tvm", REPO / "vta/python")
                               for p in sorted(folder.rglob("*.py"))},
            "graph_adapter": sha256(REPO / "vta/tutorials/frontend/thesis_canonical_graph_split.py"),
            "graph": identity["graph"], "history": identity["history"], "vta_config": identity["vta_config"],
            "sdk_setup": sha256(source_path(local["sdk_setup"])),
            "toolchain": inspect_toolchain(workspace, local)}
        write(workspace / "plans/compiler_identity.json", compiler_identity, replace=True)
        for key, target in [("bitstream", "firmware/vta_hpc.bit"), ("udmabuf_module", "firmware/u-dma-buf.ko")]:
            copy_checked(source_path(local[key]), workspace / target)
        print("Preparation workspace:", workspace, flush=True)
        yield workspace, local


def prepare_inputs(workspace, local):
    import numpy as np
    from cpu_vta_partitioning.prepare.references import freeze_reference_source
    old, reference_identity = freeze_reference_source(
        source_path(local["inputs_manifest"]), workspace, sha256(workspace / "sources/canonical_qir.json"))
    destination = workspace / "plans/inputs.json"
    if destination.exists():
        data = read(destination)
        if data.get("reference_identity") != reference_identity:
            raise ValueError("cached inputs use a different reference source")
        verify_inventory(workspace, data["files"])
        return data
    if len(old["real_inputs"]) != 64:
        raise ValueError("expected 64 frozen real inputs")
    records = []
    for index, source in enumerate(old["real_inputs"] + old["diagnostic_inputs"]):
        rec = {"index": index, "role": "calibration" if index < 16 else "evaluation" if index < 64 else "diagnostic",
               "source_id": source["id"], "input": f"inputs/{index:03}.bin", "frontiers": {}}
        copy_checked(source["input"]["path"], workspace / rec["input"], source["input"]["sha256"])
        for node, ref in source["frontiers"].items():
            name = f"references/{index:03}/{node}.npy"
            copy_checked(ref["path"], workspace / name, ref["sha256"])
            arr = np.load(workspace / name, allow_pickle=False)
            item = {"npy": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}
            # Isolated stage profiling reads raw frontier inputs, not whole images.
            if index < 16 or index >= 64:
                raw = f"inputs/frontiers/{index:03}/{node}.bin"
                path = workspace / raw
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(arr.tobytes())
                item["raw"] = raw
            rec["frontiers"][node] = item
        records.append(rec)
    data = {"records": records, "reference_identity": reference_identity,
            "files": {**{f"inputs/{k}": v for k,v in inventory(workspace / "inputs").items()},
                                            **{f"references/{k}": v for k,v in inventory(workspace / "references").items()}}}
    write(destination, data)
    return data


def compile_programs(workspace, local):
    require_prepared_toolchain(workspace, local)
    specs = read(workspace / "plans/segments.json")
    def one(index):
        name = specs[index]["name"]
        started = time.monotonic()
        log = workspace / "build_logs" / (name + ".log")
        try:
            sdk_command(local, [local["python"], "-m", "cpu_vta_partitioning.prepare.compiler",
                                "--bundle", workspace, "--index", str(index)], log)
            print(name, "OK", round(time.monotonic() - started, 1), "s", flush=True)
            return name, None
        except RuntimeError as error:
            print(name, "FAILED", str(error), flush=True)
            return name, {"error": str(error), "log": str(log.relative_to(workspace))}
    with ThreadPoolExecutor(max_workers=int(local.get("compile_jobs", 2))) as pool:
        results = list(pool.map(one, range(len(specs))))
    failures = {k: v for k,v in results if v is not None}
    programs = {s["name"]: read(workspace / "programs" / s["name"] / "record.json")
                for s in specs if s["name"] not in failures}
    write(workspace / "plans/programs.json", programs, replace=True)
    write(workspace / "plans/compile_failures.json", failures, replace=True)
    return programs


def compile_runtime(workspace, local):
    """Build runtime/driver from current sources in a private SDK CMake tree."""
    toolchain = require_prepared_toolchain(workspace, local)
    settings = "set(USE_LLVM OFF)\nset(USE_VTA_FPGA ON)\nset(USE_RPC ON)\nset(USE_LIBBACKTRACE OFF)\nset(USE_GTEST OFF)\n"
    configuration = {"toolchain": toolchain, "cmake_settings": settings}
    root = workspace / "native_build" / digest(configuration)[:24]
    root.mkdir(parents=True, exist_ok=True)
    identity_file = root / "configuration_identity.json"
    if identity_file.exists() and read(identity_file) != configuration:
        raise ValueError("native build configuration identity mismatch")
    (root / "config.cmake").write_text(settings)
    if not identity_file.exists() or not (root / "build.ninja").exists():
        sdk_command(local, ["/usr/bin/cmake", "-S", REPO, "-B", root, "-G", "Ninja",
            "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DUSE_LLVM=OFF", "-DUSE_VTA_FPGA=ON",
            "-DUSE_RPC=ON", "-DUSE_LIBBACKTRACE=OFF", "-DUSE_GTEST=OFF"], workspace / "build_logs/runtime_configure.log")
        write(identity_file, configuration, replace=True)
    sdk_command(local, ["/usr/bin/cmake", "--build", root, "--target", "tvm_runtime", "vta", "-j", str(local.get("compile_jobs", 2))],
                workspace / "build_logs/runtime_build.log")
    for name in ["libtvm_runtime.so", "libvta.so"]:
        target = workspace / "runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    source = REPO / "vta/apps/native_deploy"
    native_sources = workspace / "sources/vta/apps/native_deploy"
    native_sources.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.suffix in (".cc", ".h"):
            shutil.copy2(p, native_sources / p.name)
    copy_checked(REPO / "3rdparty/picojson/picojson.h", workspace / "sources/3rdparty/picojson/picojson.h")
    output = workspace / "runner/vta_stage_pipeline_runner"
    output.parent.mkdir(parents=True, exist_ok=True)
    # The sourced SDK supplies compiler/sysroot flags; a shell expands only its own
    # SDK variables. All paths are passed as positional parameters.
    script = 'exec $CXX -std=c++17 -O2 "$1" -o "$2" -I"$3/include" -I"$3/3rdparty/dlpack/include" -I"$3/3rdparty/dmlc-core/include" -I"$3/3rdparty/vta-hw/include" -L"$4" -Wl,-rpath,\'$ORIGIN/../runtime\' -ltvm_runtime -lvta -ldl -pthread'
    sdk_command(local, ["bash", "-c", script, "compile-runner", native_sources / "vta_stage_pipeline_runner.cc",
                        output, REPO, workspace / "runtime"], workspace / "build_logs/runner_build.log")
    write(workspace / "plans/runtime_build.json", {
        "toolchain": toolchain, "native_build_directory": str(root.relative_to(workspace)),
        "cmake_settings":settings,"runner_compiler_recipe":script,"sdk_setup_sha256":sha256(source_path(local["sdk_setup"])),
        "compiler_identity":read(workspace / "plans/compiler_identity.json"),
        "runtime_sha256":sha256(workspace / "runtime/libtvm_runtime.so"),
        "driver_sha256":sha256(workspace / "runtime/libvta.so"),"runner_sha256":sha256(output)},replace=True)
    return output
