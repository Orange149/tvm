"""Offline-only assembly of the focused CPU-strategy experiment bundle."""
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from cpu_vta_partitioning.common import (
    REPO, digest, inventory, locked, read, relative_file, sha256, verify_inventory, write,
)
from cpu_vta_partitioning.campaign import controller_identity as base_identity
from cpu_vta_partitioning.prepare.build import sdk_command
from cpu_vta_partitioning.prepare.package import elf
from cpu_vta_partitioning.prepare.payload import execution_files
from cpu_vta_partitioning.prepare.toolchain import inspect_toolchain
from .domain import ROOT, PROJECT, components, matrix, protocol, validate_candidate
from .runner_patch import BASE_RUNNER_SHA256, PATCH_VERSION, patch_runner


AVAILABLE_INDICES = (0, 16, 17, 18)
PROGRAM_NAMES = ("cpu_00_02", "vta_03_17", "cpu_18_20", "cpu_00_00", "cpu_02_02", "cpu_20_20")
DEPENDENCIES = ("numpy", "scipy", "matplotlib")


def dependency_identity():
    """Read distribution versions without importing compiler or board modules."""
    return {name: version(name) for name in DEPENDENCIES}


def _native_source():
    snapshot = ROOT.parent / "native_source"
    return snapshot if snapshot.is_dir() else REPO / "vta/apps/native_deploy"


def identity():
    files = [ROOT / "run.py", *sorted((ROOT / "cpu_strategy").glob("*.py"))]
    native = _native_source()
    return digest({
        "base_controller": base_identity(),
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "native": {p.name: sha256(p) for p in [native / "vta_stage_pipeline_runner.cc", *sorted(native.glob("*.h"))]},
        "runner_patch": {"version": PATCH_VERSION, "preimage": BASE_RUNNER_SHA256},
        "protocol": protocol(), "python": sys.version,
        "dependencies": dependency_identity(),
    })


def _copy(source, target, expected=None):
    if not source.is_file() or source.is_symlink():
        raise ValueError("missing or linked bundle source: " + str(source))
    actual = sha256(source)
    if expected is not None and actual != expected:
        raise ValueError("source hash changed: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha256(target) != actual:
        raise ValueError("copied artifact differs: " + str(target))


def _record_files(record):
    files = {record["input"]}
    for frontier in record["frontiers"].values():
        files.update(frontier[key] for key in ("raw", "npy") if key in frontier)
    return files


def _subset_payload(base_bundle, workspace, manifest):
    programs = read(base_bundle / "plans/programs.json")
    if not set(PROGRAM_NAMES) <= set(programs):
        raise ValueError("frozen bundle lacks a required CPU or VTA module")
    selected = {name: programs[name] for name in PROGRAM_NAMES}
    for name, record in selected.items():
        for filename, expected in record["files"].items():
            _copy(base_bundle / "programs" / name / filename,
                  workspace / "programs" / name / filename, expected["sha256"])
    inputs = read(base_bundle / "plans/inputs.json")
    wanted = set()
    for index in AVAILABLE_INDICES:
        wanted.update(_record_files(inputs["records"][index]))
    for name in sorted(wanted):
        expected = manifest["files"].get(name)
        if expected is None:
            raise ValueError("input/reference is outside parent manifest: " + name)
        _copy(relative_file(base_bundle, name), relative_file(workspace, name), expected["sha256"])
    input_files = {name: manifest["files"][name] for name in sorted(wanted)}
    derived = []
    # Some fine-grained diagnostic inputs were retained as canonical NPY
    # references only. Materialize their identical C-order bytes in this new
    # bundle, without touching parent records or recompiling a graph.
    for index in AVAILABLE_INDICES:
        record = inputs["records"][index]
        for p in selected.values():
            if p["begin"] == 0:
                continue
            for node in p["input_node_ids"]:
                frontier = record["frontiers"][str(node)]
                if "raw" in frontier:
                    continue
                import numpy as np
                array = np.load(relative_file(workspace, frontier["npy"]), allow_pickle=False)
                if list(array.shape) != frontier["shape"] or str(array.dtype) != frontier["dtype"]:
                    raise ValueError("NPY input metadata differs from frozen frontier")
                name = f"inputs/derived_frontiers/{index:03}/{node}.bin"
                target = relative_file(workspace, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(array.tobytes(order="C"))
                frontier["raw"] = name
                input_files[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
                derived.append({"path": name, "source_npy": frontier["npy"],
                                "source_sha256": input_files[frontier["npy"]]["sha256"],
                                **input_files[name], "dtype": frontier["dtype"], "shape": frontier["shape"]})
    # Preserve original numeric indices: parent native_command indexes this list.
    # Only the explicitly available records constitute this bundle's input closure.
    inputs["available_indices"] = list(AVAILABLE_INDICES)
    inputs["files"] = input_files
    inputs["derived_frontier_inputs"] = derived
    write(workspace / "plans/inputs.json", inputs, replace=True)
    write(workspace / "plans/programs.json", selected, replace=True)
    write(workspace / "plans/segments.json", [dict(name=name, device=p["device"], begin=p["begin"], end=p["end"])
                                               for name, p in selected.items()], replace=True)
    for folder in ("runtime", "firmware"):
        for name, expected in inventory(base_bundle / folder).items():
            _copy(base_bundle / folder / name, workspace / folder / name, expected["sha256"])
    for name in ("compiler_identity.json", "runtime_build.json", "sdk_runtime.json"):
        _copy(base_bundle / "plans" / name, workspace / "plans" / name)
    return selected


NATIVE_SELF_TEST = r'''#define VTA_STAGE_PIPELINE_NO_MAIN
#include "vta_stage_pipeline_runner.cc"
int main() {
  Args args;
  StageArgs stage;
  stage.device = "cpu";
  stage.runtime_num_threads = 2;
  stage.cpu_affinity = {0, 1, 2, 3};
  args.stages = {stage};
  bool rejected = false;
  try { ValidateStageAffinities(args); } catch (const std::runtime_error&) { rejected = true; }
  if (!rejected) return 10;
  args.allow_cpu_affinity_superset = true;
  ValidateStageAffinities(args);
  args.stages[0].runtime_num_threads = 5;
  rejected = false;
  try { ValidateStageAffinities(args); } catch (const std::runtime_error&) { rejected = true; }
  if (!rejected) return 11;
  args.stages[0].runtime_num_threads = 2;
  Frame frame;
  frame.done_ms = 12.0;
  frame.enqueue_ms = 1.0;
  frame.completion_process_cpu_ms = 7.25;
  frame.stage_timings.resize(1);
  auto json = ResultJSON(args, frame, 0, "pipeline", 0.0);
  if (json.find("\"completion_process_cpu_ms\":7.25") == std::string::npos) return 12;
  std::cout << "CPU strategy affinity and completion-clock self-test passed\n";
  return 0;
}
'''


def build_runner(workspace, local):
    native = _native_source()
    original = (native / "vta_stage_pipeline_runner.cc").read_text()
    patched, patch = patch_runner(original)
    snapshot = workspace / "sources/vta/apps/native_deploy"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in native.iterdir():
        if path.suffix in (".cc", ".h"):
            _copy(path, snapshot / path.name)
    (snapshot / "vta_stage_pipeline_runner.cc").write_text(patched)
    (workspace / "sources/runner.patch").write_text(patch.pop("unified_diff"))
    write(workspace / "plans/runner_patch.json", patch, replace=True)
    pico = workspace / "sources/3rdparty/picojson/picojson.h"
    _copy(REPO / "3rdparty/picojson/picojson.h", pico)
    output = workspace / "runner/vta_stage_pipeline_runner"
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = ('exec $CXX -std=c++17 -O2 "$1" -o "$2" -I"$3/include" '
              '-I"$3/3rdparty/dlpack/include" -I"$3/3rdparty/dmlc-core/include" '
              '-I"$3/3rdparty/vta-hw/include" -I"$5" -L"$4" '
              '-Wl,-rpath,\'$ORIGIN/../runtime\' -ltvm_runtime -lvta -ldl -pthread')
    command = ["bash", "-c", recipe, "cpu-strategy-runner", snapshot / "vta_stage_pipeline_runner.cc",
               output, REPO, workspace / "runtime", snapshot]
    sdk_command(local, command, workspace / "build_logs/runner.log")
    harness = workspace / "build_logs/native_selftest.cc"
    harness.write_text(NATIVE_SELF_TEST)
    command[4], command[5] = harness, workspace / "build_logs/native_selftest"
    sdk_command(local, command, workspace / "build_logs/native_selftest_build.log")
    return {"recipe": recipe, "runner_sha256": sha256(output), "elf": elf(output),
            "patch": patch, "selftest_sha256": sha256(workspace / "build_logs/native_selftest")}


def native_acceptance(workspace):
    emulator = shutil.which("qemu-aarch64-static")
    if not emulator:
        raise RuntimeError("QEMU is required for local loader and argument acceptance")
    loader = workspace / "runtime/ld-linux-aarch64.so.1"
    runner = workspace / "runner/vta_stage_pipeline_runner"
    dependency_records = {}
    for path in [runner, *[workspace / "programs" / n / "module.so" for n in PROGRAM_NAMES],
                 *[p for p in (workspace / "runtime").iterdir() if p.is_file()]]:
        item = elf(path)
        for name in item["needed"]:
            if not (workspace / "runtime" / name).is_file():
                raise ValueError("unresolved shared library: " + name)
        dependency_records[str(path.relative_to(workspace))] = item
    write(workspace / "plans/elf_dependencies.json", dependency_records, replace=True)
    args = [emulator, str(loader), "--library-path", str(workspace / "runtime")]
    help_result = subprocess.run([*args, str(runner), "--help"], text=True, capture_output=True, timeout=45)
    if help_result.returncode or "--allow-cpu-affinity-superset" not in help_result.stdout + help_result.stderr:
        raise RuntimeError("patched argument parser/loader failed: " + help_result.stderr[-1500:])
    selftest = subprocess.run([*args, str(workspace / "build_logs/native_selftest")],
                              text=True, capture_output=True, timeout=45)
    if selftest.returncode:
        raise RuntimeError("patched native affinity self-test failed: " + selftest.stderr[-1500:])
    # Exercise the widened mask with an actual retained CPU module and all four
    # selected inputs. QEMU adds its own helper thread; its process-level TID
    # inventory is not an actual board-affinity qualification.
    from cpu_vta_partitioning.domain import Stage
    from cpu_vta_partitioning.measurement.tasks import task, native_command
    from cpu_vta_partitioning.measurement import correctness, frames, input_sequence
    output = workspace / "build_logs/qemu_cpu_component"
    output.mkdir(parents=True, exist_ok=True)
    document = task("qualify_component", [Stage("cpu", 2, 3, 2, (0, 1, 2, 3))],
                    {"cores": 4, "poll_sleep_ns": 1000, "post_start_sleep_ns": 1000},
                    warmup=0, scored=4, tail=0, indices=list(AVAILABLE_INDICES))
    argv, lines = native_command(document, workspace, str(workspace), str(output))
    (output / "inputs.txt").write_text(lines)
    argv.append("--allow-cpu-affinity-superset")
    position = next(i for i, value in enumerate(argv) if value.endswith("ld-linux-aarch64.so.1"))
    argv.insert(position, emulator)
    numerical = subprocess.run(argv, text=True, capture_output=True, timeout=60)
    (output / "native.log").write_text(numerical.stdout + numerical.stderr)
    if numerical.returncode:
        raise RuntimeError("QEMU CPU numerical acceptance failed: " + numerical.stderr[-1500:])
    native_rows = frames(output / "result.jsonl", 4)
    input_sequence(native_rows, document)
    if not correctness(native_rows, document, workspace, output):
        raise ValueError("retained CPU module differs from canonical references")
    clocks = [r["completion_process_cpu_ms"] for r in native_rows]
    if (any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in clocks)
            or any(b < a for a, b in zip(clocks, clocks[1:])) or clocks[-1] <= clocks[0]):
        raise ValueError("completion process CPU clock is invalid")
    result = {"passed": True, "scope": "aarch64 ELF closure, QEMU parser, native affinity guard and CPU numerics; no FPGA or physical-affinity qualification",
              "runner_help": help_result.stdout + help_result.stderr, "native_selftest": selftest.stdout,
              "qemu_cpu_correctness": {"program": "cpu_02_02", "threads": 2, "allowed_cores": [0, 1, 2, 3],
                                       "inputs": list(AVAILABLE_INDICES), "passed": True,
                                       "completion_cpu_clock_monotonic": True}}
    write(workspace / "plans/native_acceptance.json", result, replace=True)
    return result


def _snapshot_controller(workspace):
    target = workspace / "sources/cpu_strategy"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(ROOT / "cpu_strategy", target / "cpu_strategy", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("run.py", "pyproject.toml", "README.md"):
        if (ROOT / name).is_file():
            _copy(ROOT / name, target / name)
    for name in ("tests", "configs", "docs"):
        if (ROOT / name).is_dir():
            shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "local.json"))
    base = workspace / "sources/base_controller/src"
    if base.exists():
        shutil.rmtree(base)
    shutil.copytree(PROJECT / "src", base, ignore=shutil.ignore_patterns("__pycache__"))
    native = workspace / "sources/native_source"
    if native.exists():
        shutil.rmtree(native)
    native.mkdir(parents=True)
    for p in _native_source().iterdir():
        if p.suffix in (".cc", ".h"):
            _copy(p, native / p.name)


def validate(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "bundle_id"})[:24] != manifest["bundle_id"]:
        raise ValueError("bundle manifest identity mismatch")
    if manifest["controller_identity"] != identity() or manifest["experiment"] != protocol()["experiment"]:
        raise ValueError("CPU strategy controller or experiment identity changed")
    verify_inventory(bundle, manifest["files"])
    if read(bundle / "plans/cpu_strategy_protocol.json") != protocol():
        raise ValueError("bundled protocol differs from the frozen implementation")
    if read(bundle / "plans/controller_dependencies.json") != dependency_identity():
        raise ValueError("CPU strategy controller dependencies changed")
    programs = read(bundle / "plans/programs.json")
    if set(programs) != set(PROGRAM_NAMES):
        raise ValueError("unexpected module coverage")
    data = matrix()
    if len(data["configs"]) != 32 or len(components()) != 34:
        raise ValueError("focused configuration/measurement coverage differs")
    for row in data["configs"].values():
        validate_candidate(row, programs)
    inputs = read(bundle / "plans/inputs.json")
    if inputs.get("available_indices") != list(AVAILABLE_INDICES):
        raise ValueError("unexpected input subset")
    required = set().union(*(_record_files(inputs["records"][i]) for i in AVAILABLE_INDICES))
    if set(inputs["files"]) != required:
        raise ValueError("input/reference subset closure differs")
    verify_inventory(bundle, inputs["files"])
    patch = read(bundle / "plans/runner_patch.json")
    if patch["preimage_sha256"] != BASE_RUNNER_SHA256 or patch["version"] != PATCH_VERSION:
        raise ValueError("runner patch source identity changed")
    if sha256(bundle / "sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc") != patch["patched_sha256"]:
        raise ValueError("patched runner source differs")
    if sha256(bundle / "sources/runner.patch") != patch["patch_sha256"]:
        raise ValueError("runner patch record differs")
    payload = read(bundle / "plans/board_payload.json")
    if payload["files"] != execution_files(bundle) or payload["unpacked_bytes"] != sum(x["bytes"] for x in payload["files"].values()):
        raise ValueError("board execution closure differs")
    readiness = read(bundle / "OFFLINE_READINESS.json")
    if readiness["status"] != "PREPARED_OFFLINE" or not readiness["tests"]["passed"]:
        raise ValueError("offline acceptance incomplete")
    return manifest


def prepare(base_bundle, local_config, config=None):
    base_bundle = Path(base_bundle).resolve()
    parent = read(base_bundle / "manifest.json")
    if digest({k: v for k, v in parent.items() if k != "bundle_id"})[:24] != parent["bundle_id"]:
        raise ValueError("parent manifest identity mismatch")
    if parent["controller_identity"] != base_identity():
        raise ValueError("base controller changed from the verified parent")
    verify_inventory(base_bundle, parent["files"])
    if config is not None and read(config) != protocol():
        raise ValueError("configuration differs from the implemented protocol")
    controller = identity()
    key = digest({"parent_bundle": parent["bundle_id"], "controller": controller})[:24]
    workspace = ROOT / "artifacts/builds" / key
    with locked(workspace):
        programs = _subset_payload(base_bundle, workspace, parent)
        actual_toolchain = inspect_toolchain(workspace, read(local_config))
        if actual_toolchain != read(base_bundle / "plans/compiler_identity.json")["toolchain"]:
            raise ValueError("SDK differs from frozen modules/runtime")
        native = build_runner(workspace, read(local_config))
        native_acceptance(workspace)
        write(workspace / "plans/cpu_strategy_protocol.json", protocol(), replace=True)
        write(workspace / "plans/controller_dependencies.json", dependency_identity(), replace=True)
        write(workspace / "plans/configurations.json", matrix(), replace=True)
        write(workspace / "plans/components.json", components(), replace=True)
        write(workspace / "plans/parent_identity.json", {
            "bundle_id": parent["bundle_id"], "manifest_sha256": sha256(base_bundle / "manifest.json"),
            "modules": {name: p["files"] for name, p in programs.items()},
            "runtime": inventory(workspace / "runtime"), "toolchain": actual_toolchain, "native_build": native,
        }, replace=True)
        write(workspace / "plans/board_execution.json", {
            "revision": "cpu-strategy-1", "udmabuf_bytes": 201326592,
            "storage_margin_bytes": 256 * 1024**2, "boot_policy": "manual_reboot_only",
            "recovery_io": "boot_session_lock", "cpu_gate_modes": ["off"],
            "analysis_evidence": "native_files_and_hashed_receipts",
        }, replace=True)
        payload = execution_files(workspace)
        write(workspace / "plans/board_payload.json", {"revision": 1, "files": payload,
            "unpacked_bytes": sum(x["bytes"] for x in payload.values()),
            "transfer": "gzip_tar_stream_no_remote_archive"}, replace=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   CPUVTA_CPU_STRATEGY_WORKSPACE=str(workspace))
        log = workspace / "build_logs/local_tests.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
                                     "--confcutdir=" + str(ROOT), str(ROOT / "tests"), "-q"],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("local CPU strategy acceptance failed: " + str(log))
        if identity() != controller:
            raise ValueError("controller source changed during preparation")
        _copy(log, workspace / "plans/local_tests.log")
        _snapshot_controller(workspace)
        write(workspace / "OFFLINE_READINESS.json", {
            "status": "PREPARED_OFFLINE", "controller_identity": controller, "base_bundle_id": parent["bundle_id"],
            "compile_coverage": {"programs": 6, "new_model_compilations": 0, "component_signatures": 34},
            "tests": {"passed": True, "log_sha256": sha256(log), "result": log.read_text()[-6000:]},
            "board_validation_pending": ["actual affinity", "native correctness", "new costs", "steady performance", "independent boots"],
            "performance_claim": None,
        }, replace=True)
        parent_dir = ROOT / "artifacts/bundles"
        parent_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".sealing-", dir=parent_dir))
        try:
            for folder in ("programs", "runtime", "runner", "firmware", "inputs", "references", "plans", "sources"):
                shutil.copytree(workspace / folder, staging / folder)
            _copy(workspace / "OFFLINE_READINESS.json", staging / "OFFLINE_READINESS.json")
            manifest = {"schema_version": 1, "experiment": protocol()["experiment"],
                        "controller_identity": controller, "parent_bundle_id": parent["bundle_id"], "files": inventory(staging)}
            manifest["bundle_id"] = digest(manifest)[:24]
            write(staging / "manifest.json", manifest)
            validate(staging)
            with tempfile.TemporaryDirectory(prefix="cpu-strategy-relocation-") as directory:
                relocated = Path(directory) / "bundle"
                shutil.copytree(staging, relocated)
                checked = subprocess.run([sys.executable, str(relocated / "sources/cpu_strategy/run.py"),
                                          "validate", "--bundle", str(relocated)],
                                         text=True, capture_output=True, timeout=180)
                if checked.returncode:
                    raise RuntimeError("relocated CPU strategy bundle failed: " + checked.stderr[-2500:])
            destination = parent_dir / manifest["bundle_id"]
            if destination.exists():
                validate(destination)
                shutil.rmtree(staging)
            else:
                staging.rename(destination)
            write(workspace / "plans/relocation_validation.json", {"passed": True, "stdout": checked.stdout,
                "bundle_id": manifest["bundle_id"]}, replace=True)
            write(ROOT / "artifacts/LATEST_BUNDLE.json", {"path": str(destination), "bundle_id": manifest["bundle_id"]}, replace=True)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)
