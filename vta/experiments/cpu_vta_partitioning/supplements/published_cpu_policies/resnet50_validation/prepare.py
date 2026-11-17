"""Resumable local builds and self-contained profile capsules; never SSH."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import re
import signal
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import digest, inventory, locked, read, sha256, sync_tree, verify_inventory, write
from cpu_vta_partitioning.prepare.build import copy_checked, sdk_command
from .assets import (DEFAULT_LOCAL_CONFIG, DEFAULT_OUTPUT, LEGACY_OUTPUT, PARENT, PROJECT, REPO,
                     ROOT, audit_assets, source_path)


REFERENCE_PROGRAMS = [
    {"name": "reference_full", "kind": "complete_graph", "device": "cpu", "eligible_search": False},
    {"name": "reference_frontiers", "kind": "complete_frontier_probe", "device": "cpu", "eligible_search": False},
]
MODULE_FILES = ("graph.json", "module.so", "params.bin")


def writable_output(value):
    path = Path(value).resolve()
    if path == LEGACY_OUTPUT.resolve() or LEGACY_OUTPUT.resolve() in path.parents:
        raise ValueError("stopped compiler-regression output is read-only")
    return path


def reuse_graph_assets(output=DEFAULT_OUTPUT, source=LEGACY_OUTPUT):
    """Copy verified graph/unit evidence; never relabel old compiled programs."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source:
        raise ValueError("native-history recovery must use a new output directory")
    output = writable_output(output)
    identity = read(source / "build/plans/compiler_identity.json")
    verify_inventory(source / "build", identity["files"])
    graph = read(source / "graph/GRAPH_STATUS.json")
    if graph["canonical_sha256"] != identity["canonical_sha256"]:
        raise ValueError("source graph differs from the previously frozen build")
    copied = {}
    for folder in ("graph", "units"):
        for relative, info in inventory(source / folder).items():
            if ".lock" in Path(relative).parts or "__pycache__" in Path(relative).parts:
                continue
            copy_checked(source / folder / relative, output / folder / relative, info["sha256"])
            copied[folder + "/" + relative] = info
    return _frozen(output / "GRAPH_ASSET_REUSE.json", {
        "source": str(source), "canonical_sha256": identity["canonical_sha256"],
        "source_compiler_identity_sha256": sha256(source / "build/plans/compiler_identity.json"),
        "files": copied, "compiled_modules_imported": False})


def _frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("frozen preparation identity changed: " + str(path))
    else:
        write(path, value)
    return value


def audit(output=None, local_config=DEFAULT_LOCAL_CONFIG):
    value = audit_assets(local_config)
    if output is not None:
        _frozen(writable_output(output) / "ASSET_AUDIT.json", value)
    return value


def initialize(graph_dir, units_dir, workspace, local_config=DEFAULT_LOCAL_CONFIG):
    graph_dir, units_dir, workspace = map(lambda p: Path(p).resolve(), (graph_dir, units_dir, workspace))
    writable_output(workspace)
    assets = audit_assets(local_config)
    local = read(local_config)
    status = read(graph_dir / "GRAPH_STATUS.json")
    if status["canonical_sha256"] != sha256(graph_dir / "canonical_qir.json"):
        raise ValueError("common graph bytes differ from graph status")
    if status["input_sha256"] != sha256(graph_dir / "input0.bin"):
        raise ValueError("common input bytes differ from graph status")
    unit_proof = read(units_dir / "UNIT_STATUS.json")
    if (unit_proof["source_sha256"] != status["canonical_sha256"]
            or not unit_proof.get("all_interval_recompositions_exact")):
        raise ValueError("typed unit proof does not cover this common graph")
    programs = read(units_dir / "programs.json")
    if isinstance(programs, dict):
        programs = list(programs.values())
    if not programs or len({p["name"] for p in programs}) != len(programs):
        raise ValueError("qualified program names must be nonempty and unique")
    from .domain import programs as domain_programs
    if programs != domain_programs(read(units_dir / "units_and_boundaries.json")):
        raise ValueError("program catalog differs from the complete declared interval domain")
    cpu = sum(p["device"] == "cpu" for p in programs)
    vta = sum(p["device"] == "vta" for p in programs)
    if cpu > 314 or vta > 299:
        raise ValueError("qualified module domain exceeds approved 314 CPU / 299 VTA budget")
    n = max(p["end"] for p in programs)
    probes = [{**p, "begin": 0, "end": n} for p in REFERENCE_PROGRAMS]
    specs = [*programs, *probes]
    for name in ("canonical_qir.json", "GRAPH_STATUS.json", "qconfig.json"):
        copy_checked(graph_dir / name, workspace / "sources" / name)
    copy_checked(graph_dir / "input0.bin", workspace / "inputs/000.bin")
    for name in ("units_and_boundaries.json", "UNIT_STATUS.json", "DOMAIN_COUNTS.json"):
        if (units_dir / name).is_file():
            copy_checked(units_dir / name, workspace / "sources/unit_proof" / name)
    copy_checked(source_path(local["tuning_history"]), workspace / "sources/tuning_history.log")
    package = Path(__file__).parent
    # Freeze only the compiler's actual closure, not a campaign still being authored.
    for name in ("compiler.py", "units.py", "graph.py"):
        copy_checked(package / name, workspace / "sources/resnet50_validation" / name)
    init = workspace / "sources/resnet50_validation/__init__.py"
    if not init.exists():
        init.write_text('"""Frozen isolated ResNet50 compiler closure."""\n')
    for name in ("vta/python/vta/top/graphpack.py", "3rdparty/vta-hw/config/vta_config.json",
                 "python/tvm/autotvm/task/dispatcher.py"):
        copy_checked(REPO / name, workspace / "sources/repository" / name)
    identity = {
        "model": "mxnet_gluon_resnet50_v2", "unit_count": n,
        "canonical_sha256": status["canonical_sha256"],
        "weights_sha256": assets["weights"]["sha256"],
        "input_sha256": status["input_sha256"],
        "host_tvm_sha256": assets["tools"]["host_tvm_library"]["sha256"],
        "sdk_setup_sha256": assets["tools"]["sdk_setup"]["sha256"],
        "reused_runner_sha256": assets["assets"]["runner/vta_stage_pipeline_runner"]["sha256"],
        "files": {name: value for name, value in inventory(workspace / "sources").items()
                  if "__pycache__" not in Path(name).parts and not name.endswith(".pyc")},
        "programs": specs, "history_usage": "native ApplyHistoryBest: model then target-key, always full template/workload; explicit default fallback",
        "runtime_or_executor_recompiled": False,
    }
    identity["files"] = {"sources/" + key: item for key, item in identity["files"].items()}
    _frozen(workspace / "plans/compiler_identity.json", identity)
    _frozen(workspace / "plans/programs.json", specs)
    _frozen(workspace / "plans/assets.json", assets)
    return local


def validate_build(workspace, *, require_all=True):
    workspace = Path(workspace)
    identity = read(workspace / "plans/compiler_identity.json")
    verify_inventory(workspace, identity["files"])
    complete, missing = {}, []
    for spec in identity["programs"]:
        marker = workspace / "programs" / spec["name"] / "record.json"
        if not marker.is_file():
            missing.append(spec["name"])
            continue
        record = read(marker)
        if (record["build_identity"] != digest(identity) or record["spec"] != spec
                or record["canonical_sha256"] != identity["canonical_sha256"]):
            raise ValueError("module identity differs: " + spec["name"])
        verify_inventory(marker.parent, record["files"])
        complete[spec["name"]] = record
    if require_all and missing:
        raise ValueError("missing compiled programs: " + ", ".join(missing[:8]))
    return {"complete": complete, "missing": missing, "identity": identity}


def next_build_log(workspace, name):
    directory = Path(workspace) / "build_logs"
    directory.mkdir(parents=True, exist_ok=True)
    old = directory / (name + ".log")
    legacy = directory / (name + ".attempt_000.log")
    if old.is_file() and not legacy.exists():
        copy_checked(old, legacy)
    index = 1
    while (directory / (name + f".attempt_{index:03d}.log")).exists():
        index += 1
    return directory / (name + f".attempt_{index:03d}.log")


def available_memory_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host MemAvailable is required for guarded compilation")


def guarded_sdk_command(local, argv, output, *, cwd, samples):
    """Original SDK invocation with monitored, owned process-group cancellation."""
    reserve, admission = 2 * 1024**3, 4 * 1024**3
    if available_memory_bytes() < admission:
        raise MemoryError("fewer than 4 GiB host memory available before compiler admission")
    env = dict(os.environ); env.pop("LD_LIBRARY_PATH", None)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [PROJECT / "src", REPO / "python", REPO / "vta/python"])),
               TVM_LIBRARY_PATH=str(source_path(local["host_tvm_library"]).parent),
               OMP_NUM_THREADS="1", TVM_NUM_THREADS="1")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    minimum, peak, count = available_memory_bytes(), 0, 0
    with output.open("w") as stream:
        child = subprocess.Popen(["bash", "-c", 'source "$1" || exit $?; shift; exec "$@"',
            "sdk", str(source_path(local["sdk_setup"])), *map(str, argv)], cwd=cwd,
            env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                available = available_memory_bytes(); minimum = min(minimum, available); count += 1
                try:
                    values = Path(f"/proc/{child.pid}/status").read_text().splitlines()
                    peak = max([peak, *(int(x.split()[1]) * 1024 for x in values if x.startswith("VmHWM:"))])
                except FileNotFoundError:
                    pass
                if available < reserve:
                    raise MemoryError("host available memory fell below the 2 GiB compiler reserve")
                time.sleep(0.5)
        except BaseException:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
            raise
        finally:
            samples.append({"program": str(argv[-1]), "minimum_host_available_bytes": minimum,
                            "peak_observed_compiler_VmHWM_bytes": peak, "samples": count})
    if child.returncode:
        raise RuntimeError(f"local tool exited {child.returncode}; see {output}")


def build(output=DEFAULT_OUTPUT, local_config=DEFAULT_LOCAL_CONFIG, *, names=None):
    output = writable_output(output)
    if not (output / "graph/GRAPH_STATUS.json").exists():
        reuse_graph_assets(output)
    workspace = output / "build"
    if shutil.disk_usage(output).free < 30 * 1024**3:
        raise RuntimeError("at least 30 GiB local free space is required before module preparation")
    started = time.monotonic()
    with locked(workspace):
        local = initialize(output / "graph", output / "units", workspace, local_config)
        specs = read(workspace / "plans/programs.json")
        requested = list(names or [p["name"] for p in specs])
        if not requested or len(set(requested)) != len(requested) or set(requested) - {p["name"] for p in specs}:
            raise ValueError("unknown or duplicate requested module")
        result, failures, memory_samples = [], {}, []
        cached_before = {name for name in requested if (workspace / "programs" / name / "record.json").exists()}
        def one(name):
            marker = workspace / "programs" / name / "record.json"
            reused = marker.exists()
            if not reused:
                log = next_build_log(workspace, name)
                guarded_sdk_command(local, [local["python"], "-m", "resnet50_validation.compiler",
                            "--workspace", workspace, "--program", name],
                            log, cwd=workspace / "sources", samples=memory_samples)
            record = read(marker)
            verify_inventory(marker.parent, record["files"])
            if record["build_identity"] != digest(read(workspace / "plans/compiler_identity.json")):
                raise ValueError("cached module identity differs")
            return {"name": name, "compile_seconds": record["compile_seconds"], "reused": reused}
        workers = max(1, min(4, int(local.get("resnet50_compile_jobs", 4))))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, name): name for name in requested}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    row = future.result()
                    result.append(row)
                    print(name, "ready", round(row["compile_seconds"], 2), "s", flush=True)
                except Exception as error:
                    failures[name] = {"type": type(error).__name__, "message": str(error)}
                    for pending in futures:
                        pending.cancel()
                    break
        checked = validate_build(workspace, require_all=False)
        # A second already-running worker may finish after cancellation. Count
        # every newly produced record once rather than lose its build cost.
        result = [{"name": name, "compile_seconds": checked["complete"][name]["compile_seconds"],
                   "reused": name in cached_before}
                  for name in requested if name in checked["complete"]]
        report = {"status": "BUILD_FAILED" if failures else
                            "LOCAL_PROGRAM_SUBSET_COMPILED" if checked["missing"] else "LOCAL_PROGRAMS_COMPILED",
                  "programs": sorted(result, key=lambda p: p["name"]), "failures": failures,
                  "remaining_programs": checked["missing"], "wall_seconds": time.monotonic() - started,
                  "new_compile_worker_seconds": sum(r["compile_seconds"] for r in result if not r["reused"]),
                  "reuse_worker_seconds_not_new_cost": sum(r["compile_seconds"] for r in result if r["reused"]),
                  "compiler_processes_limit": workers, "host_memory_admission_bytes": 4 * 1024**3,
                  "host_memory_abort_reserve_bytes": 2 * 1024**3,
                  "memory_samples": memory_samples,
                  "board_accessed": False, "runtime_or_executor_recompiled": False}
        # Keep each preparation call's measured wall cost when the status pointer
        # moves from a pilot subset to the full catalog or a recovered attempt.
        _frozen(workspace / "build_reports" / (digest(report) + ".json"), report)
        write(workspace / "BUILD_STATUS.json", report, replace=True)
        if failures:
            raise RuntimeError("stage compilation failed; preserve logs and do not reduce the domain")
        return report


def graph_storage_estimate(graph, parameter_bytes=0):
    """Conservative visible tensor-storage accounting, not a measured peak.

    Storage is max tensor bytes per storage_id, plus parameter-file bytes and
    two visible input/output sets. Runtime scratch and allocator effects remain
    unknown, so passing this check never proves board memory sufficiency.
    """
    attrs = graph["attrs"]
    shapes, dtypes, slots = (attrs[key][1] for key in ("shape", "dltype", "storage_id"))
    if not (len(shapes) == len(dtypes) == len(slots)):
        raise ValueError("graph storage metadata lengths differ")
    devices = attrs.get("device_index", [None, [-1] * len(slots)])[1]
    if len(devices) != len(slots):
        raise ValueError("graph device storage metadata lengths differ")
    sizes, storage = [], {}
    for shape, dtype, slot, device in zip(shapes, dtypes, slots, devices):
        match = re.fullmatch(r"(?:u?int|float|bfloat)(\d+)(?:x(\d+))?", dtype)
        if dtype == "bool": bits, lanes = 1, 1
        elif match: bits, lanes = int(match[1]), int(match[2] or 1)
        else: raise ValueError("unsupported graph dtype: " + dtype)
        elements = 1
        for dim in shape:
            if int(dim) < 0: raise ValueError("dynamic graph storage shape")
            elements *= int(dim)
        count = elements * ((bits + 7) // 8) * lanes
        sizes.append(count)
        if int(slot) >= 0:
            key = int(device), int(slot)
            storage[key] = max(storage.get(key, 0), count)
    rows = graph["node_row_ptr"]
    inputs = [rows[i] for i in graph["arg_nodes"] if graph["nodes"][i]["name"].startswith("edge")
              or graph["nodes"][i]["name"] == "data"]
    outputs = [rows[node] + port for node, port, _ in graph["heads"]]
    visible = sum(storage.values())
    by_device = {str(device): sum(value for (dev, _), value in storage.items() if dev == device)
                 for device in sorted({device for device, _ in storage})}
    boundary = sum(sizes[i] for i in [*inputs, *outputs])
    return {"graph_tensor_storage_bytes": visible, "params_file_bytes": int(parameter_bytes),
            "physical_device_storage_bytes": by_device,
            "visible_ext_dev_storage_bytes": by_device.get("12", 0),
            "physical_device_mapping_known": "device_index" in attrs,
            "two_boundary_sets_bytes": 2 * boundary,
            "visible_storage_plus_parameters_and_boundary_bytes": visible + int(parameter_bytes) + 2 * boundary,
            "is_measured_peak": False, "strict_upper_bound": False,
            "unknown": ["TIR workspace", "allocator overhead", "runtime temporary storage", "actual residency"]}


def inspect_module(path, runtime):
    """Check the emitted ELF and every direct shared-library dependency."""
    path, runtime = Path(path), Path(runtime)
    with path.open("rb") as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:4] != b"\x7fELF" or header[4] != 2
            or header[5] != 1 or int.from_bytes(header[18:20], "little") != 183):
        raise ValueError("compiled module is not a little-endian AArch64 ELF64: " + str(path))
    readelf = shutil.which("readelf")
    if not readelf:
        raise RuntimeError("readelf is required for local ABI inspection")
    checked = subprocess.run([readelf, "-d", str(path)], capture_output=True, text=True, timeout=30)
    if checked.returncode:
        raise ValueError("could not inspect compiled ELF dependencies")
    needed = sorted(set(re.findall(r"Shared library: \[([^\]]+)\]", checked.stdout)))
    missing = [name for name in needed if not (runtime / name).is_file()]
    if missing:
        raise ValueError("capsule lacks a shared-library dependency: " + ", ".join(missing))
    return {"class": "ELF64", "machine": "AArch64", "endianness": "little", "needed": needed,
            "direct_dependencies_present": True, "loading_on_board_proved": False}


def prepare_references(output=DEFAULT_OUTPUT):
    from .references import prepare, run_qemu
    output = writable_output(output)
    prepare(output / "build", output / "reference_capsule")
    return run_qemu(output / "reference_capsule", output / "references")


def pilot_programs(programs, units):
    """Select structural coverage without looking at costs or pipeline FPS."""
    values = list(programs.values()) if isinstance(programs, dict) else list(programs)
    cpu = [p for p in values if p["device"] == "cpu"]
    vta = [p for p in values if p["device"] == "vta"]
    n = len(units)
    projections = [u for u in units if u["kind"] == "projection"]
    mains = [u for u in units if u["kind"] == "main_preadd"]
    if len(cpu) < 6 or len(vta) < 3 or len(projections) < 2 or not mains:
        raise ValueError("structural pilot requires six CPU and three VTA modules")
    used = set()
    def choose(rows, begin, end, *, longest=False):
        eligible = [p for p in rows if p["name"] not in used and p["begin"] <= begin and p["end"] >= end]
        if not eligible:
            raise ValueError("no unused qualified pilot interval covers structural feature")
        selected = min(eligible, key=lambda p: ((-1 if longest else 1) * (p["end"] - p["begin"]), p["begin"], p["end"], p["name"]))
        used.add(selected["name"])
        return selected["name"]
    down = projections[1]
    down_main = next(u for u in mains if u["block"] == down["block"])
    mid = mains[len(mains) // 2]
    cpu_names = [choose(cpu, 0, 1),
                 choose(cpu, projections[0]["index"], projections[0]["index"] + 1),
                 choose(cpu, down_main["index"], down["index"] + 1),
                 choose(cpu, mid["index"], mid["index"] + 1),
                 choose(cpu, n - 1, n),
                 choose(cpu, 0, 1, longest=True)]
    mid_projection = projections[len(projections) // 2]
    proj_main = next(u for u in mains if u["block"] == mid_projection["block"])
    vta_names = [choose(vta, mains[0]["index"], mains[0]["index"] + 1),
                 choose(vta, proj_main["index"], mid_projection["index"] + 2),
                 choose(vta, mains[0]["index"], mains[0]["index"] + 1, longest=True)]
    return {"cpu": cpu_names, "vta": vta_names, "component_signatures": 39, "DMA_signatures": 3,
            "selection": "structural coverage before timing; no profile costs or pipeline FPS",
            "CPU_roles": ["stem", "first_projection", "first_downsampling", "middle_main", "terminal_head", "longest_prefix"],
            "VTA_roles": ["first_short_main", "middle_projection_and_merge", "longest_interval"]}


def prepare_profile(output=DEFAULT_OUTPUT):
    output = writable_output(output)
    with locked(output / "profile_capsule"):
        return _prepare_profile(output)


def _prepare_profile(output):
    output = Path(output).resolve()
    checked = validate_build(output / "build")
    proof = read(output / "references/REFERENCE_PROOF.json")
    if proof["status"] != "ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT" or proof["outputs"] != 1:
        raise ValueError("independent complete ARM output/frontier reference required")
    verify_inventory(output / "references/references", proof["files"])
    destination = output / "profile_capsule"
    if (destination / "manifest.json").exists():
        return validate_profile(destination)
    specs = [p for p in checked["identity"]["programs"] if p["kind"] == "interval"]
    parent = read(PARENT / "manifest.json")
    programs, estimates, abi = {}, {}, {}
    for spec in specs:
        name = spec["name"]; source = output / "build/programs" / name
        original = checked["complete"][name]
        if original["canonical_sha256"] != proof["canonical_sha256"]:
            raise ValueError("compiled interval and ARM reference have different common graphs")
        record = dict(original)
        record["excluded_nonexecution_proof_files"] = {k: v for k, v in original["files"].items() if k not in MODULE_FILES}
        record["files"] = {k: original["files"][k] for k in MODULE_FILES}
        record["source_record_sha256"] = sha256(source / "record.json")
        for leaf in MODULE_FILES:
            copy_checked(source / leaf, destination / "programs" / name / leaf)
        _frozen(destination / "programs" / name / "record.json", record)
        programs[name] = record
        estimates[name] = graph_storage_estimate(read(source / "graph.json"), (source / "params.bin").stat().st_size)
        abi[name] = inspect_module(source / "module.so", PARENT / "runtime")
    for name, info in parent["files"].items():
        if name.startswith(("runtime/", "firmware/")):
            copy_checked(PARENT / name, destination / name, info["sha256"])
    copy_checked(PARENT / "runner/vta_stage_pipeline_runner", destination / "runner/vta_stage_pipeline_runner",
                 parent["runner_sha256"])
    copy_checked(output / "build/inputs/000.bin", destination / "inputs/000.bin", proof["input"]["sha256"])
    frontiers, expected = {}, {}
    from .references import fnv1a64
    for node, info in proof["frontiers"].items():
        raw, npy = "inputs/frontiers/000/" + node + ".bin", "references/000/" + node + ".npy"
        copy_checked(output / "references" / info["raw"], destination / raw)
        copy_checked(output / "references" / info["npy"], destination / npy)
        frontiers[node] = {**info, "raw": raw, "npy": npy}
        data = (destination / raw).read_bytes()
        expected[node] = {"dtype": info["dtype"], "shape": info["shape"], "nbytes": len(data),
                          "fnv1a64": fnv1a64(data), "sha256": sha256(destination / raw)}
    _frozen(destination / "plans/programs.json", programs)
    _frozen(destination / "plans/inputs.json", {"records": [{"index": 0, "input": "inputs/000.bin", "frontiers": frontiers}], "available_indices": [0]})
    _frozen(destination / "plans/expected_outputs.json", expected)
    from cpu_policies.open_resources.domain import requirements
    configurations = requirements(programs)
    if len(configurations) > 2751:
        raise ValueError("component count exceeds approved budget")
    _frozen(destination / "plans/component_configurations.json", configurations)
    units = read(output / "units/units_and_boundaries.json")["units"]
    pilot = pilot_programs(programs, units)
    pilot_set = set(pilot["cpu"] + pilot["vta"])
    first = sorted(key for key, cfg in configurations.items() if cfg["stages"][0]["program"] in pilot_set)
    if len(first) != 39:
        raise ValueError("pilot does not have exactly 24 CPU + 15 VTA signatures")
    _frozen(destination / "plans/pilot.json", {**pilot, "component_keys": first})
    _frozen(destination / "plans/profile_order.json", first + sorted(set(configurations) - set(first)))
    _frozen(destination / "plans/storage_estimates.json", estimates)
    _frozen(destination / "plans/module_abi.json", abi)
    vta_visible = max((estimates[p["name"]]["visible_ext_dev_storage_bytes"] for p in specs if p["device"] == "vta"), default=0)
    _frozen(destination / "plans/board_execution.json", {"boot_policy": "one_boot_no_automatic_reboot",
        "udmabuf_bytes": 201326592, "storage_margin_bytes": 268435456,
        "calibration": {"warmup": 2, "scored": 8, "tail": 0, "input_index": 0},
        "qualification": "all-frame exact frontier hashes plus startup affinity probe",
        "thread_share_scope": "whole_native_session_including_warmup_and_startup_probe",
        "largest_visible_VTA_storage_estimate_bytes": vta_visible,
        "VTA_buffer_comparison_scope": "device_type12 tensor slots only; parameter blobs and output queues are host allocations",
        "visible_estimate_exceeds_existing_buffer": vta_visible > 201326592,
        "actual_memory_sufficiency_proved": False})
    # Reuse the proven profile validator without pretending YOLO outputs are R50 outputs.
    for folder in ("yolo_transfer",):
        for p in (ROOT / folder).rglob("*.py"):
            copy_checked(p, destination / "sources" / folder / p.relative_to(ROOT / folder))
    import cpu_vta_partitioning
    core = Path(cpu_vta_partitioning.__file__).resolve().parent
    for p in core.rglob("*.py"):
        copy_checked(p, destination / "sources/base_controller/cpu_vta_partitioning" / p.relative_to(core))
    for name in ("domain.py", "__init__.py"):
        copy_checked(ROOT / "cpu_policies/open_resources" / name, destination / "sources/open_resources" / name)
    for name in ("assets.py", "prepare.py", "references.py", "compiler.py", "units.py"):
        copy_checked(Path(__file__).parent / name, destination / "sources/resnet50_validation" / name)
    copy_checked(output / "references/REFERENCE_PROOF.json", destination / "sources/REFERENCE_PROOF.json")
    copy_checked(output / "build/plans/compiler_identity.json", destination / "sources/compiler_identity.json")
    copy_checked(output / "build/sources/tuning_history.log", destination / "sources/tuning_history.log")
    copy_checked(output / "build/sources/repository/python/tvm/autotvm/task/dispatcher.py",
                 destination / "sources/native_dispatcher.py")
    for name in ("NATIVE_HISTORY_AUDIT.json", "GRAPH_ASSET_REUSE.json", "REFERENCE_BRIDGE.json"):
        if (output / name).is_file():
            copy_checked(output / name, destination / "sources" / name)
    payload = {n: v for n, v in inventory(destination).items() if n.startswith(("programs/", "runtime/", "runner/", "firmware/", "inputs/")) and not n.endswith("/record.json")}
    payload["plans/board_execution.json"] = {"bytes": (destination / "plans/board_execution.json").stat().st_size, "sha256": sha256(destination / "plans/board_execution.json")}
    _frozen(destination / "plans/board_payload.json", {"files": payload, "unpacked_bytes": sum(v["bytes"] for v in payload.values())})
    identity = {"network": "resnet50_v2", "model": "MXNet Gluon pretrained ResNet50-v2", "unit_count": checked["identity"]["unit_count"],
                "scope": "true isolated ResNet50 component costs, no pipeline FPS",
                "canonical_sha256": proof["canonical_sha256"], "runner_sha256": parent["runner_sha256"],
                "runtime_sha256": sha256(destination / "runtime/libtvm_runtime.so"),
                "vta_runtime_sha256": sha256(destination / "runtime/libvta.so"),
                "input_sha256": proof["input"]["sha256"],
                "kernel_history_dispatcher": checked["identity"]["history_usage"],
                "kernel_history_sha256": sha256(destination / "sources/tuning_history.log"),
                "component_count": len(configurations), "files": inventory(destination, [".lock"])}
    write(destination / "manifest.json", {"bundle_id": digest(identity)[:24], **identity})
    sync_tree(destination)
    return validate_profile(destination)


def validate_profile(bundle):
    from yolo_transfer.profile import validate
    bundle = Path(bundle); manifest = validate(bundle)
    if manifest["network"] != "resnet50_v2" or manifest["unit_count"] != 38:
        raise ValueError("wrong model/unit identity")
    programs = read(bundle / "plans/programs.json")
    if (sum(p["device"] == "cpu" for p in programs.values()) != 314
            or sum(p["device"] == "vta" for p in programs.values()) != 299
            or manifest["component_count"] != 2751):
        raise ValueError("qualified profile domain is incomplete or exceeds the frozen scope")
    for name, program in programs.items():
        if program["canonical_sha256"] != manifest["canonical_sha256"]:
            raise ValueError("program graph identity differs")
        verify_inventory(bundle / "programs" / name, program["files"])
    if len(read(bundle / "plans/component_configurations.json")) != manifest["component_count"]:
        raise ValueError("component inventory count differs")
    for name in ("assets.py", "prepare.py", "references.py", "compiler.py", "units.py"):
        if sha256(Path(__file__).parent / name) != sha256(bundle / "sources/resnet50_validation" / name):
            raise ValueError("loaded ResNet50 preparation source changed: " + name)
    return manifest


def prepare(output=DEFAULT_OUTPUT, local_config=DEFAULT_LOCAL_CONFIG, *, stage="modules", names=None):
    if stage == "modules": return build(output, local_config, names=names)
    if stage == "references": return prepare_references(output)
    if stage == "profile": return prepare_profile(output)
    raise ValueError("preparation stage must be modules, references, or profile")
