#!/usr/bin/env python3
"""Label-blind prepare/local/freeze stages of the C3 reuse supplement.

Board execution is deliberately unavailable until the local gate and an explicit
board protocol are closed. Every output directory is new and hash sealed.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEMA = "c3_reuse_supplement_v1"
SALT = "c3-reuse-mechanism-v1"
SEEDS = (0, 20250901, 20260910)
MODES = {"original": 0, "input_stationary": 1, "weight_resident_barrier": 4,
         "input_weight_resident_barrier": 5}
GEOMETRIES = {
    "M1": dict(ci=64, co=64, height=28, width=28, kernel=3, stride=1),
    "M2": dict(ci=128, co=128, height=14, width=14, kernel=3, stride=1),
    "M3": dict(ci=128, co=256, height=14, width=14, kernel=1, stride=2),
    "BND": dict(ci=128, co=256, height=14, width=14, kernel=3, stride=1),
}
BITSTREAM = "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(canonical(value).decode() + "\n")


def reject_labels(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"latency_ms", "performance_label", "oracle_latency", "fpga_correct"} and item is not None:
                raise ValueError("future label in local input: " + key)
            reject_labels(item)
    elif isinstance(value, list):
        for item in value:
            reject_labels(item)


def seal(directory):
    write(directory / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(directory)): file_hash(path) for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"}})


def verify(directory):
    directory = Path(directory).resolve()
    ledger = read(directory / "artifact_hashes.json")["artifacts"]
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path.name != "artifact_hashes.json"}
    if actual != set(ledger):
        raise ValueError("unbound or missing artifact")
    for name, expected in ledger.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != expected:
            raise ValueError("artifact hash/path mismatch: " + name)
    return read(directory / "contract.json")


def source_bindings():
    files = [Path(__file__), HERE / "c3_reuse_semantics.py", HERE / "c3_reuse_experimental_template.py",
             HERE / "run_vta_c3_resnet18_local_qualification.py", HERE / "c3_candidate_identity.py",
             HERE / "qualify_vta_residency_fsim.py", HERE / "collect_vta_fsim_command_signatures.py",
             HERE / "prepare_vta_c3_resnet18_literature_candidates.py",
             HERE / "cross_compile_vta_c3_resnet18_board_pool.py",
             Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")]
    for base in (ROOT / "vta/python", ROOT / "vta/runtime", ROOT / "3rdparty/vta-hw/src",
                 ROOT / "3rdparty/vta-hw/include", ROOT / "3rdparty/vta-hw/config"):
        files += [path for path in base.rglob("*") if path.suffix in {".py", ".cc", ".h", ".json"}]
    files += [ROOT / "build/libtvm.so", ROOT / "build/libtvm_runtime.so", ROOT / "build/libvta_fsim.so",
              ROOT / "build/build.ninja"]
    return {str(path.resolve()): file_hash(path) for path in sorted(set(files))}


def check_bindings(bindings):
    for name, expected in bindings.items():
        if file_hash(name) != expected:
            raise ValueError("implementation changed: " + name)


def hardware():
    import vta
    from prepare_vta_c3_resnet18_literature_candidates import hardware_fingerprint
    env = vta.get_env()
    observed = hardware_fingerprint(env)
    expected = dict(target="axu5evb", batch=1, block_in=16, block_out=16,
                    inp_buff_size=32768, wgt_buff_size=262144, acc_buff_size=131072, uop_buff_size=32768)
    if observed != expected or (env.INP_WIDTH, env.WGT_WIDTH, env.ACC_WIDTH) != (8, 8, 32):
        raise ValueError("hardware specification mismatch")
    # Verify the retained FSim was compiled for this ABI, not merely named fsim.
    ninja = (ROOT / "build/build.ninja").read_text()
    line = next(line for line in ninja.splitlines() if line.strip().startswith("DEFINES =") and "vta_fsim_EXPORTS" in line)
    for macro, value in {"LOG_BLOCK_IN": 4, "LOG_BLOCK_OUT": 4, "LOG_BATCH": 0,
                         "LOG_INP_WIDTH": 3, "LOG_WGT_WIDTH": 3, "LOG_ACC_WIDTH": 5,
                         "LOG_INP_BUFF_SIZE": 15, "LOG_WGT_BUFF_SIZE": 18,
                         "LOG_ACC_BUFF_SIZE": 17, "LOG_UOP_BUFF_SIZE": 15, "LOG_BUS_WIDTH": 7}.items():
        if "-DVTA_{}={}".format(macro, value) not in line.split():
            raise ValueError("FSim compiled ABI mismatch: " + macro)
    return env, observed


def diversity(rows, count, key):
    from prepare_vta_c3_resnet18_literature_candidates import normalized_coordinates
    if not rows:
        return []
    entities = [row["complete_config_entity"] for row in rows]
    coords = normalized_coordinates(entities)
    rank = lambda row: digest([SALT, key, row["complete_config_entity"]])
    remaining, selected = list(rows), []
    while remaining and len(selected) < count:
        if not selected:
            choice = min(remaining, key=rank)
        else:
            def score(row):
                distances = [sum(abs(a - b) for a, b in zip(coords[digest(row["complete_config_entity"])],
                                                             coords[digest(other["complete_config_entity"])])) for other in selected]
                return min(distances), sum(distances), rank(row)
            choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return selected


def prepare(output, start):
    from prepare_vta_c3_resnet18_literature_candidates import make_workload, create_task, candidate_row
    from c3_candidate_identity import normalize_config_entity
    env, fingerprint = hardware()
    bindings = source_bindings()
    version = digest(bindings)
    pools = {}
    for name, spec in GEOMETRIES.items():
        workload = make_workload(spec, env)
        task = create_task(workload, 0, env)
        all_rows = []
        for index in range(len(task.config_space)):
            entity = normalize_config_entity(task.config_space.get(index).to_json_dict())
            family = name + "D{:04d}".format(index)
            row = candidate_row(name, spec, workload, entity, index, family, "original", version, fingerprint, env)
            # Synthetic mechanism tests have no model/network identity.
            row["model_hash"] = digest({"kind": "synthetic_mechanism", "geometry": spec})
            row["identity"] = {"workload": workload, "complete_config_entity": entity}
            values = row["visible_features"]
            co_outer = (spec["co"] // env.BLOCK_OUT) // values["tile_co"]
            spatial = (row["applicability"]["output_height"] // values["tile_h"]) * (row["applicability"]["output_width"] // values["tile_w"])
            if values["oc_nthread"] == values["h_nthread"] == 1 and co_outer >= 2 and spatial >= 2:
                all_rows.append(row)
        order = diversity(all_rows, min(64, len(all_rows)), name)
        families = []
        for row in order:
            family_rows = []
            for mode in MODES:
                candidate = candidate_row(name, spec, workload, row["complete_config_entity"], row["debug"]["config_index"],
                                          row["family_id"], mode, version, fingerprint, env)
                candidate["model_hash"] = row["model_hash"]
                candidate["identity"] = row["identity"]
                family_rows.append(candidate)
            families.append(family_rows)
        pools[name] = {"geometry": spec, "workload": workload, "config_space_count": len(task.config_space),
                       "mechanism_eligible_family_count": len(all_rows), "families": families}
    contract = {"schema": SCHEMA, "phase": "prepare", "board_contacted": False,
                "performance_labels_collected": False, "source_bindings": bindings, "schedule_version": version,
                "hardware_fingerprint": fingerprint, "bitstream_required_sha256": BITSTREAM,
                "bitstream_observed": False, "fsim_backend": str(ROOT / "build/libvta_fsim.so"),
                "fsim_host_override": "llvm (same accelerator ABI and board-target lowered TIR)",
                "max_families_per_geometry": 64, "max_boundary_original_legal": 2,
                "selection_salt": SALT, "seeds": list(SEEDS), "pools": pools,
                "fsim_warmup": {"seed": 0, "calls_per_worker": 1, "output_checked": True,
                                "cost_included": True, "reason": "separate cold initialization from three-seed steady command signatures"},
                "family_fsim_fail_fast": "remaining modes not executed when one mode fails; all four lower attempts retained",
                "selection": "first two qualified effective-dual families by knob diversity; two other qualified diverse families",
                "stop": "exhaust frozen finite order, or four selected including at least two effective-dual families",
                "timing_rounds": 8, "buffer_layout": ["output", "data", "weight"],
                "boundaries": {"W0": "process start", "T0": None, "T1": "this local phase complete"}}
    reject_labels(contract)
    check_bindings(bindings)
    write(output / "contract.json", contract)
    write(output / "summary.json", {"status": "PREPARED_LABEL_BLIND", "board_contacted": False,
                                   "wall_seconds": time.monotonic() - start,
                                   "pools": {name: {"full": pool["config_space_count"], "eligible": pool["mechanism_eligible_family_count"],
                                                     "ordered": len(pool["families"])} for name, pool in pools.items()}})


def instantiate(candidate):
    import tvm
    import vta
    from tvm import autotvm
    from c3_candidate_identity import normalize_config_entity
    import c3_reuse_experimental_template  # noqa: F401
    import vta.top.vta_conv2d_residency  # noqa: F401
    template = "conv2d_packed_reuse_control.vta" if candidate.get("experimental_reload_control") else "conv2d_packed_residency.vta"
    env = vta.get_env()
    task = autotvm.task.create(template, args=tuple(candidate["identity"]["workload"][1:]) + (candidate["implementation_mode"],),
                               target=env.target, target_host=env.target_host)
    config = task.config_space.get(candidate["debug"]["config_index"])
    if normalize_config_entity(config.to_json_dict()) != normalize_config_entity(candidate["complete_config_entity"]):
        raise ValueError("complete ConfigEntity mismatch")
    with task.target:
        return task.instantiate(config)


def candidate_build_config(candidate, **kwargs):
    """Select an explicitly named experiment-only lowering pipeline."""
    import vta

    variant = candidate.get("build_config_variant", "stock")
    if variant == "stock":
        return vta.build_config(**kwargs)
    if variant == "staged_weight_residency":
        from vta.top.vta_staged_weight import staged_weight_build_config

        return staged_weight_build_config(**kwargs)
    raise ValueError("unknown build_config_variant: " + str(variant))


def lower_candidate(candidate, output):
    import tvm
    import vta
    from c3_reuse_semantics import capture_lower, extract_requests
    from run_vta_c3_resnet18_local_qualification import dependency_audit, sync_multiplicity
    started = time.monotonic()
    row = {"candidate_id": candidate["candidate_id"], "family_id": candidate["family_id"],
           "workload_id": candidate["workload_id"], "residence_mode": candidate["residence_mode"],
           "phase": "lower", "status": "failed", "compiler_attempts": 0, "performance_label": None}
    if candidate["residence_mode"] in {"weight_resident_barrier", "input_weight_resident_barrier"} and not candidate["applicability"]["full_weight_fits_sram"]:
        row.update(status="not_applicable", reason="current conservative full-layer-weight gate; W tile may still fit",
                   fallback_policy="original must independently qualify; never a measured optimization identity")
    else:
        try:
            row["compiler_attempts"] = 1
            schedule, tensors = instantiate(candidate)
            module, storage, arenas = capture_lower(
                schedule,
                tensors,
                build_config_factory=lambda **kwargs: candidate_build_config(
                    candidate, **kwargs
                ),
            )
            dependencies = dependency_audit(module)
            if not dependencies["balanced"] or dependencies["forbidden_direct_1_3_count"]:
                raise ValueError("dependency qualification failed")
            semantics = extract_requests(module, vta.get_env())
            sync = sync_multiplicity(module)
            directory = output / "tir"
            directory.mkdir(exist_ok=True)
            path = directory / (candidate["candidate_id"] + ".json")
            path.write_text(tvm.ir.save_json(module))
            storage_path = directory / (candidate["candidate_id"] + ".storage.json")
            storage_path.write_text(tvm.ir.save_json(storage))
            row.update(status="passed", tir_sha256=file_hash(path), storage_tir_sha256=file_hash(storage_path),
                       semantics=semantics, storage_arenas=arenas, dependency_audit=dependencies,
                       sync_multiplicity=sync, explicit_residency_drains=sum(sync[:-1]) if sync else 0)
        except Exception as error:
            row["failure"] = {"type": type(error).__name__, "message": str(error)[-5000:]}
    row["wall_seconds"] = time.monotonic() - started
    return row


def worker(request, response):
    import numpy as np
    import tvm
    import vta
    from c3_reuse_semantics import capture_lower
    from qualify_vta_residency_fsim import reference_data_from_workload, array_sha256
    candidate, static, backend = request["candidate"], request["static"], request["backend"]
    ctypes.CDLL(backend, mode=ctypes.RTLD_GLOBAL)
    if tvm.get_global_func("vta.simulator.profiler_clear", True) is None:
        raise ValueError("FSim backend did not register simulator API")
    started = time.monotonic()
    result = {"status": "failed", "phase": "fsim", "candidate_id": candidate["candidate_id"],
              "seeds": [], "performance_label": None, "board_contacted": False,
              "function_invocations_attempted": 0, "function_invocations_completed": 0,
              "warmup": {"status": "not_attempted", "complete_function_invocations": 0}}
    try:
        schedule, tensors = instantiate(candidate)
        lowered, _, _ = capture_lower(
            schedule,
            tensors,
            build_config_factory=lambda **kwargs: candidate_build_config(
                candidate, **kwargs
            ),
        )
        result["relowered_tir_sha256"] = hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()
        if result["relowered_tir_sha256"] != static["tir_sha256"]:
            raise ValueError("FSim lowered identity changed")
        build_start = time.monotonic()
        with candidate_build_config(
            candidate, disabled_pass={"tir.CommonSubexprElimTIR"}
        ):
            module = vta.build(schedule, tensors, target=tvm.target.Target(vta.get_env().target, host="llvm"), name="main")
        result["build_seconds"] = time.monotonic() - build_start
        function, device = module["main"], tvm.device("ext_dev", 0)
        warm_start = time.monotonic()
        data, weight, expected = reference_data_from_workload(candidate["identity"]["workload"], 0)
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        input_buffer, weight_buffer = tvm.nd.array(data, device), tvm.nd.array(weight, device)
        result["warmup"]["status"] = "attempted"
        result["function_invocations_attempted"] += 1
        function(input_buffer, weight_buffer, output)
        result["function_invocations_completed"] += 1
        actual = output.numpy()
        result["warmup"].update(status="passed" if (actual == expected).all() else "wrong_answer",
                                complete_function_invocations=1, actual_sha256=array_sha256(actual),
                                expected_sha256=array_sha256(expected), wall_seconds=time.monotonic() - warm_start)
        if result["warmup"]["status"] != "passed":
            raise ValueError("warmup output wrong; candidate remains invalid")
        signature_profiles = []
        for seed in SEEDS:
            data, weight, expected = reference_data_from_workload(candidate["identity"]["workload"], seed)
            # The allocation order is part of the run protocol even in FSim.
            output = tvm.nd.empty(expected.shape, "int8", device=device)
            input_buffer, weight_buffer = tvm.nd.array(data, device), tvm.nd.array(weight, device)
            tvm.get_global_func("vta.runtime.profiler_clear")()
            tvm.get_global_func("vta.simulator.profiler_clear")()
            print("[C3_R18_SEED] " + str(seed), file=sys.stderr, flush=True)
            result["function_invocations_attempted"] += 1
            function(input_buffer, weight_buffer, output)
            result["function_invocations_completed"] += 1
            actual = output.numpy()
            profile = json.loads(tvm.get_global_func("vta.runtime.profiler_status")())
            sim = json.loads(tvm.get_global_func("vta.simulator.profiler_status")())
            logical = {key: value for key, value in profile.items()
                       if (key.startswith(("load_buffer_2d_", "store_buffer_2d_")) and key.endswith(("_bytes", "_calls")))
                       or key in {"synchronize_calls", "driver_run_calls"}}
            mismatches = int(np.count_nonzero(actual != expected))
            result["seeds"].append({"seed": seed, "correct": mismatches == 0, "mismatch_count": mismatches,
                                    "actual_sha256": array_sha256(actual), "expected_sha256": array_sha256(expected),
                                    "runtime_logical_dma": logical,
                                    "fsim_instruction_bytes": {key: value for key, value in sim.items() if key.endswith("_nbytes")},
                                    "complete_function_invocations": 1})
            for key, expected_value in static["semantics"]["totals"].items():
                if logical.get(key) != expected_value:
                    raise ValueError("static/runtime DMA mismatch: {} {} != {}".format(key, logical.get(key), expected_value))
            for key, runtime_key in {"inp_load_nbytes": "load_buffer_2d_inp_bytes", "wgt_load_nbytes": "load_buffer_2d_wgt_bytes",
                                     "out_store_nbytes": "store_buffer_2d_out_bytes"}.items():
                if sim[key] != logical[runtime_key]:
                    raise ValueError("instruction/runtime data bytes mismatch: " + key)
            signature_profiles.append(logical)
            if mismatches:
                raise ValueError("wrong FSim output; fail fast")
        if any(profile != signature_profiles[0] for profile in signature_profiles):
            raise ValueError("logical DMA profile depends on seed")
        result["status"] = "passed"
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)[-5000:]}
    result["wall_seconds"] = time.monotonic() - started
    write(response, result)


def fsim(candidate, static, output, backend, timeout):
    from run_vta_c3_resnet18_local_qualification import attach_command_features
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="c3_reuse_fsim_") as directory:
        request, response = Path(directory) / "request.json", Path(directory) / "response.json"
        write(request, {"candidate": candidate, "static": static, "backend": backend})
        environment = os.environ.copy()
        environment["VTA_QUEUE_DIAGNOSTICS"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request), "--worker-response", str(response)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment, check=False)
            stderr = completed.stderr
            result = read(response) if completed.returncode == 0 and response.is_file() else {
                "status": "failed", "phase": "fsim", "candidate_id": candidate["candidate_id"],
                "seeds": [], "failure": {"message": "worker exit " + str(completed.returncode)}}
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            result = {"status": "failed", "phase": "fsim", "candidate_id": candidate["candidate_id"],
                      "seeds": [], "failure": {"message": "worker timeout"}}
        logs = output / "logs"
        logs.mkdir(exist_ok=True)
        (logs / (candidate["candidate_id"] + ".stderr.log")).write_text(stderr)
        if result["status"] == "passed":
            result["implementation_mode"] = candidate["implementation_mode"]
            result = attach_command_features(result, stderr, {"sync": {"explicit_residency_drains": static["explicit_residency_drains"]}})
            if result["status"] == "passed" and not result["command_features"]["values"]["drain_check"]["matches_p3e"]:
                result.update(status="failed", failure={"message": "observed residency drains mismatch"})
        result["host_wall_seconds"] = time.monotonic() - started
        return result


def dual_effect(static):
    if set(static) != set(MODES) or any(row["status"] != "passed" for row in static.values()):
        return None
    original = static["original"]["semantics"]
    combined = static["input_weight_resident_barrier"]["semantics"]
    return all(original["reload_ratios"].get("load_" + operand, 0) > 1 and
               combined["totals"]["load_buffer_2d_" + operand + "_bytes"] < original["totals"]["load_buffer_2d_" + operand + "_bytes"]
               for operand in ("inp", "wgt"))


def matched_reload_control(resident, control):
    if resident["status"] != "passed" or control["status"] != "passed":
        return False
    if resident["sync_multiplicity"] != control["sync_multiplicity"]:
        return False
    left, right = resident["semantics"], control["semantics"]
    if not left["compute_sites"] or left["compute_sites"] != right["compute_sites"]:
        return False
    for key in ("load_buffer_2d_inp_bytes", "load_buffer_2d_inp_calls",
                "store_buffer_2d_out_bytes", "store_buffer_2d_out_calls"):
        if left["totals"].get(key) != right["totals"].get(key):
            return False
    return right["totals"]["load_buffer_2d_wgt_bytes"] > left["totals"]["load_buffer_2d_wgt_bytes"]


def cross_worker(request, response):
    import tvm
    import vta
    from tvm.contrib import cc
    from c3_reuse_semantics import capture_lower
    from cross_compile_vta_c3_resnet18_board_pool import options
    candidate = request["candidate"]
    started = time.monotonic()
    result = {"phase": "cross", "status": "failed", "candidate_id": candidate["candidate_id"],
              "board_contacted": False, "performance_label": None}
    try:
        schedule, tensors = instantiate(candidate)
        module, _, _ = capture_lower(
            schedule,
            tensors,
            build_config_factory=lambda **kwargs: candidate_build_config(
                candidate, **kwargs
            ),
        )
        sha = hashlib.sha256(tvm.ir.save_json(module).encode()).hexdigest()
        if sha != candidate["local_qualification"]["tir_sha256"]:
            raise ValueError("ARM re-lowering differs from qualified TIR")
        with candidate_build_config(
            candidate, disabled_pass={"tir.CommonSubexprElimTIR"}
        ):
            compiled = vta.build(schedule, tensors, target=tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host), name="main")
        result["build_seconds"] = time.monotonic() - started
        export_start = time.monotonic()
        compile_options = options()
        compiled.export_library(request["binary"], fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"), options=compile_options)
        compiler = subprocess.run(["aarch64-xilinx-linux-g++", "--version"], capture_output=True, text=True, check=True)
        result.update(status="passed", export_seconds=time.monotonic() - export_start,
                      binary_sha256=file_hash(request["binary"]), binary_bytes=Path(request["binary"]).stat().st_size,
                      compiler_version=compiler.stdout.splitlines()[0], compiler_path=__import__("shutil").which("aarch64-xilinx-linux-g++"),
                      compiler_sha256=file_hash(__import__("shutil").which("aarch64-xilinx-linux-g++")),
                      sysroot=os.environ.get("SDKTARGETSYSROOT"), compile_options=compile_options,
                      relowered_tir_sha256=sha)
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)[-5000:]}
    result["host_wall_seconds"] = time.monotonic() - started
    write(response, result)


def cross(candidate, output, args):
    started = time.monotonic()
    binaries = output / "binaries"
    binaries.mkdir(exist_ok=True)
    binary = binaries / (candidate["candidate_id"] + ".so")
    with tempfile.TemporaryDirectory(prefix="c3_reuse_cross_") as directory:
        request, response = Path(directory) / "request.json", Path(directory) / "response.json"
        write(request, {"candidate": candidate, "binary": str(binary)})
        command = ["bash", "-c", 'source "$1" >/dev/null && exec "$2" "$3" --cross-request "$4" --worker-response "$5"',
                   "c3-reuse-cross", args.sdk_environment, sys.executable, str(Path(__file__).resolve()), str(request), str(response)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
            stderr = completed.stderr
            result = read(response) if completed.returncode == 0 and response.is_file() else {
                "phase": "cross", "status": "failed", "candidate_id": candidate["candidate_id"],
                "failure": {"message": "cross worker exit " + str(completed.returncode)}}
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            result = {"phase": "cross", "status": "failed", "candidate_id": candidate["candidate_id"],
                      "failure": {"message": "cross worker timeout"}}
        (output / "logs").mkdir(exist_ok=True)
        (output / "logs" / (candidate["candidate_id"] + ".cross.stderr.log")).write_text(stderr)
        result["host_wall_seconds"] = time.monotonic() - started
        return result


def local(args, output, start):
    parent = Path(args.contract).resolve().parent
    contract = verify(parent)
    if contract["phase"] != "prepare":
        raise ValueError("local requires a prepared contract")
    reject_labels(contract)
    check_bindings(contract["source_bindings"])
    _, observed = hardware()
    if observed != contract["hardware_fingerprint"]:
        raise ValueError("hardware changed")
    write(output / "contract.json", {**contract, "phase": "local", "prepared_directory": str(parent),
                                    "prepared_manifest_sha256": file_hash(parent / "artifact_hashes.json")})
    families, selected, candidates, attempts = [], {}, [], []
    for name, pool in contract["pools"].items():
        good = []
        boundary_legal = 0
        for ordinal, family in enumerate(pool["families"]):
            static = {}
            for candidate in family:
                result = lower_candidate(candidate, output)
                static[candidate["residence_mode"]] = result
                append(output / "results.jsonl", result)
                attempts.append(result)
            passed = all(row["status"] == "passed" for row in static.values())
            fsim_rows = []
            if name != "BND" and passed:
                for candidate in family:
                    result = fsim(candidate, static[candidate["residence_mode"]], output,
                                  contract["fsim_backend"], args.timeout)
                    append(output / "results.jsonl", result)
                    attempts.append(result)
                    fsim_rows.append(result)
                    if result["status"] != "passed":
                        break
            record = {"workload_id": name, "family_id": family[0]["family_id"], "order": ordinal,
                      "static": static, "fsim": fsim_rows, "effective_dual_reuse": dual_effect(static),
                      "four_local_correct": len(fsim_rows) == 4 and all(row["status"] == "passed" for row in fsim_rows),
                      "original_lowered_legal": static["original"]["status"] == "passed"}
            families.append(record)
            append(output / "families.jsonl", record)
            append(output / "timeline.jsonl", {"event": "family_complete", "family_id": record["family_id"],
                                                "four_local_correct": record["four_local_correct"],
                                                "elapsed_seconds": time.monotonic() - start})
            print("{} {}/{} static4={} fsim4={} dual={}".format(name, ordinal + 1, len(pool["families"]),
                  passed, record["four_local_correct"], record["effective_dual_reuse"]), flush=True)
            if name == "BND":
                boundary_legal += record["original_lowered_legal"]
                if boundary_legal == 2:
                    break
            elif record["four_local_correct"]:
                good.append(family[0])
                effects = {row["family_id"]: row["effective_dual_reuse"] for row in families if row["workload_id"] == name}
                positive = diversity([row for row in good if effects[row["family_id"]]], 2, name + ":dual")
                others = diversity([row for row in good if row not in positive], 2, name + ":others")
                if len(positive) == len(others) == 2:
                    break
        if name == "BND":
            continue
        effects = {row["family_id"]: row["effective_dual_reuse"] for row in families if row["workload_id"] == name}
        positive = diversity([row for row in good if effects[row["family_id"]]], 2, name + ":dual")
        others = diversity([row for row in good if row not in positive], 4 - len(positive), name + ":others")
        chosen = positive + others
        selected[name] = [row["family_id"] for row in chosen]
        for family in pool["families"]:
            if family[0]["family_id"] in selected[name]:
                for candidate in family:
                    row = next(row for row in families if row["family_id"] == candidate["family_id"])
                    candidates.append({**candidate, "local_qualification": row["static"][candidate["residence_mode"]],
                                       "fsim_qualification": next(item for item in row["fsim"] if item["candidate_id"] == candidate["candidate_id"])})
    geometry_counts = {name: {"attempted_families": sum(row["workload_id"] == name for row in families),
                               "four_local_correct": sum(row["workload_id"] == name and row["four_local_correct"] for row in families),
                               "effective_dual_local_correct": sum(row["workload_id"] == name and row["four_local_correct"] and row["effective_dual_reuse"] is True for row in families),
                               "selected": selected.get(name, [])} for name in contract["pools"]}
    ready = sum(values["effective_dual_local_correct"] >= 2 for name, values in geometry_counts.items() if name != "BND") >= 2
    controls = []
    for name in ("M1", "M2"):
        if not selected.get(name):
            continue
        family_id = selected[name][0]
        for mode in ("weight_resident_barrier", "input_weight_resident_barrier"):
            resident = next(row for row in candidates if row["family_id"] == family_id and row["residence_mode"] == mode)
            control = {**resident, "experimental_reload_control": True,
                       "candidate_id": digest({"resident_candidate_id": resident["candidate_id"], "control": "reduction_scope_reload", "schedule_version": contract["schedule_version"]}),
                       "control_parent_id": resident["candidate_id"]}
            result = lower_candidate(control, output)
            result["matched_loop_sync_reload_control"] = matched_reload_control(resident["local_qualification"], result)
            append(output / "results.jsonl", result)
            attempts.append(result)
            if result["matched_loop_sync_reload_control"]:
                qualification = fsim(control, result, output, contract["fsim_backend"], args.timeout)
                append(output / "results.jsonl", qualification)
                attempts.append(qualification)
                if qualification["status"] == "passed":
                    control.update(local_qualification=result, fsim_qualification=qualification)
                    controls.append(control)
            print("control {} {} matched={}".format(name, mode, result["matched_loop_sync_reload_control"]), flush=True)
    cross_rows = []
    for candidate in candidates + controls:
        result = cross(candidate, output, args)
        append(output / "results.jsonl", result)
        attempts.append(result)
        cross_rows.append(result)
        candidate["cross_qualification"] = result
        print("cross {} {} {}".format(candidate["family_id"], candidate["residence_mode"], result["status"]), flush=True)
    write(output / "controls.json", controls)
    write(output / "candidates.json", candidates)
    check_bindings(contract["source_bindings"])
    verify(parent)
    write(output / "summary.json", {"status": "LOCAL_GATE_PASS_READY_TO_FREEZE" if ready and all(row["status"] == "passed" for row in cross_rows) else "LOCAL_GATE_INSUFFICIENT_PAUSED",
                                   "positive_combination_gate": ready, "geometry_counts": geometry_counts,
                                   "selected_identities": len(candidates), "board_contacted": False,
                                   "accepted_control_identities": len(controls), "cross_attempts": len(cross_rows),
                                   "cross_passed": sum(row["status"] == "passed" for row in cross_rows),
                                   "performance_labels_collected": False, "wall_seconds": time.monotonic() - start,
                                   "compiler_attempts": sum(row.get("compiler_attempts", 0) for row in attempts),
                                   "fsim_worker_attempts": sum(row["phase"] == "fsim" for row in attempts),
                                   "completed_fsim_invocations": sum(row.get("function_invocations_completed", 0) for row in attempts),
                                   "attempted_fsim_invocations": sum(row.get("function_invocations_attempted", 0) for row in attempts),
                                   "warmup_invocations": sum(row.get("warmup", {}).get("complete_function_invocations", 0) for row in attempts),
                                   "stage_host_wall_seconds": {phase: sum(row.get("host_wall_seconds", row.get("wall_seconds", 0)) for row in attempts if row["phase"] == phase)
                                                               for phase in ("lower", "fsim", "cross")},
                                   "cost_scope": "all local attempts including failures; not tuning time-to-quality"})


def freeze(args, output, start):
    parent = Path(args.contract).resolve().parent
    contract = verify(parent)
    if contract["phase"] != "local":
        raise ValueError("freeze requires completed local contract")
    summary = read(parent / "summary.json")
    if summary["status"] != "LOCAL_GATE_PASS_READY_TO_FREEZE":
        raise ValueError("local positive-combination/cross gate not passed; no board freeze")
    check_bindings(contract["source_bindings"])
    candidates, controls = read(parent / "candidates.json"), read(parent / "controls.json")
    if len(candidates) > 48 or len(controls) > 4:
        raise ValueError("identity cap violated")
    if any(row["cross_qualification"]["status"] != "passed" for row in candidates + controls):
        raise ValueError("unqualified binary")
    qualified_geometries = 0
    for name in ("M1", "M2", "M3"):
        families = {row["family_id"] for row in candidates if row["workload_id"] == name}
        positive = 0
        for family_id in families:
            rows = [row for row in candidates if row["family_id"] == family_id]
            if len(rows) != 4 or {row["residence_mode"] for row in rows} != set(MODES):
                raise ValueError("incomplete four-mode frozen family")
            if len({digest(row["complete_config_entity"]) for row in rows}) != 1:
                raise ValueError("same-tile freeze violated")
            positive += dual_effect({row["residence_mode"]: row["local_qualification"] for row in rows}) is True
            for row in rows:
                seeds = row["fsim_qualification"]["seeds"]
                if [seed["seed"] for seed in seeds] != list(SEEDS) or not all(seed["correct"] and seed["mismatch_count"] == 0 and seed["actual_sha256"] == seed["expected_sha256"] for seed in seeds):
                    raise ValueError("missing exact three-seed FSim qualification")
        qualified_geometries += positive >= 2
    if qualified_geometries < 2:
        raise ValueError("positive-combination gate cannot be proven")
    import shutil
    for row in candidates + controls:
        source = parent / "binaries" / (row["candidate_id"] + ".so")
        if file_hash(source) != row["cross_qualification"]["binary_sha256"]:
            raise ValueError("retained ARM binary changed")
        destination = output / "binaries" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    write(output / "candidates.json", candidates)
    write(output / "controls.json", controls)
    write(output / "contract.json", {**contract, "phase": "freeze", "local_directory": str(parent),
                                    "local_manifest_sha256": file_hash(parent / "artifact_hashes.json"),
                                    "candidate_order": [row["candidate_id"] for row in candidates],
                                    "control_order": [row["candidate_id"] for row in controls],
                                    "latin_mode_orders": [[(offset + position) % 4 for position in range(4)] for offset in range(4)] * 2,
                                    "board_protocol_implemented": False})
    check_bindings(contract["source_bindings"])
    verify(parent)
    write(output / "summary.json", {"status": "NODE_B_COMPLETE_PAUSED", "board_contacted": False,
                                   "selected_identities": len(candidates), "controls": len(controls),
                                   "wall_seconds": time.monotonic() - start})


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "prepare", "local", "freeze", "board", "analyze"))
    parser.add_argument("--contract")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--sdk-environment", default="/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")
    parser.add_argument("--cross-request")
    parser.add_argument("--worker-request")
    parser.add_argument("--worker-response")
    args = parser.parse_args()
    if args.cross_request:
        if not args.worker_response:
            parser.error("cross worker response required")
        cross_worker(read(args.cross_request), args.worker_response)
        return
    if args.worker_request:
        if not args.worker_response:
            parser.error("worker response required")
        worker(read(args.worker_request), args.worker_response)
        return
    if not args.phase or not args.output_dir:
        parser.error("phase and output-dir required")
    if args.phase not in {"prepare", "local", "freeze"}:
        parser.error("phase not implemented yet; no silent board execution or unsupported claim")
    if args.phase in {"local", "freeze"} and not args.contract:
        parser.error("local/freeze require contract")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time(), "elapsed_seconds": 0,
                                        "T0": None, "reason": "no board preparation in local phase"})
    try:
        if args.phase == "prepare":
            prepare(output, start)
        elif args.phase == "local":
            local(args, output, start)
        else:
            freeze(args, output, start)
    except BaseException as error:
        write(output / "invalid_session.json", {"phase": args.phase, "type": type(error).__name__,
                                               "message": str(error)[-5000:], "wall_seconds": time.monotonic() - start,
                                               "board_contacted": False})
        seal(output)
        raise
    append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - start})
    seal(output)
    print(json.dumps(read(output / "summary.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
