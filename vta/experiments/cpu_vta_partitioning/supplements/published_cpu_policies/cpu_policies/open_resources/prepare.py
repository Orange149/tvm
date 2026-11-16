"""Freeze a relocatable nine-context bundle using existing verified binaries.

Everything here is local.  Historical evidence is copied with its original
paths and hashes; importing it never changes the original boot or task ID.
"""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import ROOT, DEFAULT_PARENT, DEFAULT_SOURCE_RUN, identity, load_legacy, protocol


def _api(bundle=None):
    load_legacy(bundle)
    from cpu_vta_partitioning import common
    return common


def _copy(source, target, expected=None):
    api = _api()
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError("missing or linked frozen source: " + str(source))
    if expected is not None and api.sha256(source) != expected:
        raise ValueError("frozen source hash differs: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_provenance(source, target):
    """Copy only the complete calibration/qualification receipt closure."""
    api = _api()
    source, target = Path(source).resolve(), Path(target)
    costs = api.read(source / "costs.json")
    records = costs.get("costs", {})
    if not records:
        raise ValueError("source component costs are missing")
    files = {}
    for name in ("costs.json", "state.json", "protocol.json"):
        path = api.relative_file(source, name)
        files[name] = {"sha256": api.sha256(path), "bytes": path.stat().st_size}
    for item in records.values():
        for role in ("measurement", "qualification"):
            entry = item.get(role)
            if not entry:
                raise ValueError("source component lacks " + role)
            ref = entry["summary"].get("_receipt", {})
            receipt_path = api.relative_file(source, ref["path"])
            if api.sha256(receipt_path) != ref["sha256"]:
                raise ValueError("source receipt hash differs")
            receipt = api.read(receipt_path)
            if receipt["task"] != entry["document"] or receipt["boot_id"] != entry["boot_id"]:
                raise ValueError("source receipt belongs to another task or boot")
            api.verify_inventory(source, receipt["files"])
            name = str(receipt_path.relative_to(source))
            files[name] = {"sha256": ref["sha256"], "bytes": receipt_path.stat().st_size}
            for name, value in receipt["files"].items():
                if name in files and files[name] != value:
                    raise ValueError("source evidence inventory collision")
                files[name] = value
    for name, value in sorted(files.items()):
        _copy(api.relative_file(source, name), target / name, value["sha256"])
    api.verify_inventory(target, files)
    return {"files": files, "component_count": len(records),
            "source_costs_sha256": api.sha256(source / "costs.json"),
            "source_boot_ids": sorted({r["measurement"]["boot_id"] for r in records.values()}),
            "copied_original_task_and_boot_id": True}


def _full_domain_verification(path=None):
    """A synthetic full-domain proof is required before declaring readiness."""
    api = _api()
    path = Path(path or ROOT / "artifacts/open_resources_ns_full_domain_check/verification.json")
    if not path.is_file():
        raise ValueError("full-domain synthetic DP/oracle verification is not complete: " + str(path))
    value = api.read(path)
    passed = value.get("passed", value.get("verification", {}).get("passed", False))
    if (passed is not True or value.get("synthetic_only") is not True
            or value.get("counts", {}).get("enhanced") != 966240306):
        raise ValueError("full-domain synthetic verification failed or has incomplete coverage")
    if value.get("counts", {}).get("contexts") != 9:
        raise ValueError("synthetic verification lacks all nine contexts")
    implementation = value.get("implementation", {})
    for name in ("search.py", "oracle.cc", "picojson.h"):
        current = ROOT / "cpu_policies/open_resources" / name
        if not current.is_file() or implementation.get(name) != api.sha256(current):
            raise ValueError("full-domain verification implementation changed: " + name)
    return value


def validate_bundle(bundle, *, replay_imports=False):
    bundle = load_legacy(bundle)
    api = _api(bundle)
    from cpu_vta_partitioning.prepare.payload import execution_files
    manifest = api.read(bundle / "manifest.json")
    if api.digest({k: v for k, v in manifest.items() if k != "bundle_id"})[:24] != manifest["bundle_id"]:
        raise ValueError("open-resource manifest identity differs")
    if manifest.get("experiment") != protocol()["experiment"] or manifest.get("controller_identity") != identity():
        raise ValueError("open-resource controller changed; use its frozen snapshot")
    api.verify_inventory(bundle, manifest["files"])
    if api.read(bundle / "plans/open_resources_protocol.json") != protocol():
        raise ValueError("open-resource protocol differs")
    programs = api.read(bundle / "plans/programs.json")
    if len(programs) != 172 or sum(p["device"] == "cpu" for p in programs.values()) != 85:
        raise ValueError("qualified stage module coverage differs")
    executor = api.read(bundle / "plans/executors.json")["policy"]
    if api.sha256(bundle / executor["path"]) != executor["sha256"]:
        raise ValueError("frozen policy executor changed")
    inputs = api.read(bundle / "plans/inputs.json")
    if inputs["available_indices"] != [0] or len(inputs["records"]) != 1 or inputs["records"][0]["index"] != 0:
        raise ValueError("single fixed input zero required")
    api.verify_inventory(bundle, inputs["files"])
    if api.read(bundle / "plans/board_payload.json")["files"] != execution_files(bundle):
        raise ValueError("execution payload dependency closure differs")
    from .domain import contexts, requirements
    if api.read(bundle / "plans/resource_contexts.json") != list(contexts()):
        raise ValueError("nine resource contexts differ")
    wanted = requirements(programs)
    if len(wanted) != 775 or api.read(bundle / "plans/component_requirements.json") != wanted:
        raise ValueError("full representative component coverage differs")
    imported = api.read(bundle / "plans/imported_components.json")
    if len(imported.get("costs", {})) != 514 or not set(imported["costs"]) <= set(wanted):
        raise ValueError("verified historical component imports differ")
    if any(v["configuration"] != wanted[k] for k, v in imported["costs"].items()):
        raise ValueError("imported effective component configuration differs")
    if replay_imports:
        from .costs import import_verified_costs
        replayed = import_verified_costs(bundle / "provenance/source_run", target_bundle=bundle)
        if replayed != imported:
            raise ValueError("imported raw evidence replay differs")
    reused_path = bundle / "plans/reused_calibration.json"
    if reused_path.is_file():
        reused = api.read(reused_path)
        additional = reused.get("costs", {})
        if (len(additional) != 261 or set(additional).intersection(imported["costs"])
                or set(additional) | set(imported["costs"]) != set(wanted)
                or not set(reused.get("traffic", {})).issubset(
                    p for p, v in programs.items() if v["device"] == "vta")):
            raise ValueError("reused calibration coverage or duplicate component differs")
        if any(v["configuration"] != wanted[k] for k, v in additional.items()):
            raise ValueError("reused component effective configuration differs")
        if replay_imports:
            from .reuse import validate_reused_calibration
            if validate_reused_calibration(bundle) != reused:
                raise ValueError("reused calibration raw evidence replay differs")
    ready = api.read(bundle / "OFFLINE_READINESS.json")
    if ready["status"] != "PREPARED_OFFLINE" or not ready["tests"]["passed"]:
        raise ValueError("offline readiness incomplete")
    full = api.read(bundle / "plans/full_domain_verification.json")
    if (full.get("passed", full.get("verification", {}).get("passed", False)) is not True
            or full.get("synthetic_only") is not True
            or full.get("counts", {}).get("enhanced") != 966240306):
        raise ValueError("bundle lacks full-domain DP/oracle acceptance")
    for name in ("search.py", "oracle.cc", "picojson.h"):
        if full.get("implementation", {}).get(name) != api.sha256(
                bundle / "sources/open_resources_controller/cpu_policies/open_resources" / name):
            raise ValueError("bundle full-domain implementation differs: " + name)
    return manifest


def prepare(parent=None, source_run=None, *, calibration_source=None, run_tests=True):
    parent = load_legacy(parent or DEFAULT_PARENT)
    source_run = Path(source_run or DEFAULT_SOURCE_RUN).resolve()
    api = _api(parent)
    from cpu_vta_partitioning.prepare.payload import execution_files
    from cpu_vta_partitioning.prepare.package import elf
    from .domain import contexts, requirements
    from .costs import import_verified_costs
    from .history import archive_files, historical_references, summarize_history
    original = api.read(parent / "manifest.json")
    if api.digest({k: v for k, v in original.items() if k != "bundle_id"})[:24] != original["bundle_id"]:
        raise ValueError("parent manifest identity differs")
    api.verify_inventory(parent, original["files"])
    code = identity()
    full_domain = _full_domain_verification()
    calibration_source = Path(calibration_source).resolve() if calibration_source is not None else None
    calibration_fingerprint = None
    if calibration_source is not None:
        calibration_fingerprint = {
            "costs_sha256": api.sha256(calibration_source / "costs.json"),
            "traffic_files": api.inventory(calibration_source / "traffic")}
    key = api.digest({"code": code, "parent": original["bundle_id"],
                      "source_costs_sha256": api.sha256(source_run / "costs.json"),
                      "calibration_source": calibration_fingerprint})[:24]
    work = ROOT / "artifacts/open_resource_builds" / key
    with api.locked(work):
        # A failed local build is reconstructed; no files from old controllers
        # are silently retained in the new source snapshot or provenance.
        for folder in ("sources/open_resources_controller", "provenance/source_run", "provenance/calibration_source"):
            path = work / folder
            if path.exists():
                shutil.rmtree(path)
        for folder in ("programs", "runtime", "firmware", "runner"):
            for name, expected in api.inventory(parent / folder).items():
                _copy(parent / folder / name, work / folder / name, expected["sha256"])
        programs = api.read(parent / "plans/programs.json")
        inputs = deepcopy(api.read(parent / "plans/inputs.json"))
        record = next(r for r in inputs["records"] if r["index"] == 0)
        wanted_inputs = {record["input"]}
        for frontier in record["frontiers"].values():
            wanted_inputs.update(frontier[k] for k in ("raw", "npy") if k in frontier)
        inputs["records"], inputs["available_indices"] = [record], [0]
        inputs["files"] = {n: original["files"][n] for n in sorted(wanted_inputs)}
        for name in sorted(wanted_inputs):
            _copy(parent / name, work / name, original["files"][name]["sha256"])
        for filename in ("executors.json", "compiler_identity.json", "runtime_build.json",
                         "sdk_runtime.json", "policy_native_acceptance.json", "policy_runner_patch.json"):
            _copy(parent / "plans" / filename, work / "plans" / filename)
        shutil.copytree(parent / "sources", work / "sources", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        snapshot = work / "sources/open_resources_controller"
        for filename in ("run_open_resources.py", "run.py", "pyproject.toml", "README.md"):
            _copy(ROOT / filename, snapshot / filename)
        for folder in ("cpu_policies", "configs", "tests", "examples", "docs", "third_party"):
            shutil.copytree(ROOT / folder, snapshot / folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "local.json"))
        api.write(work / "plans/programs.json", programs, replace=True)
        api.write(work / "plans/inputs.json", inputs, replace=True)
        api.write(work / "plans/source_parent_manifest.json", original, replace=True)
        api.write(work / "plans/open_resources_protocol.json", protocol(), replace=True)
        # Frozen transport preflight reads this generic name; it is not a
        # controller identity or an invitation to run the previous pilot.
        api.write(work / "plans/pilot_protocol.json", protocol(), replace=True)
        api.write(work / "plans/resource_contexts.json", list(contexts()), replace=True)
        api.write(work / "plans/full_domain_verification.json", full_domain, replace=True)
        needed = requirements(programs)
        if len(needed) != 775:
            raise ValueError("unexpected representative component count")
        api.write(work / "plans/component_requirements.json", needed, replace=True)
        provenance = _copy_provenance(source_run, work / "provenance/source_run")
        imported = import_verified_costs(work / "provenance/source_run", target_bundle=work)
        if len(imported["costs"]) != 514 or not set(imported["costs"]) <= set(needed):
            raise ValueError("verified source does not provide exactly 514 matching components")
        api.write(work / "plans/imported_components.json", imported, replace=True)
        api.write(work / "plans/imported_evidence_inventory.json", provenance, replace=True)
        reused = None
        if calibration_source is not None:
            from .reuse import freeze_calibration_source
            reused = freeze_calibration_source(calibration_source, work)
            additional = reused.get("costs", {})
            if (len(additional) != 261 or set(additional).intersection(imported["costs"])
                    or set(additional) | set(imported["costs"]) != set(needed)
                    or not set(reused.get("traffic", {})).issubset(
                        p for p, v in programs.items() if v["device"] == "vta")):
                raise ValueError("calibration source must supply exactly the remaining 261 costs and valid optional traffic records")
            api.write(work / "plans/reused_calibration.json", reused, replace=True)
        api.write(work / "plans/historical_references.json", historical_references(programs), replace=True)
        api.write(work / "plans/historical_rules.json", summarize_history(imported["costs"], programs), replace=True)
        for name, path in archive_files().items():
            _copy(path, work / "provenance/history" / name)
        from .ddr import freeze_hardware_parameters
        freeze_hardware_parameters(work)
        api.write(work / "plans/board_execution.json", {
            "revision": "open-cpu-resources-1", "udmabuf_bytes": 201326592,
            "storage_margin_bytes": 256 * 1024**2, "boot_policy": protocol()["boot_policy"]}, replace=True)
        payload = execution_files(work)
        api.write(work / "plans/board_payload.json", {"revision": 1, "files": payload,
                  "unpacked_bytes": sum(v["bytes"] for v in payload.values()),
                  "transfer": "gzip_tar_stream_no_remote_archive"}, replace=True)
        closure = {}
        for path in [*(work / "runner").iterdir(), *(work / "runtime").iterdir(),
                     *(work / "programs" / name / "module.so" for name in programs)]:
            if path.is_file():
                info = elf(path)
                if any(not (work / "runtime" / name).is_file() for name in info["needed"]):
                    raise ValueError("unresolved packaged shared library")
                closure[str(path.relative_to(work))] = info
        api.write(work / "plans/elf_dependencies.json", closure, replace=True)
        emulator = shutil.which("qemu-aarch64-static")
        if not emulator:
            raise RuntimeError("QEMU unavailable for packaged loader check")
        checked = subprocess.run([emulator, str(work / "runtime/ld-linux-aarch64.so.1"),
            "--library-path", str(work / "runtime"), str(work / "runner/vta_stage_pipeline_runner_pinned"), "--help"],
            capture_output=True, text=True, timeout=45)
        help_text = checked.stdout + checked.stderr
        if (checked.returncode or "--cpu-worker-pinning" not in help_text
                or "--vta-runtime-profile-dir" not in help_text):
            raise RuntimeError("packaged loader or policy parameter check failed")
        if not run_tests:
            raise ValueError("a deployable bundle requires passing local tests")
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        tests = subprocess.run([sys.executable, "-m", "pytest", "-c", "pyproject.toml", "-q"],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        api.write(work / "plans/offline_tests.json", {"passed": tests.returncode == 0,
                  "stdout": tests.stdout, "stderr": tests.stderr}, replace=True)
        if tests.returncode:
            raise RuntimeError("open-resource tests failed:\n" + tests.stdout + tests.stderr)
        if identity() != code:
            raise RuntimeError("controller changed during preparation; prepare again after edits finish")
        api.write(work / "OFFLINE_READINESS.json", {
            "status": "PREPARED_OFFLINE", "controller_identity": code, "tests": {"passed": True},
            "programs": {"cpu": 85, "vta": 87}, "contexts": 9,
            "component_count": 775, "imported_components": 775 if reused else 514,
            "source_pilot_components": 514, "reused_calibration_components": 261 if reused else 0,
            "new_components": 0 if reused else 261,
            "traffic_tasks": 87, "imported_traffic": len(reused.get("traffic", {})) if reused else 0,
            "new_traffic": 87 - (len(reused.get("traffic", {})) if reused else 0),
            "maximum_pipeline_configurations": 43,
            "full_domain_verification": {"passed": True, "unique_enhanced_configurations": 966240306,
                                         "scope": "synthetic_costs_local_algorithm_acceptance_not_board_performance"},
            "new_model_compilations": 0, "new_executor_compilations": 0,
            "source_boot_ids": sorted(set(provenance["source_boot_ids"]) |
                                      set(reused.get("source_boot_ids", []) if reused else [])),
            "imported_measurements_relabelled_as_new": False,
            "board_validation_pending": ["actual role/worker affinity", "FPGA numerics",
                                         "full-domain rankings", "frozen pipeline pool"] +
                                        ([] if reused else ["missing components"]) +
                                        ([] if reused and len(reused.get("traffic", {})) == 87
                                         else ["VTA traffic hooks/counters"]),
            "assumptions": ["homogeneous core permutation", "uniform per-worker compute demand",
                            "control work not separately modeled"],
            "input_indices": [0], "formal_independent_confirmations": 0}, replace=True)
        files = api.inventory(work, exclude=("manifest.json", ".lock"))
        files = {n: v for n, v in files.items() if "__pycache__/" not in n and ".pytest_cache/" not in n}
        manifest = {"experiment": protocol()["experiment"], "controller_identity": code,
                    "parent_bundle_id": original["bundle_id"], "files": files}
        manifest["bundle_id"] = api.digest(manifest)[:24]
        api.write(work / "manifest.json", manifest, replace=True)
        target = ROOT / "artifacts/bundles" / manifest["bundle_id"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copytree(work, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".lock"))
        validate_bundle(target, replay_imports=True)
        return target
