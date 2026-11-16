"""Relocatable package closure, ELF checks and offline readiness gates."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

from cpu_vta_partitioning.common import PROJECT, REPO, read, write, digest, inventory, verify_inventory, sha256, now
from cpu_vta_partitioning.domain import topologies, signatures
from cpu_vta_partitioning.measurement.tasks import calibration_templates, initial_anchor, task, native_command


def elf(path):
    path = Path(path)
    header = subprocess.check_output(["/usr/bin/readelf", "-h", str(path)], text=True)
    if "AArch64" not in header or "ELF64" not in header:
        raise ValueError("not an aarch64 ELF: " + str(path))
    dynamic = subprocess.check_output(["/usr/bin/readelf", "-d", str(path)], text=True)
    return {"needed": re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic),
            "rpath": re.findall(r"\((?:RPATH|RUNPATH)\).*\[(.*?)\]", dynamic), "architecture": "aarch64"}


def runtime_closure(workspace, local):
    from cpu_vta_partitioning.prepare.toolchain import require_prepared_toolchain
    toolchain = require_prepared_toolchain(workspace, local)
    native = read(workspace / "plans/runtime_build.json")
    if native["toolchain"] != toolchain:
        raise ValueError("runtime and SDK toolchain identity mismatch")
    sysroot = Path(toolchain["sysroot"])
    provenance = {"toolchain": toolchain, "libraries": {}}
    built = {"libtvm_runtime.so": native["runtime_sha256"], "libvta.so": native["driver_sha256"]}
    runtime = workspace / "runtime"
    if any(p.is_dir() for p in runtime.iterdir()):
        raise ValueError("unexpected directory in managed runtime closure")
    # Resolve and validate a fresh closure before publishing any files. In
    # particular, an old same-SONAME library cannot satisfy a missing SDK file.
    with tempfile.TemporaryDirectory(prefix=".runtime-closure-", dir=workspace) as directory:
        staging = Path(directory)
        for name, expected in built.items():
            shutil.copy2(runtime / name, staging / name)
            if sha256(staging / name) != expected:
                raise ValueError("native runtime changed before packaging: " + name)

        def dependency(name):
            if Path(name).name != name:
                raise ValueError("invalid dependency name: " + name)
            target = staging / name
            if not target.exists():
                source = next((p for p in (sysroot / "lib" / name, sysroot / "usr/lib" / name)
                               if p.is_file()), None)
                if source is None:
                    raise FileNotFoundError("unresolved SDK runtime dependency " + name)
                expected = sha256(source)
                shutil.copy2(source, target, follow_symlinks=True)
                if sha256(target) != expected:
                    raise ValueError("SDK library changed during copy: " + name)
                provenance["libraries"]["runtime/" + name] = {
                    "source": str(source.resolve()), "sha256": expected, "bytes": target.stat().st_size}
            return target

        queue = [*[workspace / "programs" / name / "module.so" for name in read(workspace / "plans/programs.json")],
                 *[staging / name for name in built], workspace / "runner/vta_stage_pipeline_runner",
                 dependency("ld-linux-aarch64.so.1")]
        visited, records = set(), {}
        while queue:
            p = queue.pop()
            if p in visited:
                continue
            visited.add(p)
            record = elf(p)
            key = "runtime/" + p.name if p.parent == staging else str(p.relative_to(workspace))
            records[key] = record
            queue.extend(dependency(name) for name in record["needed"])
        records["firmware/u-dma-buf.ko"] = elf(workspace / "firmware/u-dma-buf.ko")
        wanted = {p.name for p in staging.iterdir()}
        for p in staging.iterdir():
            os.replace(p, runtime / p.name)
        for p in runtime.iterdir():
            if p.name not in wanted:
                p.unlink()
    # If interrupted during publication, prepare reconstructs this entire closure
    # before it can seal a bundle. Old unused SDK libraries are never carried over.
    write(workspace / "plans/sdk_runtime.json", provenance, replace=True)
    write(workspace / "plans/elf_dependencies.json", records, replace=True)
    return records


def create_plans(workspace):
    programs = read(workspace / "plans/programs.json")
    specs = read(workspace / "plans/segments.json")
    full_programs = {s["name"]:s for s in specs}
    failures = read(workspace / "plans/compile_failures.json")
    if set(programs) & set(failures) or set(programs) | set(failures) != set(full_programs):
        raise ValueError("every requested program must be compiled or explicitly failed")
    expected = {"topologies": 4623, "shared_configurations": 972528}
    def counts(domain):
        count, shared, dedicated = 0,0,0
        for topology in topologies(domain):
            count += 1
            ncpu = sum(p["device"] == "cpu" for p in topology)
            shared += 4 ** ncpu
            dedicated += 3 ** ncpu
        return {"topologies": count, "shared_configurations": shared, "dedicated_configurations": dedicated}
    original, actual = counts(full_programs), counts(programs)
    if any(original[k] != v for k,v in expected.items()) or actual["topologies"] == 0:
        raise ValueError("invalid frozen domain coverage")
    coverage = {"expected_programs": {"cpu":85,"vta":87},
                "compiled_programs": {device:sum(p["device"] == device for p in programs.values()) for device in ("cpu","vta")},
                "original_domain": original, "qualified_domain": actual,
                "shared_across_all_groups": True, "failed_programs": read(workspace / "plans/compile_failures.json")}
    write(workspace / "plans/compile_coverage.json", coverage, replace=True)
    protocol = read(workspace / "plans/protocol.json")
    main = protocol["environment"]
    sensitivity = read(workspace / "plans/sensitivity.json")
    envs = [{"cores": c, "poll_sleep_ns": p, "post_start_sleep_ns": sensitivity["post_start_sleep_ns"]}
            for c in sensitivity["cores"] for p in sensitivity["poll_sleep_ns"]]
    for e in envs:
        key = f"cores{e['cores']}_poll{e['poll_sleep_ns']}"
        templates = calibration_templates(programs,e)
        write(workspace / f"plans/calibration/{key}.json", templates, replace=True)
        # Validate all frontier inputs, dependencies and executable arguments locally.
        for t in templates:
            argv, lines = native_command(t, workspace, "/relocated/bundle", "/relocated/job")
            if len(lines.splitlines()) != 16:
                raise ValueError("incomplete calibration input pool")
        anchor = initial_anchor(programs,e)
        write(workspace / f"plans/execution/{key}.json", {
            "environment":e, "anchor":[s.to_dict() for s in anchor],
            "screening_max_candidates":47, "confirmation_max_candidates":14,
            "diagnostic_pairs":"one/two/three VTA islands; low/median/high estimated host work",
            "candidate_selection":"generated only after fresh calibration", "warmup_options":[20,100]}, replace=True)
    # Every legal adjacent frontier must match exactly, including residual carry.
    edges = set()
    for topology in topologies(programs):
        for left,right in zip(topology,topology[1:]):
            edges.add((left["name"],right["name"]))
    for a,b in edges:
        left,right = programs[a],programs[b]
        for node, kind in zip(right["input_node_ids"],right["input_types"]):
            i = left["output_node_ids"].index(node)
            if left["output_types"][i] != kind:
                raise ValueError("frontier mismatch: " + a + " -> " + b)
    write(workspace / "plans/frontier_validation.json", {"edges":len(edges),"all_live_outputs_preserved":True}, replace=True)
    from cpu_vta_partitioning.prepare.payload import create_payload_plan
    create_payload_plan(workspace)
    return coverage


def validate(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / "manifest.json")
    body = {k:v for k,v in manifest.items() if k != "bundle_id"}
    if digest(body)[:24] != manifest["bundle_id"]:
        raise ValueError("manifest identity mismatch")
    verify_inventory(bundle, manifest["files"])
    from cpu_vta_partitioning.prepare.references import validate_frozen_reference
    reference_identity = validate_frozen_reference(bundle)
    if read(bundle / "plans/inputs.json").get("reference_identity") != reference_identity:
        raise ValueError("input pool and frozen reference provenance differ")
    sdk = read(bundle / "plans/sdk_runtime.json")
    native = read(bundle / "plans/runtime_build.json")
    if sdk["toolchain"] != native["toolchain"] or sdk["toolchain"] != read(bundle / "plans/compiler_identity.json")["toolchain"]:
        raise ValueError("bundled SDK provenance mismatch")
    verify_inventory(bundle, sdk["libraries"])
    runtime_files = {"runtime/" + name for name in inventory(bundle / "runtime")}
    if runtime_files != set(sdk["libraries"]) | {"runtime/libtvm_runtime.so", "runtime/libvta.so"}:
        raise ValueError("unattributed runtime library in bundle")
    checksums = {**manifest["files"], "manifest.json":{"sha256":sha256(bundle / "manifest.json")}}
    expected_checksums = "".join(v["sha256"] + "  " + k + "\n" for k,v in sorted(checksums.items()))
    if (bundle / "manifest.sha256").read_text() != expected_checksums:
        raise ValueError("deployment checksum list mismatch")
    readiness = read(bundle / "OFFLINE_READINESS.json")
    if readiness["status"] != "PREPARED_OFFLINE" or not readiness["tests"]["passed"]:
        raise ValueError("offline acceptance gates have not passed")
    from cpu_vta_partitioning.prepare.payload import validate_payload
    payload = validate_payload(bundle)
    programs = read(bundle / "plans/programs.json")
    for name, p in programs.items():
        verify_inventory(bundle / "programs" / name, p["files"])
    deps = read(bundle / "plans/elf_dependencies.json")
    for name, record in deps.items():
        if elf(bundle / name) != record:
            raise ValueError("ELF dependency metadata mismatch")
        for library in record["needed"]:
            if not (bundle / "runtime" / library).is_file():
                raise ValueError("unresolved bundled library " + library)
    # A relocated copy must not resolve paths through the old build workspace.
    for efile in (bundle / "plans/calibration").glob("*.json"):
        for t in read(efile):
            argv, lines = native_command(t, bundle, "/new/location", "/new/job")
            for line in lines.splitlines():
                for path in line.split(","):
                    if not (bundle / path.removeprefix("/new/location/")).is_file():
                        raise FileNotFoundError(path)
    return {"status":"VALIDATED_OFFLINE", "bundle_id":manifest["bundle_id"], "programs":len(programs),
            "board_payload_bytes":payload["unpacked_bytes"], "board_validation":"pending"}


def seal(workspace, test_report):
    from cpu_vta_partitioning.campaign import controller_identity
    if not test_report.get("passed"):
        raise ValueError("cannot prepare a package without passing local tests")
    coverage = read(workspace / "plans/compile_coverage.json")
    # Ship the controller, exact source snapshots and report alongside the runnable closure.
    controller = workspace / "sources/controller"
    if controller.exists():
        shutil.rmtree(controller)
    shutil.copytree(PROJECT / "src", controller / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for folder in ("docs", "configs", "tests"):
        shutil.copytree(PROJECT / folder, controller / folder, ignore=shutil.ignore_patterns("local.json","__pycache__"))
    for filename in ("README.md","run.py","pyproject.toml"):
        shutil.copy2(PROJECT / filename, controller / filename)
    source_hashes = {str(p.relative_to(REPO)): sha256(p) for folder in (
        "src/runtime", "include", "vta/runtime", "vta/include", "3rdparty/vta-hw/src", "3rdparty/vta-hw/include")
        for p in sorted((REPO / folder).rglob("*")) if p.is_file()}
    write(workspace / "sources/runtime_source_identity.json", source_hashes, replace=True)
    write(workspace / "OFFLINE_READINESS.json", {
        "status":"PREPARED_OFFLINE", "prepared_utc":now(), "controller_identity":controller_identity(),
        "repository_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "version_policy":"working tree identified by source hashes; HEAD alone does not identify local modifications",
        "host_dependencies":{name:__import__("importlib.metadata",fromlist=["version"]).version(name) for name in ("numpy","scipy","matplotlib","pytest")},
        "compile_coverage":coverage, "tests":test_report,
        "board_validation_pending":["full numerical outputs", "affinity and active thread counts", "fresh component costs",
                                    "steady windows", "actual boot independence", "driver/kernel compatibility", "PMU availability"],
        "performance_claim":None}, replace=True)
    # Exclude caches and build logs from execution closure; retain test logs as evidence.
    payload = ["programs","runtime","runner","firmware","inputs","references","plans","sources","OFFLINE_READINESS.json"]
    import tempfile
    parent = PROJECT / "artifacts/bundles"
    parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sealing-",dir=parent))
    for folder in payload:
        p = workspace / folder
        if folder == "programs":
            (staging / folder).mkdir()
            for name in read(workspace / "plans/programs.json"):
                shutil.copytree(p / name, staging / folder / name, ignore=shutil.ignore_patterns("lowering_qir.json"))
        elif p.is_dir():
            shutil.copytree(p, staging / folder, ignore=shutil.ignore_patterns("lowering_qir.json"))
        else:
            shutil.copy2(p,staging / folder)
    programs = read(staging / "plans/programs.json")
    for name,record in programs.items():
        record["local_lowering_artifact"] = record["files"].pop("lowering_qir.json")
        write(staging / "programs" / name / "record.json",record,replace=True)
    write(staging / "plans/programs.json",programs,replace=True)
    files = inventory(staging)
    manifest = {"schema_version":1,"files":files,"controller_identity":controller_identity()}
    manifest["bundle_id"] = digest(manifest)[:24]
    write(staging / "manifest.json", manifest)
    checksums = {**files, "manifest.json":{"sha256":sha256(staging / "manifest.json")}}
    (staging / "manifest.sha256").write_text("".join(v["sha256"] + "  " + k + "\n" for k,v in sorted(checksums.items())))
    destination = parent / manifest["bundle_id"]
    if destination.exists():
        shutil.rmtree(staging)
    else:
        staging.rename(destination)
    validate(destination)
    write(PROJECT / "artifacts/LATEST_BUNDLE.json", {"bundle_id":manifest["bundle_id"],"path":str(destination),"status":"PREPARED_OFFLINE"},replace=True)
    return destination
