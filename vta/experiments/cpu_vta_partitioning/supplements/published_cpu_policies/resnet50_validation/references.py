"""Independent complete/frontier ARM references executed locally under QEMU."""
from pathlib import Path
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import digest, inventory, locked, read, sha256, sync_tree, verify_inventory, write
from cpu_vta_partitioning.prepare.build import copy_checked
from .assets import PARENT


def fnv1a64(data):
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(value, "016x")


def prepare(build, output):
    from .prepare import MODULE_FILES, validate_build, _frozen
    build, output = Path(build).resolve(), Path(output).resolve()
    checked = validate_build(build, require_all=False)
    names = ("reference_full", "reference_frontiers")
    if any(name not in checked["complete"] for name in names):
        raise ValueError("both ARM reference programs must be compiled first")
    if (output / "manifest.json").exists():
        return validate(output)
    parent = read(PARENT / "manifest.json")
    programs = {}
    for name in names:
        record = checked["complete"][name]
        for leaf in MODULE_FILES:
            copy_checked(build / "programs" / name / leaf, output / "programs" / name / leaf)
        programs[name] = {**record, "files": {n: record["files"][n] for n in MODULE_FILES},
                          "source_record_sha256": sha256(build / "programs" / name / "record.json"),
                          "excluded_nonexecution_proof_files": {n: v for n, v in record["files"].items() if n not in MODULE_FILES}}
        _frozen(output / "programs" / name / "record.json", programs[name])
    for name, entry in parent["files"].items():
        if name.startswith(("runtime/", "firmware/")):
            copy_checked(PARENT / name, output / name, entry["sha256"])
    runner_hash = copy_checked(PARENT / "runner/vta_stage_pipeline_runner", output / "runner/vta_stage_pipeline_runner",
                               parent["runner_sha256"])
    copy_checked(build / "inputs/000.bin", output / "inputs/000.bin")
    copy_checked(build / "plans/compiler_identity.json", output / "sources/compiler_identity.json")
    _frozen(output / "plans/programs.json", programs)
    _frozen(output / "plans/inputs.json", {"records": [{"index": 0, "input": "inputs/000.bin", "frontiers": {}}]})
    identity = {"network": "resnet50_v2", "scope": "independent ARM full and frontier reference under local QEMU",
                "canonical_sha256": checked["identity"]["canonical_sha256"],
                "unit_count": checked["identity"]["unit_count"], "runner_sha256": runner_hash,
                "reference_controller_sha256": sha256(__file__), "files": inventory(output, [".lock"])}
    write(output / "manifest.json", {"bundle_id": digest(identity)[:24], **identity})
    return validate(output)


def validate(bundle):
    bundle = Path(bundle); manifest = read(bundle / "manifest.json")
    if manifest["bundle_id"] != digest({k: v for k, v in manifest.items() if k != "bundle_id"})[:24]:
        raise ValueError("ARM reference capsule seal differs")
    verify_inventory(bundle, manifest["files"])
    if manifest["reference_controller_sha256"] != sha256(__file__):
        raise ValueError("ARM reference controller source changed")
    programs = read(bundle / "plans/programs.json")
    if programs["reference_full"]["canonical_sha256"] != programs["reference_frontiers"]["canonical_sha256"]:
        raise ValueError("full graph and frontier probe have different common graphs")
    return manifest


def task(bundle, name):
    manifest = validate(bundle)
    if name not in ("reference_full", "reference_frontiers"):
        raise ValueError("unknown ARM reference program")
    document = {"kind": "qualify_component", "stages": [{"program": name, "device": "cpu", "threads": 1, "mask": [0]}],
                "environment": {"poll_sleep_ns": 1000, "post_start_sleep_ns": 1000},
                "warmup": 0, "scored": 1, "tail": 0, "input_indices": [0],
                "bundle_id": manifest["bundle_id"], "purpose": "local independent ARM reference, not performance"}
    document["task_id"] = digest(document)[:24]
    return document


def output_arrays(raw, document, bundle):
    import numpy as np
    from cpu_vta_partitioning.measurement import frames, input_sequence
    raw, bundle = Path(raw), Path(bundle)
    rows = frames(raw / "result.jsonl", 1)
    input_sequence(rows, document)
    qemu_affinity(read(raw / "affinity.json"), document["stages"])
    row = rows[0]
    if len(row.get("stage_raw_outputs", [])) != 1:
        raise ValueError("reference stage outputs missing")
    dumps = row["stage_raw_outputs"][0]
    program = read(bundle / "plans/programs.json")[document["stages"][0]["program"]]
    types = program["output_types"]
    if len(dumps["output_files"]) != len(types) or len(dumps["outputs"]) != len(types):
        raise ValueError("reference frontier tensor outputs incomplete")
    arrays = []
    for i, (file, summary, typ) in enumerate(zip(dumps["output_files"], dumps["outputs"], types)):
        if (file["index"] != i or summary["index"] != i
                or any(file[k] != typ[k] or summary[k] != typ[k] for k in ("shape", "dtype"))):
            raise ValueError("reference tensor type/order differs")
        data = (raw / "outputs/run_0" / Path(file["path"]).name).read_bytes()
        expected_bytes = int(np.prod(typ["shape"])) * np.dtype(typ["dtype"]).itemsize
        if len(data) != expected_bytes or file["nbytes"] != len(data) or summary["nbytes"] != len(data):
            raise ValueError("reference tensor byte count differs")
        if summary["fnv1a64"].lower() != fnv1a64(data):
            raise ValueError("reference tensor payload hash differs")
        value = np.frombuffer(data, dtype=typ["dtype"]).reshape(typ["shape"])
        if not np.isfinite(value).all():
            raise ValueError("nonfinite ARM reference")
        arrays.append(value)
    return program, arrays


def qemu_affinity(report, stages):
    """Check active logical slots and identify zero-CPU QEMU auxiliaries.

    QEMU creates a host helper not belonging to the TVM logical pool. This is
    the same narrowly-scoped rule used by common_top20.campaign.qemu_cpu_summary;
    it never relaxes board acceptance or an active worker's placement.
    """
    if not report.get("cpu_worker_pinning") or len(report["stages"]) != len(stages):
        raise ValueError("ARM logical pinning or stage count differs")
    ignored = []
    for cfg, stage in zip(stages, report["stages"]):
        slots = sorted(stage.get("worker_slots", []), key=lambda s: s["slot"])
        if (stage["requested_cpus"] != cfg["mask"] or stage["active_runtime_threads"] != cfg["threads"]
                or [s["slot"] for s in slots] != list(range(cfg["threads"]))
                or len({s["tid"] for s in slots}) != cfg["threads"]
                or slots[0]["tid"] != stage["caller_tid"]):
            raise ValueError("ARM logical slots or caller identity differs")
        active = {s["tid"] for s in slots}
        for slot, core in zip(slots, cfg["mask"]):
            effective = [s for s in stage["effective_affinities"] if s["tid"] == slot["tid"]]
            if slot["cpus"] != [core] or len(effective) != 1 or effective[0]["cpus"] != [core]:
                raise ValueError("active ARM worker affinity differs")
        for helper in stage["effective_affinities"]:
            if helper["tid"] not in active:
                if helper["cpu_time_delta_ms"] != 0:
                    raise ValueError("nonlogical QEMU helper consumed CPU")
                ignored.append(helper)
    return ignored


def run_qemu(bundle, output, *, timeout_seconds=1800):
    import numpy as np
    from cpu_vta_partitioning.measurement.tasks import native_command
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = validate(bundle)
    qemu = shutil.which("qemu-aarch64-static")
    if not qemu:
        raise RuntimeError("qemu-aarch64-static required")
    started = time.monotonic()
    with locked(output):
        identity = {"bundle_id": manifest["bundle_id"], "input_sha256": sha256(bundle / "inputs/000.bin"),
                    "execution": "local QEMU AArch64 CPU", "board_accessed": False}
        marker = output / "run.json"
        if marker.exists() and read(marker) != identity:
            raise ValueError("reference output directory belongs to another capsule")
        if not marker.exists(): write(marker, identity)
        proof_path = output / "REFERENCE_PROOF.json"
        if proof_path.exists():
            previous = read(proof_path)
            if (previous["reference_capsule_id"] != manifest["bundle_id"]
                    or previous["canonical_sha256"] != manifest["canonical_sha256"]
                    or previous["input"]["sha256"] != identity["input_sha256"]):
                raise ValueError("completed reference proof belongs to another capsule")
            verify_inventory(output / "references", previous["files"])
            for name, source in previous["raw_sources"].items():
                receipt = Path(source["receipt"])
                if sha256(receipt) != source["receipt_sha256"]:
                    raise ValueError("reference receipt changed")
                item = read(receipt)
                if item["task"] != task(bundle, name):
                    raise ValueError("reference task changed")
                verify_inventory(item["raw"], item["files"])
                output_arrays(item["raw"], item["task"], bundle)
            return previous
        accepted, arrays = {}, {}
        for name in ("reference_full", "reference_frontiers"):
            document = task(bundle, name)
            raw = output / "raw" / name; receipt = output / "accepted" / (document["task_id"] + ".json")
            if receipt.exists():
                old = read(receipt)
                if old["task"] != document:
                    raise ValueError("reference task identity changed")
                verify_inventory(raw, old["files"])
            else:
                raw.mkdir(parents=True, exist_ok=True)
                argv, inputs = native_command(document, bundle, str(bundle), str(raw))
                (raw / "inputs.txt").write_text(inputs)
                loader = argv.index(str(bundle) + "/runtime/ld-linux-aarch64.so.1")
                variables = [*argv[2:loader], "TVM_BIND_THREADS=1", "TVM_EXCLUDE_WORKER0=1"]
                call = [qemu, *[x for value in variables for x in ("-E", value)], *argv[loader:],
                        "--cpu-worker-pinning", "--resource-affinity-probe"]
                write(raw / "task.json", document, replace=True)
                result = subprocess.run(call, capture_output=True, text=True, timeout=timeout_seconds)
                (raw / "native.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise ValueError("local ARM reference failed; preserve " + str(raw))
                output_arrays(raw, document, bundle)
                sync_tree(raw)
                write(receipt, {"task": document, "raw": str(raw), "files": inventory(raw),
                                "execution": "QEMU_AARCH64_CPU", "board_accessed": False})
            arrays[name] = output_arrays(raw, document, bundle)
            accepted[name] = {"path": str(raw), "receipt": str(receipt), "receipt_sha256": sha256(receipt)}
        full, expected = arrays["reference_full"]
        probe, observed = arrays["reference_frontiers"]
        if len(expected) != 1 or any(index is None for index in probe["terminal_output_indices"]):
            raise ValueError("ResNet50 terminal output contract is incomplete")
        for value, index in zip(expected, probe["terminal_output_indices"]):
            other = observed[index]
            if value.shape != other.shape or value.dtype != other.dtype or value.tobytes() != other.tobytes():
                raise ValueError("ARM full graph and frontier probe outputs are not byte exact")
        refs = output / "references"; refs.mkdir(exist_ok=True)
        frontiers = {}
        for node, value in zip(probe["output_node_ids"], observed):
            np.save(refs / (str(node) + ".npy"), value, allow_pickle=False)
            value.tofile(refs / (str(node) + ".bin"))
            frontiers[str(node)] = {"dtype": str(value.dtype), "shape": list(value.shape),
                                    "npy": "references/" + str(node) + ".npy",
                                    "raw": "references/" + str(node) + ".bin"}
        proof = {"status": "ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT", "network": "resnet50_v2", "outputs": 1,
                 "canonical_sha256": full["canonical_sha256"], "reference_execution_target": "AArch64 LLVM CPU under local QEMU",
                 "reference_capsule_id": manifest["bundle_id"], "frontier_tensor_count": len(frontiers),
                 "input": {"path": str(bundle / "inputs/000.bin"), "sha256": sha256(bundle / "inputs/000.bin")},
                 "frontiers": frontiers, "files": inventory(refs), "raw_sources": accepted,
                 "elapsed_seconds": time.monotonic() - started, "board_accessed": False,
                 "local_complete_graph_calls": 2, "new_board_inferences": 0, "new_performance_windows": 0}
        write(output / "REFERENCE_PROOF.json", proof, replace=True)
        return proof
