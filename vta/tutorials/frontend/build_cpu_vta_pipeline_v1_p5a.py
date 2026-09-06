#!/usr/bin/env python3
"""Build and audit the frozen V1-P5A candidate packages without using the board."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from freeze_cpu_vta_pipeline_v1 import (
    PROTOCOL_ID,
    file_sha256,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
)


DEFAULT_OUTPUT = repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
DEPLOY_SCRIPT = repo_root() / "vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py"
COMPARATOR = repo_root() / "vta/tutorials/frontend/compare_cpu_vta_pipeline_v1_reference.py"
VTA_MARKERS = (b"VTAPushGEMMOp", b"VTAPushALUOp")


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def preprocess_reference_input(image_path):
    from PIL import Image

    image = Image.open(image_path).resize((224, 224)).convert("RGB")
    array = np.asarray(image).astype("float32")
    array -= np.asarray([123.0, 117.0, 104.0], dtype="float32")
    array /= np.asarray([58.395, 57.12, 57.375], dtype="float32")
    return array.transpose((2, 0, 1))[np.newaxis, :].astype("float32")


def build_reference(output_dir):
    import mxnet as mx
    from mxnet.gluon.model_zoo import vision

    reference_dir = Path(output_dir) / "v1_p5a_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    image_path = repo_root() / "build_axu_aarch64/cat.png"
    image = preprocess_reference_input(image_path)
    model = vision.get_model("resnet18_v1", pretrained=True)
    logits = model(mx.nd.array(image)).asnumpy().astype("float32")
    if tuple(logits.shape) != (1, 1000) or not np.isfinite(logits).all():
        raise RuntimeError("independent MXNet reference produced an invalid logits tensor")
    tensor_path = reference_dir / "expected_logits_float32.bin"
    input_path = reference_dir / "expected_input_float32.bin"
    logits.tofile(tensor_path)
    image.tofile(input_path)
    payload = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5a_independent_tensor_reference",
            "protocol_id": PROTOCOL_ID,
            "model": "mxnet.gluon.model_zoo.resnet18_v1_pretrained",
            "reference_independence": "framework_model_and_full_float_logits_not_native_pipeline_hash",
            "image": str(image_path),
            "image_sha256": file_sha256(image_path),
            "preprocess": "resize224_rgb_subtract_123_117_104_divide_58.395_57.12_57.375_chw",
            "input_file": str(input_path.relative_to(output_dir)),
            "input_sha256": file_sha256(input_path),
            "tensor_file": str(tensor_path.relative_to(output_dir)),
            "tensor_sha256": file_sha256(tensor_path),
            "dtype": "float32",
            "shape": [1, 1000],
            "summary": {
                "min": float(np.min(logits)),
                "max": float(np.max(logits)),
                "mean": float(np.mean(logits, dtype="float64")),
                "sum": float(np.sum(logits, dtype="float64")),
                "top1": int(np.argmax(logits[0])),
            },
            "qualification_gate": {
                "name": "quantized_semantic_tensor_gate_v1",
                "scope": "detect_wrong_graph_or_gross_quantized_output;_not_bit_exact_float_equivalence",
                "min_cosine_similarity": 0.95,
                "max_normalized_rmse": 0.50,
                "top_k": 10,
                "min_top_k_overlap": 5,
                "top1_match_required": False,
                "thresholds_frozen_before_p5b_board_outputs": True,
            },
            "comparator": str(COMPARATOR),
            "board_output_used": False,
        }
    )
    path = Path(output_dir) / "v1_p5a_reference.json"
    write_json(path, payload)
    return payload, path


def candidate_dir_name(row):
    return "{:02d}_{}".format(int(row["frozen_execution_order"]), row["candidate_id"])


def scheme_payload(row):
    return {
        "schema_version": 1,
        "kind": "cpu_vta_pipeline_v1_p5a_scheme",
        "candidate_id": row["candidate_id"],
        "scheme_name": row["candidate_id"],
        "scheme_cfg": row["scheme_cfg"],
        "stage_runtime_threads": row["stage_runtime_threads"],
        "frozen_execution_order": row["frozen_execution_order"],
    }


def build_command(row, candidate_dir, stage_cache):
    return [
        sys.executable,
        str(DEPLOY_SCRIPT),
        "--scheme-config-json",
        str(candidate_dir / "scheme.json"),
        "--candidate-id",
        row["candidate_id"],
        "--stage-runtime-num-threads",
        ",".join(str(value) for value in row["stage_runtime_threads"]),
        "--runtime-num-threads",
        "4",
        "--runs",
        "22",
        "--queue-depth",
        "2",
        "--run-serial-before-pipeline",
        "--runner-output-mode",
        "raw",
        "--output-dump-dir",
        "correctness_outputs",
        "--stage-build-cache-dir",
        str(stage_cache),
        "--build-dir",
        str(candidate_dir),
        "--package-only",
    ]


def _stage_accounting_ids(candidate_id, stages):
    ids = []
    for stage in stages:
        if stage["device"] == "vta":
            ids.extend(
                [
                    "{}:{}:load".format(candidate_id, stage["name"]),
                    "{}:{}:compute".format(candidate_id, stage["name"]),
                    "{}:{}:store".format(candidate_id, stage["name"]),
                ]
            )
    for left, right in zip(stages, stages[1:]):
        if left["device"] != right["device"]:
            ids.append("{}:{}-{}:host_adapter".format(candidate_id, left["name"], right["name"]))
    return ids


def _hash_matches(package_dir, stage, field, relative_path):
    expected = stage["artifact_sha256"][field]
    return file_sha256(package_dir / relative_path) == expected


def prepare_package_for_p5b(candidate_dir, reference):
    package_dir = candidate_dir / "package"
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "runs": 22,
            "warmup_frames": 2,
            "scored_performance_frames": 20,
            "baseline": "external_independent_mxnet_tensor_reference",
            "correctness_reference": {
                "kind": reference["kind"],
                "artifact_sha256": reference["artifact_sha256"],
                "tensor_sha256": reference["tensor_sha256"],
                "qualification_gate": reference["qualification_gate"]["name"],
            },
        }
    )
    write_json(manifest_path, manifest)
    for name in ("run_stage_serial.sh", "run_stage_pipeline.sh"):
        path = package_dir / name
        text = path.read_text(encoding="utf-8")
        lines = []
        replaced = False
        for line in text.splitlines():
            if line.strip().startswith("--runs "):
                suffix = " \\" if line.rstrip().endswith("\\") else ""
                line = "  --runs 22{}".format(suffix)
                replaced = True
            lines.append(line)
        if not replaced:
            raise RuntimeError("{} has no generated --runs line".format(path))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boundary_contract(schema):
    slots = schema.get("slots") or []
    normalized = []
    for slot in slots:
        normalized.append(
            {
                "slot_index": slot.get("slot_index"),
                "shape": slot.get("shape"),
                "dtype": slot.get("dtype"),
            }
        )
    return {"arity": schema.get("arity"), "slots": normalized}


def _run_script_thread_map(path, stage_count):
    """Read the thread parameters that the board runner will actually receive."""
    text = Path(path).read_text(encoding="utf-8")
    result = {}
    for index in range(stage_count):
        match = re.search(
            r"--stage{}-runtime-num-threads\s+(\d+)".format(index), text
        )
        if not match:
            raise RuntimeError("{} has no stage{} thread argument".format(path, index))
        result["stage{}".format(index)] = int(match.group(1))
    return result


def audit_package(row, candidate_dir, reference):
    package_dir = candidate_dir / "package"
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("missing package manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    expected_cfg = row["scheme_cfg"]
    stages = manifest.get("stages") or []
    if manifest.get("candidate_id") != row["candidate_id"]:
        failures.append("candidate_id_mismatch")
    if len(stages) != len(expected_cfg):
        failures.append("stage_count_mismatch")
    for index, expected in enumerate(expected_cfg):
        if index >= len(stages):
            break
        stage = stages[index]
        for key in ("name", "device", "unit_names"):
            if stage.get(key) != expected.get(key):
                failures.append("stage{}_{}_mismatch".format(index, key))
        for key, rel in (
            ("graph_sha256", stage.get("graph", "")),
            ("lib_sha256", stage.get("lib", "")),
            ("params_sha256", stage.get("params", "")),
        ):
            if not rel or not (package_dir / rel).is_file() or not _hash_matches(
                package_dir, stage, key, rel
            ):
                failures.append("stage{}_{}_mismatch".format(index, key))
        if not stage.get("relay_ir_sha256") or not stage.get("compiled_module_sources"):
            failures.append("stage{}_missing_lowering_evidence".format(index))
        lib_path = package_dir / stage.get("lib", "")
        if stage.get("device") == "vta" and lib_path.is_file():
            binary = lib_path.read_bytes()
            if not any(marker in binary for marker in VTA_MARKERS):
                failures.append("stage{}_missing_vta_runtime_calls".format(index))

    expected_threads = {
        "stage{}".format(index): int(value)
        for index, value in enumerate(row["stage_runtime_threads"])
    }
    for mode in ("serial", "pipeline"):
        if manifest.get("stage_runtime_threads", {}).get(mode) != expected_threads:
            failures.append("{}_thread_map_mismatch".format(mode))
        script_path = package_dir / (
            "run_stage_serial.sh" if mode == "serial" else "run_stage_pipeline.sh"
        )
        try:
            script_threads = _run_script_thread_map(script_path, len(stages))
        except (OSError, RuntimeError) as err:
            failures.append("{}_script_thread_parse_failed: {}".format(mode, err))
        else:
            if script_threads != expected_threads:
                failures.append("{}_script_thread_map_mismatch".format(mode))
        affinities = manifest.get("stage_cpu_affinity", {}).get(mode) or {}
        for index, stage in enumerate(stages):
            key = "stage{}".format(index)
            expected_affinity = (
                list(range(expected_threads[key])) if stage.get("device") == "cpu" else []
            )
            if affinities.get(key) != expected_affinity:
                failures.append("{}_{}_affinity_mismatch".format(mode, key))
    if manifest.get("stage_cpu_affinity", {}).get("policy") != "independent_overlapping_prefix_masks_v1":
        failures.append("affinity_policy_mismatch")
    if manifest.get("runner_output_mode") != "raw":
        failures.append("runner_output_mode_not_raw")
    if manifest.get("runs") != 22 or manifest.get("warmup_frames") != 2:
        failures.append("performance_sample_count_mismatch")
    if manifest.get("scored_performance_frames") != 20:
        failures.append("scored_performance_frame_count_mismatch")
    reference_record = manifest.get("correctness_reference") or {}
    if reference_record.get("artifact_sha256") != reference["artifact_sha256"]:
        failures.append("correctness_reference_mismatch")
    if manifest.get("output_dump_dir") != "correctness_outputs":
        failures.append("output_dump_dir_missing")
    for script_name, suffix in (
        ("run_stage_serial.sh", "correctness_outputs/serial"),
        ("run_stage_pipeline.sh", "correctness_outputs/pipeline"),
    ):
        script_path = package_dir / script_name
        if not script_path.is_file() or suffix not in script_path.read_text(encoding="utf-8"):
            failures.append("{}_missing_raw_dump".format(script_name))
    input_path = package_dir / manifest.get("input_file", "input.bin")
    if not input_path.is_file() or file_sha256(input_path) != reference["input_sha256"]:
        failures.append("input_reference_mismatch")
    if stages:
        final_schema = stages[-1].get("output_schema", {})
        slots = final_schema.get("slots") or []
        if len(slots) != 1 or slots[0].get("shape") != [1, 1000] or slots[0].get("dtype") != "float32":
            failures.append("final_output_contract_mismatch")
    for left, right in zip(stages, stages[1:]):
        if _boundary_contract(left.get("output_schema", {})) != _boundary_contract(
            right.get("input_schema", {})
        ):
            failures.append("{}_{}_boundary_contract_mismatch".format(left["name"], right["name"]))
    accounting_ids = _stage_accounting_ids(row["candidate_id"], stages)
    if len(accounting_ids) != len(set(accounting_ids)):
        failures.append("duplicate_accounting_id")
    return {
        "candidate_id": row["candidate_id"],
        "frozen_execution_order": int(row["frozen_execution_order"]),
        "package_dir": str(package_dir),
        "manifest_sha256": file_sha256(manifest_path),
        "stage_count": len(stages),
        "stage_cache_hit_count": sum(bool(stage.get("stage_build_cache_hit")) for stage in stages),
        "stage_cache_miss_count": sum(not bool(stage.get("stage_build_cache_hit")) for stage in stages),
        "vta_stage_count": sum(stage.get("device") == "vta" for stage in stages),
        "tuple_boundary_count": sum(
            int((left.get("output_schema") or {}).get("arity", 0)) > 1
            for left, _ in zip(stages, stages[1:])
        ),
        "boundary_abi_audit": "ordinal_shape_dtype; tuple semantic order requires P5B tensor reference",
        "accounting_scope": "stage_level_v1",
        "accounting_ids": accounting_ids,
        "native_only_evidence": "VTA stage libraries reference VTAPushGEMMOp or VTAPushALUOp",
        "audit_failures": sorted(set(failures)),
        "audit_passed": not failures,
    }


def relink_immutable_artifacts(candidate_dir, stage_cache, content_store):
    package_dir = candidate_dir / "package"
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    linked = 0
    for stage in manifest["stages"]:
        cache_dir = Path(stage_cache) / stage["stage_build_cache_key"]
        for key, rel in (("graph.json", stage["graph"]), ("graphlib.so", stage["lib"]), ("params.params", stage["params"])):
            source = cache_dir / key
            target = package_dir / rel
            if source.is_file() and file_sha256(source) == file_sha256(target):
                target.unlink()
                os.link(source, target)
                linked += 1
    content_store.mkdir(parents=True, exist_ok=True)
    for rel in ("input.bin", "libtvm_runtime.so", "libvta.so"):
        target = package_dir / rel
        if not target.is_file():
            continue
        digest = file_sha256(target)
        source = content_store / digest
        if not source.exists():
            os.link(target, source)
        target.unlink()
        os.link(source, target)
        linked += 1
    tar_path = candidate_dir / "package.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    return linked


def run_candidate(row, packages_dir, stage_cache, reference, resume=True):
    candidate_dir = packages_dir / candidate_dir_name(row)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_json(candidate_dir / "scheme.json", scheme_payload(row))
    manifest_path = candidate_dir / "package/manifest.json"
    command = build_command(row, candidate_dir, stage_cache)
    started = time.monotonic()
    returncode = 0
    resumed = False
    if resume and manifest_path.exists():
        resumed = True
    else:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        returncode = int(proc.returncode)
        (candidate_dir / "build.log").write_text(proc.stdout, encoding="utf-8")
    elapsed = time.monotonic() - started
    record = {
        "candidate_id": row["candidate_id"],
        "frozen_execution_order": int(row["frozen_execution_order"]),
        "command": command,
        "board_argument_present": "--board" in command,
        "resumed_existing_package": resumed,
        "returncode": returncode,
        "wall_clock_s": elapsed,
    }
    if returncode != 0:
        record.update({"audit_passed": False, "audit_failures": ["build_failed"]})
        return record
    try:
        prepare_package_for_p5b(candidate_dir, reference)
        record.update(audit_package(row, candidate_dir, reference))
    except Exception as err:  # pylint: disable=broad-except
        record.update({"audit_passed": False, "audit_failures": ["audit_exception: {}".format(err)]})
    return record


def build_p5a(output_dir=DEFAULT_OUTPUT, resume=True):
    output_dir = Path(output_dir)
    state = load_sealed_artifact(
        output_dir / "v1_execution_state.json", "cpu_vta_pipeline_v1_execution_state"
    )
    if state["current_stage"] not in {"V1-P4R", "V1-P5A"}:
        raise RuntimeError("P5A requires the frozen P4R state")
    manifest = load_sealed_artifact(
        output_dir / "v1_p4r_candidate_manifest.json",
        "cpu_vta_pipeline_v1_p4r_candidate_manifest",
    )
    plan = load_sealed_artifact(
        output_dir / "v1_p4r_prospective_plan.json",
        "cpu_vta_pipeline_v1_p4r_prospective_plan",
    )
    if manifest["artifact_sha256"] != state["p4r_manifest_artifact_sha256"]:
        raise RuntimeError("P4R manifest/state hash mismatch")
    if plan["artifact_sha256"] != state["p4r_plan_artifact_sha256"]:
        raise RuntimeError("P4R plan/state hash mismatch")

    reference, reference_path = build_reference(output_dir)
    packages_dir = output_dir / "v1_p5a_packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    stage_cache = output_dir / "v1_p1_stage_cache"
    rows = sorted(manifest["rows"], key=lambda row: int(row["frozen_execution_order"]))
    results = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        print("[P5A] {}/{} {}".format(index, len(rows), row["candidate_id"]), flush=True)
        results.append(run_candidate(row, packages_dir, stage_cache, reference, resume=resume))
    total_wall = time.monotonic() - started

    linked_count = 0
    for result in results:
        if result.get("audit_passed"):
            candidate_dir = packages_dir / candidate_dir_name(
                next(row for row in rows if row["candidate_id"] == result["candidate_id"])
            )
            linked_count += relink_immutable_artifacts(
                candidate_dir, stage_cache, packages_dir / "_content_store"
            )
            # Relinking is content-preserving; re-run the hash audit to catch filesystem mistakes.
            row = next(row for row in rows if row["candidate_id"] == result["candidate_id"])
            preserved = {
                key: result[key]
                for key in ("command", "board_argument_present", "resumed_existing_package", "returncode", "wall_clock_s")
            }
            result.clear()
            result.update(preserved)
            result.update(audit_package(row, candidate_dir, reference))

    passed = sum(bool(row.get("audit_passed")) for row in results)
    audit = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5a_compile_audit",
            "protocol_id": PROTOCOL_ID,
            "p4r_manifest_artifact_sha256": manifest["artifact_sha256"],
            "p4r_plan_artifact_sha256": plan["artifact_sha256"],
            "candidate_count": len(results),
            "compile_audit_pass_count": passed,
            "compile_audit_failure_count": len(results) - passed,
            "all_commands_board_free": all(not row["board_argument_present"] for row in results),
            "performance_results_present": False,
            "reference_artifact_sha256": reference["artifact_sha256"],
            "stage_cache_dir": str(stage_cache),
            "content_preserving_hardlinks_created": linked_count,
            "total_wall_clock_s": total_wall,
            "rows": results,
        }
    )
    audit_path = output_dir / "v1_p5a_compile_audit.json"
    write_json(audit_path, audit)

    complete = passed == len(results) and len(results) == 20
    next_state = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_execution_state",
            "protocol_id": PROTOCOL_ID,
            "current_stage": "V1-P5A",
            "current_stage_status": (
                "completed_awaiting_user_confirmation" if complete else "completed_gate_failed"
            ),
            "next_stage": "V1-P5B",
            "next_stage_may_start": False,
            "advance_requires_user_confirmation": True,
            "p4r_plan_artifact_sha256": plan["artifact_sha256"],
            "p4r_manifest_artifact_sha256": manifest["artifact_sha256"],
            "p5a_reference_artifact_sha256": reference["artifact_sha256"],
            "p5a_compile_audit_artifact_sha256": audit["artifact_sha256"],
            "blocking_evidence": (
                ["P5B_board_execution_requires_explicit_user_confirmation"]
                if complete
                else ["P5A_compile_or_package_audit_failed"]
            ),
        }
    )
    state_path = output_dir / "v1_execution_state.json"
    write_json(state_path, next_state)
    return {
        reference_path.name: reference,
        audit_path.name: audit,
        state_path.name: next_state,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = build_p5a(args.output_dir, resume=not args.no_resume)
    audit = outputs["v1_p5a_compile_audit.json"]
    print(
        "[P5A] audit {}/{} passed artifact_sha256={}".format(
            audit["compile_audit_pass_count"],
            audit["candidate_count"],
            audit["artifact_sha256"],
        )
    )
    return 0 if audit["compile_audit_failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
