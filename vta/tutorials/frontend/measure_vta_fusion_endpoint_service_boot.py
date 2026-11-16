#!/usr/bin/env python3
"""Measure the nine frozen C1 endpoint segments during one board boot.

Run this program once after each independent cold boot.  It reuses the
already-qualified binaries and deterministic inputs; it does not rebuild VTA
programs or alter the C3 shared-memory runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tvm import rpc
from mxnet.gluon.model_zoo import vision
import vta

from audit_vta_compile_context import REPORT
from fusion_aware_transfer import sha256_file
from profile_split_resnet18_stages import create_stage_module
from profile_vta_stage_boundary import build_cpu_reference
from qualify_vta_segment_counters import board_snapshot
from split_resnet18_stages import (
    build_resnet18_unit_blocks,
    build_resnet18_unit_metadata,
    lower_stage_to_relay,
    make_stage_block,
    relay_inputs_for_stage,
)


TRIPLETS = tuple(
    tuple("vta:{:02d}:{:02d}".format(start, end) for end in range(start, start + 3))
    for start in (5, 10, 15)
)
SEGMENT_IDS = tuple(segment_id for triplet in TRIPLETS for segment_id in triplet)
DMA_COUNTER_KEYS = (
    "load_buffer_2d_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_acc_calls",
    "load_buffer_2d_acc_bytes",
    "store_buffer_2d_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_out_calls",
    "store_buffer_2d_out_bytes",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def output_hashes(executor):
    return [
        hashlib.sha256(executor.get_output(index).numpy().tobytes()).hexdigest()
        for index in range(executor.get_num_outputs())
    ]


def validate_qualification(path):
    qualification = json.loads(path.read_text())
    if qualification.get("status") != "passed" or len(qualification.get("samples", [])) != 9:
        raise ValueError("endpoint binary lacks the frozen 3-input qualification: {}".format(path))
    if not all(
        sample["correctness"]["correct"] and not sample["counter_differences"]
        for sample in qualification["samples"]
    ):
        raise ValueError("endpoint qualification contains an invalid sample: {}".format(path))
    return qualification


def build_reference_hashes(archive):
    """Build independent same-qconfig LLVM references before board timing."""
    env = vta.get_env()
    model = vision.get_model("resnet18_v1", pretrained=True)
    features, head = list(model.features), model.output
    units = build_resnet18_unit_blocks(features, head)
    metadata = build_resnet18_unit_metadata(units, env.BATCH, 224)
    domain = {
        row["segment_id"]: row
        for row in json.loads((REPORT / "resource_aware_maxplus/v1_profile_manifest.json").read_text())["segments"]
    }
    result = {}
    for segment_id in SEGMENT_IDS:
        segment = domain[segment_id]
        stage = {"name": "c1_endpoint_reference", "device": "vta", "unit_names": segment["unit_names"]}
        inputs = relay_inputs_for_stage(stage, metadata)
        mod, params = lower_stage_to_relay(
            make_stage_block(features, head, stage, unit_blocks=units), inputs
        )
        reference = build_cpu_reference({"relay_mod": mod, "params": params})
        local = archive / segment_id.replace(":", "_")
        result[segment_id] = {}
        for case in range(3):
            with np.load(local / "inputs_{}.npz".format(case)) as payload:
                reference.set_input(**{key: payload[key] for key in payload.files})
            reference.run()
            result[segment_id][str(case)] = [
                hashlib.sha256(reference.get_output(index).numpy().tobytes()).hexdigest()
                for index in range(reference.get_num_outputs())
            ]
    return result


def main():
    root = Path("vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--archive", type=Path, default=root / "board_e2_run1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    if args.repeat < 3 or args.warmup < 1:
        raise ValueError("E2 requires at least three repeats and one warmup")

    artifacts = {}
    qualifications = {}
    for segment_id in SEGMENT_IDS:
        local = args.archive / segment_id.replace(":", "_")
        required = [
            local / "stage.so",
            local / "graph.json",
            local / "graph.params",
            local / "qualification.json",
            *(local / "inputs_{}.npz".format(case) for case in range(3)),
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifacts[str(path.resolve())] = sha256_file(path)
        qualifications[segment_id] = validate_qualification(local / "qualification.json")
        if qualifications[segment_id]["so_sha256"] != sha256_file(local / "stage.so"):
            raise ValueError("qualified stage binary hash changed: {}".format(segment_id))
        if qualifications[segment_id]["params_sha256"] != sha256_file(local / "graph.params"):
            raise ValueError("qualified parameter hash changed: {}".format(segment_id))

    reference_hashes = build_reference_hashes(args.archive)
    args.output_dir.mkdir(parents=True)
    before = board_snapshot(args.host)
    contract = {
        "schema": "vta_c1_endpoint_service_boot_contract_v1",
        "scope": "one boot; frozen ordinary set/get-copy endpoint binaries; no C3 policy change",
        "boot_id": before["boot"],
        "segments": list(SEGMENT_IDS),
        "triplets": [list(triplet) for triplet in TRIPLETS],
        "input_cases": [0, 1, 2],
        "repeat": args.repeat,
        "warmup": args.warmup,
        "order": "case/repeat rotation of the nine endpoints",
        "runner_sha256": sha256_file(Path(__file__)),
        "service_fields": {
            "stage_service_ms": "GraphExecutor time_evaluator latency",
            "vta_driver_service_ms": "driver_run_total_us divided by two profiler invocations",
            "host_helper_plus_graph_remainder_ms": (
                "stage minus VTA-driver service; proxy includes graph dispatch and is not pure CPU-kernel time"
            ),
        },
        "qualified_sources_sha256": artifacts,
        "independent_llvm_reference_output_sha256": reference_hashes,
        "board_before": before,
    }
    write_json(args.output_dir / "contract.json", contract)

    samples = []
    remote = rpc.connect(args.host, args.port, session_timeout=300)
    try:
        remote.get_function("runtime.config_threadpool")(1, 1)
        if remote.get_function("runtime.NumThreads")() != 1:
            raise RuntimeError("remote threadpool is not fixed to one thread")
        clear = remote.get_function("vta.runtime.profiler_clear")
        status = remote.get_function("vta.runtime.profiler_status")
        loaded = {}
        for segment_id in SEGMENT_IDS:
            local = args.archive / segment_id.replace(":", "_")
            target = segment_id.replace(":", "_") + ".so"
            remote.upload(str(local / "stage.so"), target=target)
            executor, context = create_stage_module(
                segment_id,
                "vta",
                (local / "graph.json").read_text(),
                remote.load_module(target),
                remote,
            )
            executor.load_params((local / "graph.params").read_bytes())
            # All three deterministic full outputs must pass before this
            # candidate is admitted to timing on the current boot.
            for case in range(3):
                with np.load(local / "inputs_{}.npz".format(case)) as payload:
                    executor.set_input(**{key: payload[key] for key in payload.files})
                executor.run()
                observed_hashes = output_hashes(executor)
                if observed_hashes != reference_hashes[segment_id][str(case)]:
                    raise RuntimeError(
                        "pre-timing full-output mismatch {} case {}".format(segment_id, case)
                    )
            with np.load(local / "inputs_0.npz") as payload:
                executor.set_input(**{key: payload[key] for key in payload.files})
            for _ in range(args.warmup):
                executor.run()
            loaded[segment_id] = (executor, context[0])

        sample_path = args.output_dir / "samples.jsonl"
        with sample_path.open("x", encoding="utf-8") as stream:
            sequence = 0
            for case in range(3):
                for repeat in range(args.repeat):
                    offset = (case + repeat) % len(SEGMENT_IDS)
                    order = SEGMENT_IDS[offset:] + SEGMENT_IDS[:offset]
                    for segment_id in order:
                        local = args.archive / segment_id.replace(":", "_")
                        executor, timer_device = loaded[segment_id]
                        with np.load(local / "inputs_{}.npz".format(case)) as payload:
                            executor.set_input(**{key: payload[key] for key in payload.files})
                        clear()
                        measured = executor.module.time_evaluator(
                            "run", timer_device, number=1, repeat=1
                        )().results
                        if len(measured) != 1 or not np.isfinite(measured[0]) or measured[0] <= 0:
                            raise RuntimeError("invalid timer result for {}".format(segment_id))
                        raw = json.loads(status())
                        expected = qualifications[segment_id]["samples"][0]["runtime_profile"]
                        differences = {
                            key: {"expected_raw": 2 * expected[key], "observed_raw": raw.get(key)}
                            for key in DMA_COUNTER_KEYS
                            if raw.get(key) != 2 * expected[key]
                        }
                        if differences:
                            raise RuntimeError("timed DMA mismatch {}: {}".format(segment_id, differences))
                        stage_ms = float(measured[0]) * 1000.0
                        vta_ms = float(raw["driver_run_total_us"]) / 2.0 / 1000.0
                        remainder_ms = stage_ms - vta_ms
                        if remainder_ms < 0:
                            raise RuntimeError(
                                "negative host/graph remainder for {}: stage={} vta={}".format(
                                    segment_id, stage_ms, vta_ms
                                )
                            )
                        row = {
                            "schema": "vta_c1_endpoint_service_sample_v1",
                            "sequence": sequence,
                            "boot_id": before["boot"],
                            "segment_id": segment_id,
                            "input_case": case,
                            "repeat": repeat,
                            "stage_service_ms": stage_ms,
                            "profiler_invocations": 2,
                            "vta_driver_service_ms": vta_ms,
                            "host_helper_plus_graph_remainder_ms": remainder_ms,
                            "dma_counter_differences": differences,
                            "output_sha256": output_hashes(executor),
                        }
                        if row["output_sha256"] != reference_hashes[segment_id][str(case)]:
                            raise RuntimeError(
                                "post-timing full-output mismatch {} case {}".format(segment_id, case)
                            )
                        samples.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                        sequence += 1
    finally:
        remote._sess.get_function("CloseRPCConnection")()

    after = board_snapshot(args.host)
    if after["boot"] != before["boot"]:
        raise RuntimeError("board rebooted during endpoint service run")
    result = {
        "schema": "vta_c1_endpoint_service_boot_v1",
        "status": "complete",
        "boot_id": before["boot"],
        "board_before": before,
        "board_after": after,
        "samples": len(samples),
        "expected_samples": len(SEGMENT_IDS) * 3 * args.repeat,
        "sample_file": str((args.output_dir / "samples.jsonl").resolve()),
        "sample_file_sha256": sha256_file(args.output_dir / "samples.jsonl"),
        "contract_sha256": sha256_file(args.output_dir / "contract.json"),
    }
    if result["samples"] != result["expected_samples"]:
        raise RuntimeError("incomplete endpoint service sample set")
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
