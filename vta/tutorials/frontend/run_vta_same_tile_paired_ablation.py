#!/usr/bin/env python3
"""Run P6e fixed-ConfigEntity paired mechanism/control ablations on VTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.contrib import cc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_mechanism_canary_board import (
    MODE_NUMBERS,
    REQUIRED_COUNTERS,
    cross_options,
    file_sha256,
    load_json,
    load_jsonl,
    semantic_config_key,
    ssh_preflight,
    validate_source_guards,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frozen_source(source, repo_root):
    run_dir = Path(source["run_dir"])
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    path = run_dir / "results.jsonl"
    if file_sha256(path) != source["results_sha256"]:
        raise ValueError("frozen source hash mismatch: {}".format(run_dir))
    return load_jsonl(path)


def validate_inputs(protocol_path, p6b_path, p6c_path, repo_root):
    protocol = load_json(protocol_path)
    p6b = load_json(p6b_path)
    p6c = load_jsonl(p6c_path)
    if protocol.get("schema") != "c3_p6e_same_tile_paired_protocol_v1":
        raise ValueError("unexpected P6e protocol schema")
    if len(protocol.get("pairs", ())) != 6:
        raise ValueError("P6e requires six pairs")
    if len(p6c) != 9 or not all(row.get("overall_status") == "passed" for row in p6c):
        raise ValueError("P6c mechanism correctness precondition failed")
    p6c_ids = {row["candidate_id"] for row in p6c}

    p4b = load_frozen_source(p6b["source_runs"]["P4b"], repo_root)
    p4e = load_frozen_source(p6b["source_runs"]["P4e"], repo_root)
    p4f = load_frozen_source(p6b["source_runs"]["P4f"], repo_root)
    p4g = load_frozen_source(p6b["source_runs"]["P4g"], repo_root)
    sources = {row["candidate_id"]: row for row in p4b + p4g}
    q4e = {row["candidate_id"]: row for row in p4e}
    q4f = {row["candidate_id"]: row for row in p4f}
    p6b_entries = {
        row["candidate_id"]: row
        for rows in p6b["workloads"].values()
        for row in rows
    }
    common_guards = next(iter(p6b_entries.values()))["qualification_source_guards_sha256"]
    controls = {}
    mechanisms = {}
    for pair in protocol["pairs"]:
        mechanism_id = pair["mechanism_candidate_id"]
        control_id = pair["control_candidate_id"]
        if mechanism_id not in p6c_ids or mechanism_id not in p6b_entries:
            raise ValueError("mechanism lacks P6c/P6b evidence")
        mechanism = sources[mechanism_id]
        control = sources[control_id]
        for record in (mechanism, control):
            expected_id = hashlib.sha256(
                json.dumps(record["identity"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if expected_id != record["candidate_id"]:
                raise ValueError("stable candidate ID mismatch")
            if record["workload_id"] != pair["workload_id"]:
                raise ValueError("pair workload mismatch")
            if int(record["debug"]["config_index"]) != int(pair["config_index"]):
                raise ValueError("pair ConfigEntity index mismatch")
        if control["residence_mode"] != "original":
            raise ValueError("paired control is not mode 0")
        if mechanism["residence_mode"] != pair["mechanism"]:
            raise ValueError("mechanism mode mismatch")
        q_e, q_f = q4e.get(control_id), q4f.get(control_id)
        if (
            not q_e
            or q_e.get("local_status") != "fsim_passed"
            or q_e.get("correctness_seeds") != protocol["correctness_seeds_for_all_unique_candidates"]
            or not q_f
            or q_f.get("overall_status") != "passed"
        ):
            raise ValueError("control lacks P4e/P4f qualification")
        controls[control_id] = {
            "record": control,
            "mode": "original",
            "source_guards": common_guards,
        }
        mechanisms[mechanism_id] = {
            "record": mechanism,
            "mode": pair["mechanism"],
            "source_guards": p6b_entries[mechanism_id]["qualification_source_guards_sha256"],
        }
    return protocol, controls, mechanisms


def build_candidate(candidate_id, item, env, binary_dir):
    record, mode = item["record"], item["mode"]
    validate_source_guards({"qualification_source_guards_sha256": item["source_guards"]})
    workload = record["identity"]["workload"]
    task = autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (MODE_NUMBERS[mode],),
        target=env.target,
        target_host=env.target_host,
    )
    config = task.config_space.get(int(record["debug"]["config_index"]))
    expected = dict(record["identity"]["complete_config_entity"])
    expected["index"] = int(record["debug"]["config_index"])
    if semantic_config_key(config.to_json_dict()) != semantic_config_key(expected):
        raise RuntimeError("ConfigEntity mismatch: {}".format(candidate_id))
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()
    if tir_hash != record["tir_sha256"]:
        raise RuntimeError("TIR certificate mismatch: {}".format(candidate_id))
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )
    binary = binary_dir / (candidate_id + ".so")
    module.export_library(
        str(binary),
        fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
        options=cross_options(),
    )
    return binary


def classify(improvement):
    if abs(improvement) <= 0.02:
        return "equivalent"
    if improvement >= 0.05:
        return "improved"
    if improvement <= -0.05:
        return "regressed"
    return "indeterminate"


def make_orders(pair_ids, rounds, seed):
    base = list(pair_ids)
    random.Random(seed).shuffle(base)
    orders = []
    for block in range(rounds // len(base)):
        for rotation in range(len(base)):
            offset = (rotation + block) % len(base)
            orders.append(base[offset:] + base[:offset])
    if len(orders) != rounds:
        raise ValueError("rounds must be divisible by pair count")
    return orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--p6b-manifest", required=True)
    parser.add_argument("--p6c-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--dmesg-after", type=int, default=467)
    args = parser.parse_args()

    output = Path(args.output_dir)
    correctness_path, samples_path = output / "correctness.jsonl", output / "samples.jsonl"
    if correctness_path.exists() or samples_path.exists():
        raise FileExistsError("refusing to overwrite P6e observations")
    repo_root = Path(__file__).resolve().parents[3]
    protocol, controls, mechanisms = validate_inputs(
        args.protocol, args.p6b_manifest, args.p6c_results, repo_root
    )
    boot_id = protocol["board_boot_id"]
    ssh_preflight(args.host, boot_id, args.dmesg_after)
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    clear = remote.get_function("vta.runtime.profiler_clear")
    status = remote.get_function("vta.runtime.profiler_status")
    device = remote.ext_dev(0)
    unique = dict(controls)
    unique.update(mechanisms)
    loaded = {}
    try:
        with tempfile.TemporaryDirectory(prefix="c3_p6e_board_") as temporary:
            for candidate_id, item in unique.items():
                binary = build_candidate(candidate_id, item, env, Path(temporary))
                remote.upload(str(binary))
                module = remote.load_module(binary.name)
                loaded[candidate_id] = {"module": module, "item": item}

            with correctness_path.open("x", encoding="utf-8") as stream:
                for candidate_id, item in unique.items():
                    record = item["record"]
                    seed_rows = []
                    for seed in protocol["correctness_seeds_for_all_unique_candidates"]:
                        data, weight, expected = reference_data_from_workload(
                            record["identity"]["workload"], seed
                        )
                        buffers = [
                            tvm.nd.array(data, device),
                            tvm.nd.array(weight, device),
                            tvm.nd.empty(expected.shape, "int8", device),
                        ]
                        loaded[candidate_id]["module"]["main"](*buffers)
                        actual = buffers[-1].numpy()
                        mismatches = int(np.count_nonzero(actual != expected))
                        seed_rows.append(
                            {
                                "seed": seed,
                                "correct": mismatches == 0,
                                "mismatch_count": mismatches,
                                "expected_sha256": array_sha256(expected),
                                "actual_sha256": array_sha256(actual),
                            }
                        )
                        if mismatches:
                            raise AssertionError("candidate wrong answer: {}".format(candidate_id))
                    row = {
                        "candidate_id": candidate_id,
                        "workload_id": record["workload_id"],
                        "config_index": record["debug"]["config_index"],
                        "status": "passed",
                        "seeds": seed_rows,
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()

            # Use seed 0 buffers for all timed candidates, followed by post-warmup equality.
            for candidate_id, item in unique.items():
                data, weight, expected = reference_data_from_workload(
                    item["record"]["identity"]["workload"], 0
                )
                buffers = [
                    tvm.nd.array(data, device),
                    tvm.nd.array(weight, device),
                    tvm.nd.empty(expected.shape, "int8", device),
                ]
                loaded[candidate_id].update(buffers=buffers, expected=expected)
                for _ in range(protocol["warmup_runs_per_unique_candidate"]):
                    loaded[candidate_id]["module"]["main"](*buffers)
                if not np.array_equal(buffers[-1].numpy(), expected):
                    raise AssertionError("post-warmup wrong answer: {}".format(candidate_id))

            pairs = {pair["pair_id"]: pair for pair in protocol["pairs"]}
            orders = make_orders(
                list(pairs), protocol["timed_rounds_per_pair"], protocol["order_seed"]
            )
            samples = []
            with samples_path.open("x", encoding="utf-8") as stream:
                sequence = 0
                for round_index, pair_order in enumerate(orders):
                    ssh_preflight(args.host, boot_id, args.dmesg_after)
                    for pair_id in pair_order:
                        pair = pairs[pair_id]
                        ids = [pair["mechanism_candidate_id"], pair["control_candidate_id"]]
                        if (round_index + int(hashlib.sha256(pair_id.encode()).hexdigest()[:2], 16)) % 2:
                            ids.reverse()
                        for candidate_id in ids:
                            item = loaded[candidate_id]
                            clear()
                            timer = item["module"].time_evaluator("main", device, number=1, repeat=1)
                            result = timer(*item["buffers"]).results
                            if len(result) != 1 or not np.isfinite(result[0]) or result[0] <= 0:
                                raise RuntimeError("invalid timer result")
                            raw = json.loads(status())
                            missing = [name for name in REQUIRED_COUNTERS if name not in raw]
                            if missing:
                                raise RuntimeError("missing counters: {}".format(missing))
                            role = "mechanism" if candidate_id == pair["mechanism_candidate_id"] else "control"
                            sample = {
                                "schema": "c3_p6e_same_tile_sample_v1",
                                "sequence": sequence,
                                "round": round_index,
                                "pair_id": pair_id,
                                "workload_id": pair["workload_id"],
                                "config_index": pair["config_index"],
                                "role": role,
                                "candidate_id": candidate_id,
                                "latency_ms": float(result[0]) * 1000.0,
                                "profiler_invocations": 2,
                                "runtime_profile_raw": {name: raw[name] for name in REQUIRED_COUNTERS},
                                "runtime_profile": {name: raw[name] / 2 for name in REQUIRED_COUNTERS},
                            }
                            samples.append(sample)
                            stream.write(json.dumps(sample, sort_keys=True) + "\n")
                            stream.flush()
                            sequence += 1
                    print("round={}/30 complete".format(round_index + 1), flush=True)

            for candidate_id, item in loaded.items():
                if not np.array_equal(item["buffers"][-1].numpy(), item["expected"]):
                    raise AssertionError("post-timing wrong answer: {}".format(candidate_id))
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "message": str(error)[:4000],
                "traceback": traceback.format_exc()[-12000:],
            },
        )
        raise

    results = []
    for pair in protocol["pairs"]:
        pair_rows = [row for row in samples if row["pair_id"] == pair["pair_id"]]
        control = np.asarray([row["latency_ms"] for row in pair_rows if row["role"] == "control"])
        mechanism = np.asarray([row["latency_ms"] for row in pair_rows if row["role"] == "mechanism"])
        control_median = float(np.median(control))
        mechanism_median = float(np.median(mechanism))
        improvement = (control_median - mechanism_median) / control_median
        results.append(
            {
                "pair_id": pair["pair_id"],
                "workload_id": pair["workload_id"],
                "mechanism": pair["mechanism"],
                "config_index": pair["config_index"],
                "samples_per_side": len(control),
                "control_median_ms": control_median,
                "mechanism_median_ms": mechanism_median,
                "relative_improvement": float(improvement),
                "decision": classify(improvement),
            }
        )
    final = ssh_preflight(args.host, boot_id, args.dmesg_after)
    summary = {
        "schema": "c3_p6e_same_tile_summary_v1",
        "status": "passed",
        "unique_controls": len(controls),
        "unique_mechanisms": len(mechanisms),
        "correctness_seed_checks": len(unique) * 3,
        "timing_samples": len(samples),
        "pair_results": results,
        "board_after": final,
        "samples_sha256": file_sha256(samples_path),
        "correctness_sha256": file_sha256(correctness_path),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output / "summary.json", summary)
    for row in results:
        print(
            "{} control={:.6f} mechanism={:.6f} improvement={:.3%} decision={}".format(
                row["pair_id"], row["control_median_ms"], row["mechanism_median_ms"],
                row["relative_improvement"], row["decision"]
            )
        )


if __name__ == "__main__":
    main()
