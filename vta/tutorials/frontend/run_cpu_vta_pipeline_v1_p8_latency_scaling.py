#!/usr/bin/env python3
"""Measure serial ordinary-copy versus zero-copy latency for alternating CPU/VTA stages."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shlex
import shutil
import statistics
from pathlib import Path

import numpy as np

import run_cpu_vta_pipeline_v1_p8a as p8
from freeze_cpu_vta_pipeline_v1 import file_sha256, seal_artifact


MODE_ORDER = ("L0", "L1", "L1", "L0")
REPORT_ROOT = p8.REPORT_ROOT / "v1_p8_latency_island_scaling"
DEFAULT_PACKAGE = REPORT_ROOT / "three_island_build/package"
COPY_POLICIES = ("runner_default", "safe_byte_copy", "memcpy")
LEGACY_MEMCPY_SESSION_ARTIFACTS = {
    "6b4c8953091615fea7a15b7143a6f309a43a8f9e915a8e4becba44a659533293"
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-host", default="192.168.1.247")
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument(
        "--output-dir", default=str(REPORT_ROOT / "three_island_sessions_runner_default")
    )
    parser.add_argument("--remote-root", default="/var/volatile/ramps_p8_latency")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs-per-block", type=int, default=30)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument(
        "--ordinary-copy-policy",
        choices=COPY_POLICIES,
        default="runner_default",
        help=(
            "L0 copy implementation. runner_default leaves AXU5EVB_DRIVER_SAFE_COPY "
            "unset and therefore follows the native runner default."
        ),
    )
    return parser.parse_args()


def load_manifest(package):
    payload = json.loads((Path(package) / "manifest.json").read_text(encoding="utf-8"))
    stages = payload.get("stages", [])
    if len(stages) < 3 or len(stages) % 2 == 0:
        raise ValueError("latency scaling requires an odd number of alternating stages")
    for edge in range(len(stages) - 1):
        boundary_contract(payload, edge)
    return payload


def island_count(manifest):
    return sum(stage["device"] == "vta" for stage in manifest["stages"])


def boundary_contract(manifest, edge_index):
    producer = manifest["stages"][edge_index]
    consumer = manifest["stages"][edge_index + 1]
    direction = "{}_to_{}".format(producer["device"], consumer["device"])
    if direction not in {"cpu_to_vta", "vta_to_cpu"}:
        raise ValueError("edge {} is not a CPU/VTA boundary".format(edge_index))
    left = producer["output_schema"]["slots"]
    right = consumer["input_schema"]["slots"]
    if not left or len(left) != len(right):
        raise ValueError("edge {} has incompatible tensor arity".format(edge_index))
    tensors = []
    for tensor_index, (source, destination) in enumerate(zip(left, right)):
        source_contract = (str(source["dtype"]), tuple(int(x) for x in source["shape"]))
        destination_contract = (
            str(destination["dtype"]),
            tuple(int(x) for x in destination["shape"]),
        )
        if source_contract != destination_contract:
            raise ValueError("edge {} tensor {} contract differs".format(edge_index, tensor_index))
        bits = int("".join(ch for ch in source_contract[0] if ch.isdigit()))
        elements = math.prod(source_contract[1])
        tensors.append(
            {
                "tensor_index": tensor_index,
                "dtype": source_contract[0],
                "shape": list(source_contract[1]),
                "bytes": elements * bits // 8,
            }
        )
    total_bytes = sum(tensor["bytes"] for tensor in tensors)
    if total_bytes != int(producer["output_bytes"]):
        raise ValueError("edge {} output byte count is inconsistent".format(edge_index))
    return {
        "producer_stage": edge_index,
        "consumer_stage": edge_index + 1,
        "direction": direction,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "bytes": total_bytes,
    }


def graph_boundary_contract(package, manifest, edge_index):
    producer_stage = manifest["stages"][edge_index]
    consumer_stage = manifest["stages"][edge_index + 1]
    producer = json.loads((Path(package) / producer_stage["graph"]).read_text(encoding="utf-8"))
    consumer = json.loads((Path(package) / consumer_stage["graph"]).read_text(encoding="utf-8"))
    contract = boundary_contract(manifest, edge_index)
    if len(producer["heads"]) != contract["tensor_count"]:
        raise ValueError("producer graph head count differs from manifest")
    if len(consumer_stage["input_names"]) != contract["tensor_count"]:
        raise ValueError("consumer graph input count differs from manifest")
    observed = []
    for tensor_index in range(contract["tensor_count"]):
        producer_entry = p8._graph_entry_contract(producer, producer["heads"][tensor_index])
        consumer_entry = p8._graph_entry_contract(
            consumer, [consumer["arg_nodes"][tensor_index], 0, 0]
        )
        expected = contract["tensors"][tensor_index]
        observed.append(
            {
                "tensor_index": tensor_index,
                "producer": producer_entry,
                "consumer": consumer_entry,
                "shape_dtype_match": (
                    producer_entry["shape"] == expected["shape"]
                    and consumer_entry["shape"] == expected["shape"]
                    and producer_entry["dtype"] == expected["dtype"]
                    and consumer_entry["dtype"] == expected["dtype"]
                ),
                "device_views_match": (
                    producer_entry["device_index"], consumer_entry["device_index"]
                )
                == ((1, 12) if contract["direction"] == "cpu_to_vta" else (12, 1)),
            }
        )
    return observed


def build_protocol(
    manifest, warmup=10, runs_per_block=30, ordinary_copy_policy="runner_default"
):
    if ordinary_copy_policy not in COPY_POLICIES:
        raise ValueError("unknown ordinary copy policy {}".format(ordinary_copy_policy))
    edges = [
        boundary_contract(manifest, edge)
        for edge in range(len(manifest["stages"]) - 1)
    ]
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_serial_latency_protocol",
            "phase": "P8D-latency",
            "candidate_id": manifest["candidate_id"],
            "vta_island_count": island_count(manifest),
            "stage_count": len(manifest["stages"]),
            "heterogeneous_edges": edges,
            "modes": {
                "L0": "serial_ordinary_set_get_materialization",
                "L1": "serial_managed_single_slot_zero_copy",
            },
            "ordinary_copy_policy": ordinary_copy_policy,
            "measurement": {
                "block_order": list(MODE_ORDER),
                "warmup_per_block": int(warmup),
                "scored_frames_per_block": int(runs_per_block),
                "scored_frames_per_mode": int(2 * runs_per_block),
                "alternating_distinct_inputs": 2,
                "primary_metric": "per_frame_end_to_end_total_latency_ms",
                "session_estimator": "median_of_all_scored_frames_per_mode",
                "same_serial_schedule_in_both_modes": True,
            },
            "ownership": {
                "L0_materialization_bytes_per_frame": 2
                * sum(edge["bytes"] for edge in edges),
                "L1_materialization_bytes_per_frame": 0,
                "vta_internal_load_store_not_eliminated": True,
                "dual_slot_overlap_not_measured": True,
            },
            "formal_claim_requires": (
                "at_least_three_independent_boots, all functional gates pass, and paired "
                "boot latency-reduction confidence interval excludes zero"
            ),
            "candidate_throughput_used_for_fit": False,
        }
    )


def build_local_audit(package, manifest, runner_binary):
    edge_audits = []
    for edge in range(len(manifest["stages"]) - 1):
        contract = boundary_contract(manifest, edge)
        graph_tensors = graph_boundary_contract(package, manifest, edge)
        edge_audits.append({"edge": contract, "graph_tensors": graph_tensors})
    checks = {
        "all_boundaries_are_cpu_vta": all(item["edge"]["direction"] in {"cpu_to_vta", "vta_to_cpu"} for item in edge_audits),
        "all_boundary_graph_contracts_pass": all(tensor["shape_dtype_match"] for item in edge_audits for tensor in item["graph_tensors"]),
        "all_boundary_device_views_pass": all(tensor["device_views_match"] for item in edge_audits for tensor in item["graph_tensors"]),
        "runner_binary_present": Path(runner_binary).is_file(),
        "single_slot_managed_mode_present": "run_managed_serial_frame" in p8.RUNNER_SOURCE.read_text(encoding="utf-8"),
        "ordinary_serial_mode_present": "run_serial_frame" in p8.RUNNER_SOURCE.read_text(encoding="utf-8"),
    }
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_serial_latency_local_audit",
            "candidate_id": manifest["candidate_id"],
            "runner_source_sha256": file_sha256(p8.RUNNER_SOURCE),
            "runner_binary_sha256": file_sha256(runner_binary),
            "edge_audits": edge_audits,
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def make_second_input(package, manifest, output):
    shape = tuple(int(value) for value in manifest["input_shape"])
    source = np.fromfile(Path(package) / manifest["input_file"], dtype=np.float32)
    if source.size != int(np.prod(shape)):
        raise ValueError("input.bin does not match manifest shape")
    shifted = np.roll(source.reshape(shape), shift=1, axis=-1).copy()
    if np.array_equal(source, shifted.reshape(-1)):
        raise ValueError("generated second input is not distinct")
    shifted.tofile(output)


def runner_args(manifest, mode, warmup, runs, output_name, profile_dir):
    if mode not in {"L0", "L1"}:
        raise ValueError("unknown latency mode {}".format(mode))
    result = []
    threads = manifest["serial_stage_runtime_threads"]
    for index, stage in enumerate(manifest["stages"]):
        prefix = "--stage{}-".format(index)
        result.extend(
            [
                prefix + "graph", stage["graph"],
                prefix + "lib", stage["lib"],
                prefix + "params", stage["params"],
                prefix + "input-names", ",".join(stage["input_names"]),
                prefix + "name", stage["name"],
                prefix + "device", stage["device"],
                prefix + "runtime-num-threads", str(threads["stage{}".format(index)]),
            ]
        )
        affinity = manifest["stage_cpu_affinity"]["serial"]["stage{}".format(index)]
        if affinity:
            result.extend([prefix + "cpu-affinity", ",".join(str(value) for value in affinity)])
    result.extend(
        [
            "--input-list", "p8_inputs.txt",
            "--runs", str(runs),
            "--warmup-runs", str(warmup),
            "--queue-depth", "2",
            "--runtime-num-threads", "4",
            "--output-mode", "raw",
            "--output-jsonl", output_name,
            "--vta-runtime-profile-dir", profile_dir,
            "--serial",
        ]
    )
    if mode == "L1":
        result.extend(["--p8-managed-slots", "1"])
    return result


def _output_signature(row):
    return tuple(item["fnv1a64"] for item in row["raw_outputs"])


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ranges_non_overlapping(rows):
    observed = {}
    for row in rows:
        for boundary in row.get("p8_boundaries", []):
            addresses = boundary.get("physical_addresses", [boundary["physical_address"]])
            sizes = boundary.get("tensor_bytes", [boundary["boundary_bytes"]])
            if len(addresses) != len(sizes):
                return False
            for tensor_index, (address, size) in enumerate(zip(addresses, sizes)):
                key = (
                    int(boundary["edge_index"]),
                    int(boundary["slot_id"]),
                    tensor_index,
                )
                value = (int(address, 16), int(size))
                if key in observed and observed[key] != value:
                    return False
                observed[key] = value
    ranges = sorted((start, start + size) for start, size in observed.values())
    return all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


def summarize_mode(mode, blocks, manifest):
    stage_count = len(manifest["stages"])
    edge_count = stage_count - 1
    contracts = [boundary_contract(manifest, edge) for edge in range(edge_count)]
    rows = [row for block in blocks for row in block["rows"]]
    latencies = [float(row["total_latency_ms"]) for row in rows]
    edge_services = {}
    boundary_totals = [0.0] * len(rows)
    for edge, contract in enumerate(contracts):
        values = [
            float(row["stage{}_get_ms".format(edge)])
            + float(row["stage{}_set_ms".format(edge + 1)])
            for row in rows
        ]
        edge_services[str(edge)] = {
            "direction": contract["direction"],
            "bytes": contract["bytes"],
            "service_ms_median": statistics.median(values),
        }
        boundary_totals = [left + right for left, right in zip(boundary_totals, values)]
    expected_materialization = 2 * sum(item["bytes"] for item in contracts)
    profile_keys = (
        "mem_copy_from_host_bytes",
        "mem_copy_to_host_bytes",
        "load_buffer_2d_bytes",
        "store_buffer_2d_bytes",
        "driver_run_insns",
        "driver_submit_mmio_us",
        "driver_poll_wait_us",
        "synchronize_calls",
    )
    summary = {
        "mode": mode,
        "scored_frames": len(rows),
        "input_indices": sorted({int(row["input_index"]) for row in rows}),
        "output_signatures_by_input": {
            str(index): sorted({_output_signature(row) for row in rows if int(row["input_index"]) == index})
            for index in (0, 1)
        },
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95": _percentile(latencies, 95),
        "block_latency_ms_medians": [statistics.median(float(row["total_latency_ms"]) for row in block["rows"]) for block in blocks],
        "boundary_api_service_ms_median": statistics.median(boundary_totals),
        "boundary_api_service_by_edge": edge_services,
        "framework_materialization_bytes_per_frame": expected_materialization if mode == "L0" else max(int(row["p8_framework_materialization_bytes"]) for row in rows),
        "stage_service_ms_median": {
            "stage{}".format(index): statistics.median(float(row["stage{}_ms".format(index)]) for row in rows)
            for index in range(stage_count)
        },
        "vta_profile_per_scored_frame": {
            key: sum(float(block["profile"].get(key, 0)) for block in blocks) / len(rows)
            for key in profile_keys
        },
    }
    if mode == "L1":
        boundaries = [boundary for row in rows for boundary in row["p8_boundaries"]]
        summary.update(
            {
                "every_frame_has_all_edges": all(len(row["p8_boundaries"]) == edge_count for row in rows),
                "slot_ids_by_edge": {
                    str(edge): sorted({int(item["slot_id"]) for item in boundaries if int(item["edge_index"]) == edge})
                    for edge in range(edge_count)
                },
                "slot_addresses_aligned": all(int(item["physical_address"], 16) % 256 == 0 for item in boundaries),
                "slot_physical_ranges_non_overlapping": all(_ranges_non_overlapping(block["rows"]) for block in blocks),
                "slot_wait_ms_median": statistics.median(float(row["p8_total_slot_wait_ms"]) for row in rows),
            }
        )
    return summary


def _copy_policy_environment(policy):
    if policy == "runner_default":
        return ""
    if policy == "safe_byte_copy":
        return "AXU5EVB_DRIVER_SAFE_COPY=1 "
    if policy == "memcpy":
        return "AXU5EVB_DRIVER_SAFE_COPY=0 "
    raise ValueError("unknown ordinary copy policy {}".format(policy))


def _run_block(args, manifest, mode, occurrence, remote):
    name = "{}_{}".format(mode.lower(), occurrence)
    output_name = name + ".jsonl"
    profile_dir = "profile_" + name
    command = (
        "cd {root} && export LD_LIBRARY_PATH=$PWD && "
        "export LD_PRELOAD=$PWD/libtvm_runtime.so:$PWD/libvta.so && "
        "export TVM_NUM_THREADS=4 TVM_THREAD_POOL_SPIN_COUNT=0 && "
        "export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 "
        "AXU5EVB_DRIVER_POLL_SLEEP_NS=1000 {copy_policy}&& "
        "./vta_stage_pipeline_runner_latency {runner} && cat {output}"
    ).format(
        root=shlex.quote(remote),
        copy_policy=_copy_policy_environment(args.ordinary_copy_policy),
        runner=shlex.join(runner_args(manifest, mode, args.warmup, args.runs_per_block, output_name, profile_dir)),
        output=shlex.quote(output_name),
    )
    stdout = p8._ssh(args, command)
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    if len(rows) != args.runs_per_block:
        raise RuntimeError("{} returned {} scored rows".format(name, len(rows)))
    profile = json.loads(p8._ssh(args, "cat {}/profile_{}/benchmark_totals_status.json".format(shlex.quote(remote), name)))
    return {"mode": mode, "occurrence": occurrence, "rows": rows, "profile": profile}


def run_board(args, manifest, runner_binary, work_dir, output_dir):
    from collect_cpu_vta_pipeline_v1_preflight import collect as collect_preflight

    preflight_args = argparse.Namespace(board_host=args.board_host, rpc_port=args.rpc_port, ssh_user=args.ssh_user, timeout_s=min(args.timeout_s, 30))
    before = collect_preflight(preflight_args)
    if not before["passed"]:
        raise RuntimeError("latency experiment refused because board preflight failed")
    remote = args.remote_root.rstrip("/")
    archive = work_dir / "package.tar.gz"
    second_input = work_dir / "input_b.bin"
    input_list = work_dir / "p8_inputs.txt"
    make_second_input(args.package, manifest, second_input)
    input_list.write_text("input.bin\ninput_b.bin\n", encoding="ascii")
    p8._archive_package(args.package, archive)
    p8._ssh(args, "rm -rf {0} && mkdir -p {0}".format(shlex.quote(remote)))
    p8._scp(args, archive, remote + "/package.tar.gz")
    p8._ssh(args, "cd {0} && tar -xzf package.tar.gz".format(shlex.quote(remote)))
    p8._scp(args, runner_binary, remote + "/vta_stage_pipeline_runner_latency")
    p8._scp(args, second_input, remote + "/input_b.bin")
    p8._scp(args, input_list, remote + "/p8_inputs.txt")
    p8._ssh(args, "cd {0} && chmod 700 vta_stage_pipeline_runner_latency".format(shlex.quote(remote)))

    properties = p8._board_properties(args)
    blocks = []
    occurrences = {"L0": 0, "L1": 0}
    raw_artifacts = []
    for order_index, mode in enumerate(MODE_ORDER):
        occurrence = occurrences[mode]
        occurrences[mode] += 1
        block = _run_block(args, manifest, mode, occurrence, remote)
        block["order_index"] = order_index
        raw_name = "p8_latency_raw_{}_{}_{}.jsonl".format(mode.lower(), occurrence, properties["boot_id"])
        raw_path = Path(output_dir) / raw_name
        raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in block["rows"]), encoding="utf-8")
        profile_name = "p8_latency_profile_{}_{}_{}.json".format(
            mode.lower(), occurrence, properties["boot_id"]
        )
        profile_path = Path(output_dir) / profile_name
        p8.write_json(profile_path, block["profile"])
        raw_artifacts.append(
            {
                "mode": mode,
                "occurrence": occurrence,
                "path": raw_name,
                "sha256": file_sha256(raw_path),
                "profile_path": profile_name,
                "profile_sha256": file_sha256(profile_path),
            }
        )
        blocks.append(block)

    summaries = {mode: summarize_mode(mode, [block for block in blocks if block["mode"] == mode], manifest) for mode in ("L0", "L1")}
    l0, l1 = summaries["L0"], summaries["L1"]
    after = collect_preflight(preflight_args)
    edge_count = len(manifest["stages"]) - 1
    gate = {
        "outputs_match_ordinary_copy": l1["output_signatures_by_input"] == l0["output_signatures_by_input"],
        "alternating_inputs_present": l0["input_indices"] == [0, 1] and l1["input_indices"] == [0, 1],
        "zero_copy_has_every_edge": l1["every_frame_has_all_edges"],
        "single_slot_per_edge": l1["slot_ids_by_edge"] == {str(edge): [0] for edge in range(edge_count)},
        "slot_addresses_aligned": l1["slot_addresses_aligned"],
        "slot_ranges_non_overlapping": l1["slot_physical_ranges_non_overlapping"],
        "zero_copy_materialization_is_zero": l1["framework_materialization_bytes_per_frame"] == 0,
        "ordinary_materialization_is_nonzero": l0["framework_materialization_bytes_per_frame"] > 0,
        "preflight_before_and_after_passed": bool(before["passed"] and after["passed"]),
        "storage_healthy": properties.get("storage_errors") == "0",
    }
    gate["passed"] = all(gate.values())
    reduction_ms = l0["latency_ms_median"] - l1["latency_ms_median"]
    paired_block_reductions = [
        left - right
        for left, right in zip(
            l0["block_latency_ms_medians"], l1["block_latency_ms_medians"]
        )
    ]
    session = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_serial_latency_session",
            "phase": "P8D-latency",
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "board": "{}@{}".format(args.ssh_user, args.board_host),
            "boot_id": properties["boot_id"],
            "candidate_id": manifest["candidate_id"],
            "vta_island_count": island_count(manifest),
            "ordinary_copy_policy": args.ordinary_copy_policy,
            "board_properties": properties,
            "preflight_before": before,
            "preflight_after": after,
            "mode_order": list(MODE_ORDER),
            "mode_summaries": summaries,
            "effects": {
                "latency_reduction_ms": reduction_ms,
                "latency_reduction_percent": 100.0 * reduction_ms / l0["latency_ms_median"],
                "boundary_api_service_reduction_ms": l0["boundary_api_service_ms_median"] - l1["boundary_api_service_ms_median"],
                "framework_materialization_bytes_eliminated_per_frame": l0["framework_materialization_bytes_per_frame"],
                "paired_block_latency_reductions_ms": paired_block_reductions,
                "all_paired_block_reductions_positive": all(
                    value > 0 for value in paired_block_reductions
                ),
                "vta_internal_load_bytes_delta_per_frame": l1[
                    "vta_profile_per_scored_frame"
                ]["load_buffer_2d_bytes"]
                - l0["vta_profile_per_scored_frame"]["load_buffer_2d_bytes"],
                "vta_internal_store_bytes_delta_per_frame": l1[
                    "vta_profile_per_scored_frame"
                ]["store_buffer_2d_bytes"]
                - l0["vta_profile_per_scored_frame"]["store_buffer_2d_bytes"],
            },
            "raw_artifacts": raw_artifacts,
            "single_boot_gate": gate,
            "formal_latency_claim_allowed": False,
        }
    )
    if not args.keep_remote:
        p8._ssh(args, "rm -rf {}".format(shlex.quote(remote)))
    return session


def paired_boot_ci(values):
    values = [float(value) for value in values]
    if len(values) < 3:
        return None
    critical = {2: 4.302653, 3: 3.182446, 4: 2.776445}
    df = len(values) - 1
    if df not in critical:
        raise ValueError("cross-boot summary supports three to five boots")
    mean = statistics.mean(values)
    margin = critical[df] * statistics.stdev(values) / math.sqrt(len(values))
    return {"lower": mean - margin, "upper": mean + margin, "confidence": 0.95, "method": "paired_boot_student_t_95_v1"}


def _session_copy_policy(payload):
    policy = payload.get("ordinary_copy_policy")
    if policy in COPY_POLICIES:
        return policy
    if payload.get("artifact_sha256") in LEGACY_MEMCPY_SESSION_ARTIFACTS:
        return "memcpy"
    return None


def load_sessions(output_dir, candidate_id, ordinary_copy_policy="runner_default"):
    result = {}
    for path in Path(output_dir).glob("p8_latency_session_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("candidate_id") == candidate_id
            and _session_copy_policy(payload) == ordinary_copy_policy
        ):
            result[str(payload["boot_id"])] = payload
    return sorted(result.values(), key=lambda item: item.get("observed_at_utc", ""))


def build_cross_boot_summary(sessions):
    rows = [
        {
            "boot_id": item["boot_id"],
            "L0_latency_ms": item["mode_summaries"]["L0"]["latency_ms_median"],
            "L1_latency_ms": item["mode_summaries"]["L1"]["latency_ms_median"],
            "latency_reduction_ms": item["effects"]["latency_reduction_ms"],
            "latency_reduction_percent": item["effects"]["latency_reduction_percent"],
            "functional_gate_passed": item["single_boot_gate"]["passed"],
        }
        for item in sessions
    ]
    reductions = [row["latency_reduction_ms"] for row in rows]
    ci = paired_boot_ci(reductions)
    formal = bool(len(rows) >= 3 and all(row["functional_gate_passed"] for row in rows) and ci and ci["lower"] > 0)
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p8_serial_latency_cross_boot_summary",
            "independent_boot_count": len(rows),
            "boot_rows": rows,
            "aggregate": {
                "latency_reduction_ms_mean": statistics.mean(reductions) if reductions else None,
                "latency_reduction_percent_mean": statistics.mean(row["latency_reduction_percent"] for row in rows) if rows else None,
                "latency_reduction_ms_mean_ci95": ci,
            },
            "formal_latency_claim_allowed": formal,
        }
    )


def write_review(path, audit, sessions, cross_boot):
    lines = [
        "# P8 串行单帧时延审查",
        "",
        "更新日期：{}".format(datetime.date.today().isoformat()),
        "",
        "## 实验定义",
        "",
        "L0 与 L1 使用完全相同的逐 stage 串行调度。L0 使用普通 `get_output/set_input` 物化，L1 使用单 slot zero-copy；本实验不使用双 slot，也不把流水线重叠算作收益。",
        "",
        "L0 普通复制策略：`{}`。".format(
            _session_copy_policy(sessions[-1]) if sessions else "尚未执行"
        ),
        "",
        "本地审计：`{}`。".format("通过" if audit["passed"] else "失败"),
    ]
    if sessions:
        latest = sessions[-1]
        l0 = latest["mode_summaries"]["L0"]
        l1 = latest["mode_summaries"]["L1"]
        lines.extend(
            [
                "",
                "## 最新 Boot",
                "",
                "- boot：`{}`；VTA island：`{}`；功能 gate：`{}`。".format(latest["boot_id"], latest["vta_island_count"], "通过" if latest["single_boot_gate"]["passed"] else "失败"),
                "- L0/L1 单帧中位时延：`{:.3f}/{:.3f} ms`。".format(l0["latency_ms_median"], l1["latency_ms_median"]),
                "- 单帧时延差：`{:.3f} ms`（`{:.2f}%`）。".format(latest["effects"]["latency_reduction_ms"], latest["effects"]["latency_reduction_percent"]),
                "- 边界 API 服务：`{:.3f} -> {:.3f} ms`；每帧消除物化字节：`{}`。".format(l0["boundary_api_service_ms_median"], l1["boundary_api_service_ms_median"], latest["effects"]["framework_materialization_bytes_eliminated_per_frame"]),
                "",
                "该结果来自一个 boot，只是 qualification；至少三个独立 boot 且配对置信区间排除 0 后，才能正式声称单帧时延提升。",
            ]
        )
    else:
        lines.extend(["", "## 板端状态", "", "尚未执行板端 qualification。"])
    lines.extend(["", "独立 boot 数：`{}`；正式时延声明：`{}`。".format(cross_boot["independent_boot_count"], "允许" if cross_boot["formal_latency_claim_allowed"] else "不允许")])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.warmup < 0 or args.runs_per_block < 2:
        raise ValueError("warmup must be non-negative and runs-per-block at least two")
    package = Path(args.package).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/ramps_p8_latency")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    runner_binary = work_dir / "vta_stage_pipeline_runner_latency"
    p8.compile_runner(runner_binary)
    manifest = load_manifest(package)
    protocol = build_protocol(
        manifest, args.warmup, args.runs_per_block, args.ordinary_copy_policy
    )
    audit = build_local_audit(package, manifest, runner_binary)
    p8.write_json(output_dir / "p8_latency_protocol.json", protocol)
    p8.write_json(output_dir / "p8_latency_local_audit.json", audit)
    session = None
    if not args.local_only:
        session = run_board(args, manifest, runner_binary, work_dir, output_dir)
        p8.write_json(output_dir / "p8_latency_session_{}.json".format(session["boot_id"]), session)
    sessions = load_sessions(
        output_dir, manifest["candidate_id"], args.ordinary_copy_policy
    )
    cross_boot = build_cross_boot_summary(sessions)
    p8.write_json(output_dir / "p8_latency_cross_boot_summary.json", cross_boot)
    write_review(output_dir / "p8_latency_review.md", audit, sessions, cross_boot)
    print(json.dumps({"protocol": protocol, "audit": audit, "session": session, "cross_boot": cross_boot}, indent=2))


if __name__ == "__main__":
    main()
