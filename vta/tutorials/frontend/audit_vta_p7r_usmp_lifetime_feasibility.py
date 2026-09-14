#!/usr/bin/env python3
"""Fail-closed audit for a unified VTA data/control lifetime graph.

This program deliberately does not infer lifetimes from aggregate allocation
snapshots or stage wall-clock intervals.  It reports whether the frozen local
evidence can support a USMP-style conflict graph spanning graph storage,
pipeline slots, command queues, FINISH, and capture/replay.  When the evidence
is incomplete it emits a machine-readable gap report and a minimal trace
contract; it does not emit synthetic packing numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SCHEMA = "c3_p7r114_usmp_lifetime_feasibility_v1"
TRACE_CONTRACT_SCHEMA = "c3_p7r114_lifetime_trace_contract_v1"
TOPOLOGIES = ("A", "B", "C", "D")
QUEUE_PREFIX = "[VTA_QUEUE] "


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_queue_records(path: Path):
    records = []
    malformed = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.startswith(QUEUE_PREFIX):
                continue
            try:
                records.append(json.loads(line[len(QUEUE_PREFIX) :]))
            except json.JSONDecodeError as err:
                malformed.append({"line": lineno, "error": str(err)})
    return records, malformed


def all_have(records, fields):
    return bool(records) and all(all(field in record for field in fields) for record in records)


def any_have(records, fields):
    return any(any(field in record for field in fields) for record in records)


def inspect_topology(topology, queue_dir: Path, static_dir: Path):
    paths = {
        "pipeline": queue_dir / f"{topology}_default_diag.jsonl",
        "allocation": queue_dir / f"{topology}_default_diag_allocation.jsonl",
        "queue": queue_dir / f"{topology}_default_diag.stderr",
        "static": static_dir / f"{topology}.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{topology}: missing evidence: {', '.join(missing)}")

    pipeline = load_jsonl(paths["pipeline"])
    allocations = load_jsonl(paths["allocation"])
    queue, malformed = load_queue_records(paths["queue"])
    static = load_json(paths["static"])
    boundaries = [edge for row in pipeline for edge in row.get("p8_boundaries", [])]
    tensor_entries = [tensor for edge in static.get("edges", []) for tensor in edge.get("tensors", [])]

    allocation_identity_fields = ("allocation_id", "resource_id", "physical_address", "size_bytes")
    lifecycle_fields = ("event_seq", "ts_ns", "event_kind", "resource_id")
    queue_link_fields = ("frame_id", "stage_id", "invocation_id")
    command_backing_fields = ("insn_resource_id", "uop_resource_id", "insn_physical_address",
                              "uop_physical_address")
    finish_fields = ("finish_count", "finish_last_index", "finish_opcode_observed")
    replay_fields = ("replay_template_id", "replay_mode", "retained_resource_ids")

    return {
        "topology": topology,
        "input_files": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                        for name, path in paths.items()},
        "counts": {
            "pipeline_frames": len(pipeline),
            "boundary_observations": len(boundaries),
            "allocation_snapshots": len(allocations),
            "queue_records": len(queue),
            "unique_queue_ids": len({record.get("queue_id") for record in queue}),
            "static_boundary_tensors": len(tensor_entries),
        },
        "observed_local_evidence": {
            "frame_and_stage_intervals": all_have(
                pipeline, ("frame_id", "stage0_start_ms", "stage0_end_ms")
            ),
            "boundary_slot_identity_generation_address_size": all_have(
                boundaries, ("slot_id", "generation", "physical_address", "boundary_bytes")
            ),
            "aggregate_allocation_high_water": all_have(
                allocations, ("phase", "high_water_bytes", "allocation_count")
            ),
            "static_storage_alias_and_users": bool(tensor_entries) and all(
                "storage_id" in tensor and "alias_entries" in tensor
                and "readers" in tensor and "writers" in tensor for tensor in tensor_entries
            ),
            "queue_submit_sizes_and_id": all_have(
                queue, ("submit", "queue_id", "insn_bytes", "uop_bytes")
            ),
        },
        "missing_for_unified_graph": {
            "per_allocation_identity_address_size": not all_have(allocations, allocation_identity_fields),
            "allocation_and_free_event_clock": not all_have(allocations, lifecycle_fields),
            "internal_tensor_access_events": not any_have(
                pipeline, ("resource_id", "read_begin", "write_begin", "last_use")
            ),
            "static_to_runtime_resource_mapping": not all(
                "runtime_resource_id" in tensor and "physical_offset" in tensor
                for tensor in tensor_entries
            ),
            "queue_to_frame_stage_invocation_link": not all_have(queue, queue_link_fields),
            "queue_backing_resource_and_physical_range": not all_have(queue, command_backing_fields),
            "device_completion_event_separate_from_blocking_duration": not all_have(
                queue, ("submit_ts_ns", "device_done_ts_ns")
            ),
            "finish_decoded_and_observed_per_submit": not all_have(queue, finish_fields),
            "replay_retained_resource_lifetime": not all_have(queue, replay_fields),
            "queue_trace_parse_errors": bool(malformed),
        },
        "queue_peak_observations_only": {
            "insn_bytes": max((record["insn_bytes"] for record in queue), default=None),
            "uop_bytes": max((record["uop_bytes"] for record in queue), default=None),
            "sum_of_independent_component_maxima_bytes": (
                max((record["insn_bytes"] for record in queue), default=0)
                + max((record["uop_bytes"] for record in queue), default=0)
            ) if queue else None,
            "interpretation": (
                "observed per-process queue components only; not a unified lifetime-packing result, "
                "not an aligned capacity, and not evidence for multiple queue instances"
            ),
        },
        "malformed_queue_records": malformed,
    }


def inspect_profile(events_path: Path, meta_path: Path):
    events = load_json(events_path)
    meta = load_json(meta_path)
    if not isinstance(events, list):
        raise ValueError("profile events must be a JSON array")
    required_link = ("frame_id", "stage_id", "queue_id", "submit_id", "resource_id")
    return {
        "input_files": {
            "events": {"path": str(events_path.resolve()), "sha256": sha256(events_path)},
            "meta": {"path": str(meta_path.resolve()), "sha256": sha256(meta_path)},
        },
        "event_count": len(events),
        "events_limit": meta.get("events_limit"),
        "is_bounded_tail_capture": (
            isinstance(meta.get("events_limit"), int)
            and len(events) >= meta["events_limit"]
        ),
        "has_request_shape": all_have(
            events, ("seq", "ts_us", "kind", "bytes", "x_size", "y_size", "x_stride")
        ),
        "has_unified_correlation_fields": all_have(events, required_link),
        "has_resource_physical_range": all_have(
            events, ("resource_id", "physical_address", "byte_offset", "byte_length")
        ),
        "interpretation": (
            "request-shape profiler tail; it is neither a complete inference trace nor a resource "
            "lifetime ledger"
        ),
    }


def inspect_replay(attestation_path: Path, negative_path: Path):
    attestation = load_json(attestation_path)
    negative = load_json(negative_path)
    runtime = attestation.get("runtime", {})
    calls = negative.get("calls", {})
    exact_manifest_fail_closed = (
        attestation.get("manifest_id") == negative.get("manifest_id")
        and runtime.get("replay_policy") == "disabled"
        and calls.get("capture", {}).get("rejected") is True
        and calls.get("replay", {}).get("rejected") is True
    )
    return {
        "input_files": {
            "runtime_attestation": {
                "path": str(attestation_path.resolve()), "sha256": sha256(attestation_path)
            },
            "negative_control": {
                "path": str(negative_path.resolve()), "sha256": sha256(negative_path)
            },
        },
        "manifest_id": attestation.get("manifest_id"),
        "policy": runtime.get("replay_policy"),
        "exact_manifest_capture_and_replay_rejected": exact_manifest_fail_closed,
        "observed_replay_execution_lifetime": False,
        "claim_boundary": (
            "passes fail-closed policy only for the exact attested W05 manifest; provides no replay "
            "execution, retained-resource, performance, cross-manifest, or cross-boot evidence"
        ),
    }


def build_trace_contract():
    return {
        "schema": TRACE_CONTRACT_SCHEMA,
        "purpose": "construct and validate a unified data/control birth-death conflict graph",
        "clock_and_completeness": {
            "required_run_header": [
                "schema", "trace_version", "trace_complete", "event_count",
                "dropped_event_count", "monotonic_clock", "process_id", "boot_id",
                "runtime_binary_sha256", "manifest_id", "topology", "pipeline_depth",
                "warmup_frames", "measured_frame_ids",
            ],
            "requirements": [
                "single monotonic event_seq with no gaps for all participating threads",
                "timestamps in one monotonic ns clock domain",
                "trace_complete=true and dropped_event_count=0",
                "all measured frames plus pipeline fill and drain are present",
            ],
        },
        "resource_declaration": {
            "required_fields": [
                "resource_id", "resource_class", "device", "owner", "allocation_id",
                "virtual_address", "physical_address", "physical_offset", "capacity_bytes",
                "alignment_bytes", "alias_group", "queue_id", "executor_id", "stage_id",
            ],
            "resource_classes": [
                "graph_storage", "parameter", "pipeline_slot", "insn_queue", "uop_queue",
                "replay_template_insn", "replay_template_uop", "other_workspace",
            ],
        },
        "event_record": {
            "required_fields": [
                "event_seq", "ts_ns", "event_kind", "frame_id", "stage_id", "executor_id",
                "invocation_id", "queue_id", "submit_id", "resource_id", "generation",
                "byte_offset", "byte_length", "access_mode",
            ],
            "event_kinds": [
                "alloc", "bind", "read_begin", "read_end", "write_begin", "write_end",
                "submit_begin", "device_done", "release", "free", "replay_capture_begin",
                "replay_capture_end", "replay_use_begin", "replay_use_end", "invalidate",
            ],
        },
        "graph_mapping": {
            "required_fields": [
                "graph_node_id", "node_entry_id", "storage_id", "alias_group", "resource_id",
                "physical_offset", "capacity_bytes", "producer_nodes", "consumer_nodes",
                "read_write_byte_ranges", "topological_order", "dynamic_shape_upper_bound",
            ]
        },
        "command_submit": {
            "required_fields": [
                "queue_id", "submit_id", "frame_id", "stage_id", "invocation_id",
                "insn_resource_id", "uop_resource_id", "insn_used_bytes", "uop_used_bytes",
                "insn_capacity_bytes", "uop_capacity_bytes", "decoded_finish_count",
                "finish_last_index", "dependency_closure_passed", "submit_ts_ns",
                "device_done_ts_ns", "timeout",
            ],
            "safety_rules": [
                "decode the finalized submitted bytes and require exactly one last FINISH",
                "resource death is no earlier than device_done, never host submit return by assumption",
                "validate instruction and UOP used ranges before either copy or device launch",
            ],
        },
        "replay": {
            "required_fields": [
                "replay_policy", "template_id", "template_hash", "source_candidate_id",
                "retained_resource_ids", "retained_insn_bytes", "retained_uop_bytes",
                "capture_event_range", "replay_event_ranges", "fallback_events",
                "invalidation_events",
            ],
            "disabled_policy_rule": (
                "when replay_policy=disabled, capture and replay APIs must reject before retaining or "
                "launching any template resource"
            ),
        },
        "lifetime_and_conflict_rules": {
            "birth": "first alloc/bind/write that makes the resource generation live",
            "death": "last read/write completion, device_done, replay invalidation, or free, whichever is latest",
            "conflict": (
                "two logical resources conflict when their closed-open live intervals overlap and they "
                "cannot legally alias the same physical byte range"
            ),
            "packing_inputs": [
                "validated resource capacities and alignments", "complete conflict edges",
                "fixed/non-relocatable allocations", "alias legality and physical pool bounds",
            ],
        },
        "minimum_collection": {
            "scope": "one frozen topology and manifest before expanding to A/B/C/D",
            "frames": "warmup plus at least pipeline_depth + 2 measured frames and complete drain",
            "trace_limit": "unbounded for this qualification run",
            "instrumentation_points": [
                "src/runtime/graph_executor/graph_executor.cc: SetupStorage, SetupOpExecs, and op invocation",
                "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc: VTAMemAlloc/VTAMemFree/GetPhyAddr",
                "vta/runtime/runtime.cc: command append/finalize, Synchronize, DeviceRun completion, capture/replay/invalidate",
                "native pipeline runner: slot acquire/release and frame/stage/invocation correlation context",
            ],
        },
        "fail_closed_conditions": [
            "missing required field or unknown resource_id",
            "event_seq gap, truncated trace, dropped event, or clock-domain ambiguity",
            "physical overlap without an explicit legal alias relation",
            "allocation without release/free semantics or access without matching begin/end",
            "submit without separately observed device_done",
            "FINISH not decoded from the exact submitted instruction bytes",
            "enabled replay without complete retained-resource and invalidation events",
            "dynamic shape exceeds its recorded bound",
        ],
        "failure_action": (
            "do not build the conflict graph and do not report naive-sum/per-queue-max/lifetime-packing "
            "savings as comparable results"
        ),
    }


def build_audit(args):
    topology_audits = [inspect_topology(t, args.queue_run_dir, args.static_audit_dir)
                       for t in TOPOLOGIES]
    profile = inspect_profile(args.profile_events, args.profile_meta)
    replay = inspect_replay(args.runtime_attestation, args.replay_negative_control)

    missing = []
    for topology in topology_audits:
        missing.extend(
            f"{topology['topology']}:{name}"
            for name, absent in topology["missing_for_unified_graph"].items() if absent
        )
    if profile["is_bounded_tail_capture"]:
        missing.append("profile:bounded_tail_capture")
    if not profile["has_unified_correlation_fields"]:
        missing.append("profile:unified_correlation_fields")
    if not profile["has_resource_physical_range"]:
        missing.append("profile:resource_physical_range")
    if not replay["observed_replay_execution_lifetime"]:
        missing.append("replay:execution_and_retained_resource_lifetime")

    can_build = not missing
    if can_build:
        raise RuntimeError(
            "evidence unexpectedly satisfies the presence audit; packing implementation is intentionally "
            "not entered without a separately reviewed semantic validator"
        )

    return {
        "schema": SCHEMA,
        "status": "insufficient_evidence_fail_closed",
        "board_contacted": False,
        "can_build_complete_conflict_graph": False,
        "decision": (
            "The frozen artifacts do not close logical resource identity, physical byte range, birth, "
            "last device use, FINISH, and replay lifetime in one correlated trace. No lifetime graph or "
            "packing estimate is produced."
        ),
        "topologies": topology_audits,
        "request_profile": profile,
        "replay_and_finish": {
            "replay": replay,
            "finish": {
                "status": "unresolved_for_lifetime_graph",
                "observed_in_queue_trace": False,
                "available_evidence": (
                    "runtime source appends/checks FINISH and prior command certificates derive one FINISH; "
                    "the frozen queue JSON does not decode/count FINISH from each submitted byte stream"
                ),
            },
        },
        "blocking_gaps": sorted(set(missing)),
        "requested_comparison": {
            "naive_sum": {
                "status": "not_computable",
                "reason": (
                    "allocation snapshots contain aggregate high-water/count only, not a complete logical "
                    "resource ledger; observed high-water is not naive logical sum"
                ),
            },
            "per_queue_max": {
                "status": "partially_supported_existing_scope_only",
                "reason": (
                    "queue diagnostics support maxima for one queue instance per process in A/B/C/D, but "
                    "lack backing ranges and cross-resource lifetimes required for unified comparison"
                ),
                "observations": {
                    item["topology"]: item["queue_peak_observations_only"]
                    for item in topology_audits
                },
            },
            "lifetime_packing": {
                "status": "not_computable",
                "reason": (
                    "no complete birth/death/access/device-done trace or static-to-runtime physical mapping"
                ),
            },
        },
        "fail_closed_checks": {
            "replay_disabled_exact_manifest": replay["exact_manifest_capture_and_replay_rejected"],
            "finish_observed_per_submit": False,
            "packing_withheld": True,
            "overall": "passed_fail_closed",
        },
        "claim_boundary": (
            "offline feasibility audit of named frozen artifacts only; no board contact, no new memory "
            "measurement, no USMP implementation, no unified peak-memory reduction, and no performance claim"
        ),
    }


def render_gap_report(audit):
    rows = []
    for topology in audit["topologies"]:
        count = topology["counts"]
        peak = topology["queue_peak_observations_only"]
        rows.append(
            f"| {topology['topology']} | {count['pipeline_frames']} | {count['queue_records']} | "
            f"{count['unique_queue_ids']} | {peak['insn_bytes']} | {peak['uop_bytes']} |"
        )
    gaps = "\n".join(f"- `{gap}`" for gap in audit["blocking_gaps"])
    table = "\n".join(rows)
    return f"""# P7R114 USMP 式生命周期图可行性审计

## 结论

状态：`{audit['status']}`。现有冻结证据不足以构建跨 graph storage、pipeline slot、
command queue、FINISH 与 replay 的统一 birth-death conflict graph。因此本轮按预注册规则
fail closed：不生成 lifetime packing 数字，也不把聚合高水位改名为 naive sum。

现有数据能证明的是：帧/阶段时间区间、边界 slot 的地址/代际/大小、每进程一个被观测队列的
submit 大小峰值、静态 storage/alias/reader/writer 关系，以及若干阶段性的聚合内存高水位。
它们尚未由统一 resource_id、物理 byte range 和 device completion 事件串起来。

## 可保留的局部观测

| topology | frames | queue records | observed queue IDs | max insn B | max UOP B |
|---|---:|---:|---:|---:|---:|
{table}

这些峰值只是已有单进程 queue trace 的 component maxima；不是对齐后的容量，不代表多个队列，
也不是 data/control 统一 lifetime packing 结果。

## 三种比较的状态

- naive sum：`not_computable`。snapshot 只有 aggregate high-water/count，没有逐资源 ledger；
  high-water 不是 logical-size naive sum。
- per-queue max：`partially_supported_existing_scope_only`。A/B/C/D 各只观测到一个 queue_id，
  可以复核该队列的 insn/UOP submit 峰值，但无法同 graph/slot/replay 资源统一比较。
- lifetime packing：`not_computable`。缺少完整 birth/death/access/device-done 与
  static-to-runtime physical mapping。

## 阻塞缺口

{gaps}

## FINISH 与 replay

- FINISH：runtime 源码确实在正常提交前追加并检查最后一条 FINISH，已有证书也据此推导数量；
  但冻结的 `[VTA_QUEUE]` JSON 没有从每批实际提交字节流解码得到 FINISH 数量/末位置，故不能作为
  生命周期图的逐批观测证据。
- replay：P7R107/P7R108 对同一 manifest 证明 `replay_policy=disabled`，capture/replay 调用均被
  拒绝。这只支持该 exact manifest 的 fail-closed 安全主张；没有 replay execution 或 retained
  resource lifetime 证据。

## 下一步最小埋点

完整字段、事件类型、失败条件与建议埋点位置见 `instrumentation_contract.json`。最小资格运行先只做
一个冻结 topology/manifest：warmup 后覆盖至少 `pipeline_depth + 2` 帧及完整 drain，trace 不限长，
要求事件序号无缺口、零 dropped event。任何字段缺失、FINISH 未逐批解码或 device_done 未独立记录，
都继续 fail closed。

## 主张边界

{audit['claim_boundary']}。
"""


def write_outputs(output_dir: Path, audit, contract, command):
    if output_dir.exists():
        raise FileExistsError(f"immutable output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    files = {
        "audit.json": json.dumps(audit, indent=2, sort_keys=True) + "\n",
        "instrumentation_contract.json": json.dumps(contract, indent=2, sort_keys=True) + "\n",
        "GAP_REPORT.md": render_gap_report(audit),
        "STATUS.md": (
            "# P7R114 status\n\n"
            "- Status: `insufficient_evidence_fail_closed`\n"
            "- Complete conflict graph: no\n"
            "- Packing numbers emitted: no\n"
            "- Replay exact-manifest fail-closed check: passed\n"
            "- FINISH observed per submitted stream: no\n"
            "- Board contacted: no\n"
        ),
        "command.txt": command.rstrip() + "\n",
    }
    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    hashes = {name: sha256(output_dir / name) for name in sorted(files)}
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"schema": "c3_immutable_artifact_hashes_v1", "files": hashes},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv=None):
    frontend = Path(__file__).resolve().parent
    report = frontend / "report_out" / "stage_tile_cotuning"
    c3 = report / "c3_dma_residency_autotune" / "07_grouped_holdout"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-run-dir", type=Path,
                        default=report / "queue_capacity" / "q0_board_run3")
    parser.add_argument("--static-audit-dir", type=Path,
                        default=report / "c3s_buffer_reuse" / "s0_static_run1")
    parser.add_argument("--profile-events", type=Path, default=(
        report / "stage_memory_experiments" / "boot_9efb07ba" / "rank01" /
        "profile" / "pipeline" / "benchmark_totals_events.json"))
    parser.add_argument("--profile-meta", type=Path, default=(
        report / "stage_memory_experiments" / "boot_9efb07ba" / "rank01" /
        "profile" / "pipeline" / "benchmark_totals_meta.json"))
    parser.add_argument("--runtime-attestation", type=Path, default=(
        c3 / "20260911_p7r107_w05_runtime_manifest_attestation_run01" /
        "runtime_manifest_attestation.json"))
    parser.add_argument("--replay-negative-control", type=Path, default=(
        c3 / "20260911_p7r108_w05_manifest_replay_policy_fsim_run01" /
        "replay_policy_negative_control.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    audit = build_audit(args)
    contract = build_trace_contract()
    command = " ".join([Path(sys.executable).name, Path(__file__).name] + sys.argv[1:])
    write_outputs(args.output_dir, audit, contract, command)
    print(json.dumps({
        "status": audit["status"],
        "can_build_complete_conflict_graph": audit["can_build_complete_conflict_graph"],
        "blocking_gap_count": len(audit["blocking_gaps"]),
        "output_dir": str(args.output_dir.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
