#!/usr/bin/env python3
"""Recompute qualification and timing from the paired experiment's raw files."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile

from thesis_c1_memory_ablation import file_hash, finite, read_json, verify_ref, verify_seal
from thesis_e2e_ablation_protocol import window_metrics


class NumericMismatch(ValueError):
    """A retained configuration failed its common-reference numerical gate."""


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inputs_manifest(contract):
    import numpy as np
    from prepare_thesis_c1_memory_inputs import PREPROCESSING, preprocess
    ref = verify_ref(contract["artifacts"]["inputs"])
    data = read_json(ref["path"])
    verify_seal(data)
    root = Path(ref["path"]).parent
    if (data.get("schema") != "c1_memory_ablation_inputs_v1" or
            data.get("canonical_qir_sha256") != contract["artifacts"]["canonical_qir"]["sha256"]):
        raise ValueError("input/reference canonical graph mismatch")
    if len(data.get("real_inputs", [])) != 64 or len(data.get("diagnostic_inputs", [])) != 3:
        raise ValueError("64 real inputs and three diagnostic inputs required")
    if {r.get("kind") for r in data["diagnostic_inputs"]} != {"zeros", "uniform", "normal"}:
        raise ValueError("diagnostic input kinds mismatch")
    if not data.get("preprocessing") or not data.get("reference_artifacts"):
        raise ValueError("preprocessing and independent reference artifact identity required")
    if any(data["preprocessing"].get(k) != v for k, v in PREPROCESSING.items()):
        raise ValueError("preprocessing differs from the frozen existing image protocol")
    for source in data["reference_artifacts"]:
        verify_ref(source, root)
    records = data["real_inputs"] + data["diagnostic_inputs"]
    image_hashes, input_hashes, ids = set(), set(), set()
    for index, row in enumerate(records):
        if not row.get("id") or row["id"] in ids:
            raise ValueError("input identities must be unique")
        ids.add(row["id"])
        row["input"] = verify_ref(row["input"], root)
        if Path(row["input"]["path"]).stat().st_size != 1 * 3 * 224 * 224 * 4:
            raise ValueError("input must be packed NCHW float32 [1,3,224,224]")
        value = np.fromfile(row["input"]["path"], dtype="float32").reshape(1, 3, 224, 224)
        if not np.isfinite(value).all() or row["input"]["sha256"] in input_hashes:
            raise ValueError("nonfinite or duplicate frozen model input")
        input_hashes.add(row["input"]["sha256"])
        if index < 64:
            row["source_image"] = verify_ref(row["source_image"], root)
            image_hashes.add(row["source_image"]["sha256"])
            if value.tobytes() != preprocess(row["source_image"]["path"]).tobytes():
                raise ValueError("real input bytes differ from source image preprocessing")
        else:
            kind = row["kind"]
            if kind == "zeros":
                expected = np.zeros_like(value)
            else:
                seed = 59001 if kind == "uniform" else 59002
                if row.get("seed") != seed:
                    raise ValueError("diagnostic random seed differs")
                rng = np.random.Generator(np.random.PCG64(seed))
                expected = (rng.uniform(0, 1, value.shape) if kind == "uniform" else
                            rng.normal(0, 1, value.shape)).astype("float32")
            if expected.tobytes() != value.tobytes():
                raise ValueError("diagnostic bytes differ from frozen generation rule")
        if not row.get("frontiers"):
            raise ValueError("complete canonical frontier references required")
        row["frontiers"] = {node: verify_ref(value, root) for node, value in row["frontiers"].items()}
    if len(image_hashes) != 64:
        raise ValueError("real inputs must originate from 64 distinct image contents")
    return records


def _output_path(session, reported):
    # Native dump paths are absolute board paths. Extracted paths are relative to
    # the session directory, whose unique remote basename is frozen by collector.
    parts = Path(reported).parts
    try:
        index = parts.index("outputs")
    except ValueError as error:
        raise ValueError("output is outside the session output directory") from error
    relative = Path(*parts[index:])
    if ".." in relative.parts:
        raise ValueError("unsafe output path")
    path = (Path(session) / relative).resolve()
    if not path.is_relative_to(Path(session).resolve()):
        raise ValueError("unsafe output path")
    return path


def _numeric_file(session, item, reference):
    import numpy as np
    path = _output_path(session, item["path"])
    actual = np.frombuffer(path.read_bytes(), dtype=item["dtype"]).reshape(item["shape"])
    expected = np.load(reference["path"], allow_pickle=False)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise NumericMismatch("output shape/dtype differs from canonical reference")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise NumericMismatch("nonfinite output")
    exact = np.issubdtype(actual.dtype, np.integer)
    passed = np.array_equal(actual, expected) if exact else np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    if not passed:
        raise NumericMismatch("canonical frontier differs: " + str(path))
    return file_hash(path)


def audit_stage_log(path, config, queue_capacity, expected_frames=67):
    """Bind each real FIFO submission to a stage, worker, and TLS queue.

    Runtime instruction dumps and fresh completion observations are mandatory.
    The dependency auditor carries token inventory across ALL workers; switching
    TLS queues is never interpreted as a hardware reset.
    """
    from thesis_completion_dependency_audit import inventory, parse
    current, calls, transformed = None, [], []
    stages = {i for i, s in enumerate(config["identity"]["stages"]) if s["device"] == "vta"}
    for line in Path(path).read_text(errors="strict").splitlines():
        if line.startswith("[C1_MEMORY_STAGE] "):
            event = json.loads(line.split(" ", 1)[1])
            key = (event["frame"], event["stage"])
            if event["event"] == "begin":
                if current is not None:
                    raise ValueError("overlapping VTA stage scopes violate global mutex")
                current = {"begin": event, "key": key, "queues": [], "completions": []}
                transformed.append("[THESIS_CALL_BEGIN] frame_{}_stage_{}".format(*key))
            elif event["event"] == "end":
                if current is None or current["key"] != key:
                    raise ValueError("unmatched VTA stage end")
                if any(current["begin"][k] != event[k] for k in ("tid", "queue_id")):
                    raise ValueError("worker/TLS queue changed inside a stage")
                status = event["queue_status"]
                if (not status or status.get("schema") != "vta_queue_capacity_status_v1" or
                        status.get("replay_policy") != "disabled" or
                        status.get("insn_submit_threshold_configured") is not False):
                    raise ValueError("unqualified queue status or replay/early submission")
                for field, cap in (("insn", "instruction"), ("uop", "uop")):
                    if status[field + "_capacity_bytes"] != queue_capacity[cap]:
                        raise ValueError("worker queue capacity differs from frozen environment")
                    if status[field + "_peak_bytes"] > queue_capacity[cap]:
                        raise ValueError("worker queue exceeded capacity")
                if not current["queues"] or len(current["queues"]) != len(current["completions"]):
                    raise ValueError("missing worker submissions or fresh completion observations")
                if any(q["queue_id"] != event["queue_id"] or q["timeout"] for q in current["queues"]):
                    raise ValueError("queue identity/timeout mismatch")
                for queue, complete in zip(current["queues"], current["completions"]):
                    if (complete.get("after_clear_valid") != 0 or complete.get("return_valid") != 1 or
                            complete.get("return_done") != 1 or complete.get("timeout") or
                            complete.get("insn_count") != queue["insn_bytes"] // 16):
                        raise ValueError("completion is not fresh or instruction count differs")
                if event["time_ms"] <= current["begin"]["time_ms"]:
                    raise ValueError("nonpositive VTA scope duration")
                current["end"] = event
                calls.append(current)
                transformed.append("[THESIS_CALL_END] frame_{}_stage_{}".format(*key))
                current = None
            else:
                raise ValueError("unknown stage audit event")
        elif line.startswith("[VTA_QUEUE] "):
            if current is None:
                raise ValueError("unbounded worker queue submission")
            current["queues"].append(json.loads(line.split(" ", 1)[1]))
        elif "[C3_COMPLETION] " in line:
            if current is None:
                raise ValueError("unbounded completion")
            current["completions"].append(json.loads(line.split("[C3_COMPLETION] ", 1)[1]))
        elif current is not None:
            transformed.append(line)
    keys = [c["key"] for c in calls]
    if current is not None or len(keys) != len(set(keys)) or set(keys) != {
        (frame, stage) for frame in range(expected_frames) for stage in stages
    }:
        raise ValueError("incomplete frame/stage completion coverage")
    workers = {}
    for stage in stages:
        bindings = {(c["begin"]["tid"], c["begin"]["queue_id"]) for c in calls if c["key"][1] == stage}
        if len(bindings) != 1:
            raise ValueError("stage does not retain one worker/TLS queue")
        workers[str(stage)] = list(bindings.pop())
    queue_count = len({v[1] for v in workers.values()})
    if len({v[0] for v in workers.values()}) != len(stages) or queue_count not in (1, len(stages)):
        raise ValueError("worker identities or queue ownership model is inconsistent")
    # In the frozen runtime, ThreadLocal() holds a process-wide static shared_ptr.
    # Its name does not establish TLS. A shared queue is valid only under the
    # already-checked global mutex and continuous dependency inventory below.
    queue_model = "one_process_queue_shared_by_workers" if queue_count == 1 else "one_queue_per_worker"
    with tempfile.TemporaryDirectory(prefix="c1_memory_dependency_") as temp:
        bounded = Path(temp) / "bounded.log"
        bounded.write_text("\n".join(transformed) + "\n")
        blocks = parse(bounded)
    if len(blocks) != sum(len(c["queues"]) for c in calls):
        raise ValueError("actual instruction dump/submission coverage differs")
    queues = [q for call in calls for q in call["queues"]]
    if any(block["declared_count"] * 16 != q["insn_bytes"] for block, q in zip(blocks, queues)):
        raise ValueError("instruction dump differs from actual submitted queue bytes")
    findings = inventory(blocks, reset_per_condition=False)
    if any(not f["store_ack_bound_sufficient"] for f in findings):
        raise ValueError("STORE acknowledgement before FINISH is not proven across worker switches")
    return {"workers": workers, "queue_model": queue_model, "calls": len(calls), "submissions": len(blocks),
            "queue_worker_binding": True, "global_vta_mutex": True, "fresh_completion": True,
            "store_before_finish": True}


def validate_qualification(contract, config, directory, records, expected_frames=67):
    directory = Path(directory)
    rows = jsonl(directory / "result.jsonl")
    if expected_frames not in (64, 67) or len(rows) != expected_frames or len(records) != expected_frames:
        raise ValueError("qualification requires the declared 64 or 64+3 pipeline frames")
    stages = config["identity"]["stages"]
    output_hashes, generations, slot_ranges = [], {}, {}
    allocation_rows = jsonl(directory / "allocations.jsonl")
    allocations = {r["phase"]: r for r in allocation_rows}
    if len(allocations) != len(allocation_rows) or not all(phase in allocations for phase in
            ("graph_and_params_loaded", "slots_allocated", "after_steady_state")):
        raise ValueError("missing or duplicate allocator phases")
    final_alloc = allocations["after_steady_state"]
    if final_alloc["capacity_bytes"] != 192 << 20:
        raise ValueError("u-dma-buf reservation drift")
    for allocation in allocation_rows:
        if allocation.get("initialized") == 0 and allocation["phase"] == "before_graph_load":
            if any(allocation[key] != 0 for key in ("capacity_bytes", "requested_bytes", "high_water_bytes")):
                raise ValueError("uninitialized allocator reports nonzero allocation")
            continue
        if (allocation.get("initialized") != 1 or allocation["capacity_bytes"] != 192 << 20 or
                not 0 <= allocation["requested_bytes"] <= allocation["high_water_bytes"] <= 192 << 20):
            raise ValueError("allocator exceeded the fixed system reservation")
    for index, row in enumerate(rows):
        if row.get("frame_id") != index or row.get("input_index") != index or row.get("completion_index") != index:
            raise ValueError("qualification input/frame order differs")
        if row.get("mode") != "pipeline" or row.get("output_mode") != "raw_all_stages":
            raise ValueError("qualification must exercise concurrent workers and all frontiers")
        if row.get("c1_framework_materialization_bytes") != config["framework_materialization_bytes"]:
            raise ValueError("actual boundary materialization differs from frozen feature B")
        final = row.get("raw_output_files", [])
        last_nodes = stages[-1]["output_node_ids"]
        if len(final) != len(last_nodes):
            raise ValueError("missing final outputs")
        output_hashes.append([_numeric_file(directory, item, records[index]["frontiers"][str(node)])
                              for item, node in zip(final, last_nodes)])
        observed_stages = row.get("stage_raw_outputs", [])
        if len(observed_stages) != len(stages):
            raise ValueError("incomplete live frontier observations")
        for stage_index, (observed, stage) in enumerate(zip(observed_stages, stages)):
            files = observed.get("output_files", [])
            if observed.get("stage_index") != stage_index or len(files) != len(stage["output_node_ids"]):
                raise ValueError("missing stage frontier files")
            for item, node in zip(files, stage["output_node_ids"]):
                _numeric_file(directory, item, records[index]["frontiers"][str(node)])
        if config["policy"] != "ordinary_copy":
            if (row.get("p8_managed_slot_count") != 2 or
                    row.get("p8_framework_materialization_bytes") != config["framework_materialization_bytes"]):
                raise ValueError("actual shared materialization/slot count differs from frozen features")
            edges = row.get("p8_boundaries", [])
            if len(edges) != len(stages) - 1:
                raise ValueError("missing shared-edge generation observations")
            for edge_index, edge in enumerate(edges):
                if not edge or edge["edge_index"] != edge_index or edge["slot_id"] not in (0, 1):
                    raise ValueError("invalid shared slot identity")
                key = (edge_index, edge["slot_id"])
                generation = finite(edge["generation"], "slot generation", integer=True)
                if key in generations and generation <= generations[key]:
                    raise ValueError("reused or decreasing frame generation")
                generations[key] = generation
                sizes, addresses = edge.get("tensor_bytes", []), edge.get("physical_addresses", [])
                if not sizes or len(sizes) != len(addresses) or sum(sizes) != edge["boundary_bytes"]:
                    raise ValueError("missing or inconsistent physical slot tensors")
                ranges = []
                for size, address in zip(sizes, addresses):
                    size = finite(size, "slot tensor bytes", positive=True, integer=True)
                    begin = int(address, 16)
                    if (begin % 256 or begin < final_alloc["physical_base"] or
                            begin + size > final_alloc["physical_base"] + final_alloc["high_water_bytes"]):
                        raise ValueError("slot physical range is outside the certified allocation")
                    ranges.append((begin, begin + size))
                if key in slot_ranges and slot_ranges[key] != ranges:
                    raise ValueError("slot physical backing changed across frames")
                other = slot_ranges.get((edge_index, 1 - edge["slot_id"]), [])
                if any(a < d and c < b for a, b in ranges for c, d in other):
                    raise ValueError("the two slots of one edge overlap")
                slot_ranges[key] = ranges
    log = directory / "native.log"
    proof = audit_stage_log(log, config, contract["queue_capacity_bytes"], expected_frames)
    profile = read_json(directory / "profile/benchmark_totals_status.json")
    if any(field not in profile for field in ("load_buffer_2d_calls", "store_buffer_2d_calls")):
        raise ValueError("missing actual DMA counters")
    calls = sum(finite(profile[field], field, integer=True) for field in
                ("load_buffer_2d_calls", "store_buffer_2d_calls"))
    if calls != expected_frames * config["vta_dma_calls"]:
        raise ValueError("runtime DMA calls disagree with final TIR")
    slot_bytes = (allocations["slots_allocated"]["requested_bytes"] -
                  allocations["graph_and_params_loaded"]["requested_bytes"])
    if slot_bytes != config["additional_slot_bytes"]:
        raise ValueError("actual slot allocation differs from static plan (including anchor fallback)")
    return {**proof, "cross_frame_outputs": True, "slot_generation": True,
            "final_output_sha256": output_hashes, "vta_dma_calls": calls // expected_frames,
            "actual_additional_slot_bytes": slot_bytes,
            "high_water_bytes": allocations["after_steady_state"]["high_water_bytes"]}


def validate_performance(directory, config):
    directory = Path(directory)
    log = (directory / "native.log").read_text()
    if any(marker in log for marker in ("[C1_MEMORY_STAGE]", "[VTA_QUEUE]", "[C3_COMPLETION]", "INSTRUCTION 0:")):
        raise ValueError("diagnostic logging was enabled in performance window")
    rows = jsonl(directory / "result.jsonl")
    if len(rows) != 307:
        raise ValueError("performance requires five worker warmups plus a 302-row window")
    if any(r.get("frame_id") != i or r.get("completion_index") != i or
           r.get("input_index") != i % 64 or r.get("mode") != "pipeline"
           for i, r in enumerate(rows)):
        raise ValueError("performance frame/input sequence differs")
    completed = [finite(r.get("completion_ms"), "completion_ms", positive=True) / 1000 for r in rows]
    if any(right <= left for left, right in zip(completed, completed[1:])):
        raise ValueError("all warmup/window completions must increase")
    metric = window_metrics(completed[5:])
    # Independently check that physical VTA stage intervals never overlap.
    intervals = []
    for row in rows:
        for stage, spec in enumerate(config["identity"]["stages"]):
            if spec["device"] == "vta":
                a, b = row.get("stage{}_vta_mutex_acquire_ms".format(stage)), row.get("stage{}_vta_mutex_release_ms".format(stage))
                if a is None or b is None or not math_isfinite(a) or not math_isfinite(b) or b < a:
                    raise ValueError("missing actual global mutex interval")
                intervals.append((a, b))
    intervals.sort()
    if any(right[0] < left[1] - 1e-6 for left, right in zip(intervals, intervals[1:])):
        raise ValueError("overlapping physical VTA intervals")
    return {**metric, "ii_ms": metric["ii_seconds"] * 1000, "warmup_rows": 5, "raw_rows": 307}


def math_isfinite(value):
    import math
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
