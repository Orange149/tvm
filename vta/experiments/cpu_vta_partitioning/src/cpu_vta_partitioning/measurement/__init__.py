"""Strict frame/affinity/correctness validation shared by real and fake transport."""
import math
import statistics
from pathlib import Path

from cpu_vta_partitioning.common import read


def frames(path, expected):
    import json
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} frames, found {len(rows)}")
    for key in ("frame_id", "completion_index"):
        if sorted(r[key] for r in rows) != list(range(expected)):
            raise ValueError("duplicate/missing " + key)
    rows.sort(key=lambda r: r["completion_index"])
    times = [float(r["completion_ms"]) for r in rows]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a,b in zip(times, times[1:])):
        raise ValueError("invalid/nonmonotonic completion timestamps")
    return rows


def input_sequence(rows, task):
    """A valid output alone cannot prove that every scheduled input was executed."""
    indices = task["input_indices"]
    if (not indices or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("invalid scheduled input pool")
    expected = task["warmup"] + task["scored"] + task["tail"]
    if len(rows) != expected or sorted(r["frame_id"] for r in rows) != list(range(expected)):
        raise ValueError("input sequence frame coverage mismatch")
    for row in rows:
        index = row.get("input_index")
        if type(index) is not int or index != row["frame_id"] % len(indices):
            raise ValueError("frame input index differs from scheduled input sequence")
    if task["kind"].startswith("qualify") and {r["input_index"] for r in rows} != set(range(len(indices))):
        raise ValueError("qualification did not cover every scheduled input")
    return True


def window(rows, warmup=20, scored=300, tail=20):
    if len(rows) != warmup + scored + tail or scored < 2:
        raise ValueError("window size mismatch")
    kept = rows[warmup:warmup + scored]
    gaps = [b["completion_ms"] - a["completion_ms"] for a,b in zip(kept, kept[1:])]
    if min(gaps) <= 0:
        raise ValueError("completion order violation")
    return {"ii_ms": statistics.mean(gaps), "kept_timestamps": len(kept), "gaps": len(gaps),
            "first_completion_ms": kept[0]["completion_ms"], "last_completion_ms": kept[-1]["completion_ms"],
            "latency_ms": statistics.mean(r["total_latency_ms"] for r in kept),
            "p95_gap_ms": sorted(gaps)[math.ceil(.95 * len(gaps)) - 1]}


def calibration(rows, warmup=5, scored=32):
    if len(rows) != warmup + scored:
        raise ValueError("calibration window mismatch")
    kept = rows[warmup:]
    if any(r["stage_cpu_time_scope"] != "exclusive_process_cpu_time" for r in kept):
        raise ValueError("core-time must be from isolated serial runs")
    def mean(key):
        values = [r["stage0_" + key] for r in kept]
        if any(v is None or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("missing/invalid isolated core-time: " + key)
        return statistics.mean(values)
    return {"service_ms": mean("set_ms") + mean("run_ms") + mean("get_ms"),
            "run_core_ms": mean("run_process_cpu_ms"),
            "boundary_core_ms": mean("set_process_cpu_ms") + mean("get_process_cpu_ms")}


def affinity(document, stages):
    if len(document["stages"]) != len(stages):
        raise ValueError("affinity stage count mismatch")
    for s, row in zip(stages, document["stages"]):
        expected = set(s["mask"])
        actual = row["effective_affinities"]
        def cpus(value):
            if isinstance(value, list):
                return set(value)
            result = set()
            for chunk in value.split(","):
                if "-" in chunk:
                    a,b = map(int, chunk.split("-")); result.update(range(a,b+1))
                elif chunk:
                    result.add(int(chunk))
            return result
        if cpus(row["requested_cpus"]) != expected or not actual:
            raise ValueError("requested affinity mismatch")
        if any(not cpus(t["cpus"]) or not cpus(t["cpus"]) <= expected for t in actual):
            raise ValueError("effective thread affinity escapes requested mask")
        if set().union(*(cpus(t["cpus"]) for t in actual)) != expected:
            raise ValueError("effective affinity does not cover requested mask")
        if row["active_runtime_threads"] != s["threads"] or len(row["thread_ids"]) < s["threads"]:
            raise ValueError("actual TVM thread count mismatch")
    return True


def correctness(rows, task, bundle, local_result):
    import numpy as np
    input_sequence(rows, task)
    programs = read(Path(bundle) / "plans/programs.json")
    inputs = read(Path(bundle) / "plans/inputs.json")["records"]
    for row in rows:
        source = inputs[task["input_indices"][row["input_index"]]]
        dumps = row["stage_raw_outputs"]
        if len(dumps) != len(task["stages"]):
            raise ValueError("missing stage frontier output")
        for stage, dump in zip(task["stages"], dumps):
            program = programs[stage["program"]]
            if len(dump["output_files"]) != len(program["output_node_ids"]):
                raise ValueError("missing residual frontier output")
            for node, file in zip(program["output_node_ids"], dump["output_files"]):
                reference = np.load(Path(bundle) / source["frontiers"][str(node)]["npy"], allow_pickle=False)
                # Native paths are absolute remotely. Reconstruct only the known dump basename.
                raw = Path(local_result) / "outputs" / ("run_" + str(row["completion_index"])) / Path(file["path"]).name
                actual = np.fromfile(raw, dtype=file["dtype"]).reshape(file["shape"])
                if actual.shape != reference.shape or actual.dtype != reference.dtype:
                    raise ValueError("frontier dtype/shape mismatch")
                ok = np.allclose(actual, reference, rtol=1e-4, atol=1e-5) if actual.dtype.kind == "f" else np.array_equal(actual, reference)
                if not ok:
                    raise ValueError(f"incorrect full output for input {source['index']} node {node}")
    return True
