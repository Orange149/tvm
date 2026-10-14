#!/usr/bin/env python3
"""Offline evidence relations only: no RPC, build, runtime loading or tuning."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import numpy as np

HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
LATEST = "20261013_c3_banerjee_baseline_completion_run02"
LOCAL_MODEL = "20261012_c3_residency_instruction_tradeoff_run01"
SEEDS = [0, 20250901, 20260910]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


class Evidence:
    def __init__(self):
        self.bindings = {}

    def text(self, path):
        path = Path(path).resolve()
        self.bindings[str(path)] = digest(path)
        return path.read_text()

    def read(self, path):
        return json.loads(self.text(path))

    def journal(self, path):
        return [json.loads(line) for line in self.text(path).splitlines() if line]

    def bind(self, path, expected=None):
        value = digest(path)
        require(expected is None or value == expected, "identity changed: " + str(path))
        self.bindings[str(Path(path).resolve())] = value
        return value


def aggregate(calls, programs, kind="timing", program_key="program", kind_key="call_kind"):
    require(all(c.get("correct") is True for c in calls), "incorrect historical sample")
    sample = [c for c in calls if c.get(kind_key) == kind]
    require(len(sample) == 18 * len(programs), "incomplete six-round timing cohort")
    boots = {c["boot_id"] for c in sample}
    require(len(boots) == 1, "cannot splice different boots")
    cells = defaultdict(list)
    for c in sample:
        value = c["latency_ms"]
        require(c[program_key] in programs, "unexpected program")
        require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0, "invalid latency")
        cells[c["round"], c[program_key]].append(c)
    rounds = {}
    for r in range(6):
        rounds[str(r)] = {}
        for p in programs:
            cc = cells[r, p]
            require(len(cc) == 3 and sorted(c["seed"] for c in cc) == sorted(SEEDS), "missing or duplicate seed")
            rounds[str(r)][p] = median(c["latency_ms"] for c in cc)
    return {"boot_id": next(iter(boots)), "timed_calls": len(sample), "round_ms": rounds,
            "median_ms": {p: median(rounds[str(r)][p] for r in range(6)) for p in programs}}


def compare(data, candidate, baseline):
    gains = [(1 - data["round_ms"][str(r)][candidate] / data["round_ms"][str(r)][baseline]) * 100
             for r in range(6)]
    return {"candidate": candidate, "baseline": baseline, "paired_gain_pct": median(gains),
            "round_gain_pct": gains, "wins": sum(g > 0 for g in gains)}


def weight_plan(groups, resident, slots, spatial, group_bytes, capacity):
    require(all(type(v) is int for v in (groups, resident, slots, spatial, group_bytes, capacity)), "integer geometry")
    require(groups > 0 and 0 <= resident < groups and slots >= 1 and spatial >= 1 and group_bytes > 0,
            "invalid partial-residency geometry")
    allocated = (resident + slots) * group_bytes
    require(allocated <= capacity, "resident plus stream slots exceed capacity")
    loads = resident + (groups - resident) * spatial
    return {"resident_bytes": resident * group_bytes, "nominal_allocated_bytes": allocated,
            "whole_group_loads": loads, "request_bytes": loads * group_bytes,
            "max_resident_with_these_slots": capacity // group_bytes - slots,
            "not_a_compiled_capacity_certificate": True, "not_an_overlap_or_time_prediction": True}


def source_coverage(requests, memory):
    """Exact logical DDR element coverage, including request multiplicity."""
    counts, element_size = Counter(), None
    for g in requests["request_groups"]:
        if g["direction"] != "load" or g["memory_name"] != memory:
            continue
        size = g["element_bytes"]
        require(element_size in (None, size), "mixed element units")
        element_size = size
        require(g["ddr_bounds_checked"], "unresolved DDR address")
        for y in range(g["y_size"]):
            for x in range(g["x_size"]):
                offset = g["ddr_offset_elements"] + y * g["x_stride"] + x
                require(offset >= 0, "negative actual DDR access")
                counts[offset] += g["multiplicity"]
    require(counts and element_size, "no concrete operand requests")
    total = sum(counts.values()) * element_size
    require(total == requests["totals"]["load_buffer_2d_" + memory + "_bytes"], "coverage and bytes disagree")
    return counts, {"element_bytes": element_size, "unique_bytes": len(counts) * element_size,
        "requested_bytes": total, "repeat_histogram_elements": dict(Counter(counts.values())),
        "complete_contiguous_source": set(counts) == set(range(max(counts) + 1)), "not_physical_bus_traffic": True}


def load_address_spans(requests):
    """LOAD endpoint envelope, not allocation sum or dynamic occupancy."""
    ends = defaultdict(int)
    for g in requests["request_groups"]:
        if g["direction"] != "load" or g["memory_name"] not in ("inp", "wgt", "acc"):
            continue
        left, top, right, bottom = g["padding"]
        count = (g["x_size"] + left + right) * (g["y_size"] + top + bottom)
        ends[g["memory_name"]] = max(ends[g["memory_name"]],
            (g["sram_offset_elements"] + count) * g["element_bytes"])
    return dict(ends)


def identifiability(matrix, independent_validation_points, complete_features):
    x = np.asarray(matrix, dtype=float)
    require(x.ndim == 2 and np.isfinite(x).all(), "invalid design matrix")
    rank = int(np.linalg.matrix_rank(x))
    return {"rows": x.shape[0], "columns": x.shape[1], "rank": rank,
        "algebraically_identifiable": rank == x.shape[1], "independent_validation_points": independent_validation_points,
        "features_cover_all_claimed_costs": complete_features,
        "fit_permitted": rank == x.shape[1] and independent_validation_points > 0 and complete_features}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run(output):
    require(not output.exists(), "new audit directory required")
    e, latest = Evidence(), GROUPED / LATEST
    manifest = e.read(latest / "manifest.json")
    for rel, expected in manifest["files"].items():
        require(digest(latest / rel) == expected, "latest artifact changed: " + rel)
    prepared = e.read(latest / "prepared.json")["programs"]
    protocol = e.read(latest / "protocol.json")
    for path, expected in protocol["source_sha256"].items():
        e.bind(path, expected)
    config = e.read(HERE.parents[2] / "3rdparty/vta-hw/config/vta_config.json")
    capacity = {s: 1 << config["LOG_" + s + "_BUFF_SIZE"] for s in ("INP", "WGT", "ACC", "UOP")}
    selected = e.read(latest / "selected_pool.json")
    data = aggregate(e.journal(latest / "board/calls.jsonl"), selected["programs"])
    recorded = e.read(latest / "results.json")
    require(data["median_ms"] == recorded["median_ms"] and data["round_ms"] == recorded["round_medians"],
            "latest raw aggregation differs")
    commands = e.read(latest / "qualification/diagnostic_signatures.json")
    inputs = e.read(latest / "board/contract.json")["data_contract"]
    programs = {}
    for name in selected["programs"]:
        record = prepared[name]
        e.bind(record["binary"], record["binary_sha256"])
        e.bind(Path(record["directory"]) / "node24_final.tir.json", record["tir_sha256"])
        directory = Path(record["directory"])
        if name == "Input-7x7":
            directory = Path(e.read(latest / "anchors_final/Input-7x7/historical_binding.json")["directory"])
        req = e.read(directory / "request_semantics.json")
        require(req["totals"] == record["request_totals"], "request certificate differs")
        wgt, coverage = source_coverage(req, "wgt")
        require(coverage["unique_bytes"] == 294912 and coverage["complete_contiguous_source"], "weight source holes")
        spans = load_address_spans(req)
        require(all(spans[s.lower()] <= capacity[s] for s in ("INP", "WGT", "ACC")), "LOAD capacity")
        sig = commands[name]["rotations"]["0"][0]
        require(all(s == sig for s in commands[name]["rotations"]["0"]), "warm state changed")
        acc = {g["source_buffer"] for g in req["request_groups"] if g["direction"] == "load" and g["memory_name"] == "acc"}
        stores = {g["memory_name"] for g in req["request_groups"] if g["direction"] == "store"}
        programs[name] = {"median_ms": data["median_ms"][name], "request_bytes": req["totals"],
            "WGT_source_coverage": coverage, "LOAD_endpoint_span_bytes": spans, "dynamic_SRAM_utilization": None,
            "partial_sum_DDR_spill_bytes": 0 if acc == {"p2"} and stores == {"out"} else None,
            "ACC_load_source_buffers": sorted(acc), "warmed_signature": sig,
            "binary_sha256": record["binary_sha256"], "TIR_sha256": record["tir_sha256"]}
        if name in ("R13-G", "N24-R12-G2-PREFETCH"):
            r, slots = (13, 1) if name == "R13-G" else (12, 2)
            relation = weight_plan(16, r, slots, 7, 18432, capacity["WGT"])
            require(relation["whole_group_loads"] == sig["wgt_loads"] and relation["request_bytes"] == coverage["requested_bytes"],
                    "partial-residency relation mismatch")
            require(all(wgt[i] == (1 if i // 72 < r else 7) for i in range(1152)), "resident subset mismatch")
            programs[name]["partial_residency_relation"] = relation
        elif name.startswith("TPS"):
            require(set(wgt.values()) == {1}, "TPS repeated weights")
            contract = e.read(directory / "contract.json")
            programs[name]["tile"] = contract["selection"]["tile"]
            cert = e.read(latest / "local_final06" / name / "native_certificate.json")
            require(all(p["all_tokens_paired"] and p["no_queue_cycle"] for p in cert["proof"]), "native boundary")
            addr = e.read(latest / "local_final06" / name / "address_audit.json")
            programs[name]["max_kernel_uops"] = addr["max_kernel_uops"]
            programs[name]["dynamic_UOP_occupancy"] = addr["uop_effective_occupancy_peak"]
    historical = {}
    for folder, names in ((LOCAL_MODEL, ["R0-G", "R8-G", "R12-G", "R13-G", "F13-B", "Input-7x7"]),
        ("20261012_c3_five_tradeoff_selection_run01/confirmation", ["R13-G", "R12-G", "N24-R12-G2-PREFETCH", "Input-7x7"])):
        raw = GROUPED / folder / ("board/calls.jsonl" if folder == LOCAL_MODEL else "calls.jsonl")
        historical[folder] = aggregate(e.journal(raw), names)
    for folder, names in (("20261008_c3_node34_operator_boot1_run02", ["G2", "O1", "I1", "W1", "BRIDGE"]),
        ("20261009_c3_node24_group_dma_run01/board_run01", ["a13", "a13_b", "f13", "f13_b", "input_alt"]),
        ("20261009_c3_node24_group_gemm_run01/board_run02", ["f13_b", "f13_g", "input_alt"]),
        ("20261009_c3_node24_group_gemm_run01/boot2_operator_run02", ["f13_b", "f13_g", "input_alt"])):
        historical[folder] = aggregate(e.journal(GROUPED / folder / "calls.jsonl"), names,
                                      program_key="variant", kind_key="phase")
    model = e.read(GROUPED / LOCAL_MODEL / "analysis/model.json")
    validation = e.read(GROUPED / LOCAL_MODEL / "analysis/validation.json")
    old = historical[LOCAL_MODEL]
    matrix = np.asarray(model["design_matrix"])
    require(int(np.linalg.matrix_rank(matrix)) == model["rank"] == 3, "old model rank")
    targets = np.asarray([old["median_ms"][n] for n in model["calibration_programs"]])
    require(np.allclose(targets, model["calibration_targets_ms"], atol=1e-12), "old raw calibration mismatch")
    require(np.allclose(matrix @ model["coefficients_ms"], targets, atol=1e-10), "old coefficients mismatch")
    for row, r in zip(validation["rows"], (8, 12)):
        pred = old["median_ms"]["R13-G"] + (13 - r) / 13 * (old["median_ms"]["R0-G"] - old["median_ms"]["R13-G"])
        require(math.isclose(pred, row["predicted_ms"], abs_tol=1e-10) and
                math.isclose(old["median_ms"][row["program"]], row["observed_ms"], abs_tol=1e-10), "old validation")
    x = [[1, d["request_bytes"]["load_buffer_2d_inp_bytes"] / 1024, d["request_bytes"]["load_buffer_2d_wgt_bytes"] / 1024,
          d["request_bytes"]["load_buffer_2d_acc_bytes"] / 1024, d["warmed_signature"]["mac"],
          d["warmed_signature"]["uop_payload_bytes"] / 1024, d["warmed_signature"]["submissions"]] for d in programs.values()]
    model_status = {"existing_local_interpolation": {"status": "RECOMPUTED_NOT_REFITTED", "rank": 3, "validation": validation,
        "valid_domain": "old runtime, node24 compact single-slot fixed Tile; R8/R12 only",
        "not_general_organization_model": True, "not_transferred_to_new_runtime_or_TPS": True},
        "five_tradeoff_model": identifiability(x, 0, False), "no_new_latency_model_fitted": True,
        "missing": ["independent contrasts for five costs", "actual overlap durations", "unexposed validation tasks",
                    "nonconstant submission column"]}
    finish(e, output, latest, prepared, selected, programs, data, capacity, inputs, historical, model_status)


def finish(e, output, latest, prepared, selected, programs, data, capacity, inputs, historical, model_status):
    """Build plan bindings from actual graph and the already sealed candidate."""
    graph_dir = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03/G2"
    graph, build = e.read(graph_dir / "graph.json"), e.read(graph_dir / "build.json")
    stock = graph_dir.parent / "stock"
    stock_build = e.read(stock / "build.json")
    for directory, record in ((graph_dir, build), (stock, stock_build)):
        for file, key in (("graphlib.so", "graphlib_sha256"), ("graph.json", "graph_sha256"), ("params.bin", "params_sha256")):
            e.bind(directory / file, record[key])
    node = graph["nodes"][24]
    require(node["attrs"]["func_name"] == "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_3", "target binding drift")
    entries = []
    for abi, ref in zip(inputs["abi_order"], node["inputs"] + [[24, 0, 0]]):
        entry = graph["node_row_ptr"][ref[0]] + ref[1]
        entries.append({"abi": abi, "node": ref[0], "output": ref[1], "entry": entry,
            "shape": graph["attrs"]["shape"][1][entry], "dtype": graph["attrs"]["dltype"][1][entry],
            "sid": graph["attrs"]["storage_id"][1][entry], "device": graph["attrs"]["device_index"][1][entry]})
    require([v["shape"] for v in entries] == [v["shape"] for v in inputs["buffers"]], "graph/standalone ABI mismatch")
    primary = selected["primary_TPS_baseline"]
    next_plan = {"schema": "c3_minimum_deployment_correctness_plan_v1", "status": "FROZEN_REQUIRES_USER_AUTHORIZATION",
        "decision": "A_existing_stronger_program_deployment_correctness", "task": "node24 YOLOv3-tiny-224",
        "candidate": primary, "execution_task_id": e.read(latest / "local_final06" / primary / "contract.json")["route"]["execution_task_id"],
        "operator_binary": prepared[primary]["binary"], "operator_binary_sha256": prepared[primary]["binary_sha256"],
        "operator_TIR_sha256": prepared[primary]["tir_sha256"], "configuration": programs[primary]["tile"],
        "fused_contract": inputs, "graph_target_symbol": node["attrs"]["func_name"], "tensor_entries": entries,
        "stock_directory": str(stock), "current_G2_directory": str(graph_dir),
        "current_G2_binary_sha256": build["graphlib_sha256"], "stock_binary_sha256": stock_build["graphlib_sha256"],
        "common_runtime": {k: v for k, v in e.read(latest / "runtime_contract.json").items()
            if k in ("runtime_paths", "runtime_sha256", "bitstream_sha256", "queue_environment", "affinity_cpu")},
        "non_target_identity_file": str(graph_dir / "functions.json"),
        "graph_memory_contract": build["memory_contract"],
        "allowed_difference": ["node24 symbol binding and experimental metadata",
                              "target body equal to sealed operator up to ABI-preserving symbol renaming"],
        "local_gates": ["same Graph JSON and semantic parameters", "all non-target TIR equal",
            "exact target body, unchanged ABI and no silent fallback", "192 MiB static storage contract",
            "completion on actual new graph/runtime must be resolved without adding device calls",
            "experimental transforms restricted to target function, no non-target global changes"],
        "board_graph_runs": {"stock_reference": 3, "G2_qualification": 3, "candidate_qualification": 3,
                            "candidate_same_executor_input_A_B_A": 3, "total": 12},
        "seeds": SEEDS, "input_A_B_A": [0, 20250901, 0], "explicit_warmups": 0, "timed_runs": 0,
        "policies": ["current boot only, no reboot", "stock references first; reference A/B must differ",
            "all outputs checked per call", "one executor per fresh RPC; same candidate executor/input handle for ABA",
            "graph input may legally alias later output; no immutability assertion after run",
            "first output error/timeout/identity drift/unresolved gate stops all work; no fallback/reselection",
            "no device calls hidden in debug profiling or completion audit"],
        "infrastructure_extra_calls": "zero planned device calls; extra completion probes need an approved amendment",
        "success": "exact sealed candidate correctly executes in real graph context, not performance or model generalization",
        "after_pass": "report and request a separate graph-performance protocol",
        "after_fail": "retain minimal failure; no new candidates, Tile or sync patch"}
    e.bind(graph_dir / "functions.json")
    for filename in ("prepare_c3_five_tradeoff_graph.py", "run_vta_c3_node24_group_gemm_graph_prepare.py",
                     "c3_banerjee_completion.py", "c3_banerjee_uop_queue.h"):
        e.bind(HERE / filename)
    e.bind(__file__)
    old_graph = e.read(GROUPED / "20261012_c3_five_tradeoff_selection_run01/graph_node24/results.json")
    graph_rows = e.journal(GROUPED / "20261012_c3_five_tradeoff_selection_run01/graph_node24/calls.jsonl")
    require(all(c["correct"] for c in graph_rows), "historical graph error")
    med = {p: median(c["latency_ms"] for c in graph_rows if c["call_kind"] == "graph_timing" and c["program"] == p)
           for p in ("current", "Input", "rule")}
    require(med == old_graph["median_ms"], "historical graph aggregation mismatch")
    staged = e.read(GROUPED / "20260923_c3_vthread_staged_analysis_run01/comparisons.json")
    staged_summary = e.read(GROUPED / "20260923_c3_vthread_staged_analysis_run01/summary.json")
    csv_rows = list(csv.DictReader(e.text(GROUPED / "20260923_c3_vthread_staged_analysis_run01/timing_samples.csv").splitlines()))
    staged_raw = [{**c, "correct": c["correct"] == "True", "boot_id": staged_summary["boot_id"],
                   "latency_ms": float(c["latency_ms"]), "round": int(c["round"]), "seed": int(c["seed"])}
                  for c in csv_rows]
    staged_cohort = aggregate(staged_raw, ["O2", "PE", "PS"], program_key="variant", kind_key="phase")
    for c in staged:
        gains = [(1 - b / a) * 100 for a, b in zip(c["baseline_round_ms"], c["candidate_round_ms"])]
        require(math.isclose(median(gains), c["paired_median_gain_pct"], abs_tol=1e-10), "staged gain mismatch")
        expected = compare(staged_cohort, c["candidate"].split("_")[-1], c["baseline"].split("_")[-1])
        require(math.isclose(expected["paired_gain_pct"], c["paired_median_gain_pct"], abs_tol=1e-10), "staged raw mismatch")
    historical["H1_staged"] = staged_cohort
    y00 = e.read(GROUPED / "20260923_c3_y00_input_weight_dual_boot_analysis_run01/contract.json")
    y00_results = []
    for session in y00["board_sessions"]:
        directory = Path(session["directory"])
        summary = e.read(directory / "summary.json")
        require(summary["boot_id"] == session["boot_id"] and summary["all_timing_correct"], "Y00 session identity")
        cc = e.journal(directory / "timing.jsonl")
        require(all(c["complete_operator_calls"] == 1 and c["hidden_warmup_calls"] == 0 for c in cc), "Y00 call boundary")
        # Legacy journals bind boot through their same-directory summary.
        cohort = aggregate([{**c, "boot_id": summary["boot_id"]} for c in cc],
                           ["O", "I", "IW"], program_key="variant", kind_key="phase")
        y00_results.append({"cohort": cohort, "IW_vs_I": compare(cohort, "IW", "I")})
    require(len({v["cohort"]["boot_id"] for v in y00_results}) == 2, "Y00 boots not independent")
    node34_rows = e.journal(GROUPED / "20261008_c3_node34_board_qualify_run02/calls.jsonl")
    node34_requests = {}
    for name in ("W1", "I1", "BRIDGE"):
        cc = [c for c in node34_rows if c.get("phase") == "fused_replay" and c.get("variant") == name]
        require(cc and all(c["correct"] for c in cc), "node34 full-fusion qualification missing")
        fields = ("load_buffer_2d_inp_bytes", "load_buffer_2d_wgt_bytes", "driver_run_calls")
        values = [{k: c["runtime_profile"][k] for k in fields} for c in cc]
        require(all(v == values[0] for v in values), "node34 request state changes")
        node34_requests[name] = values[0]
    output.mkdir(parents=True)
    result = {"status": "LOCAL_EVIDENCE_AUDIT_COMPLETE_NO_BOARD_WORK", "latest_cohort": data, "capacity_bytes": capacity,
        "programs": programs, "historical_cohorts_separate": historical,
        "comparisons": {"latest_partial_vs_TPS": compare(data, "N24-R12-G2-PREFETCH", primary),
            "old_same_Tile_R12_dual_vs_single": compare(historical["20261012_c3_five_tradeoff_selection_run01/confirmation"],
                                                      "N24-R12-G2-PREFETCH", "R12-G")},
        "model_assessment": model_status, "old_graph_separate_runtime_cohort": old_graph,
        "staged_eager_comparisons": staged, "full_fit_Y00_two_boots": y00_results,
        "node34_actual_fused_requests": node34_requests,
        "read_only_old_files": True, "board_contacted": False, "new_measurements": 0}
    for name, value in (("analysis.json", result), ("NEXT_VALIDATION.json", next_plan), ("input_hashes.json", e.bindings)):
        write_new(output / name, value)
    print(json.dumps({"status": result["status"], "inputs_bound": len(e.bindings), "next_decision": next_plan["decision"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir.resolve())
