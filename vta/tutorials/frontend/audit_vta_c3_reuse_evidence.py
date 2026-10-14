#!/usr/bin/env python3
"""Node A: read-only audit of frozen C3 reuse evidence, without TVM or board access.

Measured pairs, resource fallback, distinct binaries and effective data reuse are
separate facts. Missing information stays unknown. Nothing in this retrospective
audit authorizes a new candidate, FPGA run, or performance-based rule change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "c3_reuse_node_a_audit_v1"
MODES = ("original", "input_stationary", "weight_resident_barrier",
         "input_weight_resident_barrier")
SOURCES = {
    "r18_analysis": "20260914_p7r513_resnet18_cheng_four_scheme_analysis_run01",
    "r18_adapter": "20260914_p7r511_resnet18_literature_replay_adapter_run01",
    "r18_board": "20260914_p7r510_resnet18_merged_board_pool_run01",
    "r18_pool": "20260914_p7r494_resnet18_merged_literature_board_pool_run01",
    "r18_cross": "20260914_p7r496_resnet18_merged_cross_compile_run01",
    "y00_board": "20260912_p7r144_y00_layout_preserving_search_board_run01",
    "y03_board": "20260912_p7r154_y03_layout_preserving_search_board_run01",
    "y02_correctness": "20260912_p7r122_y02_health_gated_correctness_run01",
    "y02_first": "20260912_p7r123_y02_b00_paired_timing_run01",
    "y02_repeat": "20260912_p7r145_y02_b00_cross_boot_paired_timing_run01",
}
COUNTERS = {
    "input_dma_bytes": ("load_buffer_2d_inp_bytes",),
    "weight_dma_bytes": ("load_buffer_2d_wgt_bytes",),
    "acc_dma_bytes": ("load_buffer_2d_acc_bytes", "load_buffer_2d_acc8_bytes"),
    "uop_dma_bytes": ("load_buffer_2d_uop_bytes",),
    "output_dma_bytes": ("store_buffer_2d_out_bytes",),
    "input_dma_calls": ("load_buffer_2d_inp_calls",),
    "weight_dma_calls": ("load_buffer_2d_wgt_calls",),
    "acc_dma_calls": ("load_buffer_2d_acc_calls", "load_buffer_2d_acc8_calls"),
    "uop_dma_calls": ("load_buffer_2d_uop_calls",),
    "output_dma_calls": ("store_buffer_2d_out_calls",),
    "total_dma_bytes": ("load_buffer_2d_bytes", "store_buffer_2d_bytes"),
    "total_dma_calls": ("load_buffer_2d_calls", "store_buffer_2d_calls"),
    "synchronize_calls": ("synchronize_calls",),
    "submissions": ("driver_run_calls",),
}
METRICS = tuple(COUNTERS) + (
    "operator_latency_ms", "driver_execution_ms", "instruction_peak_bytes",
    "uop_peak_bytes", "explicit_residency_drains",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def exact_correct(row):
    actual, expected = row.get("actual_sha256"), row.get("expected_sha256")
    return (row.get("correct") is True and row.get("mismatch_count") == 0 and
            isinstance(actual, str) and len(actual) == 64 and actual == expected and
            all(c in "0123456789abcdef" for c in actual))


def median(values):
    return statistics.median(values) if values and all(v is not None for v in values) else None


def delta(new, old):
    # A zero baseline has no percentage denominator, including the 0 -> 0 case.
    return 100 * (new / old - 1) if new is not None and old is not None and old > 0 else None


def config(candidate):
    value = candidate.get("complete_config_entity") or candidate.get("identity", {}).get(
        "complete_config_entity")
    if value is None:
        raise ValueError("missing complete ConfigEntity")
    return value


def context(candidate):
    identity = candidate.get("identity", {})
    value = {
        "hardware": candidate.get("hardware_fingerprint", identity.get("hardware_fingerprint")),
        "model": candidate.get("model_hash"),
        "workload": identity.get("workload", candidate.get("workload_features")),
        "schedule_version": identity.get("schedule_version"),
    }
    if value["hardware"] is None or value["workload"] is None:
        raise ValueError("incomplete workload/hardware context")
    return value


class FrozenReader:
    """Verify all input artifacts, including retained binary subdirectories."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.verified = {}

    def directory(self, path, expected_manifest=None):
        directory = Path(path).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("input path escapes C3 root")
        manifest = directory / "artifact_hashes.json"
        if directory in self.verified:
            actual = self.verified[directory]["manifest_sha256"]
        else:
            actual = sha256(manifest)
            value = json.loads(manifest.read_text())
            entries = value.get("artifacts", value.get("files"))
            if not isinstance(entries, dict) or not entries:
                raise ValueError("missing artifact map: {}".format(manifest))
            checked = {}

            def check(mapping, base):
                for name, expected in mapping.items():
                    target = (base / name).resolve()
                    if not target.is_relative_to(directory):
                        raise ValueError("artifact path escapes input directory")
                    if isinstance(expected, dict):
                        check(expected, target)
                    elif isinstance(expected, str):
                        if sha256(target) != expected:
                            raise ValueError("artifact hash mismatch: {}".format(target))
                        checked[str(target)] = expected
                    else:
                        raise ValueError("unsupported artifact entry: {}".format(target))

            check(entries, directory)
            self.verified[directory] = {"manifest_sha256": actual, "files": checked}
        if expected_manifest is not None and actual != expected_manifest:
            raise ValueError("linked manifest hash mismatch: {}".format(manifest))
        return directory

    def read(self, directory, name):
        directory = self.directory(directory)
        path = (directory / name).resolve()
        if str(path) not in self.verified[directory]["files"]:
            raise ValueError("file is not covered by artifact ledger: {}".format(path))
        raw = path.read_text()
        return [json.loads(line) for line in raw.splitlines() if line.strip()] if name.endswith(
            ".jsonl") else json.loads(raw)

    def recheck(self):
        for directory, entry in self.verified.items():
            if sha256(directory / "artifact_hashes.json") != entry["manifest_sha256"]:
                raise ValueError("input manifest changed during audit")
            for path, expected in entry["files"].items():
                if sha256(path) != expected:
                    raise ValueError("input artifact changed during audit: {}".format(path))


def indexed(rows):
    result = {}
    for row in rows:
        key = row["candidate_id"]
        if key in result:
            raise ValueError("duplicate candidate identity: {}".format(key))
        result[key] = row
    return result


def profile_counters(profile, divisor=1):
    result = {}
    for name, keys in COUNTERS.items():
        values = [number(profile.get(key)) for key in keys]
        result[name] = sum(values) / divisor if all(v is not None for v in values) else None
    return result


def inference_divisor(row, reference):
    declared = number(row.get("time_evaluator_total_kernel_invocations"))
    actual = profile_counters(row.get("runtime_profile_complete", {}))
    if declared is None:
        # Older Y02 records omit the timer invocation field. Certify multiplicity
        # against three-seed DIRECT correctness profiles, not submission count.
        ratios = [actual[k] / reference[k] for k in COUNTERS
                  if k != "submissions" and reference[k] is not None and reference[k] > 0
                  and actual[k] is not None]
        if len(ratios) < 4 or not all(math.isclose(r, ratios[0]) for r in ratios):
            raise ValueError("uncertified legacy profile multiplicity")
        declared = ratios[0]
    if declared <= 0 or declared != int(declared):
        raise ValueError("invalid complete-inference multiplicity")
    for key in COUNTERS:
        # Submission count can change because of empty final submissions/UOP
        # warmup. Do not use it to validate whole-function invocation count.
        if key == "submissions":
            continue
        if actual[key] is not None and reference[key] is not None and not math.isclose(
                actual[key] / declared, reference[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("timing/correctness profile mismatch: {}".format(key))
    return declared


def measured_candidate(candidate, certificate, timing, expected_rounds, binary=None, features=None):
    mode = candidate.get("residence_mode", candidate.get("public_mode"))
    result = {
        "candidate_id": candidate["candidate_id"], "mode": mode,
        "config": config(candidate), "context": context(candidate),
        "binary_sha256": binary,
        "lowered_tir_sha256": candidate.get("lowered_tir_sha256") or candidate.get("tir_sha256"),
        "metrics": dict.fromkeys(METRICS), "missing": {}, "timing_rounds": [],
        "status": "fpga_invalid", "correctness_seed_count": 0,
        "sram_allocation_span_bytes": None,
    }
    seeds = certificate.get("seeds", [])
    if certificate.get("status") != "passed" or len(seeds) != 3 or any(
            not exact_correct(seed) for seed in seeds):
        if timing:
            raise ValueError("invalid identity has timing labels")
        return result
    if len({seed["seed"] for seed in seeds}) != 3:
        raise ValueError("duplicate correctness seed")
    result["correctness_seed_count"] = 3
    if len(timing) != expected_rounds or {r["round"] for r in timing} != set(range(expected_rounds)):
        raise ValueError("incomplete or duplicate timing rounds")
    if any(not exact_correct(r) or
           number(r.get("latency_ms")) is None or r["latency_ms"] <= 0 for r in timing):
        raise ValueError("wrong or invalid timing result")
    direct = [profile_counters(seed.get("runtime_profile_complete") or seed.get(
        "runtime_profile", {})) for seed in seeds]
    reference = {}
    for key in COUNTERS:
        values = [row[key] for row in direct]
        if key != "submissions" and all(v is not None for v in values) and len(set(values)) != 1:
            raise ValueError("seed-dependent logical program counters: {}".format(key))
        reference[key] = median(values)
    divisors = [inference_divisor(row, reference) for row in timing]
    per_call = [profile_counters(row.get("runtime_profile_complete", {}), n)
                for row, n in zip(timing, divisors)]
    result["metrics"].update({k: median([row[k] for row in per_call]) for k in COUNTERS})
    result["metrics"]["operator_latency_ms"] = statistics.median(r["latency_ms"] for r in timing)
    driver = [number(r.get("runtime_profile_complete", {}).get("driver_run_total_us"))
              for r in timing]
    result["metrics"]["driver_execution_ms"] = median(
        [v / n / 1000 if v is not None else None for v, n in zip(driver, divisors)])
    features = features or {}
    for key in ("instruction_peak_bytes", "uop_peak_bytes", "explicit_residency_drains"):
        result["metrics"][key] = number(features.get(key))
    for key in ("input_dma_bytes", "weight_dma_bytes", "output_dma_bytes", "total_dma_bytes",
                "total_dma_calls"):
        expected = number(features.get(key))
        actual = result["metrics"][key]
        if expected is not None and actual is not None and not math.isclose(expected, actual):
            raise ValueError("static/runtime feature mismatch: {}".format(key))
    result.update(status="measured", timing_rounds=sorted(
        [{"round": r["round"], "latency_ms": r["latency_ms"]} for r in timing],
        key=lambda r: r["round"]), profile_divisors=divisors,
        profile_normalization="declared invocation count or ratio certified by direct three-seed profiles",
        peak_provenance="frozen local/FSim command features; not a new board peak measurement")
    result["missing"] = {key: "missing frozen field; not replaced with zero"
                         for key, value in result["metrics"].items() if value is None}
    result["missing"]["sram_allocation_span_bytes"] = (
        "actual per-SRAM allocation spans unavailable; empty lowered Allocate list is not zero SRAM")
    return result


def family_result(workload, family_id, candidates, statuses=None):
    statuses = dict(statuses or {})
    if len({c["mode"] for c in candidates}) != len(candidates):
        raise ValueError("duplicate mode in one family")
    if any(c["mode"] not in MODES for c in candidates):
        raise ValueError("unsupported nominal/feasibility mode")
    if len({digest(c["config"]) for c in candidates}) != 1 or len(
            {digest(c["context"]) for c in candidates}) != 1:
        raise ValueError("same-tile family has different ConfigEntity or execution context")
    by_mode = {c["mode"]: c for c in candidates}
    statuses.update({mode: row["status"] for mode, row in by_mode.items()})
    for mode in MODES:
        statuses.setdefault(mode, "unavailable_in_frozen_qualified_pool")
    measured = {m: c for m, c in by_mode.items() if c["status"] == "measured"}
    if "original" not in measured:
        statuses = {m: "not_applicable_fallback_original_unqualified"
                    if s == "not_applicable_fallback_original" else s
                    for m, s in statuses.items()}
    four = all(m in measured for m in MODES)
    hashes = [measured[m]["binary_sha256"] for m in MODES] if four else []
    distinct = len(set(hashes)) == 4 if four and all(hashes) else (False if not four else None)
    original, combined = measured.get("original"), measured.get(MODES[3])
    effective = None
    if original and combined:
        a, b = original["metrics"], combined["metrics"]
        if all(v is not None for v in (a["input_dma_bytes"], a["weight_dma_bytes"],
                                      b["input_dma_bytes"], b["weight_dma_bytes"])):
            effective = b["input_dma_bytes"] < a["input_dma_bytes"] and b[
                "weight_dma_bytes"] < a["weight_dma_bytes"]
    coverage = {
        "workload_id": workload, "family_id": family_id,
        "config_sha256": digest(candidates[0]["config"]),
        "context_sha256": digest(candidates[0]["context"]),
        "has_original_measured": original is not None,
        "four_resolved_with_fallback": all(statuses[m] in (
            "measured", "not_applicable_fallback_original") for m in MODES),
        "four_measured": four, "four_distinct_programs": distinct,
        "effective_dual_reuse": effective,
        "sram_allocation_span_bytes": None,
        **{m + "_status": statuses[m] for m in MODES},
    }
    pairs = []
    if original:
        for mode in MODES[1:]:
            if mode not in measured:
                continue
            other = measured[mode]
            pair = {k: coverage[k] for k in ("workload_id", "family_id", "config_sha256",
                                            "context_sha256")}
            pair.update(mode=mode, original_candidate_id=original["candidate_id"],
                        variant_candidate_id=other["candidate_id"],
                        original_correctness_seed_count=3, variant_correctness_seed_count=3,
                        peak_provenance=original["peak_provenance"],
                        config=original["config"], missing={
                            "original": original["missing"], "variant": other["missing"]})
            for key in METRICS:
                a, b = original["metrics"][key], other["metrics"][key]
                pair["original_" + key], pair["variant_" + key] = a, b
                pair[key + "_delta_percent"] = delta(b, a)
            pair["latency_improvement_percent"] = -pair["operator_latency_ms_delta_percent"]
            a = {r["round"]: r["latency_ms"] for r in original["timing_rounds"]}
            b = {r["round"]: r["latency_ms"] for r in other["timing_rounds"]}
            common = set(a) & set(b)
            pair["paired_round_count"] = len(common)
            pair["paired_round_wins"] = sum(b[r] < a[r] for r in common)
            pair["raw_median_win_not_significance"] = pair["latency_improvement_percent"] > 0
            pairs.append(pair)
    return coverage, pairs


def timing_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["candidate_id"]].append(row)
    return groups


def frozen_features(candidate):
    """Use non-null legacy aliases, not null deferred fields as measurements."""
    result = dict(candidate.get("static_features") or {})
    totals = (candidate.get("dma_features") or {}).get("totals", {})
    for name, keys in COUNTERS.items():
        if number(result.get(name)) is None:
            values = [number(totals.get(key)) for key in keys]
            if all(v is not None for v in values):
                result[name] = sum(values)
    command = (candidate.get("command_features") or {}).get("values", {})
    peaks = command.get("peaks", {})
    for name, alias in (("instruction_peak_bytes", "insn_bytes"), ("uop_peak_bytes", "uop_bytes")):
        if number(result.get(name)) is None and number(peaks.get(alias)) is not None:
            result[name] = peaks[alias]
    if number(result.get("explicit_residency_drains")) is None:
        result["explicit_residency_drains"] = command.get("drain_check", {}).get(
            "observed_residency_drain_count")
    return result


def collect_r18(reader, dirs):
    analysis = reader.read(dirs["r18_analysis"], "results.jsonl")
    analysis_contract = reader.read(dirs["r18_analysis"], "contract.json")
    reader.directory(dirs["r18_adapter"], analysis_contract["source_adapter_manifest_sha256"])
    adapter = reader.read(dirs["r18_adapter"], "contract.json")
    for label, role in (("board_run", "r18_board"), ("board_pool", "r18_pool"),
                        ("cross_compile", "r18_cross")):
        reader.directory(dirs[role], adapter["input_hashes"][label])
    raw = indexed(reader.read(dirs["r18_pool"], "candidates.jsonl"))
    certificates = indexed(reader.read(dirs["r18_board"], "correctness.jsonl"))
    cross = indexed(reader.read(dirs["r18_cross"], "results.jsonl"))
    timing = timing_groups(reader.read(dirs["r18_board"], "timing.jsonl"))
    contract = reader.read(dirs["r18_board"], "contract.json")
    if set(raw) != set(certificates) or set(raw) != set(cross):
        raise ValueError("R18 pool/certificate identities disagree")
    candidates = {}
    for key, row in raw.items():
        if cross[key]["status"] != "passed" or cross[key]["relowered_tir_sha256"] != (
                row.get("lowered_tir_sha256") or row.get("tir_sha256")):
            raise ValueError("cross binary/TIR certificate disagreement")
        candidate = measured_candidate(row, certificates[key], timing.get(key, []),
                                       contract["timing_rounds"], cross[key]["binary"]["sha256"],
                                       frozen_features(row))
        candidates[key] = candidate
    coverage, pairs = [], []
    # Audit all qualified families, including those without an original control.
    groups = defaultdict(list)
    for key, row in raw.items():
        groups[(row["workload_id"], row["family_id"])].append(candidates[key])
    old = {(r["workload_id"], r["family_id"]): r for r in analysis}
    for (workload, family), rows in sorted(groups.items()):
        legacy = old.get((workload, family))
        if legacy:
            for scheme in legacy["schemes"]:
                if scheme["status"] == "not_applicable_fallback_original":
                    original = next((c for c in rows if c["mode"] == "original"), None)
                    if original is None or scheme[
                            "candidate_id"] != original["candidate_id"] or scheme.get(
                                "fallback_to") != "original":
                        raise ValueError("fallback does not identify actual same-tile original")
                if scheme["status"] == "measured":
                    c = candidates[scheme["candidate_id"]]
                    if c["mode"] != scheme["mode"] or c["config"] != legacy["config_entity"]:
                        raise ValueError("legacy family identity disagreement")
                    for metric in ("operator_latency_ms", "input_dma_bytes", "weight_dma_bytes",
                                   "total_dma_bytes", "total_dma_calls"):
                        if not math.isclose(c["metrics"][metric], scheme[metric]):
                            raise ValueError("legacy aggregate/raw measurement mismatch")
        statuses = {s["mode"]: s["status"] for s in legacy["schemes"]} if legacy else {}
        entry, ps = family_result(workload, family, rows, statuses)
        if legacy and entry["four_resolved_with_fallback"] != legacy["all_four_semantically_resolved"]:
            raise ValueError("legacy fallback classification disagreement")
        entry["session"] = "r18_board"
        coverage.append(entry)
        pairs.extend(dict(p, session="r18_board") for p in ps)
    return coverage, pairs


def collect_yolo_pool(reader, directory, session):
    contract = reader.read(directory, "contract.json")
    link = contract["p7r132"]
    source = reader.directory(link["path"], link["artifact_ledger_sha256"])
    collection = reader.read(source, "board_collection_contract.json")
    raw = indexed(collection["candidates"])
    pool = reader.read(directory, "completed_pool.json")
    cross = reader.read(directory, "fresh_cross_certificates.json")
    certificates = indexed(reader.read(directory, "correctness.jsonl"))
    timing = timing_groups(reader.read(directory, "timing.jsonl"))
    coverage, pairs = [], []
    for workload, data in pool["workloads"].items():
        groups = defaultdict(list)
        for row in data["candidates"]:
            key = row["candidate_id"]
            candidate = raw[key]
            if row["residence_mode"] != candidate["public_mode"]:
                raise ValueError("YOLO mode certificate disagreement")
            features = row["predispatch"]["validity"]
            translated = {name: features.get("static_" + name) for name in (
                "input_dma_bytes", "weight_dma_bytes", "output_dma_bytes", "total_dma_bytes",
                "total_dma_calls", "submissions")}
            translated.update(instruction_peak_bytes=features.get("static_insn_peak_bytes"),
                              uop_peak_bytes=features.get("static_uop_peak_bytes"),
                              explicit_residency_drains=features.get("static_residency_drains"))
            c = measured_candidate(candidate, certificates[key], timing[key],
                                   contract["timing_rounds"], cross[key]["binary_sha256"], translated)
            c["lowered_tir_sha256"] = cross[key]["tir_sha256"]
            groups[candidate["family_id"]].append(c)
        for family, rows in sorted(groups.items()):
            entry, ps = family_result(workload, family, rows)
            coverage.append(dict(entry, session=session))
            pairs.extend(dict(p, session=session) for p in ps)
    return coverage, pairs


def collect_y02(reader, dirs, session):
    directory = dirs[session]
    contract = reader.read(directory, "contract.json")
    link = contract["inputs"]
    source = reader.directory(link["p7r122_dir"], link["p7r122_artifact_ledger_sha256"])
    certificates = indexed(reader.read(source, "candidate_correctness.jsonl"))
    timing = timing_groups(reader.read(directory, "candidate_timing.jsonl"))
    groups = defaultdict(list)
    for candidate in contract["candidates"]:
        cert = candidate["fresh_axu_cross_certificate"]
        c = measured_candidate(candidate, certificates[candidate["candidate_id"]],
                               timing[candidate["candidate_id"]], contract["protocol"]["rounds"],
                               cert["binary_sha256"])
        c["lowered_tir_sha256"] = cert["tir_sha256"]
        groups[(candidate["workload_id"], candidate["family_id"])].append(c)
    coverage, pairs = [], []
    for (workload, family), rows in groups.items():
        entry, ps = family_result(workload, family, rows)
        coverage.append(dict(entry, session=session))
        pairs.extend(dict(p, session=session) for p in ps)
    return coverage, pairs


def summarize(coverage, pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["workload_id"], row["session"], row["mode"])].append(row)
    stats = []
    for (workload, session, mode), rows in sorted(groups.items()):
        stats.append({"workload_id": workload, "session": session, "mode": mode,
                      "pair_count": len(rows), "raw_median_wins": sum(
                          r["raw_median_win_not_significance"] for r in rows),
                      "median_latency_improvement_percent": statistics.median(
                          r["latency_improvement_percent"] for r in rows),
                      "median_deltas_percent": {key: median([r[key + "_delta_percent"] for r in rows])
                                                for key in COUNTERS}})
    counts = {}
    for workload in sorted({c["workload_id"] for c in coverage}):
        rows = [c for c in coverage if c["workload_id"] == workload]
        counts[workload] = {"family_session_count": len(rows), "unique_tile_context_count": len({
            (r["config_sha256"], r["context_sha256"]) for r in rows}),
            **{name: sum(r[name] is True for r in rows) for name in (
                "four_resolved_with_fallback", "four_measured", "four_distinct_programs",
                "effective_dual_reuse")},
            "unknown_distinct_program_count": sum(r["four_distinct_programs"] is None for r in rows),
            "unknown_dual_reuse_count": sum(r["effective_dual_reuse"] is None for r in rows)}
    return {"coverage": counts, "paired_effects": stats}


def input_rectangle_check(height=56, width=56, kernel=3, tile_h=8, tile_w=1):
    """Independent scalar coordinate enumeration vs clipped rectangle formula.

    Limited to stride-one odd-kernel rectangular input LOAD envelopes. This does
    not claim halo retention or validate every VTA access/UOP consumer.
    """
    enumerated, rectangular = 0, 0
    pad = kernel // 2
    for h0 in range(0, height, tile_h):
        for w0 in range(0, width, tile_w):
            hs, ws = min(tile_h, height - h0), min(tile_w, width - w0)
            points = {(h + kh - pad, w + kw - pad)
                      for h in range(h0, h0 + hs) for w in range(w0, w0 + ws)
                      for kh in range(kernel) for kw in range(kernel)
                      if 0 <= h + kh - pad < height and 0 <= w + kw - pad < width}
            enumerated += len(points)
            rectangular += (min(height, h0 + hs + pad) - max(0, h0 - pad)) * (
                min(width, w0 + ws + pad) - max(0, w0 - pad))
    if enumerated != rectangular:
        raise ValueError("independent input-region formula mismatch")
    return enumerated


def reference_checks(pairs):
    h1 = next(r for r in pairs if r["family_id"] == "R18-H1D0580")
    # Derive actual tile dimensions and CO outer count from the frozen config.
    knobs = {name: value[-1] if kind == "sp" else value
             for name, kind, value in h1["config"]["entity"]}
    area = input_rectangle_check(tile_h=knobs["tile_h"], tile_w=knobs["tile_w"])
    expected_input = area * 64 * (64 // (16 * knobs["tile_co"]))
    if expected_input != h1["original_input_dma_bytes"]:
        raise ValueError("H1 input rectangle count does not match frozen runtime")
    y02 = next(r for r in pairs if r["session"] == "y02_first")
    wknobs = {name: value[-1] if kind == "sp" else value
              for name, kind, value in y02["config"]["entity"]}
    unique_weight = 384 * 256 * 3 * 3
    spatial_groups = math.ceil(26 / wknobs["tile_h"]) * math.ceil(26 / wknobs["tile_w"])
    if unique_weight * spatial_groups != y02["original_weight_dma_bytes"] or unique_weight != y02[
            "variant_weight_dma_bytes"]:
        raise ValueError("Y02 weight reload count does not match frozen runtime")
    return {"H1_original_input_rectangle": {
        "enumerated_and_formula_source_pixels_over_tiles": area,
        "expected_and_runtime_input_bytes": expected_input,
        "scope": "one stride-one original LOAD-envelope check; not a general dependency proof"},
        "Y02_weight_reload": {"unique_weight_bytes": unique_weight,
                              "original_spatial_reload_groups": spatial_groups,
                              "resident_reload_groups": 1,
                              "scope": "one frozen same-tile weight case"}}


def counterexample_report(pairs):
    selected = [next(r for r in pairs if r["family_id"] == "R18-H1D0580"),
                next(r for r in pairs if r["workload_id"] == "R18-H2" and
                     r["original_input_dma_bytes"] == r["variant_input_dma_bytes"] and
                     r["original_weight_dma_bytes"] == r["variant_weight_dma_bytes"]),
                next(r for r in pairs if r["session"] == "y02_first"),
                next(r for r in pairs if r["family_id"] == "R18-H3D0103" and
                     r["mode"] == "weight_resident_barrier")]
    explanations = [
        "输入优先将空间循环放到输出通道循环之外，扩大输入加载作用域；该例输入-50%、权重+50%，"
        "总字节更多却更快。请求数也变化，不能把速度只归因于bytes或从相关性单独推出DMA启动成本。",
        "该tile的输入优先模式没有改变input/weight搬运量；模式flag不等于有效复用。"
        "小延迟差异不作显著性结论，仍需查具体LOAD/循环结构。",
        "权重compute_at提升到输出通道驻留作用域，在全部空间位置计算完成后drain；"
        "同一权重从跨13个空间组重复加载变为每层一次，新增同步仍取得净加速。"
        "两个历史会话分层报告；RPC本身未提供boot_id，跨启动声明来自原治理记录，不伪造新观测。",
        "原始tile已无可消除的重复权重字节；驻留减少请求并新增submission，driver累计时间下降，"
        "算子反而变慢。与同步/执行组织代价一致，但未做纯同步控制，不能精确归因。",
    ]
    lines = ["# 节点 A：既有输入/权重复用正反案例", "",
             "本报告只读聚合旧证据，不新增FPGA结果，不证明统一地址/消费者/覆盖安全。", ""]
    for row, explanation in zip(selected, explanations):
        lines += ["## {} / {} / {}".format(row["workload_id"], row["family_id"], row["mode"]), "",
                  "| 指标 | original | variant |", "|---|---:|---:|"]
        for metric in ("input_dma_bytes", "weight_dma_bytes", "input_dma_calls", "weight_dma_calls",
                       "total_dma_bytes", "total_dma_calls", "submissions", "synchronize_calls",
                       "instruction_peak_bytes", "uop_peak_bytes", "driver_execution_ms",
                       "operator_latency_ms"):
            lines.append("| {} | {} | {} |".format(metric, row["original_" + metric],
                                                   row["variant_" + metric]))
        lines += ["", explanation, "", "身份与来源：`{}` → `{}`，session `{}`。".format(
            row["original_candidate_id"], row["variant_candidate_id"], row["session"]), ""]
    lines += ["## 结论边界", "",
              "- 19组资源回退可解析不等于19组实际四模式实测；实际仅1组，且未形成双重复用。",
              "- 同tile配对只能说明该程序变换的组合效应，不隔离所有DMA、计算和同步分量。",
              "- 实际SRAM分配跨度、部分命令peak及异步覆盖关系仍有缺口；不将缺失值补零。",
              "- 新增组合实验优先选择四模式适用且双方原本都有重复加载的family。", ""]
    return "\n".join(lines)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "unknown" if value is None else json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def run(args):
    started = time.monotonic()
    root = Path(args.c3_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite audit output")
    reader = FrozenReader(root)
    dirs = {role: root / "07_grouped_holdout" / name for role, name in SOURCES.items()}
    for path in dirs.values():
        if output.is_relative_to(path.resolve()):
            raise ValueError("output cannot be inside frozen input directory")
        reader.directory(path)
    coverage, pairs = collect_r18(reader, dirs)
    for role in ("y00_board", "y03_board"):
        cs, ps = collect_yolo_pool(reader, dirs[role], role)
        coverage.extend(cs)
        pairs.extend(ps)
    for role in ("y02_first", "y02_repeat"):
        cs, ps = collect_y02(reader, dirs, role)
        coverage.extend(cs)
        pairs.extend(ps)
    stats = summarize(coverage, pairs)
    r18 = [r for r in pairs if r["session"] == "r18_board" and r["mode"] == "input_stationary"]
    if len(r18) != 40 or sum(c["four_resolved_with_fallback"] for c in coverage if c[
            "session"] == "r18_board") != 19 or sum(c["four_measured"] for c in coverage if c[
                "session"] == "r18_board") != 1:
        raise ValueError("Node A expected evidence counts not reproduced")
    checks = reference_checks(pairs)
    report = counterexample_report(pairs)
    reader.recheck()
    output.mkdir(parents=True)
    contract = {"schema": SCHEMA, "phase": "audit", "board_contacted": False,
                "new_candidates_frozen": False, "retrospective_only": True,
                "source_sha256": sha256(__file__), "sources": {k: str(v) for k, v in dirs.items()},
                "input_write_policy": "read-only and verified before/after",
                "next_action": "pause_after_node_A; Node B not started"}
    write_json(output / "contract.json", contract)
    write_json(output / "input_bindings.json", {str(k): v for k, v in reader.verified.items()})
    write_csv(output / "reuse_pairs.csv", pairs)
    write_csv(output / "family_coverage.csv", coverage)
    (output / "results.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pairs))
    write_json(output / "reference_checks.json", checks)
    (output / "counterexamples.md").write_text(report)
    summary = {"schema": SCHEMA, "status": "NODE_A_COMPLETE_PAUSED", "board_contacted": False,
               "new_fpga_dispatches": 0, "new_compiler_attempts": 0,
               "pair_session_count": len(pairs), "verified_input_directory_count": len(reader.verified),
               "verified_input_file_count": sum(len(v["files"]) for v in reader.verified.values()),
               "input_recheck_passed": True, "reference_checks": checks, **stats,
               "unknown_metrics": dict(Counter(k for r in pairs for side in r["missing"].values()
                                               for k in side)),
               "claim_boundary": "retrospective pair explanations; no new search or graph performance claim"}
    write_json(output / "summary.json", summary)
    (output / "timeline.jsonl").write_text(json.dumps({"event": "audit_complete", "wall_seconds":
                                                     time.monotonic() - started}) + "\n")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit",), default="audit")
    parser.add_argument("--c3-root", default=str(Path(__file__).parent /
                        "report_out/stage_tile_cotuning/c3_dma_residency_autotune"))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
