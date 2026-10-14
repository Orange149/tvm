#!/usr/bin/env python3
"""Read-only evidence checks and final accounting; never contacts the board.

This is a post-acquisition auditor, not part of the frozen execution source set.
It cannot select another candidate, acquire samples, or repair failed samples.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def audit(root):
    protocol = read(root / "protocol.json")
    prepared = read(root / "prepared.json")
    selected = read(root / "selected_pool.json")
    result = read(root / "results.json")
    qualification = read(root / "qualification/summary.json")
    formal = read(root / "board/summary.json")
    boot = selected["boot_id"]
    require(not list(root.rglob("invalid_attempt.json")), "invalid board attempt exists")
    for name, expected in protocol["source_sha256"].items():
        require(sha(name) == expected, "frozen execution source changed: " + name)
    require(sha(protocol["document"]) == protocol["document_sha256"], "protocol text changed")
    require(sha(root / "protocol.json") == prepared["protocol_sha256"], "prepared protocol mismatch")
    require(sha(root / "prepared.json") == qualification["prepared_sha256"], "qualification identity mismatch")
    for helper in protocol["helpers"].values():
        for key in ("source", "binary"):
            require(sha(helper[key]) == helper[key + "_sha256"], "helper identity changed")
        require(helper["defines_VTA_runtime"] is False, "helper embeds a second runtime")
    for name, item in prepared["programs"].items():
        require(sha(item["binary"]) == item["binary_sha256"], "program binary changed: " + name)
    runtime = read(root / "runtime_contract.json")
    for name, path in runtime["runtime_paths"].items():
        require(sha(path) == protocol["runtime_sha256"][name], "runtime binary changed: " + name)
    for folder in ("runtime_sim", "runtime_board_attempt02"):
        build = read(root / folder / "build.json")
        require(build["returncode"] == 0 and build["enabled"], "runtime build failed or disabled")
        require("-DC3_BANERJEE_VALUE_UOPS=1" in build["command"] and
                "-DC3_BANERJEE_GREEDY_UOPS=1" in build["command"], "runtime optimization flags missing")

    previous = root.parent / "20261013_c3_banerjee_baseline_completion_run01"
    protected = read(previous / "protected.json")
    for name, expected in protected.items():
        require(sha(name) == expected, "protected production/history changed: " + name)
    local = read(root / "local_index.json")
    dirs = [Path(p) for p in local.values()] + [root / "anchors_attempt02/R13-G",
        root / "anchors_attempt02/N24-R12-G2-PREFETCH", root / "anchors_final/Input-7x7"]
    for directory in dirs:
        cert = read(directory / "native_certificate.json")
        require("PASSED" in cert["status"], "native certificate not passed")
        if "result_sha256" in cert:
            require(sha(directory / "result.json") == cert["result_sha256"], "local result identity mismatch")
        else:
            require(cert["all_tokens_paired"] and len(cert["proof"]) == 6, "anchor proof incomplete")
        result_local = read(directory / "result.json")
        require(result_local["runtime_sha256"] == sha(root / "runtime_sim/libvta_fsim.so"),
                "local certificate used a different runtime")
        rr = [read(directory / ("call_%02d.json" % i)) for i in range(1, 7)]
        require(all(c["correct"] and c["arguments_unchanged"] and c["same_handles"] for c in rr),
                "local fusion/immutable argument failure")
        require(rr[3]["output_sha256"] == rr[5]["output_sha256"] != rr[4]["output_sha256"],
                "local same-handle ABA intervention not observable")

    stage_rows, session_count, kind_counts = {}, {}, {}
    ids = set()
    for stage, expected_calls, expected_sessions in (("qualification", 159, 27),
            ("selection", 60, 10), ("board", 216, 36)):
        calls = rows(root / stage / "calls.jsonl")
        sessions = rows(root / stage / "sessions.jsonl")
        require(len(calls) == expected_calls and len(sessions) == expected_sessions, "incomplete " + stage)
        for session in sessions:
            require(session["status"] == "PASS", "failed session")
            require(session["runtime_attestation"] == protocol["runtime_sha256"], "runtime attestation mismatch")
            if session["program"] != "helper_regression":
                require("libvta.so" in session["loaded_maps"], "actual VTA DSO mapping missing")
            require(session["bitstream_load_record"]["sha256"] == runtime["bitstream_sha256"],
                    "bitstream changed")
            for field in ("environment", "end_environment"):
                env = session[field]
                require(env["boot_id"] == boot and env["masks"] and set(env["masks"]) == {"3"},
                        "boot or actual worker CPU affinity changed")
        for c in calls:
            require(c["invocation_id"] not in ids, "duplicate invocation")
            ids.add(c["invocation_id"])
            require(c["boot_id"] == boot and c["status"] == "PASS" and c["correct"] and
                    c["arguments_unchanged"], "call not correct or identity changed")
            if c["call_kind"] != "helper_regression":
                require(c["complete_operator_calls"] == 1 and c["hidden_warmup_calls"] == 0,
                        "hidden or repeated complete operator")
                require(c["output_sha256"] == c["expected_sha256"], "output mismatch")
                require(c["runtime_profile"]["driver_timeout_calls"] == 0, "device timeout")
                if "buffer_handles" in c:
                    related = [x for x in calls if x["session_id"] == c["session_id"]]
                    require(all(x["buffer_handles"] == c["buffer_handles"] for x in related), "handles changed")
            timed = c["call_kind"] in ("timing", "selection_timing")
            require((isinstance(c["latency_ms"], (int, float)) and math.isfinite(c["latency_ms"])
                     and c["latency_ms"] > 0) if timed else c["latency_ms"] is None, "invalid time field")
        stage_rows[stage] = calls
        session_count[stage] = len(sessions)
        kind_counts[stage] = dict(Counter(c["call_kind"] for c in calls))
    require(kind_counts["qualification"] == qualification["call_counts"], "qualification accounting mismatch")
    require(kind_counts["board"] == formal["call_counts"], "formal accounting mismatch")
    for name in prepared["programs"]:
        q = [c for c in stage_rows["qualification"] if c["program"] == name]
        require(len(q) == 12, "qualification count wrong")
        aba = [c for c in q if c["call_kind"] == "weight_ABA"]
        require(aba[0]["output_sha256"] == aba[2]["output_sha256"] != aba[1]["output_sha256"],
                "board ABA references not distinguishing")

    require(sha(root / "selection/calls.jsonl") == selected["selection_data_sha256"], "selection data changed")
    scores = {}
    for name in local:
        cells = [c for c in stage_rows["selection"] if c["program"] == name]
        require(len(cells) == 6, "selection panel incomplete")
        timings = [c for c in cells if c["call_kind"] == "selection_timing"]
        require(len(timings) == 3 and [c["seed"] for c in timings] == protocol["seeds"], "selection seeds wrong")
        scores[name] = median(c["latency_ms"] for c in timings)
    require(scores == selected["scores_ms"], "selection scores mismatch")
    winners = [min((n for n in scores if n.startswith(prefix)), key=lambda n: (scores[n], n))
               for prefix in ("TPS-single", "TPS-oc-reuse", "TPS-h-reuse")]
    require(winners + ["R13-G", "N24-R12-G2-PREFETCH", "Input-7x7"] == selected["programs"],
            "sealed winners not from selection panel")
    require(min(scores, key=lambda n: (scores[n], n)) == selected["primary_TPS_baseline"], "primary changed")
    contract = read(root / "board/contract.json")
    # The inherited contract retains all 13 qualification bindings; the sealed
    # orders and the actual call log, not this inventory, define the six-arm pool.
    require(set(contract["programs"]) == set(prepared["programs"]) and contract["orders"] == selected["orders"],
            "confirmation differs from sealed pool")
    require(all(contract["programs"][name] == prepared["programs"][name] for name in selected["programs"]),
            "confirmation program identity differs from qualification")
    programs = selected["programs"]
    round_ms, profile_summary = {}, {}
    for rnd, order in enumerate(selected["orders"]):
        units = [c for c in stage_rows["board"] if c["round"] == rnd]
        require([units[i]["program"] for i in range(0, len(units), 6)] == order, "execution order mismatch")
        round_ms[str(rnd)] = {}
        for name in programs:
            cells = [c for c in units if c["program"] == name]
            require([c["call_kind"] for c in cells] == ["warmup"] * 3 + ["timing"] * 3,
                    "warmup/timing protocol mismatch")
            require([c["seed"] for c in cells] == protocol["seeds"] * 2, "formal seed order mismatch")
            require([c["call_position"] for c in cells] == list(range(1, 7)), "call position mismatch")
            round_ms[str(rnd)][name] = median(c["latency_ms"] for c in cells[3:])
    require(round_ms == result["round_medians"], "formal aggregation mismatch")
    require({name: median(round_ms[str(r)][name] for r in range(6)) for name in programs}
            == result["median_ms"], "median of round medians mismatch")
    require(sha(root / "board/calls.jsonl") == result["raw_calls_sha256"], "formal raw SHA mismatch")
    counts = rows(root / "board/helper_counts.jsonl")
    require(len(counts) == 108 and all(c["count"] == c["expected"] for c in counts), "helper counts mismatch")
    require({c["invocation_id"] for c in counts} == {c["invocation_id"] for c in stage_rows["board"]
            if c["call_kind"] == "timing"}, "helper counts not bound to timed calls")
    for comparison in result["comparisons"]:
        gains = [(1 - round_ms[str(r)][comparison["candidate"]] / round_ms[str(r)][comparison["baseline"]]) * 100
                 for r in range(6)]
        require(gains == comparison["round_improvement_pct"] and median(gains) == comparison["median_improvement_pct"]
                and sum(x > 0 for x in gains) == comparison["wins"], "paired result mismatch")

    signatures = read(root / "qualification/diagnostic_signatures.json")
    for name in programs:
        times = [c for c in stage_rows["board"] if c["program"] == name and c["call_kind"] == "timing"]
        signature = signatures[name]["rotations"]["0"][0]
        require(all(s == signature for s in signatures[name]["rotations"]["0"]), "warmed signature changes")
        request = prepared["programs"][name]["request_totals"]
        profile_summary[name] = dict(signature,
            logical_INP_KiB=request["load_buffer_2d_inp_bytes"] / 1024,
            logical_WGT_KiB=request["load_buffer_2d_wgt_bytes"] / 1024,
            median_device_run_wait_ms=median(c["runtime_profile"]["device_run_wait_us"] / 1000 for c in times),
            median_call_minus_device_wait_ms=median(c["latency_ms"] - c["runtime_profile"]["device_run_wait_us"] / 1000
                                                 for c in times),
            not_pure_CPU_time_or_hardware_idle=True)
    tests = ET.parse(root / "tests.xml").getroot()
    suites = list(tests.iter("testsuite"))
    require(sum(int(s.attrib.get("tests", 0)) for s in suites) == 32 and
            not any(int(s.attrib.get("errors", 0)) or int(s.attrib.get("failures", 0)) for s in suites),
            "Python regression failure")
    cpp = Path("/tmp/c3_banerjee_uop_queue_test")
    regression = subprocess.run([str(cpp)], check=True, capture_output=True, text=True)
    value = {"status": "FINAL_EVIDENCE_AND_ACCOUNTING_PASSED", "board_contacted_by_final_audit": False,
        "boot_id": boot, "frozen_execution_source_count": len(protocol["source_sha256"]),
        "protected_history_and_production_files": len(protected), "local_native_certificates": len(dirs),
        "stage_call_counts": kind_counts, "stage_sessions": session_count,
        "target_FPGA_calls": 432, "CPU_helper_regression_calls": 3, "total_actual_calls": len(ids),
        "selection_panel_recomputed_and_disjoint_from_evaluation": True,
        "primary_TPS_baseline": selected["primary_TPS_baseline"], "all_correct": True,
        "formal_positive_timing_samples": 108, "formal_aggregate_and_pairwise_results_recomputed": True,
        "profile_summary": profile_summary, "Python_tests_passed": 32,
        "cpp_queue_regression": {"exit_code": regression.returncode, "binary_sha256": sha(cpp),
            "source_sha256": sha(Path(__file__).with_name("c3_banerjee_uop_queue_test.cc")),
            "header_sha256": sha(Path(__file__).with_name("c3_banerjee_uop_queue.h"))},
        "limitations": ["one Linux boot", "fixed node24 full fusion", "software adaptation, not author hardware",
            "restricted TPS tie panel, not full-space oracle", "no new graph timing"],
        "auditor_sha256": sha(__file__), "results_sha256": sha(root / "results.json")}
    require(len(ids) == 435, "total invocation accounting mismatch")
    write_new(root / "final_audit.json", value)
    print(json.dumps({k: value[k] for k in ("status", "total_actual_calls", "Python_tests_passed", "primary_TPS_baseline")}))


def seal(root):
    require(read(root / "final_audit.json")["status"] == "FINAL_EVIDENCE_AND_ACCOUNTING_PASSED", "audit missing")
    require((root / "REPORT.md").is_file(), "report missing")
    write_new(root / "manifest.json", {"schema": "c3_banerjee_completion_final_manifest_v1",
        "status": "COMPLETE_PAUSED", "files": {p.relative_to(root).as_posix(): sha(p)
          for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"},
        "auditor_sha256": sha(__file__)})


def verify(root):
    manifest = read(root / "manifest.json")
    for name, expected in manifest["files"].items():
        require(sha(root / name) == expected, "sealed artifact changed: " + name)
    require(sha(__file__) == manifest["auditor_sha256"], "auditor changed")
    print("FINAL_MANIFEST_VERIFIED", len(manifest["files"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "seal", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"audit": audit, "seal": seal, "verify": verify}[args.phase](args.root.resolve())
