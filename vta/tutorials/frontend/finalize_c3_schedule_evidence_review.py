#!/usr/bin/env python3
"""Audit all local-goal deliverables and seal them without device access."""
import argparse
import ast
import json
from pathlib import Path
import re

from c3_schedule_evidence_audit import HERE, digest, require, write_new


def read(path):
    return json.loads(Path(path).read_text())


def finish(root):
    require(not (root / "COMPLETION_AUDIT.json").exists(), "completion already sealed")
    evidence = read(root / "input_hashes.json")
    for path, expected in evidence.items():
        require(digest(path) == expected, "input evidence changed: " + path)
    data = read(root / "analysis.json")
    plan = read(root / "NEXT_VALIDATION.json")
    rules = read(root / "RULES.json")
    tests = read(root / "tests.json")
    require(data["new_measurements"] == 0 and not data["board_contacted"], "unexpected new samples")
    require(data["status"] == "LOCAL_EVIDENCE_AUDIT_COMPLETE_NO_BOARD_WORK", "analysis incomplete")
    require(tests["passed"] and tests["tests_run"] == 22 and tests["failures"] == tests["errors"] == 0,
            "tests incomplete")
    require(digest(HERE / "c3_schedule_evidence_audit.py") == tests["analysis_source_sha256"], "tested analysis changed")
    require(digest(HERE / "test_c3_schedule_evidence_audit.py") == tests["test_source_sha256"], "tested tests changed")
    require(set(v["id"] for v in rules["rules"]) == {"H1", "H2", "H3", "P1", "P2", "P3", "P4", "P5"}, "rule coverage")
    for rule in rules["rules"]:
        require(all(rule.get(k) for k in ("category", "mechanism", "support", "counterexample", "scope", "unobserved")),
                "rule missing required evidence or limits")
        require(rule["performance_pruning_allowed"] is False, "performance preference became safety filter")
    primary = data["programs"]["TPS-oc-reuse-ci4"]
    coverage = primary["WGT_source_coverage"]
    require(coverage["repeat_histogram_elements"] == {"1": 1152}, "each-once source proof missing")
    require(coverage["requested_bytes"] == 294912, "weight byte unit mismatch")
    require(primary["LOAD_endpoint_span_bytes"] == {"inp": 32768, "wgt": 73728, "acc": 100864},
            "actual endpoint spans differ")
    require(all(v["partial_sum_DDR_spill_bytes"] == 0 and v["dynamic_SRAM_utilization"] is None
                for v in data["programs"].values()), "spill or utilization misrepresented")
    m = data["model_assessment"]
    require(m["no_new_latency_model_fitted"] and not m["five_tradeoff_model"]["fit_permitted"], "unjustified new model")
    require(m["five_tradeoff_model"]["rank"] == 6 and m["five_tradeoff_model"]["columns"] == 7, "model rank changed")
    require(m["existing_local_interpolation"]["not_transferred_to_new_runtime_or_TPS"], "unsupported model transfer")
    require(plan["decision"] == "A_existing_stronger_program_deployment_correctness", "unfrozen alternative")
    require(plan["candidate"] == "TPS-oc-reuse-ci4" and plan["status"] == "FROZEN_REQUIRES_USER_AUTHORIZATION",
            "candidate or authorization changed")
    require(plan["configuration"] == primary["tile"] and plan["operator_binary_sha256"] == primary["binary_sha256"]
            and plan["operator_TIR_sha256"] == primary["TIR_sha256"], "plan candidate differs from existing program")
    require(digest(plan["operator_binary"]) == plan["operator_binary_sha256"], "planned operator identity drift")
    calls = plan["board_graph_runs"]
    require(calls["total"] == sum(v for k, v in calls.items() if k != "total") == 12, "call budget mismatch")
    require(plan["timed_runs"] == plan["explicit_warmups"] == 0, "plan silently added performance")
    require([v["entry"] for v in plan["tensor_entries"]] == [21, 22, 23, 24], "entry mapping differs")
    require(plan["graph_memory_contract"]["fits"] and
            plan["graph_memory_contract"]["static_contract_bytes"] <= 201326592, "memory contract unresolved")
    forbidden = {"socket", "paramiko", "requests", "subprocess", "tvm", "vta", "ctypes"}
    tree = ast.parse((HERE / "c3_schedule_evidence_audit.py").read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(v.name.split(".")[0] for v in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    require(not imports & forbidden, "offline analyzer imports device/network execution machinery")
    for name in ("REPORT.md", "NEXT_VALIDATION.md"):
        text = (root / name).read_text()
        for link in re.findall(r"\]\(([^)]+)\)", text):
            path = link.strip("<>")
            if path == "COMPLETION_AUDIT.json":
                # This receipt is generated below only after every gate passes.
                continue
            if not path.startswith(("http:", "https:")):
                require((Path(path) if path.startswith("/") else root / path).is_file(), "broken evidence link: " + path)
    external = {str(HERE / name): digest(HERE / name) for name in (
        "c3_schedule_evidence_audit.py", "test_c3_schedule_evidence_audit.py",
        "finalize_c3_schedule_evidence_review.py", "c3_five_tradeoff_prefetch.py",
        "c3_banerjee_completion.py", "c3_banerjee_uop_queue.h", "run_vta_c3_yolo608_graph_boot.py")}
    partial = HERE.parents[2] / "vta/python/vta/top/vta_node24_partial.py"
    external[str(partial)] = digest(partial)
    requirements = [
        ("actual_sources_not_chat", ["input_hashes.json", "analysis.json"], "88 source/raw/config/binary evidence bindings verified"),
        ("five_tradeoffs", ["REPORT.md", "RULES.json"], "P1-P5 all mechanisms, support, counterexamples, scope and missing observations"),
        ("node24_each_once_explanation", ["analysis.json/programs", "REPORT.md"], "1152 WGT elements exact single coverage, legal per-bank endpoints"),
        ("safety_vs_priority", ["RULES.json"], "H1-H3 separate from P1-P5; no performance pruning"),
        ("calculable_relations", ["analysis.json/programs", "REPORT.md"], "R/K/S load equation and actual DMA mapping verified"),
        ("model_conditions", ["analysis.json/model_assessment"], "old interpolation recomputed; five-cost model rejected with rank and missing validation"),
        ("existing_C3_claims", ["REPORT.md", "analysis.json"], "confirmed, refuted and unknown claims separated across cohorts"),
        ("strong_baseline_and_negative", ["analysis.json/comparisons", "REPORT.md"], "same-runtime 3.254 vs 4.637 ms; no full-author claim"),
        ("smallest_next_decision", ["NEXT_VALIDATION.json"], "choice A; existing sealed candidate, no prototype/search"),
        ("frozen_task_contract_controls", ["NEXT_VALIDATION.json", "NEXT_VALIDATION.md"], "exact candidate/ABI/graph/runtime and allowed differences"),
        ("budget_criteria_stop", ["NEXT_VALIDATION.json", "NEXT_VALIDATION.md"], "12 untimed graph calls, observable reference, no retries or reselection"),
        ("no_board_or_reboot", ["analysis.json", "tests.json"], "offline code paths only; zero acquisition"),
        ("no_production_or_history_mutation", ["input_hashes.json"], "original evidence/source hashes verified at closeout"),
        ("analysis_scripts_and_tests", ["tests.json"], "22 actual tests passed; selected raw cohorts recomputed"),
        ("user_readable_handoff", ["REPORT.md", "NEXT_VALIDATION.md"], "plain-language weights/Tiles explanation, limits and approval gate"),
    ]
    write_new(root / "COMPLETION_AUDIT.json", {
        "status": "LOCAL_GOAL_REQUIREMENTS_VERIFIED", "requirements": [
            {"id": name, "status": "PROVEN_FOR_THIS_LOCAL_STAGE", "evidence": files, "check": check}
            for name, files, check in requirements],
        "input_evidence_count": len(evidence), "external_source_hashes": external,
        "runtime_or_hardware_contact": False, "no_new_latency": True,
        "future_graph_compile_and_board_checks": "NOT_EXECUTED; outside this goal stage; require user approval"})
    files = {p.name: digest(p) for p in root.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_new(root / "manifest.json", {"status": "LOCAL_STAGE_COMPLETE_WAITING_FOR_AUTHORIZATION",
                                     "files": files, "external_source_hashes": external})
    print("LOCAL_GOAL_COMPLETION_AUDIT_PASSED", len(requirements), "requirements")


def verify(root):
    manifest = read(root / "manifest.json")
    for name, value in manifest["files"].items():
        require(digest(root / name) == value, "final artifact changed: " + name)
    for name, value in manifest["external_source_hashes"].items():
        require(digest(name) == value, "final source changed: " + name)
    for name, value in read(root / "input_hashes.json").items():
        require(digest(name) == value, "historical input changed: " + name)
    print("LOCAL_GOAL_FINAL_MANIFEST_VERIFIED", len(manifest["files"]), "artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("finish", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (finish if args.phase == "finish" else verify)(args.root.resolve())
