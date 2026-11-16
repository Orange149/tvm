"""Small prospective phase-gap check, not a baseline or complete Top20 study.

Two unseen resource layouts per network, stratified by the two largest front
thread counts still available. One short performance session per layout.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from cpu_vta_partitioning.common import read, write, sha256, digest, verify_inventory
from finite_service.common import freeze
from cpu_policies.open_resources.domain import canonical_tuple
from policy_branch_eval.common import isolated_inputs_only
from prepare_finite_service_holdout import collect_known, fingerprint
from compiled_sleep_model import CompiledFinite
from isolated_output_control_model import IsolatedOutputFinite
from scheduler_gap_audit.curves import csv_write
import finite_service_holdout_board as parent

ROOT = parent.ROOT
OUTPUT = parent.OUTPUT
CONFIG = parent.CONFIG
POOL = ROOT/"artifacts/compiled_gap_validation_01"
RUN = ROOT/"runs/compiled_gap_validation_01"
BOOT = "6018b3b7-bb29-4f80-92c1-0bfd8c81bb99"


def sources():
    return {**parent.sources(), **{str(ROOT/p):sha256(ROOT/p) for p in
        ("run_compiled_gap_validation.py", "compiled_sleep_model.py", "isolated_output_control_model.py",
         "prepare_finite_service_holdout.py", "tests/test_compiled_gap_validation.py")}}


def choose(cases, known):
    selected = {}
    for network, group in cases.items():
        selected[network] = []
        available = sorted({c["configuration"]["stages"][0]["threads"] for c in group.values()
                            if fingerprint(c["configuration"]) not in known})
        if len(available) < 2:
            raise ValueError("insufficient distinct unseen thread strata")
        for threads in available[-2:]:
            rows = [c for c in group.values() if c["configuration"]["stages"][0]["threads"]==threads
                    and fingerprint(c["configuration"]) not in known]
            if not rows:
                raise ValueError("insufficient unseen cases in a declared thread stratum")
            selected[network].append(min(rows, key=lambda c:(c["static_score_ns"], canonical_tuple(c["configuration"]))))
    if sum(map(len, selected.values())) != 4:
        raise ValueError("expected exactly four prospective diagnostic layouts")
    return selected


def rank():
    POOL.mkdir(parents=True, exist_ok=True)
    if (POOL/"SELECTION_FREEZE.json").exists():
        selection(); return dict(cached=True, selected_configurations=4)
    if not (POOL/"KNOWN_EXECUTIONS.json").exists():
        known = collect_known(POOL)
    else:
        known = set(read(POOL/"KNOWN_EXECUTIONS.json")["fingerprints"])
    with isolated_inputs_only():
        src = ROOT/"artifacts/device_completion_holdout_01"
        inputs = read(src/"INPUTS.json")
        p = read(ROOT/"artifacts/compiled_sleeper_01/COMPILED_SLEEPER.json")
        costs = read(ROOT/"artifacts/isolated_output_replay_01/ISOLATED_COSTS.json")["costs"]
        old = read(src/"SELECTION_FREEZE.json")
        chosen = choose(inputs["networks"], known)
        cases = {n:{c["configuration"]["id"]:c for c in rows} for n,rows in chosen.items()}
        freeze(POOL/"INPUTS.json", dict(networks=cases, window=inputs["window"], quanta_ns=inputs["quanta_ns"],
            output_costs=costs, parameters=p, contains_pipeline_performance=False))
        freeze(POOL/"PLAN.json", dict(sources=sources(), inputs_sha256=sha256(POOL/"INPUTS.json"),
            known_metadata_sha256=sha256(POOL/"KNOWN_EXECUTIONS.json"), expected_boot_id=BOOT,
            scope="phase prediction of unseen layouts under two fixed known topologies; no full-domain claim",
            selection="original isolated-score best in each of the two largest still-unmeasured front-thread strata; same rule for both networks",
            thread_strata={n:[c["configuration"]["stages"][0]["threads"] for c in rows] for n,rows in chosen.items()},
            methods=["original", "compiled", "compiled_output_ring1", "compiled_output_ring4"],
            primary="compiled_output_ring1", ring4="cache-state sensitivity, never chosen by native performance",
            qualification=dict(warmup=0, scored=1, tail=0), performance=dict(warmup=8, scored=16, tail=4),
            max_configurations=4, max_native_sessions=8, max_DNN_inferences=116,
            no_anchors=True, no_drift_rejection=True, no_reboot=True, no_automatic_repeats=True,
            no_formula_refit_or_full_domain_search=True,
            success="report each front-stage error and per-network absolute-error change; do not infer Top20 gain"))
        files = {}
        for network, rows in cases.items():
            for cid, c in rows.items():
                baseline = src/"predictions"/network/(cid+"_midpoint.json")
                expected = old["prediction_files"][str(baseline.relative_to(src))]
                if sha256(baseline) != expected:
                    raise ValueError("cached original prediction changed")
                files[network+"/"+cid+"/original"] = dict(path=str(baseline), sha256=expected)
                for method in ("compiled", "compiled_output_ring1", "compiled_output_ring4"):
                    path = POOL/"predictions"/network/(cid+"_"+method+".json")
                    if not path.exists():
                        common = dict(quanta=inputs["quanta_ns"], wake=True, sleeper=True,
                            wakeup_granularity_ns=p["wakeup_granularity_ns"], sleeper_credit_ns=p["sleeper_credit_ns"],
                            placement="wake_boundary", completion_model="independent", completion_point="midpoint",
                            runnable_completion_wait=True, controls=None, gate="off", host_order="before", **inputs["window"])
                        if method=="compiled":
                            result = CompiledFinite(c["stages"], **common).run()
                        else:
                            result = IsolatedOutputFinite(c["stages"],
                                isolated_output_cost=costs[network+"_ring"+method[-1]],
                                collector_mask=c["configuration"]["control_mask"], **common).run()
                        freeze(path, dict(result=result, plan_sha256=sha256(POOL/"PLAN.json"), no_pipeline_values_read=True))
                    if read(path)["plan_sha256"] != sha256(POOL/"PLAN.json"):
                        raise ValueError("prospective prediction identity changed")
                    files[network+"/"+cid+"/"+method] = dict(path=str(path), sha256=sha256(path))
                print("未测布局阶段预测", network, cid, flush=True)
        freeze(POOL/"SELECTION_FREEZE.json", dict(plan_sha256=sha256(POOL/"PLAN.json"), prediction_files=files,
            selected_configurations={n:[c["configuration"] for c in rows.values()] for n,rows in cases.items()},
            no_pipeline_performance_read=True, selection_before_native=True, DNN_budget=116))
    return dict(selected_configurations=4, native_sessions=8, DNN_inferences=116)


def selection():
    s = read(POOL/"SELECTION_FREEZE.json"); p = read(POOL/"PLAN.json")
    if p["sources"] != sources() or sha256(POOL/"PLAN.json") != s["plan_sha256"]:
        raise ValueError("prospective controller changed")
    if sha256(POOL/"INPUTS.json") != p["inputs_sha256"]:
        raise ValueError("prospective inputs changed")
    known = set(read(POOL/"KNOWN_EXECUTIONS.json")["fingerprints"])
    if sum(map(len, s["selected_configurations"].values())) != 4 or s["DNN_budget"] != 116:
        raise ValueError("prospective budget changed")
    for item in s["prediction_files"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("prospective prediction changed")
    for network, configs in s["selected_configurations"].items():
        if sorted(c["stages"][0]["threads"] for c in configs) != p["thread_strata"][network] or len(set(p["thread_strata"][network])) != 2:
            raise ValueError("front-stage thread strata incomplete")
        if any(fingerprint(c) in known for c in configs):
            raise ValueError("prospective layout already in known metadata")
    return s


def protocol():
    r = parent.protocol()
    r.update(experiment="compiled_gap_validation_01", purpose="prospective stage gap, not global baseline comparison")
    return r


def adapt(name):
    f = getattr(parent, name)
    g = FunctionType(f.__code__, {**f.__globals__, **globals()}, f.__name__, f.__defaults__)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def specs(): return adapt("specs")()
def prepare(): return adapt("prepare")()
def validate(bundle): return adapt("validate")(bundle)
def make_task(bundle, spec): return adapt("make_task")(bundle, spec)
def native_command(doc, bundle, remote_bundle, remote_job): return adapt("native_command")(doc,bundle,remote_bundle,remote_job)
def matched_ssh(config, *, execute_board=False): return adapt("matched_ssh")(config, execute_board=execute_board)
def qemu(): return adapt("qemu")()
summary = parent.summary
accept = parent.accept


def board(*, execute_board=False):
    if read(POOL/"BOARD_PLAN.json")["expected_boot_id"] != BOOT:
        raise ValueError("reused board preparation boot differs")
    return adapt("board")(execute_board=execute_board)


def analyze():
    s = selection(); state = read(RUN/"state.json")
    if state["status"] != "COMPLETE" or state["boot_id"] != BOOT or state["accepted_sessions"] != 8:
        raise ValueError("prospective native cohort incomplete")
    rows, receipts, qualified = [], {}, {}
    for spec in specs():
        bundle=POOL/spec["network"]/"capsule";task=make_task(bundle,spec)
        path=RUN/"accepted"/BOOT/(task["task_id"]+".json");r=read(path)
        if r["task"]!=task or r["boot_id"]!=BOOT:
            raise ValueError("native task identity mismatch")
        verify_inventory(r["raw"],r["files"])
        if summary(r["raw"],task,bundle)!=r["summary"] or (Path(r["raw"])/"task_hash").read_text().strip()!=digest(task):
            raise ValueError("native raw data no longer replay")
        key=spec["network"]+"/"+spec["case"];receipts[str(path)]=sha256(path)
        if spec["phase"]=="qualify":
            qualified[key]=r["summary"]["output_fingerprints"];continue
        if qualified[key]!=r["summary"]["output_fingerprints"]:
            raise ValueError("qualification/output mismatch")
        for method in read(POOL/"PLAN.json")["methods"]:
            p=read(s["prediction_files"][key+"/"+method]["path"])["result"]["replay"]
            n=r["summary"]
            rows.append(dict(network=spec["network"],configuration_id=spec["case"],method=method,
                CPU_threads=[v["threads"] for v in task["stages"] if v["device"]=="cpu"],
                CPU_masks=[v["mask"] for v in task["stages"] if v["device"]=="cpu"],
                predicted_front_ms=p["stage_service_ns"][0]/1e6,native_front_ms=n["stage_service_ms"][0],
                front_error_ms=p["stage_service_ns"][0]/1e6-n["stage_service_ms"][0],
                predicted_II_ms=p["interval_ns"]/1e6,native_II_ms=n["ii_ms"],native_FPS=n["fps"],
                all_native_stage_service_ms=n["stage_service_ms"],all_predicted_stage_service_ms=[v/1e6 for v in p["stage_service_ns"]]))
    result=dict(rows=rows,receipts=receipts,source_sha256=sha256(__file__),
        selection_sha256=sha256(POOL/"SELECTION_FREEZE.json"),DNN_inferences=116,native_sessions=8,
        independently_frozen_unseen_layout_validation=True,no_full_DP_or_Top20_or_13_FPS_remeasurement_claim=True)
    write(POOL/"RESULTS.json",result,replace=True);csv_write(POOL/"phase_comparison.csv",rows)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("rank","prepare","validate","qemu","board","status","analyze"))
    p.add_argument("--execute-board",action="store_true");a=p.parse_args()
    if a.command=="board":r=board(execute_board=a.execute_board)
    elif a.command=="validate":r={n:validate(POOL/n/"capsule") for n in selection()["selected_configurations"]}
    elif a.command=="status":r=read(RUN/"state.json") if (RUN/"state.json").exists() else dict(status="NOT_STARTED")
    else:r=globals()[a.command]()
    print(json.dumps(r,ensure_ascii=False,indent=2))
