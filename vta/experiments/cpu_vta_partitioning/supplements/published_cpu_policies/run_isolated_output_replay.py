"""Frozen development replay: compiled wake/sleep rules plus isolated output work.

All original and both predeclared cache regimes are reported on all seven known
cases. No FPS-based coefficient, input refit, native DNN, or full-domain search.
"""
from pathlib import Path
import json
import shutil
import statistics as st
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cpu_vta_partitioning.common import read, write, sha256, digest
from finite_service.common import freeze
from policy_branch_eval.common import isolated_inputs_only
from phase_model.simulator import rational
from isolated_output_control_model import IsolatedOutputFinite
from scheduler_gap_audit.curves import csv_write
import run_output_control_profile as control

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/isolated_output_replay_01"


def project():
    control.validate()
    state = read(control.RUN / "state.json")
    if state["status"] != "COMPLETE" or state["accepted_sessions"] != 4 or state["DNN_inferences"]:
        raise ValueError("isolated CPU-only cohort incomplete")
    plan = read(control.OUT / "PLAN.json")
    costs = {}
    for spec in control.specs():
        key = spec["network"] + "_ring" + str(spec["ring_count"])
        receipt = control.RUN / (key + ".json")
        r = read(receipt)
        if (r["spec"] != spec or r["boot_id"] != state["boot_id"]
            or r["plan_sha256"] != sha256(control.OUT / "PLAN.json") or r["DNN_inferences"]
            or r["FPGA_calls"] or r["output_sha256"] != digest(r["output"])):
            raise ValueError("isolated control identity changed")
        value = control.validate_output(r["output"], spec)
        cpu = (sum(s["CPU_ns"] for s in value["samples"]) + 7) // 8
        cost = dict(schema="isolated-output-control-v1", origin="isolated_output_serialization",
            network=spec["network"], ring_count=spec["ring_count"], CPU_ns=cpu,
            warmup=2, scored=8, output_bytes=value["output_bytes"], output_count=len(value["raw_outputs"]),
            receipt_sha256=sha256(receipt), plan_sha256=sha256(control.OUT / "PLAN.json"),
            runtime_sha256=sha256(control.BUNDLE / "runtime/libtvm_runtime.so"),
            serializer_source_sha256=next(v for p, v in plan["source_files"].items()
                if p.endswith("vta_stage_pipeline_runner_phase.cc")),
            contains_pipeline_data=False, measured_scope="ResultJSON_and_deferred_string_insertion")
        costs[key] = cost
    return freeze(OUT / "ISOLATED_COSTS.json", dict(costs=costs,
        no_DNN_or_pipeline_measurement_used=True, both_predeclared_cache_regimes_retained=True))


def run():
    started = time.monotonic()
    project()
    with isolated_inputs_only():
        costs = read(OUT / "ISOLATED_COSTS.json")["costs"]
        params = read(ROOT / "artifacts/compiled_sleeper_01/COMPILED_SLEEPER.json")
        three = read(ROOT / "artifacts/three_thread_host_gap_01/existing_model_diagnostic/INPUTS.json")
        four = read(ROOT / "artifacts/device_completion_holdout_01/INPUTS.json")
        old_three = read(ROOT / "artifacts/compiled_sleeper_01/REPLAY_FREEZE.json")
        old_four = read(ROOT / "artifacts/compiled_kernel_crosscheck_01/REPLAY_FREEZE.json")
        if three["window"] != four["window"] or three["quanta_ns"] != four["quanta_ns"]:
            raise ValueError("known replay windows or platform granules differ")
        cases = {}
        for cid, stages in three["cases"].items():
            cases["host_layout/"+cid] = dict(network="yolov3_tiny416", stages=stages,
                configuration_id=cid, collector_mask=[0,1,2,3], reference=old_three["paths"]["both/"+cid])
        for key, reference in old_four["paths"].items():
            network, cid = key.split("/")
            c = four["networks"][network][cid]
            cases["cross_network/"+key] = dict(network=network, stages=c["stages"], configuration_id=cid,
                collector_mask=c["configuration"]["control_mask"], reference=reference)
        if len(cases) != 7:
            raise ValueError("expected every previously defined development case")
        for c in cases.values():
            if sha256(c["reference"]["path"]) != c["reference"]["sha256"]:
                raise ValueError("frozen no-control comparison changed")
        plan = dict(cases=cases, window=three["window"], quanta_ns=three["quanta_ns"],
            source_hashes={str(p):sha256(p) for p in (Path(__file__), ROOT/"isolated_output_control_model.py",
                ROOT/"compiled_sleep_model.py")}, isolated_costs_sha256=sha256(OUT/"ISOLATED_COSTS.json"),
            params_sha256=sha256(ROOT/"artifacts/compiled_sleeper_01/COMPILED_SLEEPER.json"),
            variants=["no_control", "ring1", "ring4"],
            all_variants_required_no_favorable_cache_choice=True,
            no_other_profile_or_scheduler_change=True, no_native_DNN=True,
            source_backed_missing_work_test_not_unseen_validation=True)
        freeze(OUT/"PLAN.json", plan)
        paths = {}
        for key, c in cases.items():
            for ring in (1,4):
                path = OUT/"replays"/(key.replace("/", "_")+"_ring"+str(ring)+".json")
                if not path.exists():
                    r = IsolatedOutputFinite(c["stages"], quanta=three["quanta_ns"],
                        wake=True, sleeper=True, wakeup_granularity_ns=params["wakeup_granularity_ns"],
                        sleeper_credit_ns=params["sleeper_credit_ns"],
                        isolated_output_cost=costs[c["network"]+"_ring"+str(ring)],
                        collector_mask=c["collector_mask"], placement="wake_boundary", completion_model="independent",
                        completion_point="midpoint", runnable_completion_wait=True, controls=None,
                        gate="off", host_order="before", **three["window"]).run()
                    freeze(path, dict(result=r, plan_sha256=sha256(OUT/"PLAN.json"), no_pipeline_values_read=True))
                if read(path)["plan_sha256"] != sha256(OUT/"PLAN.json"):
                    raise ValueError("cached replay identity differs")
                paths[key+"/ring"+str(ring)] = dict(path=str(path), sha256=sha256(path))
        freeze(OUT/"REPLAY_FREEZE.json", dict(paths=paths, plan_sha256=sha256(OUT/"PLAN.json"),
            all_predictions_frozen_before_native_results_open=True))
    # Only analysis below can read already-known native pipeline values.
    native_three = read(ROOT/"artifacts/compiled_sleeper_01/RESULTS.json")
    native_four = read(ROOT/"artifacts/compiled_kernel_crosscheck_01/RESULTS.json")
    observed = {r["configuration_id"]:r for r in native_three["rows"] if r["variant"]=="both"}
    observed.update({r["configuration_id"]:r for r in native_four["rows"]})
    rows = []
    for key, c in cases.items():
        n = observed[c["configuration_id"]]
        for variant in ("no_control", "ring1", "ring4"):
            r = read(c["reference"]["path"] if variant=="no_control" else paths[key+"/"+variant]["path"])["result"]
            front = r["replay"]["stage_service_ns"][0]/1e6
            rows.append(dict(case=key, network=c["network"], configuration_id=c["configuration_id"], variant=variant,
                predicted_front_ms=front, native_front_ms=n["native_front_ms"], front_error_ms=front-n["native_front_ms"],
                predicted_II_ms=r["replay"]["interval_ns"]/1e6, native_II_ms=n["native_II_ms"],
                native_FPS=n["native_FPS"],
                modeled_collector_CPU_ms=sum(float(rational(t["CPU"])) for t in r["actor_totals"]
                    if t["role"]=="_collector")/three["window"]["scored"]/1e6))
    result = dict(rows=rows, new_native_DNN=0, new_FPGA_calls=0, known_cases_only=True,
        no_DP_or_Top20_or_new_throughput_claim=True, all_cache_regimes_reported=True,
        no_fitting_from_pipeline=True, control_profile_native_sessions=4,
        independent_control_scope="output serialization only, other controls remain unknown",
        offline_seconds=time.monotonic()-started, source_sha256=sha256(__file__),
        plan_sha256=sha256(OUT/"PLAN.json"))
    write(OUT/"RESULTS.json", result, replace=True)
    csv_write(OUT/"stage_comparison.csv", rows)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
