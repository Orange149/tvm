"""Read-only alignment of frozen scheduler replays with accepted worker traces.

No input generation, model fitting, native execution, or ranking is performed.
The model's ready interval starts before its first CPU service. Native callback
CPU queue counters start inside the callback, so their totals are not identical
measurements and must not be subtracted to construct a penalty.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import json
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cpu_vta_partitioning.common import read, sha256, write
from phase_model.simulator import rational
from scheduler_gap_audit.curves import csv_write
from finite_service.common import freeze

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/compiled_gap_alignment_01"


def model_path(result, frame, kernel, native_slot):
    """One completion path per kernel; never sum concurrent workers' waits."""
    tasks = [t for t in result["tasks"] if t["stage"] == 0
             and t["frame"] == frame and t["kernel"] == kernel
             and t["operation"] == "worker_compute"]
    if len(tasks) != 3 or {t["slot"] for t in tasks} != {0, 1, 2}:
        raise ValueError("missing or duplicate worker task")
    if any(t.get("generation") != 0 for t in tasks):
        raise ValueError("internal barrier requires a different path projection")
    joins = [j for j in result["kernel_joins"] if j["stage"] == 0
             and j["frame"] == frame and j["kernel"] == kernel]
    if len(joins) != 1 or joins[0]["generation"] != 0:
        raise ValueError("missing or duplicate completion join")
    join = joins[0]
    if join["tasks"] != 3 or join["join_ns"] != max(t["end_ns"] for t in tasks):
        raise ValueError("join does not wait for all workers")
    chosen = min(t["slot"] for t in tasks if t["end_ns"] == join["join_ns"])
    last = next(t for t in tasks if t["slot"] == chosen)
    same = next(t for t in tasks if t["slot"] == native_slot)
    for t in tasks:
        if not t["ready_ns"] <= t["first_service_ns"] <= t["end_ns"]:
            raise ValueError("model worker timestamps out of order")
        residual = (t["end_ns"] - t["ready_ns"] - rational(t["CPU"])
                    - rational(t["ready_not_running"]))
        if abs(residual) > 2:
            raise ValueError("rounded timestamps do not conserve model CPU and ready time")
    initial = same["first_service_ns"] - same["ready_ns"]
    return dict(
        chosen_slot=chosen,
        identity_agrees=native_slot in join["last_slots"],
        critical_CPU_ns=rational(last["CPU"]),
        critical_ready_ns=rational(last["ready_not_running"]),
        same_native_slot_ready_ns=rational(same["ready_not_running"]),
        same_native_slot_initial_ready_ns=Fraction(initial),
        same_native_slot_post_first_service_ready_ns=rational(same["ready_not_running"]) - initial,
        completion_span_ns=join["join_ns"] - join["ready_ns"],
    )


def run():
    # Frozen predictions already exist before any diagnostic trace is opened.
    sources = ROOT / "artifacts/compiled_sleeper_01"
    seal = read(sources / "REPLAY_FREEZE.json")
    plan = read(sources / "PLAN.json")
    if seal["plan_sha256"] != sha256(sources / "PLAN.json"):
        raise ValueError("prediction plan changed")
    for path, expected in plan["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("prediction implementation changed")
    for item in seal["paths"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("frozen prediction changed")
    from run_three_thread_worker_gap_compat import verified_receipts
    from analyze_three_thread_worker_gap import parse_front
    import run_three_thread_worker_gap as native
    selected = native.selection()
    bundle = native.POOL / "yolov3_tiny416/capsule"
    receipts = verified_receipts()
    freeze(OUT / "PLAN.json", dict(
        prediction_freeze_sha256=sha256(sources / "REPLAY_FREEZE.json"),
        native_receipts={str(p): sha256(p) for p, _ in receipts},
        source_sha256=sha256(__file__),
        prediction_parameters_and_inputs_not_changed=True,
        comparison_is_known_case_diagnosis_only=True,
        native_queue_and_model_ready_have_different_markers=True,
        new_DNN=0, new_native_sessions=0,
    ))
    summaries, kernels = [], []
    native_summary = read(native.POOL / "analysis/RESULTS.json")
    quality = {r["role"]: r for r in native_summary["rows"]}
    for _, receipt in receipts:
        task = receipt["task"]
        if task["spec"]["phase"] != "observed" or task["spec"]["scope"] != "pipeline":
            continue
        cid = task["configuration"]["id"]
        role = selected["roles"][task["spec"]["case"]]
        observed = parse_front(receipt, bundle)
        for variant in ("original", "wake_only", "sleep_only", "both"):
            result = read(seal["paths"][variant + "/" + cid]["path"])["result"]
            paths = []
            for k in observed["kernels"]:
                p = model_path(result, k["frame"], k["kernel"], k["last_worker"])
                paths.append(p)
                kernels.append(dict(
                    role=role, variant=variant, frame=k["frame"], kernel=k["kernel"],
                    function_offset=k["function_offset"], native_last_slot=k["last_worker"],
                    native_entry_ms=k["entry_delay_ns"] / 1e6,
                    native_critical_CPU_ms=k["callback_CPU_ns"] / 1e6,
                    native_callback_wall_minus_CPU_ms=k["callback_wall_minus_CPU_ns"] / 1e6,
                    **{(key[:-3] + "_ms" if key.endswith("_ns") else key):
                       (float(value) / 1e6 if key.endswith("_ns") else value)
                       for key, value in p.items()},
                ))
            q = quality[role]
            frames = len(observed["frames"])
            row = dict(role=role, variant=variant, configuration_id=cid,
                predicted_front_ms=result["replay"]["stage_service_ns"][0] / 1e6,
                native_clean_front_ms=q["clean_front_service_ms"],
                native_probe_front_ms=q["probe_front_service_ms"],
                native_probe_disturbance_ms=q["probe_front_disturbance_ms"],
                quantitative_remaining_error_attribution_supported=q["quantitative_comparison_supported"],
                native_callback_queue_lower_ms=q["sum_last_finisher_queue_lower_ms"],
                native_callback_queue_upper_ms=q["sum_last_finisher_queue_upper_ms"],
                native_entry_ms=q["entry_delay_ms"],
                native_callback_wall_minus_CPU_ms=q["callback_wall_minus_CPU_ms"],
                last_slot_agreement=st.mean(p["identity_agrees"] for p in paths),
                model_last_slot_counts=dict(Counter(p["chosen_slot"] for p in paths)),
                native_last_slot_counts=observed["last_workers"],
            )
            for key in ("critical_CPU_ns", "critical_ready_ns", "same_native_slot_ready_ns",
                        "same_native_slot_initial_ready_ns", "same_native_slot_post_first_service_ready_ns",
                        "completion_span_ns"):
                row[key.replace("_ns", "_ms_per_frame")] = float(sum(p[key] for p in paths)) / frames / 1e6
            summaries.append(row)
    result = dict(rows=summaries, per_kernel=kernels, plan_sha256=sha256(OUT / "PLAN.json"),
        source_sha256=sha256(__file__), new_DNN=0, new_native_sessions=0,
        no_predicted_penalty_from_native_wait=True, no_new_parameter_or_profile=True,
        no_full_domain_or_selection_gain_claim=True,
        native_probe_quality_does_not_support_exact_remaining_gap_attribution=True,
        no_wall_times_summed_across_concurrent_workers=True)
    write(OUT / "RESULTS.json", result, replace=True)
    csv_write(OUT / "worker_paths.csv", summaries)
    csv_write(OUT / "per_kernel.csv", kernels)
    return result


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run().items() if k != "per_kernel"}, ensure_ascii=False, indent=2))
