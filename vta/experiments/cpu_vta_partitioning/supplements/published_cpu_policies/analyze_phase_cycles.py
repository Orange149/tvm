"""Separate stage work, between-frame gaps, and finite-window latency drift."""
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"artifacts/phase_cycle_audit_01"


def ns(value):
    exact=Decimal(str(value))*1000000
    if exact!=exact.to_integral_value():
        raise ValueError("timestamp precision exceeds integer ns")
    return int(exact)


def decompose(rows, stage=0, warmup=8, scored=16):
    ids=[r["frame_id"] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate frame")
    by={r["frame_id"]:r for r in rows}
    keep=list(range(warmup,warmup+scored))
    if scored<2 or any(f not in by for f in keep):
        raise ValueError("missing retained frame")
    starts={f:ns(by[f][f"stage{stage}_start_ms"]) for f in keep}
    ends={f:ns(by[f][f"stage{stage}_end_ms"]) for f in keep}
    completions={f:ns(by[f]["completion_ms"]) for f in keep}
    if (any(ends[f]<starts[f] or completions[f]<ends[f] for f in keep)
        or any(starts[b]<ends[a] or completions[b]<completions[a] for a,b in zip(keep,keep[1:]))):
        raise ValueError("stage sequence or completion timestamps inconsistent")
    first,last=keep[0],keep[-1];gaps=scored-1
    service=sum(ends[f]-starts[f] for f in keep[:-1])
    between=sum(starts[b]-ends[a] for a,b in zip(keep,keep[1:]))
    stage_cycle=starts[last]-starts[first]
    completion_interval=completions[last]-completions[first]
    latency_change=(completions[last]-starts[last])-(completions[first]-starts[first])
    if service+between!=stage_cycle or stage_cycle+latency_change!=completion_interval:
        raise ValueError("time decomposition does not telescope")
    def avg(v):return float(Fraction(v,gaps))/1e6
    endpoints=[]
    count=by[first]["stage_count"]
    for f in (first,last):
        c=completions[f]
        completed=[sum(ns(r[f"stage{i}_end_ms"])<=c for r in rows) for i in range(count)]
        endpoints.append(dict(frame=f,completed_by_stage=completed,
            final_completed=sum(ns(r["completion_ms"])<=c for r in rows),
            latency_from_stage_start_ms=(c-starts[f])/1e6))
    return dict(stage=stage,first_frame=first,last_frame=last,completion_gaps=gaps,
        stage_service_ms_per_cycle=avg(service),
        between_frame_gap_ms=avg(between),stage_start_cycle_ms=avg(stage_cycle),
        downstream_latency_change_per_gap_ms=avg(latency_change),
        completion_interval_ms=avg(completion_interval),
        latency_change_across_window_ms=latency_change/1e6,
        endpoints=endpoints,exact_integer_ns_identity_passed=True,
        latency_drift_is_not_additional_stage_CPU_work=True)


def run():
    import run_compiled_gap_validation as v
    state=read(v.RUN/"state.json")
    if state["status"]!="COMPLETE" or state["boot_id"]!=v.BOOT:
        raise ValueError("native validation incomplete")
    rows=[];proof={}
    for spec in v.specs():
        if spec["phase"]!="clean":continue
        bundle=v.POOL/spec["network"]/"capsule";task=v.make_task(bundle,spec)
        p=v.RUN/"accepted"/v.BOOT/(task["task_id"]+".json");r=read(p)
        if r["task"]!=task or r["boot_id"]!=v.BOOT:raise ValueError("native identity changed")
        verify_inventory(r["raw"],r["files"])
        raw=Path(r["raw"])/"result.jsonl"
        frames=[json.loads(s,parse_float=Decimal) for s in raw.read_text().splitlines()]
        if len(frames)!=28:raise ValueError("window differs")
        for i,s in enumerate(task["stages"]):
            rows.append(dict(network=spec["network"],configuration_id=spec["case"],
                program=s["program"],device=s["device"],threads=s["threads"],mask=s["mask"],
                **decompose(frames,i)))
        proof[str(p)]=sha256(p);proof[str(raw)]=sha256(raw)
    result=dict(rows=rows,source_files=proof,source_sha256=sha256(__file__),
        no_new_DNN=True,no_new_model_or_penalty=True,
        exact_cycle_identity="completion_interval = stage_service + inter-frame_gap + change_in_downstream_latency_per_gap",
        no_claim_of_stationary_or_steady_state_from_short_window=True)
    write(OUT/"RESULTS.json",result,replace=True);csv_write(OUT/"cycles.csv",rows)
    return result


if __name__=="__main__":
    x=run();print(json.dumps({**x,"rows":[r for r in x["rows"] if r["stage"]==0]},ensure_ascii=False,indent=2))
