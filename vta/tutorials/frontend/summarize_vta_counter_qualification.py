#!/usr/bin/env python3
"""Join disjoint E2 batches and independently check every stored counter/output result."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",nargs="+",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    read=lambda p:json.loads(p.read_text())
    source=read(args.archive/"summary.json")
    census={r["segment_id"]:r for r in read(args.archive/"logical_dma_census_v2.json")["rows"]}
    rows=[]
    for directory in args.runs:
        batch=read(directory/"summary.json")
        assert batch["complete"]
        assert {r["segment_id"] for r in batch["results"]}==set(read(directory/"preregistered.json")["segments"])
        rows+=batch["results"]
    assert len(rows)==len({r["segment_id"] for r in rows})
    assert {r["segment_id"] for r in rows}=={r["segment_id"] for r in source["rows"]}
    checks=elements=0
    details=[]
    for row in rows:
        assert row["status"]=="passed" and row["build_tir_equal"]
        samples=row["samples"]
        assert len(samples)==9 and {(s["input_case"],s["repeat"]) for s in samples}=={
            (case,repeat) for case in range(3) for repeat in range(3)}
        c=census[row["segment_id"]]
        expected=dict(c["logical_descriptor_totals"])
        expected.update(push_alu_op_calls=c["uop_scope_calls"].get("VTAPushALUOp",0),
                        push_gemm_op_calls=c["uop_scope_calls"].get("VTAPushGEMMOp",0))
        for sample in samples:
            assert not sample["counter_differences"] and sample["correctness"]["correct"]
            for key,value in expected.items():
                assert sample["runtime_profile"][key]==value,(row["segment_id"],key)
                checks+=1
            for output in sample["correctness"]["outputs"]:
                assert output["mismatched"]==0 and output["max_abs_error"]==0
                elements+=math.prod(output["shape"])
        details.append({"segment_id":row["segment_id"],"samples":9,"counter_fields":len(expected),
                        "load_calls":expected["load_buffer_2d_calls"],"load_bytes":expected["load_buffer_2d_bytes"],
                        "acc_load_bytes":expected["load_buffer_2d_acc_bytes"],
                        "store_calls":expected["store_buffer_2d_calls"],"store_bytes":expected["store_buffer_2d_bytes"],
                        "alu_push_calls":expected["push_alu_op_calls"],"gemm_push_calls":expected["push_gemm_op_calls"]})
    snapshots=[r[k] for r in rows for k in ("before","after")]
    boots=sorted({s["boot"] for s in snapshots})
    assert len(boots)==1
    result={"complete":True,"scope":"All frozen representative functions/counters; one boot, no timing/FPS or physical DDR claim",
            "segments":len(rows),"profiled_runs":len(rows)*9,"warmup_runs":len(rows)*6,
            "input_cases":len(rows)*3,"output_element_comparisons_including_repeats":elements,
            "counter_field_comparisons":checks,"counter_mismatches":0,"output_mismatches":0,
            "boot_ids":boots,"frequency_khz":sorted({s["frequency"] for s in snapshots}),
            "governors":sorted({s["governor"] for s in snapshots}),
            "temperature_ps_c_sampled_range":[min(s["temperature_ps_c"] for s in snapshots),max(s["temperature_ps_c"] for s in snapshots)],
            "sources_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [*(d/"summary.json" for d in args.runs),*(d/"preregistered.json" for d in args.runs),
                 args.archive/"logical_dma_census_v2.json",Path(__file__)]},
            "rows":sorted(details,key=lambda r:r["segment_id"])}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows","sources_sha256"}},indent=2))


if __name__=="__main__":
    main()
