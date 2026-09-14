#!/usr/bin/env python3
"""Compare full representative LOAD/STORE against frozen bare-workload aggregation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census",type=Path,required=True)
    parser.add_argument("--scan",type=Path,required=True)
    parser.add_argument("--workloads",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise FileExistsError(args.output)
    read=lambda p:json.loads(p.read_text())
    key=lambda w:json.dumps(w,sort_keys=True)
    baseline={(key(r["workload"]),r["config_index"]):r for r in read(args.workloads)["rows"]}
    scan={r["segment_id"]:r for r in read(args.scan/"summary.json")["rows"]}
    records=[]
    for row in read(args.census)["rows"]:
        aggregate=Counter()
        convs=scan[row["segment_id"]]["convolutions"]
        for conv in convs:
            b=baseline[(key(conv["workload"]),conv["config"]["index"])]
            for metric,value in b["static_dma"]["totals"].items():
                if metric.endswith(("_calls","_bytes")) and "average" not in metric:
                    aggregate[metric]+=value
        full=row["logical_descriptor_totals"]
        delta={k:full.get(k,0)-aggregate.get(k,0) for k in set(full)|set(aggregate)}
        unexplained={k:v for k,v in delta.items() if v and k not in {
            "load_buffer_2d_calls","load_buffer_2d_bytes","load_buffer_2d_small_calls",
            "load_buffer_2d_strided_calls","load_buffer_2d_acc_calls","load_buffer_2d_acc_bytes"}}
        acc_calls=full.get("load_buffer_2d_acc_calls",0)
        acc_bytes=full.get("load_buffer_2d_acc_bytes",0)
        # Also reconstruct small/strided residual directly from ACC descriptors.
        acc_small=acc_strided=0
        for f in row["functions"].values():
            for d in f["static_dma"]["request_descriptors"]:
                if d["direction"]=="load" and d["memory_name"]=="acc":
                    mult=f["invocations"]*d["multiplicity"]
                    acc_small+=mult*(d["bytes_per_request"]<4096)
                    acc_strided+=mult*(d["x_stride"]!=d["x_size"])
        explained=not unexplained and all(delta.get(k,0)==v for k,v in {
            "load_buffer_2d_calls":acc_calls,"load_buffer_2d_bytes":acc_bytes,
            "load_buffer_2d_small_calls":acc_small,"load_buffer_2d_strided_calls":acc_strided}.items())
        records.append({"segment_id":row["segment_id"],"convolutions":len(convs),
                        "full_load_calls":full.get("load_buffer_2d_calls",0),
                        "full_load_bytes":full.get("load_buffer_2d_bytes",0),
                        "bare_load_calls":aggregate["load_buffer_2d_calls"],
                        "bare_load_bytes":aggregate["load_buffer_2d_bytes"],
                        "acc_calls":acc_calls,"acc_bytes":acc_bytes,
                        "full_store_bytes":full.get("store_buffer_2d_bytes",0),
                        "residual_exactly_acc":explained,"delta":delta})
    result={"scope":"Logical LOAD/STORE static-to-static reconciliation; not new board qualification or ALU counts",
            "sources_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [args.census,args.scan/"summary.json",args.workloads,Path(__file__)]},
            "segments":len(records),"segments_with_only_acc_residual":sum(r["residual_exactly_acc"] for r in records),
            "rows":records}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    fields=[k for k in records[0] if k!="delta"]
    with args.output.with_suffix(".csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in records)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
