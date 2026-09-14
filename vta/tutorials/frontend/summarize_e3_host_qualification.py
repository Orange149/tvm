#!/usr/bin/env python3
"""Aggregate E3 host evidence, including exact independent-producer reuse check."""
import argparse
from collections import Counter
import json
from pathlib import Path

import tvm
from qualify_once_quantized_tail_split import save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=Path(
        "vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline=args.audit/"e3_once_quantized_run2"
    released=args.audit/"e3_barrier_release_run1"
    sources=[Path(__file__),Path(__file__).with_name("qualify_once_quantized_tail_split.py")]
    for directory in (baseline,released):
        protocol=directory/"preregistered.json"
        for path,digest in json.loads(protocol.read_text())["sources_sha256"].items():
            assert sha(Path(path))==digest,("Source changed",path)
        sources.extend([protocol,directory/"summary.json"])
    all_rows=[]
    for layer,start in ((2,5),(3,10),(4,15)):
        local=baseline/f"layer{layer}"
        inventory={}
        for arm in ("A","B","int8_producer","int8_consumer","float32_producer","float32_consumer"):
            path=local/arm/"fusion.json"
            sources.append(path)
            snapshots=json.loads(path.read_text())
            assert len(snapshots)==1 and snapshots[0]
            inventory[arm]=Counter(snapshots[0])
        fp=local/"float32_producer/relay.json"
        old=args.audit/"e3_tail_pilot_run1"/f"vta_{start:02d}_{start+1:02d}"/"quantized.json"
        sources.extend([fp,old,local/"result.json",released/f"layer{layer}"/"graph.json"])
        lhs=tvm.ir.load_json(fp.read_text())["main"].body
        rhs=tvm.ir.load_json(old.read_text())["main"].body
        exact=tvm.ir.structural_equal(lhs,rhs,map_free_vars=True)
        row={"layer":layer,"independent_producer_body_structurally_equal":bool(exact),
             "A_B_primitive_multisets_equal":inventory["A"]==inventory["B"],
             "A_native_split_primitive_multisets_equal":inventory["A"]==inventory["int8_producer"]+inventory["int8_consumer"],
             "primitive_counts":{k:sum(v.values()) for k,v in inventory.items()}}
        case=json.loads((local/"result.json").read_text())["cases"][0]
        row["one_edge_payload_bytes"]={label:sum(t["bytes"] for t in tensors) for label,tensors in case["transport"].items()}
        assert exact and row["A_B_primitive_multisets_equal"] and row["A_native_split_primitive_multisets_equal"]
        all_rows.append(row)
    baseline_summary=json.loads((baseline/"summary.json").read_text())
    released_summary=json.loads((released/"summary.json").read_text())
    assert baseline_summary["complete"] and baseline_summary["all_exact"] and released_summary["complete"]
    assert all(c["mismatches"]==0 for r in released_summary["rows"] for c in r["cases"])
    save(args.output,{"complete":True,"rows":all_rows,
        "six_arm_comparison_elements":baseline_summary["comparison_elements"],
        "barrier_release_comparison_elements":sum(c["elements"] for r in released_summary["rows"] for c in r["cases"]),
        "source_manifest_hashes_verified":True,"full_e3_complete":False,"new_board_runs":0,
        "sources_sha256":{str(p):sha(p) for p in sources}})


if __name__=="__main__":
    main()
