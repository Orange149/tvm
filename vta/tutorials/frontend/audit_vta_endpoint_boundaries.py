#!/usr/bin/env python3
"""Contract ledger for three complete CPU-prefix/VTA/CPU-suffix endpoint pairs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def slot_bytes(slot):
    assert slot["dtype"]=="float32"
    return 4*math.prod(slot["shape"])


def edge(producer,consumer):
    out,inp=producer["output_contract"],consumer["input_contract"]
    descriptor=lambda c:[(s["shape"],s["dtype"]) for s in c["slots"]]
    assert descriptor(out)==descriptor(inp),"Adjacent contract mismatch"
    sizes=[slot_bytes(s) for s in out["slots"]]
    return {"producer":producer["segment_id"],"consumer":consumer["segment_id"],
            "direction":producer["device"]+"_to_"+consumer["device"],
            "contract":out,"tensor_payload_bytes":sizes,"total_payload_bytes":sum(sizes),
            "theoretical_k2_aligned_slot_bytes":2*sum((n+255)//256*256 for n in sizes),
            "materialization_accounting":"one-transfer logical payload only; actual API copy multiplicity depends on runner",
            "zero_copy_status":"shape/dtype necessary conditions only; storage/alignment/physical binding unqualified",
            "live_interval":"producer write/ready -> consumer completion/release, independently per frame generation"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    profile=json.loads(args.profile.read_text())
    by_id={r["segment_id"]:r for r in profile["segments"]}
    pairs=[]
    for start in (5,10,15):
        versions={}
        for label,end in [("tail_outside",start+1),("tail_inside",start+2)]:
            ids=[f"cpu:00:{start-1:02d}",f"vta:{start:02d}:{end:02d}",f"cpu:{end+1:02d}:20"]
            stages=[by_id[i] for i in ids]
            flattened=[u for s in stages for u in s["unit_names"]]
            assert len(flattened)==21 and len(set(flattened))==21
            edges=[edge(a,b) for a,b in zip(stages,stages[1:])]
            versions[label]={"path":ids,"cpu_threads":[1,1],"thread_choice_scope":"fixed structural control, not optimized ranking",
                "stages":[{k:s[k] for k in ("segment_id","unit_names","input_contract","output_contract")} for s in stages],
                "edges":edges,"boundary_payload_bytes":sum(e["total_payload_bytes"] for e in edges),
                "theoretical_k2_slot_bytes":sum(e["theoretical_k2_aligned_slot_bytes"] for e in edges),
                "compiled_full_pipeline":False,"numerical_equivalence_qualified":False}
        left,right=versions["tail_outside"],versions["tail_inside"]
        assert left["edges"][0]==right["edges"][0] or all(
            left["edges"][0][k]==right["edges"][0][k] for k in ("contract","total_payload_bytes","theoretical_k2_aligned_slot_bytes"))
        pairs.append({"layer":(start//5)+1,"versions":versions,
            "boundary_payload_delta_bytes":right["boundary_payload_bytes"]-left["boundary_payload_bytes"],
            "theoretical_k2_slot_delta_bytes":right["theoretical_k2_slot_bytes"]-left["theoretical_k2_slot_bytes"],
            "tail_owner_outside":left["path"][2],"tail_owner_inside":"CPU helper inside "+right["path"][1],
            "mandatory_followup":"Compare full CPU suffix lowering and quantization; do not add or drop tail compute twice"})
    result={"scope":"Six complete manifest paths, contract-only; not 4623-topology E6-S completion",
            "profile_sha256":hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cost_ownership":{"vta_dma":"LOAD/STORE only within VTA stage",
                              "internal_cpu_helpers":"owned by VTA stage execution, not charged again as edge copy",
                              "external_cpu_tail":"owned by CPU suffix only in tail_outside version",
                              "boundary":"one contract payload per edge; copy API count separately qualified",
                              "slot_capacity":"theoretical two aligned buffers per live edge, not allocator high-water"},
            "pairs":pairs}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps([{k:v for k,v in p.items() if k!="versions"} for p in pairs],ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
