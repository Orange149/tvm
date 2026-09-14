#!/usr/bin/env python3
"""Independent E3 board sample, byte ledger, hash and range audit."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from qualify_once_quantized_tail_split import save,sha
from qualify_vta_cpu_tail_handoff import check_ranges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    destination=args.run/"audited_summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol=json.loads((args.run/"preregistered.json").read_text())
    for path,digest in protocol["sources_sha256"].items():
        assert sha(Path(path))==digest,("Changed source",path)
    raw=json.loads((args.run/"summary.json").read_text())
    assert raw["complete"] and len(raw["rows"])==3
    totals=Counter()
    rows=[]
    for row in raw["rows"]:
        assert row["passed"] and len(row["samples"])==27
        assert row["before"]["boot"]==row["after"]["boot"]==protocol["board_before"]["boot"]
        assert row["cpu_tail_sha256"]==sha(args.run/f"layer{row['layer']}/cpu_tail.so")
        assert row["buffer_probe_sha256"]==sha(args.run/"buffer_probe.so")
        ranges=row["shared_ranges"]
        check_ranges(ranges,protocol["udmabuf_base"],int(protocol["board_before"]["udmabuf"]))
        expected_keys={(case,repeat,arm) for case in range(3) for repeat in range(3) for arm in ("mono","shared","copy")}
        assert {(s["case"],s["repeat"],s["arm"]) for s in row["samples"]}==expected_keys
        by_key={(s["case"],s["repeat"],s["arm"]):s for s in row["samples"]}
        payload=sum(r["bytes"] for r in ranges)
        dma_keys=[k for k in row["samples"][0]["profile_after_producer"]
                  if k.startswith(("load_buffer_2d_","store_buffer_2d_"))
                  and k.endswith(("_calls","_bytes"))]
        for sample in row["samples"]:
            assert sample["correctness"]["correct"] and not sample["counter_differences"] and not sample["tail_counter_differences"]
            outputs=sample["correctness"]["outputs"]
            assert all(o["mismatched"]==0 for o in outputs)
            totals["sampled_vta_runs"]+=1
            totals["final_output_element_comparisons"]+=sum(math.prod(o["shape"]) for o in outputs)
            copy=sample["arm"]=="copy"
            assert sample["boundary_copy_api_calls"]==(2 if copy else 0)
            assert sample["boundary_copy_payload_bytes"]==(payload if copy else 0)
            totals["boundary_copy_api_calls"]+=sample["boundary_copy_api_calls"]
            totals["boundary_copy_payload_bytes"]+=sample["boundary_copy_payload_bytes"]
            if sample["arm"]!="mono":
                assert sample["producer_correctness"]["correct"]
                totals["cpu_tail_samples"]+=1
                totals["producer_element_comparisons"]+=sum(math.prod(o["shape"]) for o in sample["producer_correctness"]["outputs"])
                mono=by_key[sample["case"],sample["repeat"],"mono"]["profile_after_producer"]
                assert all(sample["profile_after_producer"][k]==mono[k] for k in dma_keys)
                assert all(sample["profile_after_tail"][k]==sample["profile_after_producer"][k] for k in dma_keys)
        rows.append({"layer":row["layer"],"samples":27,"all_exact":True,
                     "all_arms_logical_DMA_identical":True,"CPU_tail_adds_no_VTA_DMA":True,
                     "copy_boundary_payload_per_run":payload,"shared_boundary_copy_payload_per_run":0,
                     "shared_ranges":ranges,
                     "temperature_ps_before_after":[row["before"]["temperature_ps_c"],row["after"]["temperature_ps_c"]]})
    sources=[Path(__file__),Path(__file__).with_name("qualify_vta_cpu_tail_handoff.py"),
             args.run/"preregistered.json",args.run/"summary.json",args.run/"buffer_probe.so"]
    for directory in sorted(args.run.glob("layer*")):
        sources.extend(p for p in directory.iterdir() if p.is_file())
    save(destination,{"complete":True,"totals":dict(totals),"rows":rows,
        "warmup_vta_runs":18,"new_cpu_tail_builds":3,"reused_frozen_vta_binaries":6,
        "boot":protocol["board_before"]["boot"],"full_E3_complete":False,
        "no_new_pipeline_FPS":True,"single_inflight_not_K2_pipeline":True,
        "boundary_bytes_scope":"API ledger, excludes diagnostic reads; not physical DDR traffic",
        "sources_sha256":{str(p):sha(p) for p in sources}})
    print(json.dumps(dict(totals),indent=2))


if __name__=="__main__":
    main()
