#!/usr/bin/env python3
"""E6-S: exhaustive contract ledger; frozen M1 ranking, no physical binding claims."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

from build_cpu_vta_pipeline_v1_p1 import enumerate_reachable
from freeze_cpu_vta_pipeline_v1 import load_sealed_artifact, UNIT_ORDER
from solve_cpu_vta_pipeline_v1_p3 import (CPU_THREAD_CHOICES, build_context, candidate_id,
    label_from_path, path_from_scheme, topology_id)
from iterate_cpu_vta_pipeline_v1_p5b_iteration2 import apply_atomic_cpu_costs, summarize_atomic_profiles
from audit_vta_endpoint_boundaries import edge


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def numeric_sum(items):
    result=Counter()
    for item,count in items:
        for key,value in item.items():
            result[key]+=value*count
    return dict(result)


def class_library(archive):
    census={r["segment_id"]:r for r in read(archive/"logical_dma_census_v2.json")["rows"]}
    checked=read(archive/"context_verification_v2.json")
    assert not checked["missing_contexts"] and not checked["varying_primitive_classes"]
    cpu={r["segment_id"]:{f["function"]:f for f in r["cpu_helpers"]} for r in checked["rows"]}
    library={}
    for sid,row in census.items():
        local=archive/sid.replace(":","_")
        inventory=read(local/"primitive_inventory.json")
        hashes={p["compiler_hash"]:p for p in inventory}
        for n in read(local/"graph.json")["nodes"]:
            if n["op"]!="tvm_op":
                continue
            p=hashes[n["attrs"]["hash"]]
            name=n["attrs"]["func_name"]
            f=row["functions"][name]
            helper=cpu[sid].get(name)
            if helper:
                assert helper["status"]!="unsupported"
            value={"dma":{k:v for k,v in f["static_dma"]["totals"].items() if "average" not in k},
                   "push_calls":f["uop_scope_calls"],
                   "cpu_helper_logical_access":helper["accesses_per_invocation"] if helper else {}}
            signature=p["signature"]
            if signature in library:
                assert library[signature]["cost"]==value,("Context counters differ",signature)
            else:
                library[signature]={"cost":value,"reference":{"segment":sid,"function":name}}
    assert len(library)==72
    return library,census


def compose_segments(scan,library,census):
    result={}
    for row in read(scan/"summary.json")["rows"]:
        sid=row["segment_id"]
        counts=Counter(p["signature"] for p in read(scan/sid.replace(":","_")/"primitives.json"))
        costs={key:numeric_sum((library[s]["cost"][key],n) for s,n in counts.items())
               for key in ("dma","push_calls","cpu_helper_logical_access")}
        # Empty CPU-function counters must not alter DMA zeros/averages.
        costs["dma"]={k:v for k,v in costs["dma"].items() if k.endswith(("_calls","_bytes"))}
        if sid in census:
            expected=census[sid]["logical_descriptor_totals"]
            assert all(costs["dma"].get(k,0)==expected.get(k,0) for k in set(costs["dma"])|set(expected))
        result[sid]={"primitive_invocations":dict(counts),"context_signature":digest(dict(counts)),
                     **costs,"evidence":"direct_TIR_and_board" if sid in census else "class_composition_only_not_full_build"}
    assert len(result)==87
    return result


def pareto_ids(rows):
    """Exact minimization front, retaining all ties; slot objective checked separately."""
    ordered=sorted(rows,key=lambda r:(r["m1_score_ms"],r["boundary_payload_bytes"],r["topology_id"]))
    front=[]
    best_bytes=math.inf
    prior_score=None
    same_score_best=math.inf
    for r in ordered:
        score,size=r["m1_score_ms"],r["boundary_payload_bytes"]
        if score!=prior_score:
            best_bytes=min(best_bytes,same_score_best)
            same_score_best=math.inf
            prior_score=score
        if size<best_bytes and size<=same_score_best:
            front.append(r["topology_id"])
        same_score_best=min(same_score_best,size)
    return front


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource",type=Path,default=Path("vta/tutorials/frontend/report_out/resource_aware_maxplus"))
    parser.add_argument("--audit",type=Path,default=Path("vta/tutorials/frontend/report_out/stage_tile_cotuning/compile_context_audit"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    profile_path=args.resource/"v1_profile_manifest.json"
    unit_path=args.resource/"v1_unit_and_boundary_schema.json"
    cost_path=args.resource/"v1_local_cost_table.json"
    units=load_sealed_artifact(unit_path,"cpu_vta_pipeline_v1_unit_and_boundary_schema")
    profile=load_sealed_artifact(profile_path,"cpu_vta_pipeline_v1_profile_manifest")
    costs=load_sealed_artifact(cost_path,"cpu_vta_pipeline_v1_local_cost_table")
    base=build_context(profile,costs)
    wall,core,atomic_sessions,atomic_reference=summarize_atomic_profiles(args.resource)
    context,_=apply_atomic_cpu_costs(base,wall,core)
    by_id={s["segment_id"]:s for s in profile["segments"]}
    scan=args.audit/"precodegen87_run1"
    archive=args.audit/"representatives23_run1"
    assert read(args.audit/"board_e2_all23_summary.json")["complete"]
    library,census=class_library(archive)
    segments=compose_segments(scan,library,census)
    save(args.output/"class_library.json",library)
    save(args.output/"vta_segments.json",segments)
    sources=[profile_path,unit_path,cost_path,archive/"context_verification_v2.json",
             archive/"logical_dma_census_v2.json",scan/"summary.json",args.audit/"board_e2_all23_summary.json",
             Path(__file__),Path(__file__).with_name("solve_cpu_vta_pipeline_v1_p3.py"),
             Path(__file__).with_name("iterate_cpu_vta_pipeline_v1_p5b_iteration2.py")]
    sources.extend(Path(s["path"]) for s in atomic_sessions)
    sources.append(Path(atomic_sessions[0]["path"]).parent.parent/"t1/reference_comparison.json")
    sources.extend(scan.glob("vta_*/primitives.json"))
    sources.extend(archive.glob("vta_*/primitive_inventory.json"))
    sources.extend(archive.glob("vta_*/graph.json"))
    sources.extend(Path(__file__).with_name(name) for name in (
        "audit_vta_endpoint_boundaries.py", "build_cpu_vta_pipeline_v1_p1.py",
        "freeze_cpu_vta_pipeline_v1.py"))
    save(args.output/"atomic_cpu_inputs.json",{"sessions":atomic_sessions,"reference":atomic_reference})
    save(args.output/"preregistered.json",{"scope":"4623 contract paths and frozen M1 model; no external writes or board runs",
        "expected_configurations":972528,"K":2,"alignment":256,
        "ranking":"reuse unchanged M1, enumerate CPU thread parameters, no new cost fitting",
        "pareto":"minimize best-thread M1 II proxy, one-handoff boundary payload, theoretical aligned K2 slots",
        "physical_fields":"unknown until E6-R; no Graph storage IDs or physical addresses invented",
        "sources_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    schemes,_,_=enumerate_reachable(units)
    assert len(schemes)==4623
    edges,contracts,topologies={}, {}, []
    retained=[]
    configurations=0
    for index,scheme in enumerate(schemes):
        neutral=path_from_scheme(scheme,[1]*sum(s["device"]=="cpu" for s in scheme))
        sids=[sid for sid,_ in neutral]
        assert [u for sid in sids for u in by_id[sid]["unit_names"]]==list(UNIT_ORDER)
        edge_ids=[]
        for left,right in zip(sids,sids[1:]):
            item=edge(by_id[left],by_id[right])
            spec={"direction":item["direction"],"contract":item["contract"],
                  "external_layout":"NCHW" if all(len(s["shape"])==4 for s in item["contract"]["slots"]) else "rank-defined",
                  "required_alignment_bytes":256,"K":2,
                  "physical_reachability":None,"cache_coherence_mode":None,
                  "observed_alignment":None,"storage_id_exclusivity":None,
                  "cross_executor_layout_adapter_required_by_shape_dtype":False,
                  "quantization_equivalence":"not qualified by contract matching",
                  "handoff":"copy or shared-slot; implementation legality pending E6-R"}
            contract_id=digest(spec)
            contracts.setdefault(contract_id,spec)
            vta_sid=left if left.startswith("vta:") else right
            cpu_sid=right if left.startswith("vta:") else left
            key=digest({"contract":contract_id,"producer":left,"consumer":right,
                        "vta_context":segments[vta_sid]["context_signature"],"cpu_context_uncompiled":cpu_sid})
            if key not in edges:
                edges[key]={**item,"contract_signature":contract_id,"vta_context_signature":segments[vta_sid]["context_signature"],
                            "cpu_segment_context":"uncompiled; conservatively keyed by "+cpu_sid,
                            "context_key_is_minimal_equivalence_class":False,"topology_occurrences":0}
            edges[key]["topology_occurrences"]+=1
            edge_ids.append(key)
        best=None
        cpu_count=sum(s["device"]=="cpu" for s in scheme)
        for allocation in itertools.product(CPU_THREAD_CHOICES,repeat=cpu_count):
            path=path_from_scheme(scheme,allocation)
            label=label_from_path(path,context)
            score=max(label.max_cpu_stage_ms,label.vta_service_sum_ms+label.vta_mutex_boundary_sum_ms,
                      label.cpu_core_pool_lower_bound_ms)
            record={"candidate_id":candidate_id(path),"topology_id":topology_id(path),
                    "path":[list(x) for x in path],"m1_score_ms":score}
            configurations+=1
            key=lambda r:(r["m1_score_ms"],r["candidate_id"])
            if best is None or key(record)<key(best):
                best=record
            if len(retained)<20 or key(record)<key(retained[-1]):
                retained.append(record)
                retained.sort(key=key)
                del retained[20:]
        vtas=[sid for sid in sids if sid.startswith("vta:")]
        best={**best,"edge_ids":edge_ids,"vta_islands":len(vtas),
              "boundary_payload_bytes":sum(edges[e]["total_payload_bytes"] for e in edge_ids),
              "theoretical_k2_slot_bytes":sum(edges[e]["theoretical_k2_aligned_slot_bytes"] for e in edge_ids),
              "vta_logical_dma":numeric_sum((segments[sid]["dma"],1) for sid in vtas),
              "vta_internal_cpu_helper_logical_access":numeric_sum((segments[sid]["cpu_helper_logical_access"],1) for sid in vtas),
              "full_topology_binary_verified":False}
        topologies.append(best)
        if (index+1)%500==0:
            print("[AUDIT]",index+1,"/4623 configurations",configurations,flush=True)
            save(args.output/"progress.json",{"complete":False,"topologies":index+1,"configurations":configurations})
    assert configurations==972528
    old_path=args.audit.parent/"stage_memory_experiments/memory_model_ablation.json"
    old_models=read(old_path)["models"]
    old=old_models["m1"]["top20"]
    assert [r["candidate_id"] for r in retained]==[r["candidate_id"] for r in old],"M1 ranking changed"
    assert all(abs(a["m1_score_ms"]-b["m1_score_ms"])<1e-9 for a,b in zip(retained,old))
    proportional=all(r["theoretical_k2_slot_bytes"]==2*r["boundary_payload_bytes"] for r in topologies)
    assert proportional,"Use full 3D Pareto if alignment breaks capacity/payload proportionality"
    front=pareto_ids(topologies)
    for row in topologies:
        row["on_ii_payload_slot_pareto"]=row["topology_id"] in front
    lookup={row["topology_id"]:row for row in topologies}
    alignment={}
    for model,results in old_models.items():
        alignment[model]=[]
        for historical in results["top20"]:
            row=lookup[historical["topology_id"]]
            assert historical["boundary_bytes"]==row["boundary_payload_bytes"]
            alignment[model].append({"rank":historical["rank"],
                "candidate_id":historical["candidate_id"],"topology_id":row["topology_id"],
                "legacy_score_ms":historical[model+"_score_ms"],
                "boundary_payload_bytes":row["boundary_payload_bytes"],
                "theoretical_k2_slot_bytes":row["theoretical_k2_slot_bytes"],
                "on_static_pareto":row["on_ii_payload_slot_pareto"]})
    save(args.output/"legacy_model_alignment.json",{
        "scope":"Historical M0/M1/M2 top20 alignment, not refitted or remeasured models",
        "source_sha256":hashlib.sha256(old_path.read_bytes()).hexdigest(),"models":alignment})
    save(args.output/"edge_contexts.json",edges)
    save(args.output/"contract_classes.json",contracts)
    save(args.output/"topologies.json",topologies)
    summary={"complete":True,"topologies":len(topologies),"configurations":configurations,
        "edge_occurrences":sum(len(r["edge_ids"]) for r in topologies),"contract_classes":len(contracts),
        "conservative_edge_context_keys":len(edges),"context_keys_are_not_proven_minimal_fusion_classes":True,
        "boundary_payload_range_bytes":[min(r["boundary_payload_bytes"] for r in topologies),max(r["boundary_payload_bytes"] for r in topologies)],
        "theoretical_k2_slot_range_bytes":[min(r["theoretical_k2_slot_bytes"] for r in topologies),max(r["theoretical_k2_slot_bytes"] for r in topologies)],
        "slots_exactly_twice_payload_for_all":proportional,"pareto_topologies":len(front),"pareto_ids":front,
        "vta_segment_evidence":{"direct_TIR_and_board":23,"class_composition_only":64},
        "m1_top20_unchanged":True,"ranking_source_sha256":hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "m1_top20":retained,"e6_r_complete":False,"physical_traffic_or_peak_memory_measured":False}
    save(args.output/"summary.json",summary)
    save(args.output/"progress.json",{"complete":True,"topologies":len(topologies),"configurations":configurations})
    print(json.dumps({k:v for k,v in summary.items() if k not in {"m1_top20","pareto_ids"}},indent=2),flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for islands in (1,2,3):
        selected=[r for r in topologies if r["vta_islands"]==islands]
        ax.scatter([r["boundary_payload_bytes"]/2**20 for r in selected],
                   [r["m1_score_ms"] for r in selected],s=9,alpha=.35,label=f"{islands} VTA island(s)")
    selected=sorted([r for r in topologies if r["on_ii_payload_slot_pareto"]],key=lambda r:r["boundary_payload_bytes"])
    ax.plot([r["boundary_payload_bytes"]/2**20 for r in selected],[r["m1_score_ms"] for r in selected],
            "k.-",label="Static Pareto (not safe pruning)")
    ax.set(xlabel="One-handoff boundary payload (MiB/frame)",ylabel="Frozen M1 II proxy (ms)",
           title="4623 topologies; theoretical K2 slots = 2 x payload")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/"ii_payload_pareto.svg")
    fig.savefig(args.output/"ii_payload_pareto.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    main()
