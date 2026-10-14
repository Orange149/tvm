#!/usr/bin/env python3
"""Retrospective search limitations and required Node-D tables; no board calls."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from run_vta_c3_reuse_supplement import verify,read,write,append,seal,file_hash
from analyze_vta_c3_reuse_dual_boot import csv_file, MODES, O, I, W, IW


def dominates(a,b,boot):
    keys=(f"C{boot}_median_ms","logical_data_DMA_bytes","command_backing_bytes")
    return all(a[k]<=b[k] for k in keys) and any(a[k]<b[k] for k in keys)


def rank_diagnostics(rows):
    result=[]
    for geometry in ("M1","M2","M3"):
        rr=[r for r in rows if r["workload_id"]==geometry and not r["control"]]
        ranked=sorted(rr,key=lambda r:(r["logical_data_DMA_bytes"],r["logical_data_DMA_calls"],r["candidate_id"]))
        first=ranked[0]
        for boot in (1,2):
            oracle=min(r[f"C{boot}_median_ms"] for r in rr)
            result.append({"geometry":geometry,"boot":boot,"selected_family":first["family_id"],"selected_mode":first["mode"],
                           "selected_latency_ms":first[f"C{boot}_median_ms"],"oracle_ms":oracle,
                           "regret_pct":100*(first[f"C{boot}_median_ms"]/oracle-1),
                           "within_2pct":first[f"C{boot}_median_ms"]<=1.02*oracle,
                           "first_pool_2pct_rank":next(n+1 for n,r in enumerate(ranked) if r[f"C{boot}_median_ms"]<=1.02*oracle),
                           "scope":"retrospective exposed mechanism pool, not an online search result"})
    return result


def numeric_rows(path):
    rr=list(csv.DictReader(path.open()))
    for r in rr:
        for k,v in list(r.items()):
            if v in ("True","False"):r[k]=v=="True"
            elif v=="unknown":r[k]=None
            else:
                try:r[k]=float(v) if "." in v else int(v)
                except ValueError:pass
    return rr


def run(args):
    started=time.monotonic()
    source=Path(args.analysis_dir).resolve()
    root=source.parent
    verify(source)
    rows=numeric_rows(source/"four_mode_space_dma_latency.csv")
    families=read(source/"family_classification.json")
    diagnostics=rank_diagnostics(rows)
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    append(output/"timeline.jsonl",{"event":"analysis_start","wall_unix":time.time()})
    write(output/"contract.json",{"phase":"node_D_retrospective_audit","input_directory":str(source),
          "input_manifest_sha256":file_hash(source/"artifact_hashes.json"),"source_sha256":file_hash(__file__),
          "new_search_or_FPGA_claim":False,"board_contacted":False})
    coverage=read(source/"legacy_and_new_coverage.json")
    keys=sorted({k for r in coverage for k in r})
    csv_file(output/"legacy_and_new_coverage.csv",[{k:r.get(k) for k in keys} for r in coverage])
    classification=[]
    for f in families:
        classification.append({"family_id":f["family_id"],"workload_id":f["workload_id"],"four_measured_each_boot":True,
            "four_distinct_programs":f["four_distinct_programs"],"effective_dual_reuse":f["effective_dual_reuse"],
            "fastest_C1":f["fastest_modes"][0],"fastest_C2":f["fastest_modes"][1],
            "stable_I_vs_O":f["effects"][I]["stable_acceleration_vs_original"],
            "stable_W_vs_O":f["effects"][W]["stable_acceleration_vs_original"],
            "stable_IW_vs_O":f["effects"][IW]["stable_acceleration_vs_original"],
            "combination_fastest_both":f["combination_fastest_both_boots"]})
    csv_file(output/"four_mode_correctness_reuse.csv",classification)
    bootcost=[]
    for n,name in enumerate(("20260915_c3_reuse_fixed_driver_c1_board_run01","20260915_c3_reuse_fixed_driver_c2_board_run02"),1):
        verify(root/name)
        s=read(root/name/"summary.json")
        bootcost.append({"boot":n,"boot_id":s["boot_id"],"W0_T1_seconds":s["W0_to_T1_seconds"],
                        "T0_T1_seconds":s["T0_to_T1_seconds"],"operator_calls":s["all_completed_operator_calls"],
                        "timed_calls":s["timed_complete_operator_calls"],"correct_identities":s["correct_identities"],
                        "prior_qualification_diagnosis_recovery_included":False})
    csv_file(output/"two_boot_benefit_and_cost.csv",bootcost)
    csv_file(output/"retrospective_bytes_rule.csv",diagnostics)
    pareto=[]
    for geometry in ("M1","M2","M3"):
        rr=[r for r in rows if r["workload_id"]==geometry and not r["control"]]
        for boot in (1,2):
            for r in rr:
                pareto.append({"geometry":geometry,"boot":boot,"candidate_id":r["candidate_id"],"family_id":r["family_id"],
                               "mode":r["mode"],"nondominated":not any(dominates(a,r,boot) for a in rr),
                               "latency_ms":r[f"C{boot}_median_ms"],"logical_DMA_bytes":r["logical_data_DMA_bytes"],
                               "aligned_command_backing_bytes":r["command_backing_bytes"],
                               "scope":"latency/data/command Pareto in measured pool; fixed SRAM feasibility separate"})
    csv_file(output/"measured_pool_pareto.csv",pareto)
    log=Path(args.tophub_log)
    workloads={r["workload_id"]:r["identity"]["workload"] for r in read(root/"20260915_c3_reuse_node_b_freeze_run02/candidates.json")}
    matches=Counter()
    malformed=0
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):continue
            try:
                value=json.loads(line); inp=value["input"];workload=[inp[1]]+inp[2]
            except (ValueError,KeyError,IndexError,TypeError):
                malformed+=1;continue
            for key,w in workloads.items():
                if workload==w:matches[key]+=1
    write(output/"tophub_availability.json",{"log_path":str(log),"log_sha256":file_hash(log) if log.exists() else None,
          "exact_workload_match_counts":{m:matches[m] for m in workloads},"malformed_records":malformed,
          "exact_bitstream_qualification":False,"hardware_FPGA_calls":0,
          "reference_line_used":False,"scope":"local cached log audit only; workload hit is not hardware qualification"})
    figures={name:{"sha256":file_hash(source/name),"svg_root":ET.parse(source/name).getroot().tag} for name in
             ("01_load_consumer_scopes.svg","02_four_mode_dma_calls_latency.svg","03_space_sync_latency_tradeoff.svg","04_best_original_extended.svg")}
    if any(v["svg_root"]!="{http://www.w3.org/2000/svg}svg" for v in figures.values()):raise ValueError("invalid figure")
    write(output/"figure_validation.json",figures)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    selected=[r for r in rows if r["workload_id"]=="M3" and r["mode"]==IW and not r["control"]]
    for boot,ax in enumerate(axes,1):
        for r in selected:
            ax.scatter(r["logical_data_DMA_bytes"]/1024,r[f"C{boot}_median_ms"],s=50)
            ax.annotate(r["family_id"]+f'\n{r["inp_calls"]} input calls',
                        (r["logical_data_DMA_bytes"]/1024,r[f"C{boot}_median_ms"]),xytext=(5,4),textcoords="offset points",fontsize=8)
        ax.set_xlim(47,62)
        ax.set_ylim(.3,2.1)
        ax.set_xlabel("Logical input + weight + output DMA KiB")
        ax.set_ylabel("Operator latency ms")
        ax.set_title(f"C{boot}: smaller bytes do not imply faster")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"M3_bytes_counterexample.svg")
    fig.savefig(output/"M3_bytes_counterexample.png",dpi=130)
    plt.close(fig)
    for r in diagnostics:append(output/"results.jsonl",r)
    report={"status":"NODE_D_RETROSPECTIVE_AUDIT_COMPLETE","board_contacted":False,
            "minimum_bytes_first_within_2pct_geometries":sum(all(r["within_2pct"] for r in diagnostics if r["geometry"]==m) for m in workloads),
            "M3_bytes_first_regret_pct":[r["regret_pct"] for r in diagnostics if r["geometry"]=="M3"],
            "new_search_claim":False,"four_required_SVGs_validated":4,"required_tables_generated":3,
            "tophub_workload_matches":dict(matches),"analysis_wall_seconds":time.monotonic()-started}
    write(output/"summary.json",report)
    verify(source)
    append(output/"timeline.jsonl",{"event":"analysis_end","elapsed_seconds":time.monotonic()-started})
    seal(output)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--tophub-log",default="/home/orange/.tvm/tophub/vta_v0.10.log")
    run(p.parse_args())
