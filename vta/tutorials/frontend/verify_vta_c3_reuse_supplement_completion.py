#!/usr/bin/env python3
"""Requirement-level completion audit, with fresh local tests and immutable log."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

from run_vta_c3_reuse_supplement import verify,read,write,append,seal,file_hash
from analyze_vta_c3_reuse_dual_boot import validate_board,BOARDS,FREEZE


def run(args):
    root=Path(args.root).resolve()
    here=Path(__file__).resolve().parent
    out=Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    append(out/"timeline.jsonl",{"event":"audit_W0","wall_unix":time.time()})
    ledger={}
    requirements=[]

    def verified(name):
        path=root/name
        c=verify(path)
        ledger[str(path)]=file_hash(path/"artifact_hashes.json")
        return path,c

    def check(label,ok,evidence):
        requirements.append({"requirement":label,"complete":bool(ok),"evidence":evidence})
        if not ok:raise ValueError("incomplete: "+label)

    try:
        a,_=verified("20260915_c3_reuse_node_a_audit_run02")
        old=read(a/"summary.json")
        check("A1: 40 R18 input pairs, 19 resolved / 1 measured / 0 distinct-program groups",
              sum(x["pair_count"] for x in old["paired_effects"] if x["mode"]=="input_stationary" and x["workload_id"].startswith("R18"))==40
              and sum(x["four_resolved_with_fallback"] for key,x in old["coverage"].items() if key.startswith("R18"))==19
              and sum(x["four_measured"] for key,x in old["coverage"].items() if key.startswith("R18"))==1
              and sum(x["four_distinct_programs"] for key,x in old["coverage"].items() if key.startswith("R18"))==0,str(a/"summary.json"))
        check("A1/A2: pair/coverage/counterexample artifacts, reference-region checks and read-only recheck",
              all((a/n).exists() for n in ("reuse_pairs.csv","family_coverage.csv","counterexamples.md","reference_checks.json","input_bindings.json"))
              and old["input_recheck_passed"],str(a))
        b,_=verified("20260915_c3_reuse_node_b_local_run02")
        bs=read(b/"summary.json")
        check("B1/B2: bounded M1/M2/M3 selection, original spaces and branch-aware static/FSim qualification",
              [bs["geometry_counts"][m]["effective_dual_local_correct"] for m in ("M1","M2","M3")]==[4,3,4]
              and all(bs["geometry_counts"][m]["attempted_families"]<=64 for m in ("M1","M2","M3")),str(b/"summary.json"))
        check("B2/B4: real compiler/FSim costs include failed attempts",bs["compiler_attempts"]==468 and bs["attempted_fsim_invocations"]==208 and bs["completed_fsim_invocations"]==200,str(b/"summary.json"))
        freeze,frozen=verified(FREEZE)
        candidates,controls=read(freeze/"candidates.json"),read(freeze/"controls.json")
        check("B3/B4: frozen 44 main + 4 matching reload controls; source and binaries bound",len(candidates)==44 and len(controls)==4,str(freeze))
        c1=[]
        for n in BOARDS:
            directory,_=verified(n)
            contract,summary,rows=validate_board(directory,candidates,controls,frozen)
            c1.append(contract["boot_id"])
            check("C: "+n+" complete same-layout 3-seed qualification / 8-round Latin / independent profile / checked warmup",
                  summary["correct_identities"]==48 and len(rows)==659 and summary["timed_complete_operator_calls"]==416,str(directory))
        check("C2: real different boot, no stitched timing",c1[0]!=c1[1],c1)
        micro,_=verified("20260915_c3_completion_fresh_valid_board_run01_trace_analysis")
        ms=read(micro/"summary.json")
        check("Approved driver amendment: 42-call, 90-submit fresh-completion micro qualification",ms["status"]=="HANDSHAKE_MICRO_QUALIFICATION_PASS" and ms["driver_submissions_verified"]==90,str(micro))
        d,_=verified("20260915_c3_reuse_node_d_dual_boot_run01")
        ds=read(d/"summary.json")
        families=read(d/"family_classification.json")
        controls_analysis=read(d/"reload_control_pairs.json")
        check("D1: per-tensor / stable double-boot pairs / explicit additive interaction",
              len(families)==11 and all("additive_interaction_ms" in r for r in families)
              and ds["stable_mode_gains_vs_original"]=={"input_stationary":11,"weight_resident_barrier":10,"input_weight_resident_barrier":11},str(d))
        check("D1/D4: all 4 independent matched reload effects retain physical-confound caveat",
              len(controls_analysis)==4 and all(r["stable_resident_gain"] and not r["physical_address_isolation_claim"] for r in controls_analysis),str(d/"reload_control_pairs.json"))
        check("B3/D2: over-capacity records and per-domain SRAM / command upper bounds without live-peak claim",
              len(read(d/"boundary_rejections.json"))>0 and not ds["unified_live_peak_claim"] and (d/"four_mode_space_dma_latency.csv").exists(),str(d))
        dx,_=verified("20260915_c3_reuse_node_d_retrospective_audit_run01")
        xs=read(dx/"summary.json")
        top=read(dx/"tophub_availability.json")
        check("D3: identical measured tile-space minima / Pareto; cached TopHub exact workloads absent",
              (dx/"measured_pool_pareto.csv").exists() and all(v==0 for v in top["exact_workload_match_counts"].values()) and not top["reference_line_used"],str(dx))
        check("D4: retrospective bytes-prior M3 failure retained; no new search/graph claim",
              all(v>200 for v in xs["M3_bytes_first_regret_pct"]) and not xs["new_search_claim"] and not ds["new_tuner_or_graph_claim"],str(dx))
        check("Section 7: four primary SVGs parsed and three primary tables generated",xs["four_required_SVGs_validated"]==4 and xs["required_tables_generated"]==3,str(dx/"figure_validation.json"))
        check("Section 7: scope/target/request-shape evidence and all 832 raw timed calls",(d/"scope_evidence.json").exists() and ds["verified_complete_calls"]==1318 and ds["verified_timed_calls"]==832,str(d))
        check("C2/Section 7: old failures/diagnostics/recovery separately hash-bound; unknown total not zero",
              (d/"all_attempt_cost_ledger.csv").exists() and ds["complete_total_including_unknown_cost_seconds"] is None and len(ds["unknown_non_derived_cost_directories"])>0,str(d/"all_attempt_cost_ledger.csv"))
        for name in ("20260915_C3_REUSE_NODE_D_CLOSEOUT.md","C3_INPUT_WEIGHT_REUSE_SUPPLEMENT_PLAN.md","C3_CCF_C_SHARED_MEMORY_INNOVATION.md"):
            path=root.parent/"00_governance"/name
            ledger[str(path)]=file_hash(path)
        tests=["test_analyze_vta_c3_reuse_dual_boot.py","test_audit_vta_c3_reuse_node_d.py",
               "test_analyze_vta_c3_reuse_board.py","test_audit_vta_c3_reuse_evidence.py",
               "test_run_vta_c3_reuse_supplement.py","test_run_vta_c3_reuse_board.py",
               "test_audit_vta_c3_reuse_completion.py","test_build_vta_c3_completion_probe.py",
               "test_build_vta_c3_completion_handshake.py","test_run_vta_c3_completion_probe.py",
               "test_run_vta_c3_reuse_fixed_driver.py"]
        test_begin=time.monotonic()
        result=subprocess.run([sys.executable,"-m","pytest","-q",*[str(here/n) for n in tests]],capture_output=True,text=True,timeout=90)
        write(out/"tests.json",{"argv":[sys.executable,"-m","pytest","-q",*tests],"returncode":result.returncode,
                                 "stdout":result.stdout,"stderr":result.stderr,"wall_seconds":time.monotonic()-test_begin,
                                 "test_source_sha256":{n:file_hash(here/n) for n in tests}})
        check("Section 7: identity / fallback / zero-vs-unknown / conditional DMA / FSim modes / round/layout / interruption / input immutability tests",
              result.returncode==0 and "88 passed" in result.stdout and "skipped" not in result.stdout,"tests.json")
        check("Section 8: positive-combination gate at least two geometries with two families and all boundaries respected",
              ds["effective_dual_reuse"]==11 and sum(bs["geometry_counts"][m]["effective_dual_local_correct"]>=2 for m in ("M1","M2","M3"))>=2,str(d/"summary.json"))
        write(out/"contract.json",{"phase":"supplement_completion_audit","input_hashes":ledger,
              "source_sha256":file_hash(__file__),"board_contacted":False,"no_scope_expansion":True})
        for r in requirements:append(out/"results.jsonl",r)
        write(out/"summary.json",{"status":"SUPPLEMENT_PLAN_COMPLETE_VERIFIED","requirements_passed":len(requirements),
              "tests_passed":88,"skipped":0,"wall_seconds":time.monotonic()-started,
              "scope":"revised input-weight mechanism supplement A/B/C1/C2/D; not universal tuner or production driver",
              "retained_gaps":["historical missing timer/early contract metadata disclosed, not reconstructed",
                               "generated valid-register mapping not attested","unified asynchronous overwrite proof out of scope",
                               "no new online search, fullgraph or physical AXI experiment"]})
        print("SUPPLEMENT_PLAN_COMPLETE_VERIFIED",len(requirements),"requirements; 88 tests",flush=True)
    except BaseException as error:
        write(out/"invalid_session.json",{"message":str(error),"completion_claim":False})
        raise
    finally:
        append(out/"timeline.jsonl",{"event":"audit_T1","elapsed_seconds":time.monotonic()-started})
        seal(out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--output-dir",required=True)
    run(p.parse_args())
