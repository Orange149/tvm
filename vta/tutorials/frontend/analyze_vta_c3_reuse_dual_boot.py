#!/usr/bin/env python3
"""Node D: exact-round, read-only mechanism and cost audit of the two boots."""
import argparse
from collections import Counter, defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics
import time

from run_vta_c3_reuse_supplement import read, write, append, seal, verify, file_hash, check_bindings, MODES, SEEDS
from analyze_vta_c3_reuse_local import table_row
from audit_vta_c3_reuse_evidence import FrozenReader

O, I, W, IW = MODES
CAPACITY = {"inp": 32768, "wgt": 262144, "acc": 131072}
BOARDS = ("20260915_c3_reuse_fixed_driver_c1_board_run01", "20260915_c3_reuse_fixed_driver_c2_board_run02")
FREEZE = "20260915_c3_reuse_node_b_freeze_run02"


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def round_values(rows):
    if len(rows) != 8 or {r["round"] for r in rows} != set(range(8)):
        raise ValueError("requires eight distinct explicit rounds")
    ordered = sorted(rows, key=lambda r: r["round"])
    if any(r["seed"] != SEEDS[r["round"] % 3] for r in ordered):
        raise ValueError("round seed mismatch")
    if any(not r["correct"] or r["mismatch_count"] != 0 or r["actual_sha256"] != r["expected_sha256"] for r in ordered):
        raise ValueError("wrong output cannot supply latency")
    values = [r["latency_ms"] for r in ordered]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("missing/nonfinite/nonpositive latency")
    return values


def paired(old, new):
    if len(old) != 8 or len(new) != 8:
        raise ValueError("eight paired rounds required")
    return {"old_median_ms": statistics.median(old), "new_median_ms": statistics.median(new),
            "improvement_pct": 100 * (1 - statistics.median(new) / statistics.median(old)),
            "wins": sum(b < a for a, b in zip(old, new)),
            "paired_improvements_pct": [100 * (1 - b / a) for a, b in zip(old, new)],
            "median_paired_improvement_pct": statistics.median(100 * (1 - b / a) for a, b in zip(old, new))}


def stable(stats):
    return len(stats) == 2 and all(s["improvement_pct"] >= 2 and s["wins"] >= 6 for s in stats)


def aligned_capacity(peak):
    if not isinstance(peak, int) or peak <= 0:
        raise ValueError("missing command peak")
    return ((peak + 4095) // 4096) * 4096


def validate_board(directory, candidates, controls, frozen):
    contract = verify(directory)
    if (directory / "invalid_session.json").exists():
        raise ValueError("invalid session cannot supply formal timing")
    check_bindings(contract["source_bindings"])
    summary = read(directory / "summary.json")
    if not summary["status"].startswith(("C1_COMPLETE", "C2_COMPLETE")):
        raise ValueError("incomplete board session")
    if not contract.get("all_modes_same_experimental_driver") or contract["trace_during_formal_timing"]:
        raise ValueError("formal timing must use common trace-off runtime")
    if read(directory / "final_state.json")["boot_id"] != contract["boot_id"]:
        raise ValueError("boot changed")
    all_ids = {r["candidate_id"]: r for r in candidates + controls}
    results = jsonl(directory / "results.jsonl")
    timeline = jsonl(directory / "timeline.jsonl")
    begins = [r for r in timeline if r["event"] == "dispatch_begin"]
    ends = [r for r in timeline if r["event"] == "dispatch_complete"]
    if len(results) != len(begins) or len(results) != len(ends) or len(results) != summary["all_completed_operator_calls"]:
        raise ValueError("unaccounted dispatch")
    for r, begin, end in zip(results, begins, ends):
        if any(r[k] != begin[k] for k in ("candidate_id", "phase", "seed")) or r["candidate_id"] != end["candidate_id"]:
            raise ValueError("dispatch identity/sequence mismatch")
        if (r["complete_operator_calls"] != 1 or r["hidden_warmup_calls"] or not r["correct"]
                or r["mismatch_count"] or r["actual_sha256"] != r["expected_sha256"]):
            raise ValueError("incorrect or hidden invocation")
        if r["candidate_id"] in all_ids:
            c = all_ids[r["candidate_id"]]
            if (r["family_id"] != c["family_id"] or r["residence_mode"] != c["residence_mode"]
                    or r["control"] != c.get("experimental_reload_control", False)):
                raise ValueError("mode/control identity mismatch")
            p = r["runtime_profile_complete"]
            for key, value in c["local_qualification"]["semantics"]["totals"].items():
                if p.get(key) != value:
                    raise ValueError("actual/static DMA mismatch")
            if p["driver_run_calls"] != c["fsim_qualification"]["command_features"]["values"]["submissions"]:
                raise ValueError("actual/static submission mismatch")
            if p["driver_timeout_calls"]:
                raise ValueError("runtime timeout")
        elif r["phase"] != "health":
            raise ValueError("unexpected identity")
    if Counter(r["phase"] for r in results) != {"health": 3, "correctness": 144, "warmup": 48, "timing": 416, "profile": 48}:
        raise ValueError("incomplete phase coverage")
    for cid in all_ids:
        qualified = [r for r in results if r["phase"] == "correctness" and r["candidate_id"] == cid]
        if [r["seed"] for r in qualified] != list(SEEDS):
            raise ValueError("incomplete three-seed correctness")
    main = [r for r in results if r.get("block") == "main"]
    expected = [(family, mode, n) for family in frozen["frozen_family_order"]
                for n, order in enumerate(frozen["frozen_mode_orders"]) for mode in order]
    if [(r["family_id"], r["residence_mode"], r["round"]) for r in main] != expected:
        raise ValueError("frozen Latin/family order changed")
    reload = [r for r in results if r.get("block") == "reload_control"]
    expected = []
    for family in dict.fromkeys(r["family_id"] for r in controls):
        block = [next(r["candidate_id"] for r in rows if r["family_id"] == family and r["residence_mode"] == mode)
                 for mode in (W, IW) for rows in (candidates, controls)]
        for n in range(8):
            order = block[n % 4:] + block[:n % 4]
            expected.extend((cid, n) for cid in order)
    if [(r["candidate_id"], r["round"]) for r in reload] != expected:
        raise ValueError("independent reload Latin order changed")
    for a in jsonl(directory / "allocations.jsonl"):
        if a["allocation_events"] != 3 or a["allocation_order"] != ["output", "data", "weight"]:
            raise ValueError("qualified layout/reuse contract changed")
    return contract, summary, results


def figure_scopes(candidates, output):
    selected = [r for r in candidates if r["family_id"] == "M3D0046"]
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="560" viewBox="0 0 1320 560">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:monospace;font-size:12px}.title{font-size:16px;font-weight:bold}</style>',
              '<text x="20" y="25" class="title">M3D0046: exact lowered producer scope and consumer-loop context</text>']
    for index, mode in enumerate(MODES):
        row = next(r for r in selected if r["residence_mode"] == mode)
        sem = row["local_qualification"]["semantics"]
        x = 20 + index * 325
        pieces.extend([f'<rect x="{x}" y="45" width="310" height="465" fill="#f5f7fa" stroke="#8798aa"/>',
                       f'<text x="{x+10}" y="70" class="title">{("O", "I", "W", "IW")[index]}</text>'])
        for operand, y in (("inp", 105), ("wgt", 210)):
            site = next(s for s in sem["load_store_sites"] if s["extern"] == "VTALoadBuffer2D"
                        and s["symbolic_arguments"][-1] == ("2" if operand == "inp" else "1"))
            pieces.append(f'<text x="{x+10}" y="{y}">{operand.upper()} LOAD enclosing loops:</text>')
            scopes = [f'{s["var"]} x {s["extent"]}' for s in site["loop_scope"]] or ["function scope"]
            for n, label in enumerate(scopes):
                pieces.append(f'<text x="{x+15}" y="{y+20+n*17}">{escape(label)}</text>')
            pieces.append(f'<text x="{x+10}" y="{y+78}">reload ratio: {sem["reload_ratios"]["load_"+operand]:g}</text>')
        consumer = max(sem["compute_sites"], key=lambda s: len(s["loop_scope"]))
        pieces.append(f'<text x="{x+10}" y="320">UOP consumer emission context:</text>')
        for n, s in enumerate(consumer["loop_scope"]):
            pieces.append(f'<text x="{x+15}" y="{340+17*n}">{escape(s["var"])} x {s["extent"]}</text>')
        pieces.extend([f'<text x="{x+10}" y="420">INP/WGT source reuse: {sem["reload_ratios"]["load_inp"]:g}/{sem["reload_ratios"]["load_wgt"]:g}</text>',
                       f'<text x="{x+10}" y="445">Source rectangles + SRAM targets</text>',
                       f'<text x="{x+10}" y="465">are retained in scope_evidence.json.</text>'])
    pieces.extend(['<text x="20" y="535">This is scope/context evidence, not a complete address-to-consumer or asynchronous overwrite proof.</text>', '</svg>'])
    (output / "01_load_consumer_scopes.svg").write_text("\n".join(pieces))


def plots(tables, groups, families, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = sorted(families)
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    for j, mode in enumerate(MODES):
        xx = np.arange(len(labels)) + (j-1.5)*.19
        rr = [next(t for t in tables if t["family_id"] == f and t["mode"] == mode and not t["control"]) for f in labels]
        bottom = np.zeros(len(labels))
        for operand, hatch in (("inp", ""), ("wgt", "//"), ("out", "xx")):
            values = np.array([r[operand+"_bytes"]/1024 for r in rr])
            axes[0].bar(xx, values, .18, bottom=bottom, color=f"C{j}", hatch=hatch, label=f'{("O","I","W","IW")[j]} {operand}')
            bottom += values
        bottom = np.zeros(len(labels))
        for operand, hatch in (("inp", ""), ("wgt", "//"), ("out", "xx")):
            values = np.array([r[operand+"_calls"] for r in rr])
            axes[1].bar(xx, values, .18, bottom=bottom, color=f"C{j}", hatch=hatch)
            bottom += values
        for boot in range(2):
            for x, r in zip(xx, rr):
                v = round_values(groups[(boot, "main", r["candidate_id"])])
                axes[boot+2].scatter([x]*8, v, s=9, alpha=.5, color=f"C{j}")
                axes[boot+2].scatter(x, statistics.median(v), color=f"C{j}", marker="_", s=180)
    for ax, label in zip(axes, ("Logical data DMA KiB", "Logical data DMA calls", "C1 latency ms (all 8 rounds)", "C2 latency ms (all 8 rounds)")):
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xticks(range(len(labels)), labels, rotation=30)
    fig.suptitle("Same-tile four-mode DMA and balanced FPGA latency; no slow rounds dropped")
    fig.tight_layout()
    fig.savefig(output / "02_four_mode_dma_calls_latency.svg")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for n, operand in enumerate(("inp", "wgt", "acc")):
        for j, mode in enumerate(MODES):
            rr = [next(t for t in tables if t["family_id"] == f and t["mode"] == mode and not t["control"]) for f in labels]
            axes[n, 0].bar(np.arange(len(labels))+(j-1.5)*.19,
                           [100*r[operand+"_arena_bytes"]/CAPACITY[operand] for r in rr], .18, label=("O","I","W","IW")[j])
        axes[n, 0].axhline(100, color="red", linestyle="--")
        axes[n, 0].set_ylabel(operand.upper()+" arena span / fixed capacity %")
        axes[n, 0].set_xticks(range(len(labels)), labels, rotation=35, fontsize=8)
    axes[0, 0].legend(ncol=4)
    for j, mode in enumerate(MODES):
        rr = [r for r in tables if r["mode"] == mode and not r["control"]]
        axes[0, 1].scatter([r["submissions"] for r in rr], [r["C2_median_ms"] for r in rr], label=("O","I","W","IW")[j])
        axes[1, 1].scatter([r["command_backing_bytes"]/1024 for r in rr], [r["C2_median_ms"] for r in rr], label=("O","I","W","IW")[j])
    axes[0, 1].set_xlabel("Submissions per complete operator")
    axes[0, 1].set_ylabel("C2 latency ms (association, not isolated cause)")
    axes[1, 1].set_xlabel("4-KiB-aligned instruction + UOP backing KiB")
    axes[1, 1].set_ylabel("C2 latency ms")
    axes[0, 1].legend()
    axes[2, 1].axis("off")
    axes[2, 1].text(.02, .95, "Boundary: full layer weights 294,912 B > 256 KiB.\nW/IW full-layer gate rejects; no FPGA timing.\n\nCounterexample M1D0034:\nW uses fewer weight DMA bytes but is slower than O.\nIW uses least data DMA but I is slightly faster.\n\nArena sums are upper bounds, not simultaneous peaks.\nFixed physical SRAM and 192 MiB backing do not shrink.", va="top", fontsize=11)
    fig.suptitle("Fixed-capacity space / submission / command / latency tradeoffs")
    fig.tight_layout()
    fig.savefig(output / "03_space_sync_latency_tradeoff.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for boot, ax in enumerate(axes):
        for j, workload in enumerate(("M1", "M2", "M3")):
            rr = [r for r in tables if r["workload_id"] == workload and not r["control"]]
            vals = [min(r[f"C{boot+1}_median_ms"] for r in rr if r["mode"] == O), min(r[f"C{boot+1}_median_ms"] for r in rr)]
            ax.bar([j-.18,j+.18], vals, .36, color=["C0","C3"])
            ax.text(j, max(vals)*1.03, f'{100*(1-vals[1]/vals[0]):.2f}%', ha="center")
        ax.set_xticks(range(3), ["M1", "M2", "M3"])
        ax.set_title(f"C{boot+1}: best-original (blue) / best-extended (red)")
        ax.set_ylabel("Operator latency ms")
    fig.suptitle("Identical measured tile set only; not full ConfigSpace or TopHub")
    fig.tight_layout()
    fig.savefig(output / "04_best_original_extended.svg")
    plt.close(fig)


def csv_file(path, rows):
    if not rows:
        raise ValueError("empty required table")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "unknown" if v is None else json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def run(args):
    started = time.monotonic()
    root = Path(args.root).resolve()
    reader = FrozenReader(root)
    freeze = root / FREEZE
    frozen = verify(freeze)
    check_bindings(frozen["source_bindings"])
    reader.directory(freeze)
    candidates, controls = read(freeze / "candidates.json"), read(freeze / "controls.json")
    tables = [table_row(c) for c in candidates + controls]
    for c in candidates + controls:
        if file_hash(freeze / "binaries" / (c["candidate_id"]+".so")) != c["cross_qualification"]["binary_sha256"]:
            raise ValueError("frozen binary changed")
    family_ids = sorted({r["family_id"] for r in candidates})
    for f in family_ids:
        rows = [r for r in candidates if r["family_id"] == f]
        if len(rows) != 4 or {r["residence_mode"] for r in rows} != set(MODES) or any(r["identity"] != rows[0]["identity"] for r in rows):
            raise ValueError("not same full workload/ConfigEntity")
        if len({r["local_qualification"]["tir_sha256"] for r in rows}) != 4 or len({r["cross_qualification"]["binary_sha256"] for r in rows}) != 4:
            raise ValueError("four distinct programs required")
    boards = []
    groups = defaultdict(list)
    phase_costs = []
    raw_timing = []
    for boot, name in enumerate(BOARDS):
        reader.directory(root / name)
        contract, summary, results = validate_board(root / name, candidates, controls, frozen)
        if contract["parent_manifest_sha256"] != file_hash(freeze / "artifact_hashes.json"):
            raise ValueError("parent freeze changed")
        boards.append((contract, summary))
        for r in results:
            if r["phase"] == "timing":
                groups[(boot, r["block"], r["candidate_id"])].append(r)
                raw_timing.append({"boot": boot+1, "boot_id": contract["boot_id"], **r})
        for phase in dict.fromkeys(r["phase"] for r in results):
            rr = [r for r in results if r["phase"] == phase]
            phase_costs.append({"boot": boot+1, "phase": phase, "operator_calls": len(rr),
                "wrong_outputs": 0, "host_call_wall_seconds": sum(r["host_wall_seconds"] for r in rr),
                "logical_load_bytes": sum(r["runtime_profile_complete"]["load_buffer_2d_bytes"] for r in rr),
                "logical_store_bytes": sum(r["runtime_profile_complete"]["store_buffer_2d_bytes"] for r in rr),
                "logical_dma_calls": sum(r["runtime_profile_complete"]["load_buffer_2d_calls"]+r["runtime_profile_complete"]["store_buffer_2d_calls"] for r in rr),
                "FPGA_submissions": sum(r["runtime_profile_complete"]["driver_run_calls"] for r in rr)})
    if boards[0][0]["boot_id"] == boards[1][0]["boot_id"]:
        raise ValueError("same boot not independent confirmation")
    for key in ("parent_manifest_sha256", "dispatch_context_sha256", "runtime_sha256"):
        if boards[0][0][key] != boards[1][0][key]:
            raise ValueError("implementation/hardware context mismatch")
    by_id = {t["candidate_id"]: t for t in tables}
    for table in tables:
        table["command_backing_bytes"] = aligned_capacity(table["instruction_peak_bytes"])+aligned_capacity(table["uop_peak_bytes"])
        table["arena_sum_upper_bound_bytes"] = sum(table[m+"_arena_bytes"] for m in CAPACITY)
        table["logical_data_DMA_bytes"] = table["inp_bytes"]+table["wgt_bytes"]+table["out_bytes"]
        table["logical_data_DMA_calls"] = table["inp_calls"]+table["wgt_calls"]+table["out_calls"]
        for m in CAPACITY:
            table[m+"_arena_capacity_pct"] = 100*table[m+"_arena_bytes"]/CAPACITY[m]
        for boot in range(2):
            block = "reload_control" if table["control"] else "main"
            rr = groups[(boot, block, table["candidate_id"])]
            table[f"C{boot+1}_median_ms"] = statistics.median(round_values(rr))
            table[f"C{boot+1}_driver_total_us_median"] = statistics.median(r["runtime_profile_complete"]["driver_run_total_us"] for r in rr)
    families, pairs, mode_minimum = [], [], []
    for f in family_ids:
        tt = {t["mode"]: t for t in tables if t["family_id"] == f and not t["control"]}
        effects = {}
        for mode in (I, W, IW):
            stats = [paired(round_values(groups[(b,"main",tt[O]["candidate_id"])]), round_values(groups[(b,"main",tt[mode]["candidate_id"])])) for b in range(2)]
            effects[mode] = {"per_boot": stats, "stable_acceleration_vs_original": stable(stats)}
            pairs.append({"family_id": f, "workload_id": tt[O]["workload_id"], "mode": mode,
                          "input_delta_pct": 100*(tt[mode]["inp_bytes"]/tt[O]["inp_bytes"]-1),
                          "weight_delta_pct": 100*(tt[mode]["wgt_bytes"]/tt[O]["wgt_bytes"]-1),
                          "C1_improvement_pct": stats[0]["improvement_pct"], "C2_improvement_pct": stats[1]["improvement_pct"],
                          "C1_wins": stats[0]["wins"], "C2_wins": stats[1]["wins"], "stable_vs_original": stable(stats)})
        record = {"family_id": f, "workload_id": tt[O]["workload_id"], "four_resolved_with_fallback": True,
                  "four_measured": True, "four_distinct_programs": 4,
                  "effective_dual_reuse": tt[IW]["inp_bytes"] < tt[O]["inp_bytes"] and tt[IW]["wgt_bytes"] < tt[O]["wgt_bytes"],
                  "effects": effects,
                  "fastest_modes": [min(MODES,key=lambda m:tt[m][f"C{b+1}_median_ms"]) for b in range(2)],
                  "additive_interaction_ms": [tt[IW][f"C{b+1}_median_ms"]-tt[I][f"C{b+1}_median_ms"]-tt[W][f"C{b+1}_median_ms"]+tt[O][f"C{b+1}_median_ms"] for b in range(2)]}
        record["combination_fastest_both_boots"] = record["fastest_modes"] == [IW,IW]
        families.append(record)
        min_bytes = min(t["logical_data_DMA_bytes"] for t in tt.values())
        least = [m for m in MODES if tt[m]["logical_data_DMA_bytes"] == min_bytes]
        mode_minimum.append({"family_id": f, "minimum_data_access_modes": least, "fastest_modes": record["fastest_modes"],
                             "minimum_matches_fastest_both": all(m in least for m in record["fastest_modes"])})
    reload_pairs = []
    for c in controls:
        resident = next(r for r in candidates if r["family_id"]==c["family_id"] and r["residence_mode"]==c["residence_mode"])
        a,b = [by_id[r["candidate_id"]] for r in (resident,c)]
        sa,sb = [r["local_qualification"]["semantics"] for r in (resident,c)]
        if sa["compute_sites"] != sb["compute_sites"] or a["submissions"] != b["submissions"]:
            raise ValueError("reload matched compute/sync control not preserved")
        for operand in ("inp", "out"):
            if any(a[operand+"_"+metric] != b[operand+"_"+metric] for metric in ("bytes","calls")):
                raise ValueError("reload changed input/output DMA")
        if b["wgt_bytes"] <= a["wgt_bytes"]:
            raise ValueError("reload did not increase actual weights")
        stats = [paired(round_values(groups[(boot,"reload_control",c["candidate_id"])]),round_values(groups[(boot,"reload_control",resident["candidate_id"])])) for boot in range(2)]
        reload_pairs.append({"family_id":c["family_id"], "mode":c["residence_mode"], "resident_id":resident["candidate_id"],
                             "reload_id":c["candidate_id"], "resident_weight_bytes":a["wgt_bytes"], "reload_weight_bytes":b["wgt_bytes"],
                             "matched_submissions":a["submissions"], "per_boot":stats,
                             "stable_resident_gain":stable(stats), "physical_address_isolation_claim":False})
    local = reader.directory(root / "20260915_c3_reuse_node_b_local_run02")
    local_rows = jsonl(local / "results.jsonl")
    boundary = [r for r in local_rows if r.get("workload_id") == "BND"]
    legacy = reader.directory(root / "20260915_c3_reuse_node_a_audit_run02")
    old = read(legacy / "summary.json")
    coverage = [{"scope": key, **value} for key,value in old["coverage"].items()]
    coverage += [{"scope": "new_"+m, "family_session_count":sum(r["workload_id"]==m for r in families),
                  "four_measured":sum(r["workload_id"]==m for r in families), "four_distinct_programs":sum(r["workload_id"]==m for r in families),
                  "effective_dual_reuse":sum(r["workload_id"]==m and r["effective_dual_reuse"] for r in families)} for m in ("M1","M2","M3")]
    # Every retained attempt is disclosed. Derived reports never re-add their parent board time.
    costs = []
    for directory in sorted(root.iterdir()):
        if not directory.name.startswith(("20260915_c3_reuse", "20260915_c3_completion")):
            continue
        if not (directory / "artifact_hashes.json").exists():
            costs.append({"directory":str(directory),"derived":True,"invalid":True,"wall_seconds":None,"timer_source":None,"ledger_available":False})
            continue
        reader.directory(directory)
        s = read(directory / "summary.json") if (directory / "summary.json").exists() else {}
        fail = read(directory / "invalid_session.json") if (directory / "invalid_session.json").exists() else {}
        derived = any(word in directory.name for word in ("analysis", "audit", "closeout"))
        keys = ("analysis_wall_seconds",) if derived else ("W0_to_T1_seconds", "W0_to_failure_seconds", "wall_seconds")
        seconds, source = None, None
        for container, label in ((s,"summary"),(fail,"invalid_session")):
            for key in keys:
                if isinstance(container.get(key), (int,float)):
                    seconds,source = container[key],label+"."+key
                    break
            if seconds is not None:
                break
        costs.append({"directory":str(directory), "derived":derived, "invalid":bool(fail), "wall_seconds":seconds,
                      "timer_source":source,"ledger_available":True})
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    append(out / "timeline.jsonl", {"event":"analysis_W0","wall_unix":time.time(),"elapsed_seconds":time.monotonic()-started})
    write(out / "contract.json", {"phase":"dual_boot_mechanism_analysis", "source_sha256":file_hash(__file__),
          "input_manifests":{str(p):v["manifest_sha256"] for p,v in reader.verified.items()}, "board_contacted":False,
          "stable_rule":{"min_improvement_pct_each_boot":2,"min_round_wins_each_boot":6,"rounds":8},
          "sessions_not_spliced":True,"new_search_claim":False})
    csv_file(out / "four_mode_space_dma_latency.csv", tables)
    csv_file(out / "stable_same_tile_pairs.csv", pairs)
    csv_file(out / "boot_phase_costs.csv", phase_costs)
    csv_file(out / "all_attempt_cost_ledger.csv", costs)
    # Coverage entries have legacy-specific keys; retain them without fabricating missing fields.
    write(out / "legacy_and_new_coverage.json", coverage)
    write(out / "family_classification.json",families)
    write(out / "reload_control_pairs.json",reload_pairs)
    write(out / "minimum_access_vs_fastest.json",mode_minimum)
    write(out / "boundary_rejections.json",boundary)
    write(out / "scope_evidence.json", [{"candidate_id":r["candidate_id"],"family_id":r["family_id"],"mode":r["residence_mode"],
          "load_store_sites":r["local_qualification"]["semantics"]["load_store_sites"],
          "compute_contexts":r["local_qualification"]["semantics"]["compute_sites"],
          "request_groups":r["local_qualification"]["semantics"]["request_groups"],
          "storage_arenas":r["local_qualification"]["storage_arenas"]} for r in candidates+controls])
    for r in raw_timing:
        append(out / "results.jsonl", r)
    figure_scopes(candidates,out)
    plots(tables,groups,family_ids,out)
    summary = {"status":"NODE_D_ANALYSIS_COMPLETE", "board_contacted":False, "main_families":len(families),
          "effective_dual_reuse":sum(r["effective_dual_reuse"] for r in families),
          "combination_fastest_both_boots":sum(r["combination_fastest_both_boots"] for r in families),
          "stable_mode_gains_vs_original":{m:sum(r["effects"][m]["stable_acceleration_vs_original"] for r in families) for m in (I,W,IW)},
          "stable_reload_control_gains":sum(r["stable_resident_gain"] for r in reload_pairs),
          "reload_pairs":len(reload_pairs), "minimum_access_fastest_disagreements":sum(not r["minimum_matches_fastest_both"] for r in mode_minimum),
          "verified_complete_calls":sum(s["all_completed_operator_calls"] for _,s in boards), "verified_timed_calls":len(raw_timing),
          "known_non_derived_investment_lower_bound_seconds":sum(r["wall_seconds"] for r in costs if not r["derived"] and r["wall_seconds"] is not None),
          "complete_total_including_unknown_cost_seconds":None,
          "unknown_non_derived_cost_directories":[r["directory"] for r in costs if not r["derived"] and r["wall_seconds"] is None],
          "physical_AXI_claim":False,"unified_live_peak_claim":False,"new_tuner_or_graph_claim":False,
          "register_generated_map_attested":False,"analysis_wall_seconds":time.monotonic()-started}
    write(out / "summary.json",summary)
    lines = ["# 双启动四方案机制分析", "", "两个独立boot、相同冻结实现。稳定门槛为每启动至少2%改善且至少6/8轮获胜；不是统计显著性保证。", "",
             "| family | I稳定/O | W稳定/O | IW稳定/O | 两启动最快 | 交互 C1/C2 ms |", "|---|---|---|---|---|---|"]
    for r in families:
        lines.append("| {} | {} | {} | {} | {} | {:.4f}/{:.4f} |".format(r["family_id"],
                     *[r["effects"][m]["stable_acceleration_vs_original"] for m in (I,W,IW)],
                     "/".join(r["fastest_modes"]),*r["additive_interaction_ms"]))
    lines += ["", "## 匹配循环与同步的独立reload块", "", "| family | mode | resident/reload weight B | C1改善/wins | C2改善/wins | 稳定 |", "|---|---|---|---|---|---|"]
    for r in reload_pairs:
        a,b=r["per_boot"]
        lines.append(f'| {r["family_id"]} | {r["mode"]} | {r["resident_weight_bytes"]}/{r["reload_weight_bytes"]} | {a["improvement_pct"]:.2f}%/{a["wins"]} | {b["improvement_pct"]:.2f}%/{b["wins"]} | {r["stable_resident_gain"]} |')
    lines += ["", "主块与reload块不混配；空间索引和allocation仍可能变化，不声称完全隔离物理因果。",
              "", "## 使用边界", "", "无新搜索器、整网或物理AXI结果。命令容量是FSim观测峰值的分页需求，不是新runtime定容认证。",
              "arena求和只是上界，完整地址—消费者—异步覆盖证明不在本计划范围。",
              "已暴露池只可作机制解释和规则开发，不能再包装为标签不可见的搜索确认。",
              "失败和恢复逐目录分账；未知时间不填零，known投资为已记录部分的下界，不是连续time-to-quality。"]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n")
    reader.recheck()
    append(out / "timeline.jsonl", {"event":"analysis_T1","elapsed_seconds":time.monotonic()-started})
    seal(out)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--output-dir",required=True)
    run(p.parse_args())
