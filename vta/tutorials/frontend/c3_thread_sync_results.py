"""Read-only, identity-bound one/two-boot analysis. No performance-driven retry."""
import csv
from pathlib import Path
import statistics

import run_vta_c3_reuse_supplement as s
import c3_thread_sync_protocol as p


def read_lines(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args, out):
    frozen = Path(args.contract).resolve().parent
    contract = s.verify(frozen)
    if contract.get("phase") != "freeze":
        raise ValueError("analysis requires frozen contract")
    prepared = Path(contract["prepared_directory"])
    s.verify(prepared)
    if s.file_hash(prepared/"artifact_hashes.json") != contract["prepared_manifest_sha256"]:
        raise ValueError("preparation manifest changed")
    rows = s.read(prepared/"candidates.json")
    controls = s.read(prepared/"controls.json")
    boots, rejected = [], []
    for directory in args.boot_dirs:
        try:
            bc = s.verify(directory)
            if (directory/"invalid_session.json").exists():
                raise ValueError("invalid whole session")
            summary = s.read(directory/"summary.json")
            if bc["frozen_manifest_sha256"] != s.file_hash(frozen/"artifact_hashes.json") or not summary["timing_valid"]:
                raise ValueError("different identity or invalid timing")
            times = read_lines(directory/"timing.jsonl")
            p.validate_timing(times, summary["accepted"])
            boots.append(dict(directory=str(directory.resolve()), boot_id=summary["boot_id"],
                              boot_index=bc["boot_index"], accepted=summary["accepted"], times=times))
        except Exception as error:
            rejected.append(dict(directory=str(directory), status="invalid_session", reason=str(error)))
    if len(boots) > 2 or len({b["boot_id"] for b in boots}) != len(boots) or len({b["boot_index"] for b in boots}) != len(boots):
        raise ValueError("need at most two distinct boots, not repeat sessions")
    boots.sort(key=lambda b:b["boot_index"])
    comparisons = []
    specs = [(tile+"_"+a,tile+"_"+b,kind,controls.get(tile+":"+a+"/"+b,{"matched":True})["matched"])
             for tile in ("T8","T7") for a,b,kind in p.COMPARISONS]
    specs += [("T8_"+v,"T7_"+v,"tile_conditionality",True) for v in p.LETTERS]
    for baseline,candidate,kind,matched in specs:
        observations = []
        for boot in boots:
            if baseline in boot["accepted"] and candidate in boot["accepted"]:
                observations.append({**p.paired(boot["times"], baseline, candidate), "boot_id":boot["boot_id"]})
        status = p.classify(observations, matched)
        if not observations and matched:
            status = "no_qualified_pair"
        comparisons.append(dict(baseline=baseline,candidate=candidate,kind=kind,matched=matched,status=status,boots=observations))
    table=[]
    for row in rows:
        item=dict(candidate_id=row["candidate_id"],tile=row["tile"],version=row["version"],status=row["status"],rejection=row.get("rejection",""))
        item.update(row.get("arenas",{}))
        item.update(row.get("local_qualification",{}).get("semantics",{}).get("totals",{}))
        command=row.get("command_qualification",{}).get("instruction_audits",{}).get("0")
        if command:
            item.update(fsim_uop_load_bytes=command["uop_load_bytes"],fsim_submission_count=command["submission_count"])
            for category in ("raw_opcode","semantic"):
                for block in command["submissions"]:
                    for name,value in block[category].items():
                        key="fsim_"+category+"_"+name
                        item[key]=item.get(key,0)+value
        for boot in boots:
            values=[x["latency_ms"] for x in boot["times"] if x["candidate_id"]==row["candidate_id"]]
            item[f'boot{boot["boot_index"]}_median_ms']=statistics.median(values) if values else None
        table.append(item)
    write_csv(out/"candidate_space_commands_time.csv",table)
    flat=[]
    for comparison in comparisons:
        for boot in comparison["boots"]:
            flat.append({k:v for k,v in comparison.items() if k!="boots"} |
                        {k:v for k,v in boot.items() if k in ("boot_id","paired_median_gain_pct","wins","losses")})
    write_csv(out/"paired_comparisons.csv",flat)
    s.write(out/"comparisons.json",comparisons)
    s.write(out/"contract.json",dict(schema=contract["schema"],phase="analyze",frozen_directory=str(frozen),
                    frozen_manifest_sha256=s.file_hash(frozen/"artifact_hashes.json"),
                    boot_manifests={str(d):s.file_hash(d/"artifact_hashes.json") for d in args.boot_dirs}))
    s.write(out/"summary.json",dict(valid_boots=len(boots),rejected_sessions=rejected,
                 status="TWO_BOOT_ANALYSIS" if len(boots)==2 else "AWAITING_SECOND_BOOT" if boots else "LOCAL_ONLY",
                 whole_graph_claim=False,hardware_overlap_cycles_claim=False))
    lines=["# R18-H1 线程／同步／驻留对照", "",f"有效启动：{len(boots)}；旧时间未复用。", "",
           "| 比较（基线→测试） | 类型 | 控制匹配 | 判定 | 各启动逐轮改善率中位数 |",
           "|---|---|---|---|---|"]
    for c in comparisons:
        gains=", ".join(f'{b["paired_median_gain_pct"]:.3f}% ({b["wins"]}/8 wins)' for b in c["boots"]) or "unknown"
        lines.append(f'| {c["baseline"]} → {c["candidate"]} | {c["kind"]} | {c["matched"]} | {c["status"]} | {gains} |')
    lines += ["", "这是预热后板端算子调用耗时及程序级对照，不是纯硬件时间、逐周期重叠或整网结论。",
              "空间是静态 arena 占比；UOP 装载累计量不是片上有效占用峰值。未达到门槛不构成重跑理由。", ""]
    (out/"README.md").write_text("\n".join(lines))
    if boots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,tile in zip(axes,("T8","T7")):
            for boot in boots:
                xs,ys=[],[]
                for i,label in enumerate(p.LETTERS):
                    values=[x["latency_ms"] for x in boot["times"] if x["candidate_id"]==tile+"_"+label]
                    if values: xs.append(i);ys.append(statistics.median(values))
                ax.plot(xs,ys,"o-",label=f'boot {boot["boot_index"]}')
            ax.set_xticks(range(8),list(p.LETTERS));ax.set_title(tile);ax.set_ylabel("board operator call (ms)");ax.legend()
        fig.tight_layout();fig.savefig(out/"latency_by_tile.svg");plt.close(fig)
