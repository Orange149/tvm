"""CSV, JSON and portable plots generated from accepted independent boots."""
import csv
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import read, write
from cpu_vta_partitioning.analysis.accounting import cost_accounting


def export(root):
    from cpu_vta_partitioning.campaign import analyze_environment, validate_run_identity
    root = Path(root)
    state = validate_run_identity(root)
    outputs = {}
    for envdir in sorted(root.glob("env*")):
        if not envdir.is_dir():
            continue
        if (len(list((envdir / "confirmation").glob("boot*/round*.json"))) != 12 or
                len(list((envdir / "confirmation").glob("boot*/diagnostic.json"))) != 6):
            outputs[envdir.name] = {"status":"INCOMPLETE", "inference":"not generated before six complete confirmation boots"}
            continue
        result = analyze_environment(envdir)
        write(envdir / "analysis.json",result,replace=True)
        selection = read(envdir / "selection.json")
        rows = [{"group":g,"budget":b,"candidate":key,"boot_id":boot,"ii_ms":value}
                for g,budgets in sorted(selection.items()) for b,key in sorted(budgets.items())
                for boot,value in sorted(result["boot_means"][key].items())]
        with (envdir / "budget_1_5.csv").open("w") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        ranking = read(envdir / "ranking.json")
        with (envdir / "model_components.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["group","rank","candidate","score_ms","cpu_core_ms","host_core_ms","boundary_core_ms","local_cpu"])
            for group, candidates in sorted(ranking["rankings"].items()):
                for index,c in enumerate(candidates):
                    writer.writerow([group,index+1,c["id"],c["score_ms"],*[c["components"][k] for k in ("cpu_core_ms","host_core_ms","boundary_core_ms","local_cpu")]])
        with (envdir / "host_work_ablation.csv").open("w") as stream:
            writer=csv.writer(stream)
            writer.writerow(["comparison","gain","ci95_low","ci95_high","secondary_holm_p","same_configuration","passes_gain_gate"])
            for key,row in result["host_work_ablation_secondary"].items():
                writer.writerow([key,row["gain"],*row["gain_ci95"],row["p_holm_secondary"],row["same_configuration"],row["passes_gain_gate"]])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,budget in zip(axes,("1","5")):
            groups = sorted(selection)
            values = [list(result["boot_means"][selection[g][budget]].values()) for g in groups]
            ax.boxplot(values)
            ax.set_xticks(range(1,len(groups)+1),groups)
            ax.set_title("Budget " + budget + ": independent boots")
            ax.set_ylabel("Initiation interval (ms)")
        fig.tight_layout(); fig.savefig(envdir / "performance.pdf"); fig.savefig(envdir / "performance.png",dpi=160); plt.close(fig)
        summary = ["CPU–VTA independent confirmation", "", "Direction: " + result["direction"],
                   "Budgets passing all primary comparisons: " + str(result["eligible_budgets"]),
                   "Host-work selection effect (secondary H/C or O/A): " + str(result["host_work_gate"]),
                   "Local CPU constraint selection effect (O/H): " + str(result["local_cpu_constraint_gate"]),
                   "Mechanism supported: " + str(result["mechanism_supported"]), "",
                   "Six kernel boots; rounds averaged within boot; frames are not statistical replicates.",
                   "", "comparison,gain,95% CI,Holm p,same configuration"]
        summary += [f"{k},{v['gain']:.4f},{v['gain_ci95']},{v['p_holm']:.6g},{v['same_configuration']}"
                    for k,v in result["comparisons"].items()]
        summary += ["", "Equivalence is reported separately using the prespecified ±5% margin.",
                    "Diagnostic CPU/runtime and runqueue counters cover the 300-completion steady window.",
                    "Polling wall time is not pure hardware service; unexplained residual is not assigned to DDR."]
        (envdir / "report.txt").write_text("\n".join(summary) + "\n")
        outputs[envdir.name] = {"status":"ANALYZED", "directory":str(envdir),"direction":result["direction"]}
    accounting = cost_accounting(root)
    bundle = Path(state["bundle"])
    if (bundle / "plans/programs.json").exists():
        programs=read(bundle / "plans/programs.json")
        accounting["stage_compile_wall_seconds_sum"] = sum(p.get("compile_seconds",0) for p in programs.values())
        accounting["compiled_programs"] = len(programs)
    accounting["provenance"] = {k:state[k] for k in ("bundle_id", "controller_identity")}
    write(root / "cost_accounting.json",accounting,replace=True)
    return {"environments":outputs,"cost_accounting":accounting}
