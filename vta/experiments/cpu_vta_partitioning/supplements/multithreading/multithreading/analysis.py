"""Boot-level comparisons, crossover attribution and plain-language exports."""
import csv
import math
import statistics
from pathlib import Path

from cpu_vta_partitioning.common import read,write,sha256
from cpu_vta_partitioning.analysis import paired,holm
from cpu_vta_partitioning.analysis.accounting import cost_accounting
from cpu_vta_partitioning.domain import Stage
from .domain import ENVIRONMENTS,ENV_NAMES,METHODS,NAMES,signatures
from .evidence import Evidence
from .campaign import validate_state,rounds,grouped,validate_diagnostic,validate_warmup


def effect(a,b):
    value = paired(a,b)
    value["throughput_gain"] = 1/(1-value["gain"])-1
    value["throughput_gain_ci95"] = [1/(1-v)-1 for v in value["gain_ci95"]]
    return value


def adjust(values,field):
    ps = holm({k:v["p_two_sided"] for k,v in values.items()})
    for key,value in values.items():
        value[field] = ps[key]
        value["passes_gain_gate"] = value["gain"]>=.05 and ps[key]<.05 and value["gain_ci95"][0]>0


def inference(frozen_by_env,measures_by_env,diagnostics_by_env):
    primary,secondary,crossovers,mechanisms,history,capacity_diagnostic = {},{},{},{},{},{}
    for env in ENVIRONMENTS:
        name = env["name"]
        frozen,measures = frozen_by_env[name],measures_by_env[name]
        selected = frozen["selection"]
        boot_sets = {tuple(sorted(v)) for v in measures.values()}
        if len(boot_sets)!=1 or len(next(iter(boot_sets)))!=6:
            raise ValueError("six matching confirmation boots are required")
        for budget in ("1","5"):
            for other in ("single_core","fixed_groups_open","subset_capacity"):
                a,b = selected["existing"][budget],selected[other][budget]
                primary[f"{name}:existing:{other}:budget{budget}"] = {
                    **effect(measures[a],measures[b]),"environment":name,"budget":int(budget),
                    "method":NAMES["existing"],"comparator":NAMES[other],"same_configuration":a==b}
            for a_method,b_method in (("fixed_groups_open","single_core"),
                                      ("fixed_groups_serial","fixed_groups_open"),
                                      ("subset_capacity","total_capacity"),
                                      ("subset_capacity","fixed_groups_open"),
                                      ("existing","fixed_groups_serial")):
                a,b = selected[a_method][budget],selected[b_method][budget]
                secondary[f"{name}:{a_method}:{b_method}:budget{budget}"] = {
                    **effect(measures[a],measures[b]),"same_configuration":a==b}
            serial = selected["fixed_groups_serial"][budget]
            open_ = frozen["ungated_serial_selections"][budget]
            secondary[f"{name}:same_config_gate:budget{budget}"] = effect(measures[serial],measures[open_])
        capacity_diagnostic[name] = effect(measures[selected["total_capacity"]["1"]],measures[frozen["legacy_independent"]])
        capacity_diagnostic[name]["scope"] = "same arbitrary-mask domain, both static budget one; descriptive unadjusted CI"
        cells = frozen["crossovers"]
        pairings = (("single_partition_multi","single_partition_single"),
                    ("multi_partition_multi","multi_partition_single"),
                    ("multi_partition_single","single_partition_single"),
                    ("multi_partition_multi","single_partition_multi"))
        crossovers[name] = {a+"_vs_"+b:effect(measures[cells[a]],measures[cells[b]]) for a,b in pairings}
        crossovers[name]["same_partition"] = frozen["same_crossover_partition"]
        values = [math.log((measures[cells["single_partition_single"]][boot]/measures[cells["single_partition_multi"]][boot])/
                           (measures[cells["multi_partition_single"]][boot]/measures[cells["multi_partition_multi"]][boot]))
                  for boot in next(iter(boot_sets))]
        from scipy.stats import t
        mean = statistics.mean(values)
        radius = float(t.ppf(.975,5))*statistics.stdev(values)/math.sqrt(6)
        crossovers[name]["interaction_log_ratio"] = {"mean":mean,"ci95":[mean-radius,mean+radius],"n_boots":6}
        crossovers[name]["inference_scope"] = "exploratory unadjusted CIs; new confirmation data, two preselected partitions"
        diagnostics = diagnostics_by_env[name]
        if len(diagnostics)!=6 or {d["boot_id"] for d in diagnostics}!=set(next(iter(boot_sets))):
            raise ValueError("mechanism boots must match confirmation")
        placement = {p:{d["boot_id"]:d["measurements"][p+"_plain"]["ii_ms"] for d in diagnostics}
                     for p in ("overlap","separate")}
        perturbation = max(abs(d["measurements"][p+"_profile"]["ii_ms"]/d["measurements"][p+"_plain"]["ii_ms"]-1)
                           for p in placement for d in diagnostics)
        scheduler = {}
        for p in placement:
            samples = [s for d in diagnostics for s in d["measurements"][p+"_profile"].get("scheduler_by_stage",[]) if s["device"]=="cpu"]
            scheduler[p] = {"all_tids_observed":bool(samples) and all(s["all_tids_observed"] for s in samples),
                            "mean_cpu_runqueue_wait_ns":statistics.mean(s["runqueue_wait_ns"] for s in samples) if samples else None}
        mechanisms[name] = {**effect(placement["separate"],placement["overlap"]),
                            "max_profile_perturbation":perturbation,"scheduler":scheduler}
        if frozen["historical"]:
            h = frozen["historical"]
            history[name] = effect(measures[h["historical_one_island"]],measures[h["historical_three_islands"]])
            history[name]["old_result_scope"] = "45.67% throughput from one historical boot, three within-boot rounds; kept separate"
            history[name]["fair_comparators"] = {m:effect(measures[h["historical_one_island"]],measures[selected[m]["1"]])
                                                 for m in ("single_core","fixed_groups_open","subset_capacity")}
    adjust(primary,"p_holm_primary")
    adjust(secondary,"p_holm_secondary")
    adjust(mechanisms,"p_holm_mechanism")
    for value in mechanisms.values():
        a,b = value["scheduler"]["overlap"],value["scheduler"]["separate"]
        value["mechanism_supported"] = (value["passes_gain_gate"] and value["max_profile_perturbation"]<=.02
            and a["all_tids_observed"] and b["all_tids_observed"]
            and a["mean_cpu_runqueue_wait_ns"]>b["mean_cpu_runqueue_wait_ns"])
    gates = {env["name"]:{budget:all(primary[f"{env['name']}:existing:{m}:budget{budget}"]["passes_gain_gate"]
                     for m in ("single_core","fixed_groups_open","subset_capacity")) for budget in ("1","5")}
             for env in ENVIRONMENTS}
    performance = any(any(v.values()) for v in gates.values())
    return {"primary":primary,"secondary":secondary,"crossovers":crossovers,"mechanisms":mechanisms,
            "history":history,"capacity_selection_diagnostic":capacity_diagnostic,
            "budget_gates":gates,"performance_gate":performance,
            "direction":"existing_method_has_performance_evidence" if performance else "no_extra_method_advantage_established",
            "publication_claim":None,"causal_scope":"fixed-partition affinity interventions only"}


def csv_file(path,rows):
    rows = list(rows)
    if not rows: return
    with Path(path).open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def component_curves(costs):
    result = []
    for key,c in costs["costs"].items():
        if not key.startswith("cpu_"): continue
        program,threads,mask = key.split("_t")[0],int(key.split("_t")[1].split("_")[0]),key.split("_a")[1]
        core = min(map(int,mask.split("-")))
        baseline = costs["costs"][program+f"_t1_a{core}"]
        result.append({"阶段":program,"线程数":threads,"核心":mask,"阶段时间毫秒":c["service_ms"],
                       "输入交接壁钟毫秒":c.get("set_wall_ms"),"计算壁钟毫秒":c.get("run_wall_ms"),
                       "输出交接壁钟毫秒":c.get("get_wall_ms"),
                       "计算核毫秒":c["run_core_ms"],"交接核毫秒":c["boundary_core_ms"],
                       "相对单线程加速比":baseline["service_ms"]/c["service_ms"],
                       "计算核时比":c["run_core_ms"]/baseline["run_core_ms"] if baseline["run_core_ms"] else None})
    return result


def analyze(root):
    root = Path(root).resolve()
    state = validate_state(root)
    if state["boot_index"]!=10: raise ValueError("analysis requires all ten completed boots")
    programs = read(Path(state["bundle"])/"plans/programs.json")
    evidence = Evidence(root,state["bundle_id"])
    # Verify development/confirmation separation and all source summaries before
    # writing a single analysis output.
    cal = rounds(root,"calibration",ENVIRONMENTS[0],programs,evidence)
    if {r["boot_id"] for r in cal}!=set(state["completed_boots"][:2]): raise ValueError("calibration boot mismatch")
    frozen,measures,diagnostics = {},{},{}
    summary,per_boot = [],[]
    for env in ENVIRONMENTS:
        name = env["name"]
        validate_warmup(root,env,programs,evidence)
        screen = rounds(root,"screening",env,programs,evidence)
        confirm = rounds(root,"confirmation",env,programs,evidence)
        if {r["boot_id"] for r in screen}!=set(state["completed_boots"][2:4]): raise ValueError("screen boot mismatch")
        if {r["boot_id"] for r in confirm}!=set(state["completed_boots"][4:]): raise ValueError("confirmation boot mismatch")
        frozen[name] = read(root/name/"frozen.json")
        measures[name] = grouped(confirm)
        diagnostics[name] = [validate_diagnostic(root,root/name/"confirmation"/f"boot{i}"/"diagnostic.json",programs,evidence) for i in range(4,10)]
        for method in METHODS:
            for budget,key in frozen[name]["selection"][method].items():
                values = measures[name][key]
                ii = statistics.mean(values.values())
                summary.append({"环境":ENV_NAMES[name],"方法":NAMES[method],"实测预算":int(budget),
                                "平均完成间隔毫秒":ii,"吞吐每秒":1000/ii,"配置":key})
                per_boot.extend({"环境":ENV_NAMES[name],"方法":NAMES[method],"实测预算":int(budget),
                                 "启动":boot,"完成间隔毫秒":value,"配置":key} for boot,value in values.items())
    result = inference(frozen,measures,diagnostics)
    result["provenance"] = {"bundle_id":state["bundle_id"],"controller_identity":state["controller_identity"],
        "completed_boots":state["completed_boots"],"cost_sha256":sha256(root/"costs.json"),
        "frozen_sha256":{e["name"]:sha256(root/e["name"]/"frozen.json") for e in ENVIRONMENTS}}
    result["boot_means"] = measures
    destination = root/"analysis"
    destination.mkdir(exist_ok=True)
    csv_file(destination/"method_summary.csv",summary)
    csv_file(destination/"boot_measurements.csv",per_boot)
    curves = component_curves(read(root/"costs.json"))
    csv_file(destination/"stage_speedups.csv",curves)
    from .models import load_costs,score
    costs = load_costs(read(root/"costs.json"),state["bundle_id"])
    components,cells,mechanism_rows = [],[],[]
    for env in ENVIRONMENTS:
        name = env["name"]
        for method in METHODS:
            for budget,key in frozen[name]["selection"][method].items():
                value = score(frozen[name]["pool"][key],costs,env)
                c = value["components"]
                components.append({"环境":ENV_NAMES[name],"方法":NAMES[method],"预算":int(budget),
                    "阶段约束毫秒":c["service_ms"],"CPU计算与交接核毫秒":c["cpu_core_ms"],
                    "VTA主机与交接核毫秒":c["host_core_ms"],
                    "最大固定组服务毫秒":max(c["fixed_group_services"].values()),
                    "最大子集容量毫秒":max(c["subset_bounds"].values()),"配置":key})
        for cell,key in frozen[name]["crossovers"].items():
            cells.append({"环境":ENV_NAMES[name],"划分来源":"单线程所选划分" if cell.startswith("single_partition") else "多线程所选划分",
                          "映射方式":"单线程" if cell.endswith("_single") else "多线程",
                          "平均完成间隔毫秒":statistics.mean(measures[name][key].values()),"配置":key})
        m = result["mechanisms"][name]
        mechanism_rows.append({"环境":ENV_NAMES[name],"分开放置改善":m["gain"],"区间下界":m["gain_ci95"][0],
            "区间上界":m["gain_ci95"][1],"Holm校正p":m["p_holm_mechanism"],
            "最大日志扰动":m["max_profile_perturbation"],"机制条件通过":m["mechanism_supported"]})
    csv_file(destination/"model_components.csv",components)
    csv_file(destination/"crossover.csv",cells)
    csv_file(destination/"mechanism.csv",mechanism_rows)
    write(destination/"analysis.json",result,replace=True)
    write(destination/"cost_accounting.json",cost_accounting(root),replace=True)
    lines = ["ResNet18 单任务多线程与共享核心竞争补充实验", "",
        "已完成：十次不同内核启动；两次标定、两次筛选、六次确认。",
        "统计单位为独立启动；同启动两轮先平均。主要十二项比较统一进行 Holm 校正。",
        "阶段独立评分仅为旧简化对照；单线程物理映射是受限域上的论文设备模型适配。", ""]
    for r in summary:
        lines.append(f"{r['环境']}｜{r['方法']}｜预算{r['实测预算']}：{r['平均完成间隔毫秒']:.4f} 毫秒，{r['吞吐每秒']:.3f} 帧/秒")
    lines.extend(["","现有方法相对公平对照："])
    for key,v in result["primary"].items():
        lines.append(f"{ENV_NAMES[v['environment']]}｜预算{v['budget']}｜对{v['comparator']}：完成间隔改善 {v['gain']:.2%}，"
                     f"95%区间 [{v['gain_ci95'][0]:.2%}, {v['gain_ci95'][1]:.2%}]，Holm p={v['p_holm_primary']:.6g}。")
    lines.extend(["","核心放置干预："])
    for name,m in result["mechanisms"].items():
        lines.append(f"{ENV_NAMES[name]}：分开放置的完成间隔改善 {m['gain']:.2%}，95%区间 "
                     f"[{m['gain_ci95'][0]:.2%}, {m['gain_ci95'][1]:.2%}]；机制条件{'通过' if m['mechanism_supported'] else '未通过'}。")
    for name,h in result["history"].items():
        lines.extend(["","历史两个配置重新测量：",
            f"单岛配置相对三岛配置的吞吐改善 {h['throughput_gain']:.2%}，95%区间 "
            f"[{h['throughput_gain_ci95'][0]:.2%}, {h['throughput_gain_ci95'][1]:.2%}]。"])
    lines.extend(["","性能门槛通过。" if result["performance_gate"] else "未建立现有方法超出强对照的额外性能优势。",
        "交叉实验和核心重叠干预分别报告；选择相同的配置共用测量。",
        "本轮结论限于 ResNet18，不自动证明方法新颖性；历史 45.67% 与本轮独立确认结果分列。"])
    (destination/"REPORT_zh.txt").write_text("\n".join(lines)+"\n")
    plots(destination,summary,result,curves,frozen,measures)
    return result


def plots(destination,summary,result,curves,frozen,measures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for candidate in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = ["DejaVu Sans",font_manager.FontProperties(fname=candidate).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig,axes = plt.subplots(1,2,figsize=(15,6),layout="constrained")
    for ax,env in zip(axes,ENVIRONMENTS):
        rows = [r for r in summary if r["环境"]==ENV_NAMES[env["name"]] and r["实测预算"]==5]
        ax.barh([r["方法"] for r in rows],[r["平均完成间隔毫秒"] for r in rows])
        ax.set_title("四核共享" if env["name"]=="shared4" else "三核计算，一核专用主机")
        ax.set_xlabel("完成间隔（毫秒，越小越好；预算五次）")
    fig.savefig(destination/"performance.pdf");fig.savefig(destination/"performance.png",dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,5),layout="constrained")
    for ax,env in zip(axes,ENVIRONMENTS):
        name = env["name"];cells = frozen[name]["crossovers"]
        matrix = [[statistics.mean(measures[name][cells[p+"_"+m]].values()) for m in ("single","multi")]
                  for p in ("single_partition","multi_partition")]
        image = ax.imshow(matrix,cmap="viridis")
        ax.set_xticks([0,1],["单线程映射","多线程映射"])
        ax.set_yticks([0,1],["单线程所选划分","多线程所选划分"])
        ax.set_title("四核共享" if name=="shared4" else "三核计算，一核主机")
        for i in range(2):
            for j in range(2): ax.text(j,i,f"{matrix[i][j]:.2f} ms",ha="center",va="center",color="white")
        fig.colorbar(image,ax=ax)
    fig.savefig(destination/"crossover.pdf");fig.savefig(destination/"crossover.png",dpi=180);plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,5),layout="constrained")
    programs = sorted({r["阶段"] for r in curves})
    for program in programs:
        rows = [r for r in curves if r["阶段"]==program and r["核心"]=="-".join(map(str,range(r["线程数"])))]
        ax.plot([r["线程数"] for r in rows],[r["相对单线程加速比"] for r in rows],alpha=.2)
    ax.set_xlabel("线程数（前缀核心集合）");ax.set_ylabel("阶段加速比");ax.set_xticks([1,2,3,4])
    fig.savefig(destination/"stage_speedups.pdf");fig.savefig(destination/"stage_speedups.png",dpi=180);plt.close(fig)
