"""Turn archived observations into explicit allocation rules, not a winner list."""
from pathlib import Path

from .costs import Costs, file_sha256, read
from .domain import candidate, configuration, stage, validate_candidate


ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENTS = ROOT.parent
RECOVERY = SUPPLEMENTS / "cpu_strategy/runs/resnet18_scheduler_recovery_01"
SHORT = SUPPLEMENTS / "multithreading/runs/resnet18_multithread_short_02"
PILOT = ROOT / "runs/resnet18_published_policy_dp_pilot_01"


def archive_files():
    """Small archival source closure copied into the new package by preparation."""
    paths = [RECOVERY / "protocol.json", RECOVERY / "analysis/RESOURCE_RULE_FINDINGS.json",
             RECOVERY / "analysis/REPORT.json", SHORT / "analysis.json", PILOT / "rankings.json"]
    return {str(p.relative_to(SUPPLEMENTS)): p for p in paths}


def historical_references(programs=None):
    """Exactly three external references; none consumes or replaces a Top-20 slot."""
    old = read(RECOVERY / "protocol.json")["configurations"]
    result = []
    for label, ctx in (("pinned_host4", "shared4"), ("pinned_host2", "host_shared_2")):
        original = old[label]
        rows = [stage(s["device"], s["begin"], s["end"], s["threads"], s["mask"])
                for s in original["stages"]]
        cfg = candidate(rows, ctx, original["execution_mode"])
        validate_candidate(cfg, programs)
        result.append(cfg)
    original = read(PILOT / "rankings.json")["methods"]["prefix"]["rankings"][0]
    rows = [stage(s["device"], s["begin"], s["end"], s["threads"], s["mask"])
            for s in original["stages"]]
    cfg = candidate(rows, "shared4", original["execution_mode"])
    validate_candidate(cfg, programs)
    result.append(cfg)
    return result


def summarize_history(records, programs=None):
    """Development evidence only: ranking receives no pipeline FPS from this file."""
    costs = records if isinstance(records, Costs) else Costs(records, programs)
    findings = read(RECOVERY / "analysis/RESOURCE_RULE_FINDINGS.json")
    short = read(SHORT / "analysis.json")
    refs = historical_references(programs)
    sources = {name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
               for name, path in archive_files().items()}
    curves = {}
    for begin, end in ((0, 3), (20, 21)):
        name = f"cpu_{begin:02}_{end - 1:02}"
        curves[name] = {}
        for n in range(1, 5):
            value = costs.get(stage("cpu", begin, end, n, tuple(range(n))))
            curves[name][str(n)] = {k: value[k] for k in
                                    ("service_ms", "run_core_ms", "boundary_core_ms")}
    loads = {}
    vta = 0.
    for row in refs[0]["stages"]:
        value = costs.get(row, refs[0])
        if row["device"] == "cpu":
            key = ",".join(map(str, row["mask"]))
            loads[key] = loads.get(key, 0.) + value["service_ms"]
        else:
            vta += value["service_ms"]
    recovery_rows = {r["configuration"]: r for r in findings["recovery"]["rows"]}
    host_gain = findings["recovery"]["effects"]["host2_when_slots_pinned"]
    light_host_gain = findings["transfer"]["light_tail"]["incremental_host_gain"]
    rule_list = [
        {"id": "stage_specific_thread_curve", "label": "线程收益随编译子图改变",
         "status": "supported_on_profiled_subgraphs",
         "selection_rule": "Use each compiled subgraph's own T_i(n) and measured CPU core work; allocate extra cores only through the joint bottleneck objective.",
         "model_features": ["service_ms_by_threads", "run_core_ms_by_threads", "boundary_core_ms"],
         "evidence": {"curves": curves},
         "not_a_rule": "All early stages use three/four threads or all final stages use one thread."},
        {"id": "physical_overlap", "label": "相同线程数必须区分实际核心重叠",
         "status": "supported_for_one_fixed_partition_within_one_boot",
         "selection_rule": "Represent actual core sets and accumulate fixed CPU demand on their physical cores.",
         "model_features": ["worker_core_map", "per_core_compute_demand", "caller_boundary_demand"],
         "evidence": {"threads": [2, 2], "overlap_masks": [[0, 1], [0, 1]],
                      "separate_masks": [[0, 1], [2, 3]],
                      "throughput_gain": short["separate_vs_overlap"]["throughput_gain"],
                      "scope": short["separate_vs_overlap"]["scope"]}},
        {"id": "reused_group_balance", "label": "按复用核组累计服务平衡",
         "status": "historical_configuration_supported_model_explanation_is_retrospective",
         "selection_rule": "Minimize max_g sum_{i assigned to g} T_i(|g|), together with VTA and other resource bounds; count each distinct physical group once.",
         "model_features": ["group_services_ms", "group_reuse_assignment", "physical_core_budget"],
         "fixed_group_sizes_four_cores": [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]],
         "evidence": {"cpu_threads": [3, 1, 3, 1],
                      "group_services_ms": loads, "vta_service_sum_ms": vta,
                      "scope": "current input0 isolated component lookback, not historical direct group timing",
                      "historical_shared_host_fps": recovery_rows["pinned_host4"]["fps"]},
         "not_a_rule": "Always use 3/1/3/1 or require sum of all stage thread parameters <= 4."},
        {"id": "host_compute_overlap", "label": "主机位置取决于计算热点",
         "status": "conditional_support_with_counterexample",
         "selection_rule": "Evaluate host placement jointly with the computed per-core demand, retaining shared and genuinely reserved contexts.",
         "model_features": ["host_mask", "compute_pool", "reserved_cores", "per_core_compute_demand"],
         "evidence": {"host2_gain_multistage": host_gain,
                      "host2_gain_light_tail": light_host_gain,
                      "historical_fixed_host2_fps": recovery_rows["pinned_host2"]["fps"]},
         "not_a_rule": "Physical core2 is generally preferable or binding a host to one core reserves that core."},
        {"id": "binding_and_gate", "label": "绑定与串行机制需要匹配执行语义",
         "status": "conditional_support",
         "selection_rule": "Use fixed one-thread-per-core CPU mappings in this study; keep compatible same-group serialization optional, with matching accounting.",
         "model_features": ["affinity_mode", "execution_mode", "core_partition"],
         "evidence": {"pinning_gain_shared_host": findings["recovery"]["effects"]["pinning_when_host_shared"],
                      "open_4_4_ii_ms": short["methods"]["fixed_groups_open"]["ii_ms"],
                      "serial_4_4_ii_ms": short["methods"]["fixed_groups_serial"]["ii_ms"]}},
    ]
    return {"scope": "historical development evidence and component-model lookback; no pipeline-FPS fitting",
            "sources": sources, "rules": rule_list,
            "corrected_62pct_interpretation": {
                "configuration": "two CPU stages each four threads sharing all four cores, gate off",
                "throughput_gain_vs_single": short["fixed_multithread_vs_single"]["throughput_gain"],
                "existing_4_2_gain_vs_shared_4_4": short["existing_vs"]["固定核心组多线程，无门控"]["throughput_gain"],
                "two_by_two_separate_vs_overlap_gain": short["separate_vs_overlap"]["throughput_gain"]},
            "historical_references": [
                {"id": refs[0]["id"], "source_id": "cfg_b2c8e00fd77dc33f5580c966", "historical_fps": recovery_rows["pinned_host4"]["fps"]},
                {"id": refs[1]["id"], "source_id": "cfg_e454a325cb74cab099ee4154", "historical_fps": recovery_rows["pinned_host2"]["fps"]},
                {"id": refs[2]["id"], "source_id": "cfg_1ca0ada47d98caa4738fb34a", "role": "previous prefix static winner"}],
            "frozen_selection_inputs": ["isolated component profiles", "static model", "generic resource strata"],
            "pipeline_fps_used_for_ranking": False,
            "historical_reference_insertion_into_top20": False,
            "formal_cross_boot_confirmation_claim": False}
