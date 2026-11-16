#!/usr/bin/env python3
"""Independently replay raw C1 evidence, all masks, and capacity controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import statistics
import tarfile
import uuid

from thesis_c1_memory_ablation import (
    BOOT_SCHEMA, CAPACITIES, evaluate_orders, file_hash, read_json, score_order,
    seal, validate_contract, verify_seal, write_new,
)
from thesis_c1_memory_evidence import inputs_manifest, validate_performance, validate_qualification
from thesis_supplement_run import RunStore


def verify_session(root, token, expected):
    path = Path(token)
    if path.is_absolute() or len(path.parts) != 1 or path.name in (".", ".."):
        raise ValueError("session must be a single relative directory name")
    directory = root / "raw" / path
    observed = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*") if p.is_file()}
    if not expected or observed != expected:
        raise ValueError("raw session files drift or coverage mismatch: " + token)
    return directory


def _check_binding(directory, contract, config, boot, diagnostic):
    session = read_json(directory / "session_contract.json")
    if (session.get("configuration_id") != config["configuration_id"] or
            session.get("boot_id") != boot["boot_id"] or session.get("boot_index") != boot["boot_index"] or
            session.get("selection_contract_sha256") != contract["artifact_sha256"] or
            session.get("diagnostic") is not diagnostic):
        raise ValueError("raw session/boot/configuration binding differs")
    # Rebuild every argument and environment instead of trusting a summary flag.
    from run_thesis_c1_memory_ablation import native_environment, session_argv
    argv = session["argv"]
    remote = str(Path(argv[0]).parent)
    boundary = session.get("capacity_bytes") is not None and "--runs" in argv and argv[argv.index("--runs") + 1] == "64"
    expected = session_argv(contract, config, remote,
                            "inputs_real.txt" if boundary or not diagnostic else "inputs_all.txt",
                            remote + "/" + directory.name, diagnostic,
                            session.get("capacity_bytes"), boundary)
    if argv != expected:
        raise ValueError("native arguments differ from frozen execution protocol")
    if session["environment"] != native_environment(contract, remote, diagnostic):
        raise ValueError("session environment differs from the frozen clean environment")


def verify_costs(root, boot, contract):
    """Bind raw sessions to the executed reset/native scripts and action journal."""
    from run_thesis_c1_memory_ablation import native_session_script, reset_script, ssh_argv
    store = RunStore(root)
    events = store.events()
    ends = [e for e in events if e["event"] == "action_end"]
    if not ends or ends != boot.get("cost_actions"):
        raise ValueError("cost actions differ from the complete append-only journal")
    if (store.contract["boot_index"] != boot["boot_index"] or
            store.contract["campaign_id"] != contract["artifact_sha256"]):
        raise ValueError("cost journal belongs to another boot or selection")
    starts, by_action, positions, costs = {}, {}, {}, {}
    for index, event in enumerate(events):
        if event["event"] == "action_start":
            if event["action_id"] in starts:
                raise ValueError("duplicate cost action start")
            starts[event["action_id"]] = (index, event)
        if event["event"] != "action_end":
            continue
        start_index, start = starts.get(event["action_id"], (-1, {}))
        if (start_index < 0 or start_index >= index or start["event_id"] != event["parent_event_id"] or
                start["argv"] != event["argv"] or start["cost_class"] != event["cost_class"] or
                event["action"] in by_action or event["status"] != "ok" and event.get("returncode") != 3):
            raise ValueError("cost action identity, sequence or status is invalid")
        transport = ssh_argv(store.contract["board_host"])
        if event["argv"][:len(transport)] != transport or (event["action"] != "upload_bundle" and
                event["argv"] != transport + ["sh", "-s"]):
            raise ValueError("cost action was not executed through the frozen board transport")
        seconds = (event["end_monotonic_ns"] - event["begin_monotonic_ns"]) / 1e9
        if seconds < 0 or abs(event["wall_seconds"] - seconds) > 1e-9:
            raise ValueError("cost duration differs from measured interval")
        for stream in ("stdout", "stderr"):
            if file_hash(root / event[stream]) != event[stream + "_sha256"]:
                raise ValueError("cost action raw output drift")
        by_action[event["action"]] = (event, start)
        positions[event["action"]] = (start_index, index)
        costs[event["cost_class"]] = costs.get(event["cost_class"], 0.0) + seconds
    if len(starts) != len(ends) or "kernel_identity" not in by_action:
        raise ValueError("incomplete action journal or missing actual kernel identity")
    identity = by_action["kernel_identity"][0]
    if (root / identity["stdout"]).read_text().strip() != boot["boot_id"]:
        raise ValueError("boot identity differs from actual SSH output")
    tokens = [r["qualification"] for r in boot["configurations"]]
    tokens += [r["session"] for r in boot["capacity_boundaries"]]
    tokens += [r["performance"] for r in boot["configurations"] if r.get("performance")]
    previous_end = -1
    for token in tokens:
        directory = root / "raw" / token
        session = read_json(directory / "session_contract.json")
        remote = str(Path(session["argv"][0]).parent)
        expected = {token + "_reset": reset_script(contract, remote, boot["boot_id"]),
                    token: native_session_script(boot["boot_id"], remote + "/" + token,
                                                 session["argv"], session["environment"])}
        for action, script in expected.items():
            if action not in by_action:
                raise ValueError("missing executed reset/native action")
            end, start = by_action[action]
            path = root / "raw" / (action + ".sh")
            if path.read_text() != "set -eu\n" + script + "\n" or file_hash(path) != start["stdin_sha256"]:
                raise ValueError("executed reset/native script differs from frozen session")
            if positions[action][0] <= previous_end:
                raise ValueError("reset/native sessions were not executed in the declared order")
            previous_end = positions[action][1]
        reset = by_action[token + "_reset"][0]
        marker = "C1_CONTEXT_RESET_READY " + boot["boot_id"] + " " + contract["artifacts"]["bitstream"]["sha256"]
        if reset["status"] != "ok" or marker not in (root / reset["stdout"]).read_text().splitlines():
            raise ValueError("missing successful same-boot bitstream reset receipt")
        # Recomputed session evidence must be exactly what SSH returned.
        action = by_action[token][0]
        archived = {}
        with tarfile.open(root / action["stdout"]) as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts or str(path) in archived:
                    raise ValueError("unsafe or duplicate raw session archive member")
                content = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        content.update(chunk)
                archived[str(path)] = content.hexdigest()
        extracted = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*")
                     if p.is_file() and p.name != "session_contract.json"}
        if extracted != archived:
            raise ValueError("extracted raw evidence differs from actual SSH archive")
    return costs


def analyze(contract, boot_paths):
    validate_contract(contract)
    records = inputs_manifest(contract)
    if len(boot_paths) != 3:
        raise ValueError("exactly three independent boots required")
    by_id = {r["configuration_id"]: r for r in contract["support"]}
    labels, points, boot_ids, boot_indices, costs = {key: [] for key in by_id}, [], set(), set(), {}
    capacity_results = []
    skipped = not contract["identifiability"]["ranking_windows_allowed"]
    for path in boot_paths:
        boot = read_json(path)
        verify_seal(boot)
        if boot.get("schema") != BOOT_SCHEMA or boot.get("status") != "complete":
            raise ValueError("incomplete or incompatible boot; do not manufacture a full-pool oracle")
        if boot.get("selection_contract_sha256") != contract["artifact_sha256"]:
            raise ValueError("selection contract differs across boots")
        boot_id, index = boot["boot_id"], boot["boot_index"]
        uuid.UUID(boot_id)
        if boot_id in boot_ids or index in boot_indices or index not in (0, 1, 2):
            raise ValueError("three distinct kernel boot identities/indices required")
        boot_ids.add(boot_id)
        boot_indices.add(index)
        rows = boot["configurations"]
        if [r["configuration_id"] for r in rows] != contract["measurement_orders"][str(index)]:
            raise ValueError("configuration coverage or measurement order differs")
        root = path.parent
        boot_labels, pairs = {}, {}
        for row in rows:
            key = row["configuration_id"]
            if row["status"] != "qualified":
                raise ValueError("qualification incomplete; configuration retained, claim unavailable")
            directory = verify_session(root, row["qualification"], row["qualification_files"])
            _check_binding(directory, contract, by_id[key], boot, True)
            proof = validate_qualification(contract, by_id[key], directory, records)
            pair_id = by_id[key]["pair_id"]
            if pair_id in pairs and pairs[pair_id] != proof["final_output_sha256"]:
                raise ValueError("copy/shared full outputs differ")
            pairs[pair_id] = proof["final_output_sha256"]
            if skipped:
                if row.get("performance"):
                    raise ValueError("performance was collected despite the identifiability stop")
            else:
                directory = verify_session(root, row["performance"], row["performance_files"])
                _check_binding(directory, contract, by_id[key], boot, False)
                metric = validate_performance(directory, by_id[key])
                labels[key].append(metric["ii_ms"])
                boot_labels[key] = metric["ii_ms"]
        expected_boundaries = {(key, budget) for key in contract["capacity_boundary_ids"] for budget in
                               (max(0, by_id[key]["additional_slot_bytes"] - 256), by_id[key]["additional_slot_bytes"])}
        observations = boot.get("capacity_boundaries", [])
        if len(observations) != 4 or {(r["configuration_id"], r["budget_bytes"]) for r in observations} != expected_boundaries:
            raise ValueError("missing capacity boundary observations")
        for row in observations:
            config = by_id[row["configuration_id"]]
            directory = verify_session(root, row["session"], row["files"])
            _check_binding(directory, contract, config, boot, True)
            if read_json(directory / "session_contract.json")["capacity_bytes"] != row["budget_bytes"]:
                raise ValueError("capacity observation differs from the actual attempted budget")
            if row["budget_bytes"] < config["additional_slot_bytes"]:
                text = (directory / "native.log").read_text()
                if "[C1_SLOT_CAPACITY_REJECTED]" not in text or (directory / "result.jsonl").exists():
                    raise ValueError("unsafe or unobserved capacity rejection")
            else:
                validate_qualification(contract, config, directory, records[:64], expected_frames=64)
            capacity_results.append({"boot_id": boot_id, **row})
        for category, seconds in verify_costs(root, boot, contract).items():
            costs[category] = costs.get(category, 0.0) + seconds
        points.append({"boot_id": boot_id, "boot_index": index, "labels_ii_ms": boot_labels,
                       "evaluation": None if skipped else evaluate_orders(contract, boot_labels)})
    medians = {key: statistics.median(values) for key, values in labels.items()} if not skipped else {}
    evaluation = evaluate_orders(contract, medians) if not skipped else None
    capacity = {}
    for budget in CAPACITIES:
        capacity[str(budget)] = {
            "extra_slot_budget_bytes": budget,
            "eligible_configurations": sum(r["additional_slot_bytes"] <= budget for r in contract["support"]),
            "deployable_fraction": sum(r["additional_slot_bytes"] <= budget for r in contract["support"]) / 16,
            "quality_bands": {str(band): {policy: score_order(contract["orders"]["BCDI"], medians, band,
                                                              budget, contract["support"], prefilter)
                                          for policy, prefilter in (("filter_before_selection", True), ("reject_on_attempt", False))}
                              for band in (0.02, 0.05)} if not skipped else None}
    return seal({"schema": "c1_memory_ablation_analysis_v1", "status": "mechanism_only" if skipped else "analyzed",
                 "selection_contract_sha256": contract["artifact_sha256"], "boot_points": sorted(points, key=lambda r: r["boot_index"]),
                 "candidate_median_ii_ms": medians, "evaluation": evaluation, "capacity": capacity,
                 "capacity_boundary_observations": capacity_results, "identifiability": contract["identifiability"],
                 "cost_class_action_wall_seconds": costs, "online_time_to_quality_measured": False,
                 "ranking_task_count": 1, "independent_boot_count": 3,
                 "performance_windows": 0 if skipped else 48, "completion_rows": 0 if skipped else 14496,
                 "raw_completion_rows_including_warmups": 0 if skipped else 14736,
                 "warmup_rows": 0 if skipped else 240,
                 "sources": [{"path": str(p.resolve()), "sha256": file_hash(p)} for p in boot_paths]})


def artifacts(output, result):
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "analysis.json", result)
    evaluation = result["evaluation"]
    lines = ["# C1 共享内存感知消融结果", "", "状态：" + result["status"], "",
             "单模型、一个排序任务、三个独立 kernel boot；未测真实在线 time-to-quality。", ""]
    if evaluation:
        with (output / "all_masks.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["quality_band", "mask", "first_hit"] +
                            ["{}_at_{}".format(field, k) for k in range(1, 6) for field in ("hit", "regret")])
            for band, values in evaluation.items():
                for mask, score in values["methods"].items():
                    writer.writerow([band, mask, score["first_hit"]] +
                                    [row[field] for row in score["curve"][:5] for field in ("hit", "regret")])
        with (output / "factor_marginals.csv").open("x", newline="") as stream:
            fields = ["quality_band", "factor_added", "from_mask", "to_mask", "first_hit_before", "first_hit_after",
                      "regret_at_5_before", "regret_at_5_after", "improved_by_primary_rule"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for band, values in evaluation.items():
                writer.writerows({"quality_band": band, **row} for row in values["marginal_factor_effects"])
        main = evaluation["0.02"]
        lines += ["Full 相对 M1 的预注册工程门：{}。".format(main["full_vs_M1_improved"]), "",
                  "首次命中：M1={}，Full={}；超过五个候选不计为预算内成功。".format(
                      main["methods"]["NONE"]["first_hit"], main["methods"]["BCDI"]["first_hit"]), ""]
        plot_figures(output, result)
    else:
        lines += ["Full 与 M1 前五相同，按合同跳过性能窗口；容量与功能证据单独报告。", ""]
        plot_capacity_figure(output, result)
    lines += ["各因素的条件贡献、全部开关、随机对照及容量结果见 analysis.json。",
              "三启动重复不构成跨任务泛化或统计显著性证明；离线前缀不等于实际搜索墙钟。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plot_figures(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = result["evaluation"]["0.02"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mask in ("NONE", "BCDI", "CDI", "BDI", "BCI", "BCD"):
        curve = values["methods"][mask]["curve"][:5]
        axes[0].plot([r["k"] for r in curve], [int(r["hit"]) for r in curve], marker="o", label=mask)
        axes[1].plot([r["k"] for r in curve], [r["regret"] for r in curve], marker="o", label=mask)
    random = list(values["random"].values())
    for ax, field in zip(axes, ("hit", "regret")):
        averages = [statistics.mean(float(r["curve"][k][field]) for r in random) for k in range(5)]
        ax.plot(range(1, 6), averages, linestyle="--", label="random mean (20 fixed-first seeds)")
        ax.set_xlabel("Candidate budget K")
        ax.set_xticks(range(1, 6))
    axes[0].set_ylabel("Hit@K (oracle +2%)")
    axes[1].set_ylabel("II regret")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("quality_by_budget." + suffix))
    plt.close(fig)
    plot_capacity_figure(output, result)


def plot_capacity_figure(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([int(k) / (1 << 20) for k in result["capacity"]],
           [v["eligible_configurations"] for v in result["capacity"].values()])
    ax.set(xlabel="Additional shared-slot budget (MiB)", ylabel="Eligible configurations", ylim=(0, 16))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("capacity_coverage." + suffix))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--boot", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifacts(args.output_dir, analyze(read_json(args.contract), args.boot))


if __name__ == "__main__":
    main()
