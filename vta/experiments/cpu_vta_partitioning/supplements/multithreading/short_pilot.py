"""Small, explicitly exploratory experiment; the full campaign stays frozen."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / "src")]

from cpu_vta_partitioning.common import read, write, digest, sha256, locked, verify_inventory, now
from cpu_vta_partitioning.domain import Stage
from cpu_vta_partitioning.models import Cost
from cpu_vta_partitioning.campaign import measured, retry_cleanup
from cpu_vta_partitioning.measurement.tasks import task
from multithreading.domain import ENVIRONMENTS, METHODS, NAMES, masks, configuration
from multithreading.evidence import Evidence
from multithreading.prepare import identity as parent_identity, validate
from multithreading.search import rank, reference

ENV = ENVIRONMENTS[0]
PROGRAM_NAMES = ("cpu_00_02", "vta_03_17", "cpu_18_20")
PROTOCOL = {
    "experiment": "resnet18_multithread_short_pilot", "revision": 1,
    "scope": "exploratory_single_boot_fixed_partition", "environment": ENV,
    "programs": list(PROGRAM_NAMES), "seed": 20261104,
    "calibration": {"input_indices": [0], "warmup": 3, "scored": 8, "tail": 0},
    "performance": {"input_indices": [16], "warmup": 8, "extended_warmup": 24,
                    "scored": 32, "tail": 4, "warmup_tolerance": .05},
    "qualification": {"component_indices": [0], "pipeline_indices": [16]},
    "rounds": 3, "anchor_drift_limit": .05, "max_round_attempts": 2,
    "selection": "static_first_only_complete_fixed_partition_mapping_domain",
    "methods": list(METHODS), "practical_gain": .05,
    "formal_significance": False, "manual_reboot": False,
}


def identity():
    return digest({"source": sha256(__file__), "parent": parent_identity(), "protocol": PROTOCOL})


def component_stages(programs):
    result = []
    for name in PROGRAM_NAMES:
        p = programs[name]
        options = masks(range(4)) if p["device"] == "cpu" else [(0, 1, 2, 3)]
        result.extend(Stage(p["device"], p["begin"], p["end"],
                            len(m) if p["device"] == "cpu" else 1, m) for m in options)
    if len(result) != 31:
        raise ValueError("pilot needs exactly 31 component signatures")
    return result


def make_task(kind, state, *, candidate=None, stages=None, **options):
    mode = candidate["execution_mode"] if candidate else "off"
    return task(kind, candidate["stages"] if candidate else stages, ENV,
                experiment=PROTOCOL["experiment"], pilot_identity=state["pilot_identity"],
                execution_mode=mode, **options)


def select(programs, costs):
    small = {name: programs[name] for name in PROGRAM_NAMES}
    ranking = rank(small, costs, ENV, k=1, seed=20261104, topology_filter=PROGRAM_NAMES)
    independent = reference(small, costs, ENV, k=1)
    for method in (*METHODS, "legacy_independent"):
        actual = [(r["score_ms"], r["id"]) for r in ranking["rankings"][method]]
        expected = independent[method]
        if (len(actual) != len(expected) or any(a[1] != b[1] or
                not math.isclose(a[0], b[0], rel_tol=0, abs_tol=1.1e-9)
                for a, b in zip(actual, expected))):
            raise ValueError("vector/scalar mapping search mismatch: " + method)
        # Python round and numpy round can disagree at a half nanosecond in
        # millisecond units. Identities must match; canonicalize displayed scores
        # using the independent scalar calculation after the tolerance check.
        for row, (value, _) in zip(ranking["rankings"][method], expected):
            row["score_ms"] = value
    selected = {m: ranking["rankings"][m][0]["id"] for m in METHODS}
    pool = {ranking["rankings"][m][0]["id"]: ranking["rankings"][m][0] for m in METHODS}
    interventions = {}
    topology = [programs[name] for name in PROGRAM_NAMES]
    for placement, allocation in (("overlap", [(0, 1), (0, 1)]),
                                  ("separate", [(0, 1), (2, 3)])):
        row = configuration(topology, allocation, ENV)
        pool[row["id"]] = row
        interventions[placement] = row["id"]
    if ranking["counts"]["subset_capacity"] != 225:
        raise ValueError("fixed-partition mapping domain incomplete")
    return {"ranking": ranking, "selection": selected, "pool": pool,
            "interventions": interventions, "scope": PROTOCOL["scope"]}


def pin(root, state, names):
    for name in names:
        item = Path(root) / name
        record = {"sha256": sha256(item), "bytes": item.stat().st_size}
        previous = state.setdefault("frozen_files", {}).get(name)
        if previous and previous != record:
            raise ValueError("frozen pilot artifact changed: " + name)
        state["frozen_files"][name] = record
    write(Path(root) / "state.json", state, replace=True)


def check_state(root):
    state = read(Path(root) / "state.json")
    if state["pilot_identity"] != identity() or read(Path(root) / "protocol.json") != PROTOCOL:
        raise ValueError("pilot source or protocol changed; preserve this batch")
    if read(Path(state["bundle"]) / "manifest.json")["bundle_id"] != state["bundle_id"]:
        raise ValueError("pilot bundle changed")
    verify_inventory(root, state.get("frozen_files", {}))
    return state


def calibration_context(root, state):
    """Import only verified component data, with explicit original identities."""
    marker = state.get("calibration_import")
    if not marker:
        return Path(root), state
    source = Path(marker["run_dir"])
    original = read(source / "state.json")
    protocol = read(source / "protocol.json")
    archived_identity = digest({"source": sha256(source / "controller_source.py"),
                                "parent": parent_identity(), "protocol": protocol})
    if (archived_identity != original["pilot_identity"] or archived_identity != marker["pilot_identity"]
            or original["bundle_id"] != state["bundle_id"] or original["boot_id"] != state["boot_id"]
            or protocol != PROTOCOL or sha256(source / "costs.json") != marker["costs_sha256"]):
        raise ValueError("imported calibration identity changed")
    verify_inventory(source, original.get("frozen_files", {}))
    return source, original


class Recorder:
    def __init__(self, root, state, board, remote):
        self.root, self.state, self.board, self.remote = Path(root), state, board, remote
        self.evidence = Evidence(root, state["bundle_id"])

    def saved(self, name, document):
        path = self.root / "records" / (name + ".json")
        if path.exists():
            record = read(path)
            if record["document"] != document or record["boot_id"] != self.state["boot_id"]:
                raise ValueError("pilot record belongs to another task/boot")
            self.evidence.summary(record["summary"], document, record["boot_id"])
            return record["summary"]
        if self.board is None:
            raise ValueError("missing pilot record: " + name)
        print("[短窗口预实验] " + name, flush=True)
        summary = measured(document, self.state["bundle"], self.remote, self.root,
                           self.state["boot_id"], self.board)
        self.evidence.summary(summary, document, self.state["boot_id"])
        write(path, {"document": document, "boot_id": self.state["boot_id"], "summary": summary})
        return summary


def component_documents(stage, state):
    return (make_task("qualify_component", state, stages=[stage], indices=[0], warmup=0, scored=1, tail=0),
            make_task("calibration", state, stages=[stage], indices=[0], warmup=3, scored=8, tail=0))


def pipeline_document(kind, state, candidate, warmup, **labels):
    return make_task(kind, state, candidate=candidate, indices=[16], warmup=warmup,
                     scored=32, tail=4, **labels)


def run(root, bundle, connection, board, calibration_source=None):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    with locked(root):
        if not (root / "state.json").exists():
            manifest = validate(bundle)
            write(root / "protocol.json", PROTOCOL)
            state = {"experiment": PROTOCOL["experiment"], "pilot_identity": identity(),
                     "bundle": str(bundle), "bundle_id": manifest["bundle_id"], "connection": connection,
                     "boot_id": board.boot_id(), "status": "RUNNING", "frozen_files": {}, "created": now()}
            if calibration_source is not None:
                source = Path(calibration_source).resolve()
                if source == root: raise ValueError("calibration source must be a different batch")
                original = read(source / "state.json")
                state["calibration_import"] = {"run_dir": str(source), "pilot_identity": original["pilot_identity"],
                                                "costs_sha256": sha256(source / "costs.json")}
                calibration_context(root, state)
                write(root / "calibration_import.json", state["calibration_import"])
            (root / "controller_source.py").write_bytes(Path(__file__).read_bytes())
            pin(root, state, ["protocol.json", "controller_source.py"])
            if calibration_source is not None: pin(root, state, ["calibration_import.json"])
        state = check_state(root)
        if state["status"] == "COMPLETE":
            return analyze(root)
        if board.boot_id() != state["boot_id"]:
            raise ValueError("pilot boot changed: keep this evidence and use a new run ID")
        validate(bundle)
        board.wait_idle(state["boot_id"])
        if hasattr(board, "mount_storage"):
            board.mount_storage()
        retry_cleanup(root, board)
        remote = board.deploy(bundle, root)
        if hasattr(board, "verify_payload"):
            board.verify_payload(remote, state["boot_id"])
        board.qualify_environment(remote, state["boot_id"], root, bundle)
        recorder = Recorder(root, state, board, remote)
        calibration_root, calibration_state = calibration_context(root, state)
        cal_recorder = recorder if calibration_root == root else Recorder(calibration_root, calibration_state, None, None)
        programs = read(bundle / "plans/programs.json")
        costs = {}
        for stage in component_stages(programs):
            qdoc, cdoc = component_documents(stage, calibration_state)
            cal_recorder.saved("component_correctness/" + stage.signature, qdoc)
            costs[stage.signature] = cal_recorder.saved("calibration/" + stage.signature, cdoc)
        cost_path = root / "costs.json"
        document = {"source": "exploratory_single_boot_board_calibration", "boot_id": state["boot_id"],
                    "bundle_id": state["bundle_id"], "costs": costs,
                    "calibration_import": state.get("calibration_import")}
        if cost_path.exists():
            if read(cost_path) != document: raise ValueError("pilot calibration changed")
        else:
            write(cost_path, document)
        selection = select(programs, {k: Cost.from_dict(v) for k, v in costs.items()})
        selection_path = root / "selection.json"
        if selection_path.exists():
            if read(selection_path) != selection: raise ValueError("pilot selection changed")
        else:
            write(selection_path, selection)
        pin(root, state, ["costs.json", "selection.json"])
        anchor = selection["pool"][selection["interventions"]["overlap"]]
        warmups = {}
        for count in (8, 24):
            warmups[str(count)] = recorder.saved("warmup/" + str(count),
                pipeline_document("pilot_warmup", state, anchor, count))
        chosen = 24 if abs(warmups["8"]["ii_ms"] / warmups["24"]["ii_ms"] - 1) > .05 else 8
        warmup_record = {"warmup": chosen, "checks": warmups}
        path = root / "warmup.json"
        if path.exists():
            if read(path) != warmup_record: raise ValueError("warmup rule changed")
        else:
            write(path, warmup_record)
        pin(root, state, ["warmup.json"])
        for key, candidate in sorted(selection["pool"].items()):
            recorder.saved("pipeline_correctness/" + digest(key)[:16],
                make_task("qualify_pipeline", state, candidate=candidate,
                          indices=[16], warmup=0, scored=1, tail=0))
        for round_index in range(3):
            path = root / f"round{round_index}.json"
            if path.exists(): continue
            for attempt in range(2):
                if (root / f"invalid_round{round_index}_attempt{attempt}.json").exists(): continue
                labels = {"pilot_round": round_index, "pilot_attempt": attempt}
                before = recorder.saved(f"round{round_index}/attempt{attempt}/before",
                    pipeline_document("pilot_anchor", state, anchor, chosen, position="before", **labels))
                keys = sorted(selection["pool"])
                random.Random(20261104 + round_index).shuffle(keys)
                results = {key: recorder.saved(f"round{round_index}/attempt{attempt}/" + digest(key)[:16],
                    pipeline_document("pilot_performance", state, selection["pool"][key], chosen, **labels)) for key in keys}
                after = recorder.saved(f"round{round_index}/attempt{attempt}/after",
                    pipeline_document("pilot_anchor", state, anchor, chosen, position="after", **labels))
                value = {"round": round_index, "attempt": attempt, "boot_id": state["boot_id"],
                         "order": keys, "before": before, "after": after, "measurements": results}
                if abs(after["ii_ms"] / before["ii_ms"] - 1) > .05:
                    write(root / f"invalid_round{round_index}_attempt{attempt}.json", value)
                    continue
                write(path, value)
                pin(root, state, [path.name])
                break
            else:
                raise ValueError("short pilot drift exceeded limit; evidence retained")
        for placement in ("overlap", "separate"):
            candidate = selection["pool"][selection["interventions"][placement]]
            for detailed in (False, True):
                recorder.saved("diagnostic/" + placement + ("_profile" if detailed else "_plain"),
                    pipeline_document("pilot_diagnostic", state, candidate, chosen,
                                      diagnostic_placement=placement, detailed=detailed))
        if board.boot_id() != state["boot_id"]: raise ValueError("boot changed before completion")
        if hasattr(board, "verify_payload"):
            board.verify_payload(remote, state["boot_id"])
        board.idle_command("sync", boot=state["boot_id"])
        pin(root, state, ["round0.json", "round1.json", "round2.json"])
        result = analyze(root)
        state["status"] = "COMPLETE"
        state["completed"] = now()
        write(root / "state.json", state, replace=True)
        return result


def analyze(root):
    root = Path(root)
    state = check_state(root)
    recorder = Recorder(root, state, None, None)
    calibration_root, calibration_state = calibration_context(root, state)
    cal_recorder = Recorder(calibration_root, calibration_state, None, None)
    programs = read(Path(state["bundle"]) / "plans/programs.json")
    cost_document = read(root / "costs.json")
    for stage in component_stages(programs):
        qdoc, cdoc = component_documents(stage, calibration_state)
        cal_recorder.saved("component_correctness/" + stage.signature, qdoc)
        value = cal_recorder.saved("calibration/" + stage.signature, cdoc)
        if value != cost_document["costs"][stage.signature]: raise ValueError("cost provenance changed")
    selection = read(root / "selection.json")
    if selection != select(programs, {k: Cost.from_dict(v) for k, v in cost_document["costs"].items()}):
        raise ValueError("selection does not match pilot calibration")
    anchor = selection["pool"][selection["interventions"]["overlap"]]
    warm = read(root / "warmup.json")
    for count in (8, 24):
        value = recorder.saved("warmup/" + str(count), pipeline_document("pilot_warmup", state, anchor, count))
        if value != warm["checks"][str(count)]: raise ValueError("warmup evidence changed")
    expected = 24 if abs(warm["checks"]["8"]["ii_ms"] / warm["checks"]["24"]["ii_ms"] - 1) > .05 else 8
    if warm["warmup"] != expected: raise ValueError("warmup decision mismatch")
    values = {key: [] for key in selection["pool"]}
    for key, candidate in selection["pool"].items():
        recorder.saved("pipeline_correctness/" + digest(key)[:16],
            make_task("qualify_pipeline", state, candidate=candidate, indices=[16], warmup=0, scored=1, tail=0))
    for index in range(3):
        value = read(root / f"round{index}.json")
        attempt = value["attempt"]
        if (value["round"] != index or attempt not in (0, 1) or value["boot_id"] != state["boot_id"]
                or set(value["measurements"]) != set(selection["pool"])):
            raise ValueError("pilot round identity mismatch")
        labels = {"pilot_round": index, "pilot_attempt": attempt}
        for position in ("before", "after"):
            summary = recorder.saved(f"round{index}/attempt{attempt}/" + position,
                pipeline_document("pilot_anchor", state, anchor, expected, position=position, **labels))
            if summary != value[position]: raise ValueError("anchor evidence changed")
        if abs(value["after"]["ii_ms"] / value["before"]["ii_ms"] - 1) > .05:
            raise ValueError("pilot round drift exceeded limit")
        for key, candidate in selection["pool"].items():
            summary = recorder.saved(f"round{index}/attempt{attempt}/" + digest(key)[:16],
                pipeline_document("pilot_performance", state, candidate, expected, **labels))
            if summary != value["measurements"][key]: raise ValueError("performance evidence changed")
            values[key].append(summary["ii_ms"])
    def compare(a, b):
        aa, bb = values[a], values[b]
        gains = [y / x - 1 for x, y in zip(aa, bb)]
        return {"throughput_gain": statistics.mean(bb) / statistics.mean(aa) - 1,
                "round_gains": gains, "same_configuration": a == b,
                "at_least_5pct_all_rounds": min(gains) >= .05,
                "scope": "three_within_boot_rounds_no_formal_significance"}
    methods = {m: {"name": NAMES[m], "configuration": selection["selection"][m],
                    "ii_ms": statistics.mean(values[selection["selection"][m]]),
                    "round_ii_ms": values[selection["selection"][m]]} for m in METHODS}
    comparisons = {NAMES[m]: compare(selection["selection"]["existing"], selection["selection"][m])
                   for m in ("single_core", "fixed_groups_open", "subset_capacity")}
    fixed_gain = compare(selection["selection"]["fixed_groups_open"], selection["selection"]["single_core"])
    intervention = compare(selection["interventions"]["separate"], selection["interventions"]["overlap"])
    diagnostics = {}
    for placement in ("overlap", "separate"):
        candidate = selection["pool"][selection["interventions"][placement]]
        diagnostics[placement] = {}
        for detailed in (False, True):
            suffix = "profile" if detailed else "plain"
            diagnostics[placement][suffix] = recorder.saved("diagnostic/" + placement + "_" + suffix,
                pipeline_document("pilot_diagnostic", state, candidate, expected,
                                  diagnostic_placement=placement, detailed=detailed))
        d = diagnostics[placement]
        d["profile_perturbation"] = abs(d["profile"]["ii_ms"] / d["plain"]["ii_ms"] - 1)
    result = {"scope": PROTOCOL["scope"], "boot_id": state["boot_id"],
              "pilot_identity": state["pilot_identity"], "bundle_id": state["bundle_id"],
              "component_signatures": 31, "mapping_configurations": 225, "input_images": 2,
              "calibration_import": state.get("calibration_import"),
              "formal_significance": False, "methods": methods, "existing_vs": comparisons,
              "fixed_multithread_vs_single": fixed_gain, "separate_vs_overlap": intervention,
              "diagnostics": diagnostics,
              "claim_limits": ["one fixed partition", "one evaluation image", "one kernel boot",
                               "no global partition advantage or paper-level significance claim"]}
    lines = ["ResNet18 少帧探索实验", "范围：固定 CPU 单元0–2 / VTA单元3–17 / CPU单元18–20，四核共享。",
             "组件标定使用1张图片；流水使用另一张固定图片。三轮均来自同一次开机，不是三个独立启动。",
             "每个性能窗口预热%d帧、保留32个完成时间戳、尾部4帧。" % expected, ""]
    for item in methods.values():
        cpu = [s for s in selection["pool"][item["configuration"]]["stages"] if s["device"] == "cpu"]
        lines.append("%s：平均完成间隔 %.4f ms，吞吐 %.3f 张/秒；前后CPU核心 %s / %s。" %
                     (item["name"], item["ii_ms"], 1000/item["ii_ms"], cpu[0]["mask"], cpu[1]["mask"]))
    lines.append("\n现有方法相对各对照的吞吐变化（正值表示现有方法更快）：")
    for name, value in comparisons.items():
        lines.append("%s：%+.2f%%；三轮范围 %+.2f%% 至 %+.2f%%%s。" %
                     (name, 100*value["throughput_gain"], 100*min(value["round_gains"]),
                      100*max(value["round_gains"]), "；选中相同配置" if value["same_configuration"] else ""))
    lines.append("固定核心组多线程相对单线程物理映射：吞吐 %+.2f%%。" % (100*fixed_gain["throughput_gain"]))
    lines.append("保持两段各两线程，仅由重叠放置改为分开放置：吞吐 %+.2f%%。" % (100*intervention["throughput_gain"]))
    for placement in ("overlap", "separate"):
        lines.append(("重叠" if placement == "overlap" else "分开") + "放置详细日志扰动：%.2f%%。" %
                     (100*diagnostics[placement]["profile_perturbation"]))
    lines.extend(["", "这些结果用于判断此固定划分下的收益趋势；不作全局划分优越性、跨输入稳定性或统计显著性结论。",
                  "差异小或不一致时，应报告当前证据不足，不把它解释为等效，也不为得到显著结果反复追加测量。"])
    write(root / "analysis.json", result, replace=True)
    (root / "REPORT_zh.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "analyze"))
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/resnet18_multithread_short_01")
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/bundles/fe70c42fd53a73a83039d03b")
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--calibration-source", type=Path)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(check_state(args.run_dir), ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        analyze(args.run_dir)
    else:
        if not args.execute_board: parser.error("board access requires --execute-board")
        from multithreading.transport import SSH
        if (args.run_dir / "state.json").exists():
            state = check_state(args.run_dir)
            bundle, connection = Path(state["bundle"]), state["connection"]
        else:
            bundle = args.bundle
            connection = read(ROOT.parents[1] / "configs/local.json")["board"]
        run(args.run_dir, bundle, connection, SSH(connection, execute_board=True), args.calibration_source)


if __name__ == "__main__":
    main()
