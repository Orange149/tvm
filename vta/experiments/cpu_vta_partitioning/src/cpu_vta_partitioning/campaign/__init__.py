"""Persistent campaign controller. A kernel boot, not a process, is the replicate."""
from collections import defaultdict
from pathlib import Path
import random
import statistics

from cpu_vta_partitioning.common import read, write, digest, inventory, verify_inventory, event, locked, now, PROJECT, sync_tree
from cpu_vta_partitioning.domain import Stage, signatures, candidate_id, topology_id
from cpu_vta_partitioning.measurement import frames, window, calibration, affinity, correctness, input_sequence
from cpu_vta_partitioning.measurement.tasks import task, initial_anchor, calibration_templates
from cpu_vta_partitioning.models import load_costs
from cpu_vta_partitioning.search import rank, screening_pool, freeze_selection
from cpu_vta_partitioning.analysis import report, paired
from cpu_vta_partitioning.campaign.evidence import Evidence, linked_summary


def controller_identity():
    import sys
    from importlib.metadata import version
    sources = {
        str(p.relative_to(PROJECT)): __import__("hashlib").sha256(p.read_bytes()).hexdigest()
        for p in sorted((PROJECT / "src").rglob("*.py"))}
    return digest({"sources":sources,"python":sys.version,
                   "dependencies":{name:version(name) for name in ("numpy","scipy","matplotlib")}})


def validate_run_identity(root):
    """Require the frozen controller for both measurement and offline analysis."""
    state = read(Path(root) / "state.json")
    if state["controller_identity"] != controller_identity():
        raise ValueError("controller changed: use the frozen controller or create a new campaign")
    manifest = read(Path(state["bundle"]) / "manifest.json")
    if (manifest["bundle_id"] != state["bundle_id"] or
            manifest["controller_identity"] != state["controller_identity"]):
        raise ValueError("run and bundle identity mismatch")
    return state


def create(bundle, run_dir, connection):
    bundle, root = Path(bundle).resolve(), Path(run_dir)
    if root.exists():
        raise FileExistsError("run already exists; use board resume")
    manifest = read(bundle / "manifest.json")
    if manifest["controller_identity"] != controller_identity():
        raise ValueError("controller differs from frozen package; prepare a new bundle")
    protocol = read(bundle / "plans/protocol.json")
    main = protocol["environment"]
    sensitivity = read(bundle / "plans/sensitivity.json")
    envs = [main] + [{"cores": c, "poll_sleep_ns": p, "post_start_sleep_ns": sensitivity["post_start_sleep_ns"]}
                    for c in sensitivity["cores"] for p in sensitivity["poll_sleep_ns"]
                    if {"cores": c, "poll_sleep_ns": p, "post_start_sleep_ns": sensitivity["post_start_sleep_ns"]} != main]
    root.mkdir(parents=True)
    state = {"bundle": str(bundle), "bundle_id": manifest["bundle_id"], "controller_identity": controller_identity(),
             "connection": connection, "environments": envs, "environment_index": 0, "phase": "calibration",
             "boot_index": 0, "active_boot": None, "completed_boots": [], "status": "READY", "created": now()}
    write(root / "state.json", state)
    return state


def acknowledge_result(document, root, boot, transport):
    if not hasattr(transport, "acknowledge"):
        return
    pending = Path(root) / "cleanup_pending" / boot / (document["task_id"] + ".json")
    try:
        transport.acknowledge(document, root, boot)
    except (RuntimeError, OSError) as error:
        write(pending, {"task":document,"boot_id":boot}, replace=True)
        event(root, "cleanup_pending", boot_id=boot, task_id=document["task_id"], error=str(error))
    else:
        pending.unlink(missing_ok=True)


def retry_cleanup(root, transport):
    for pending in sorted(Path(root).glob("cleanup_pending/*/*.json")):
        item = read(pending)
        receipt = read(Path(root) / "accepted" / item["boot_id"] / pending.name)
        if receipt["task"] != item["task"] or receipt["boot_id"] != item["boot_id"]:
            raise ValueError("cleanup receipt identity mismatch")
        verify_inventory(root, receipt["files"])
        acknowledge_result(item["task"], root, item["boot_id"], transport)


def measured(document, bundle, remote, root, boot, transport):
    document = {k:v for k,v in document.items() if k != "task_id"}
    document["bundle_id"] = read(Path(bundle) / "manifest.json")["bundle_id"]
    document["task_id"] = digest(document)[:24]
    receipt = Path(root) / "accepted" / boot / (document["task_id"] + ".json")
    def acknowledge():
        acknowledge_result(document, root, boot, transport)
    if receipt.exists():
        cached = read(receipt)
        if cached["task"] != document or cached["boot_id"] != boot:
            raise ValueError("cached task identity changed")
        verify_inventory(Path(root), cached["files"])
        acknowledge()
        return linked_summary(Path(root), receipt, cached)
    event(root, "session_start", boot_id=boot, task_id=document["task_id"], task_kind=document["kind"],
          planned_native_frames=document["warmup"] + document["scored"] + document["tail"])
    try:
        result = transport.measure(document, bundle, remote, root, boot)
        rows = frames(result / "result.jsonl", document["warmup"] + document["scored"] + document["tail"])
        input_sequence(rows, document)
        affinity(read(result / "affinity.json"), document["stages"])
        if document["kind"].startswith("qualify"):
            summary = {"correct": correctness(rows, document, bundle, result)}
        elif document["kind"] == "calibration":
            summary = calibration(rows, document["warmup"], document["scored"])
        else:
            summary = window(rows, document["warmup"], document["scored"], document["tail"])
            if document.get("detailed"):
                summary["profile"] = read(result / "profile.json")
                summary["affinity"] = read(result / "affinity.json")
                summary["profile_completion_window"] = [document["warmup"], document["warmup"] + document["scored"] - 1]
                kept = rows[document["warmup"]:document["warmup"] + document["scored"]]
                summary["stage_timings"] = {k: statistics.mean(r[k] for r in kept if r[k] is not None)
                    for k in kept[0] if k.startswith("stage") and k.endswith("_ms") and
                    any(r.get(k) is not None for r in kept)}
                threads = {t["tid"]:t for t in summary["profile"].get("threads",[])}
                summary["scheduler_by_stage"] = []
                for stage, assigned in zip(document["stages"], summary["affinity"]["stages"]):
                    samples = [threads[tid] for tid in assigned["thread_ids"] if tid in threads]
                    summary["scheduler_by_stage"].append({"program":stage["program"],"device":stage["device"],
                        "all_tids_observed":len(samples)==len(assigned["thread_ids"]),
                        "runtime_ns":sum(s["runtime_ns"] for s in samples),
                        "runqueue_wait_ns":sum(s["runqueue_wait_ns"] for s in samples)})
        files = {str(result.relative_to(root) / name): value for name,value in inventory(result).items()}
        sync_tree(result)
        write(receipt, {"task": document, "boot_id": boot, "summary": summary, "files": files})
        event(root, "session_accepted", boot_id=boot, task_id=document["task_id"])
        acknowledge()
        return linked_summary(Path(root), receipt, read(receipt))
    except Exception as error:
        event(root, "session_failed", boot_id=boot, task_id=document["task_id"], error=str(error))
        raise


def grouped(rounds):
    rows = defaultdict(lambda: defaultdict(list))
    for round_ in rounds:
        for key, item in round_["measurements"].items():
            rows[key][round_["boot_id"]].append(item["ii_ms"])
    return {key: {boot: statistics.mean(values) for boot,values in boots.items()} for key,boots in rows.items()}


def phase_rounds(directory, phase):
    return Evidence(directory).rounds(phase)


def diagnosis_candidates(ranking, costs, cores):
    candidates = screening_pool(ranking)
    chosen = []
    for islands, quantile in ((1, 0.), (2, .5), (3, 1.)):
        pool = []
        for row in candidates.values():
            stages = [Stage.from_dict(s) for s in row["stages"]]
            if sum(s.device == "vta" for s in stages) != islands:
                continue
            baseline = tuple(Stage(s.device, s.begin, s.end, min(s.threads, cores-1) if s.device == "cpu" else 1,
                                   tuple(range(min(s.threads, cores-1))) if s.device == "cpu" else tuple(range(cores))) for s in stages)
            host = sum(costs[s.signature].run_core_ms for s in baseline if s.device == "vta")
            pool.append((host, candidate_id(baseline), baseline))
        if pool:
            pool.sort()
            baseline = pool[round(quantile * (len(pool)-1))][2]
            isolated = tuple(Stage(s.device,s.begin,s.end,s.threads,s.mask if s.device == "cpu" else (cores-1,)) for s in baseline)
            chosen.append({"islands": islands, "shared": [s.to_dict() for s in baseline],
                           "dedicated": [s.to_dict() for s in isolated]})
    return chosen


def diagnostic_order(boot_index, islands, seed):
    """Three randomized orders, each followed on the next boot by its reverse."""
    order = [(mode, detailed) for mode in ("shared", "dedicated") for detailed in (False, True)]
    random.Random(seed + (boot_index // 2) * 101 + islands * 17).shuffle(order)
    return list(reversed(order)) if boot_index % 2 else order


def run_diagnostics(execute, pairs, phase_dir, anchor, environment, warmup, common, boot, seed):
    """Qualification and drift-guarded intervention blocks; each retry has new IDs."""
    from cpu_vta_partitioning.prepare.payload import EXECUTION_POLICY
    pairs = list(pairs)
    random.Random(seed + common["boot_index"] // 2).shuffle(pairs)
    if common["boot_index"] % 2:
        pairs.reverse()
    measurements, blocks = {}, []
    for pair in pairs:
        islands = pair["islands"]
        result_file = phase_dir / f"diagnostic-islands{islands}.json"
        if result_file.exists():
            block = read(result_file)
        else:
            for mode in ("shared", "dedicated"):
                execute(task("qualify_pipeline", pair[mode], environment, warmup=0, scored=3, tail=0,
                             indices=[64,65,66], diagnostic_islands=islands, diagnostic_mode=mode, **common))
            for attempt in range(EXECUTION_POLICY["diagnostic_replacement_blocks"] + 1):
                bad = phase_dir / f"invalid-diagnostic-islands{islands}-attempt{attempt}.json"
                if bad.exists():
                    continue
                label = {**common, "diagnostic_islands":islands,"diagnostic_attempt":attempt}
                before = execute(task("anchor", anchor, environment, warmup=warmup, position="diagnostic_before", **label))
                values = {}
                order = diagnostic_order(common["boot_index"], islands, seed)
                for mode, detailed in order:
                    key = f"islands{islands}_{mode}_{'profile' if detailed else 'plain'}"
                    values[key] = execute(task("diagnostic", pair[mode], environment, warmup=warmup,
                                               detailed=detailed, diagnostic_key=key, **label))
                after = execute(task("anchor", anchor, environment, warmup=warmup, position="diagnostic_after", **label))
                block = {"boot_id":boot,"islands":islands,"attempt":attempt,"order":[list(x) for x in order],
                         "before":before,"after":after,"measurements":values}
                if abs(after["ii_ms"] / before["ii_ms"] - 1) > EXECUTION_POLICY["diagnostic_anchor_drift"]:
                    write(bad, {**block,"reason":"diagnostic_anchor_drift_exceeds_5pct"})
                    continue
                write(result_file, block)
                break
            else:
                raise RuntimeError("diagnostic drift exhausted replacement blocks; reboot and resume this replicate")
        measurements.update(block["measurements"])
        blocks.append({k:v for k,v in block.items() if k != "measurements"})
    return {"boot_id":boot,"measurements":measurements,"blocks":blocks,"execution_revision":EXECUTION_POLICY["revision"]}


def run_current(root, transport):
    root = Path(root).resolve()
    with locked(root):
        state = validate_run_identity(root)
        bundle = Path(state["bundle"])
        from cpu_vta_partitioning.prepare.package import validate
        validate(bundle)
        boot = transport.boot_id()
        if hasattr(transport, "wait_idle"):
            transport.wait_idle(boot)
        retry_cleanup(root, transport)
        if state["status"] == "COMPLETE":
            return state
        if boot in state["completed_boots"]:
            state["status"] = "WAITING_NEW_BOOT"
            write(root / "state.json", state, replace=True)
            return state
        if state["active_boot"] and state["active_boot"] != boot:
            # An unexpected reboot invalidates the incomplete replicate as a whole.
            event(root, "incomplete_boot_abandoned", boot_id=state["active_boot"], replacement=boot)
            if hasattr(transport, "recover_abandoned"):
                transport.recover_abandoned(root, state["active_boot"])
            state["boot_index"] = state.get("boot_index",0)
        state["active_boot"], state["status"] = boot, "RUNNING"
        write(root / "state.json", state, replace=True)
        remote = transport.deploy(bundle, root)
        transport.qualify_environment(remote, boot, root, bundle)
        environment = state["environments"][state["environment_index"]]
        envdir = root / f"env{state['environment_index']}"
        envdir.mkdir(exist_ok=True)
        programs = read(bundle / "plans/programs.json")
        protocol = read(bundle / "plans/protocol.json")
        anchor = initial_anchor(programs, environment)
        common = {"campaign_phase": state["phase"], "boot_index": state["boot_index"]}

        def execute(t):
            print(f"[{state['environment_index']} {state['phase']} boot {state['boot_index']+1}] {t['kind']} {t['task_id']}", flush=True)
            return measured(t, bundle, remote, root, boot, transport)

        # Full outputs across all 64 real and diagnostic inputs, once per boot.
        execute(task("qualify_pipeline", anchor, environment, warmup=0, scored=67, tail=0, indices=list(range(67)), **common))
        if state["phase"] == "calibration" and state["boot_index"] == 0:
            for p in programs.values():
                threads = environment["cores"] if p["device"] == "cpu" else 1
                s = Stage(p["device"], p["begin"], p["end"], threads, tuple(range(environment["cores"])))
                execute(task("qualify_component", [s], environment, warmup=0, scored=19, tail=0,
                             indices=[*range(16),64,65,66], **common))

        phase = state["phase"]
        phase_dir = envdir / phase / ("boot" + str(state["boot_index"]))
        phase_dir.mkdir(parents=True, exist_ok=True)
        # Never mix rounds from an abandoned kernel boot with a replacement boot.
        for old in phase_dir.glob("*.json"):
            if read(old)["boot_id"] != boot:
                archive = phase_dir / ("abandoned-" + read(old)["boot_id"])
                archive.mkdir(exist_ok=True)
                old.rename(archive / old.name)

        if phase != "calibration":
            ranking = read(envdir / "ranking.json")
            pool = screening_pool(ranking)
            if phase == "confirmation":
                selection = read(envdir / "selection.json")
                wanted = {key for budgets in selection.values() for key in budgets.values()}
                pool = {key: pool[key] for key in sorted(wanted)}
            # Every newly selected composition must pass all-output verification.
            for key, row in pool.items():
                execute(task("qualify_pipeline", row["stages"], environment, warmup=0, scored=3, tail=0,
                             indices=[64,65,66], candidate=key, **common))

        warmup_file = envdir / "warmup.json"
        if not warmup_file.exists():
            a = execute(task("warmup_check", anchor, environment, warmup=20, **common))
            b = execute(task("warmup_check", anchor, environment, warmup=100, **common))
            warmup = 100 if abs(a["ii_ms"] / b["ii_ms"] - 1) > .02 else 20
            write(warmup_file, {"warmup": warmup, "boot_id": boot, "short": a, "long": b})
        warmup = read(warmup_file)["warmup"]

        for round_index in range(2):
            result_file = phase_dir / f"round{round_index}.json"
            if result_file.exists():
                Evidence(envdir).round(result_file)
                continue
            accepted = False
            for attempt in range(3):
                bad = phase_dir / f"invalid-round{round_index}-attempt{attempt}.json"
                if bad.exists():
                    continue
                label = {**common, "round": round_index, "attempt": attempt}
                before = execute(task("anchor", anchor, environment, warmup=warmup, position="before", **label))
                measurements = {}
                if phase == "calibration":
                    templates = calibration_templates(programs, environment)
                    random.Random(protocol["seed"] + state["boot_index"] * 31 + round_index).shuffle(templates)
                    for template in templates:
                        template = {k:v for k,v in template.items() if k != "task_id"}
                        measurements[template["signature"]] = execute(task(**template, **label))
                else:
                    keys = sorted(pool)
                    random.Random(protocol["seed"] + state["boot_index"] * 31 + round_index).shuffle(keys)
                    for key in keys:
                        measurements[key] = execute(task(phase, pool[key]["stages"], environment, warmup=warmup, candidate=key, **label))
                after = execute(task("anchor", anchor, environment, warmup=warmup, position="after", **label))
                record = {"boot_id": boot, "round": round_index, "attempt": attempt,
                          "before": before, "after": after, "measurements": measurements}
                if abs(after["ii_ms"] / before["ii_ms"] - 1) > .05:
                    write(bad, {**record, "reason": "anchor_drift_exceeds_5pct"})
                    continue
                write(result_file, record)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("anchor drift exhausted two replacement rounds; reboot and resume this replicate")

        if phase == "confirmation":
            diagnostic_file = phase_dir / "diagnostic.json"
            if not diagnostic_file.exists():
                diagnostics = run_diagnostics(execute, read(envdir / "diagnostic_selection.json"), phase_dir,
                                              anchor, environment, warmup, common, boot, protocol["seed"])
                write(diagnostic_file, diagnostics)

        state["completed_boots"].append(boot)
        state["active_boot"] = None
        state["boot_index"] += 1
        limit = 6 if phase == "confirmation" else 2
        if state["boot_index"] == limit:
            state["boot_index"] = 0
            if phase == "calibration":
                rounds = phase_rounds(envdir, phase)
                boot_ids = sorted({r["boot_id"] for r in rounds})
                costs = {key: {field: statistics.mean(r["measurements"][key][field] for r in rounds)
                               for field in ("service_ms", "run_core_ms", "boundary_core_ms")}
                         for key in rounds[0]["measurements"]}
                document = {"source": "board_calibration", "bundle_id": state["bundle_id"], "environment": environment,
                            "boot_ids": boot_ids, "costs": costs}
                write(envdir / "costs.json", document, replace=True)
                costs = load_costs(document, environment=environment, bundle_id=state["bundle_id"])
                ranking = rank(programs, costs, environment["cores"])
                write(envdir / "ranking.json", ranking, replace=True)
                write(envdir / "diagnostic_selection.json", diagnosis_candidates(ranking, costs, environment["cores"]), replace=True)
                state["phase"] = "screening"
            elif phase == "screening":
                write(envdir / "selection.json", freeze_selection(read(envdir / "ranking.json"), grouped(phase_rounds(envdir, phase))), replace=True)
                state["phase"] = "confirmation"
            else:
                result = analyze_environment(envdir)
                write(envdir / "analysis.json", result, replace=True)
                if state["environment_index"] == 0 and not result["continue_sensitivity"]:
                    state["status"] = "COMPLETE"
                elif state["environment_index"] == len(state["environments"]) - 1:
                    state["status"] = "COMPLETE"
                else:
                    state["environment_index"] += 1
                    state["phase"] = "calibration"
        if state["status"] != "COMPLETE":
            state["status"] = "WAITING_NEW_BOOT"
        write(root / "state.json", state, replace=True)
        return state


def analyze_environment(envdir):
    evidence = Evidence(envdir)
    rounds, diagnostics = evidence.confirmation()
    selection = read(envdir / "selection.json")
    measures = grouped(rounds)
    mechanism = {}
    for islands in (1,2,3):
        prefix = f"islands{islands}_"
        if not diagnostics or prefix + "shared_plain" not in diagnostics[0]["measurements"]:
            continue
        shared = {d["boot_id"]: d["measurements"][prefix + "shared_plain"]["ii_ms"] for d in diagnostics}
        dedicated = {d["boot_id"]: d["measurements"][prefix + "dedicated_plain"]["ii_ms"] for d in diagnostics}
        overhead = max(abs(d["measurements"][prefix + mode + "_profile"]["ii_ms"] /
                           d["measurements"][prefix + mode + "_plain"]["ii_ms"] - 1)
                       for d in diagnostics for mode in ("shared", "dedicated"))
        effect = paired(dedicated, shared)
        mechanism[str(islands)] = {"host_isolation": effect, "max_profile_overhead": overhead,
                                    "interpretation": "host placement intervention; scheduler/PMU counters span the retained completion window"}
        scheduler = {}
        for mode in ("shared", "dedicated"):
            samples = [s for d in diagnostics for s in d["measurements"][prefix + mode + "_profile"].get("scheduler_by_stage",[]) if s["device"] == "vta"]
            scheduler[mode] = {"all_tids_observed":bool(samples) and all(s["all_tids_observed"] for s in samples),
                               "mean_host_runqueue_wait_ns":statistics.mean(s["runqueue_wait_ns"] for s in samples) if samples else None}
        mechanism[str(islands)]["scheduler"] = scheduler
    from cpu_vta_partitioning.analysis import holm
    diagnostic_p = holm({k:m["host_isolation"]["p_two_sided"] for k,m in mechanism.items()})
    for k,m in mechanism.items():
        m["host_isolation"]["p_holm_diagnostic"] = diagnostic_p[k]
    supported = any(m["host_isolation"]["gain"] >= .05 and diagnostic_p[k] < .05
                    and m["max_profile_overhead"] <= .02
                    and all(s["all_tids_observed"] for s in m["scheduler"].values())
                    and m["scheduler"]["shared"]["mean_host_runqueue_wait_ns"] > m["scheduler"]["dedicated"]["mean_host_runqueue_wait_ns"]
                    for k,m in mechanism.items())
    result = report(selection, measures, mechanism_supported=supported)
    result["mechanism_diagnostics"] = mechanism
    result["boot_means"] = measures
    result["provenance"] = {"bundle_id": evidence.bundle_id,
                            "controller_identity": evidence.controller_identity}
    return result
