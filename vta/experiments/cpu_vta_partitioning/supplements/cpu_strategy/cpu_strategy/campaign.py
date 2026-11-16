"""Frozen component profiles, exact static DP and bounded pipeline verification."""
from pathlib import Path
import random
import re
import statistics

from cpu_vta_partitioning.common import (
    digest, event, inventory, locked, now, read, sha256, sync_tree, verify_inventory, write,
)
from .domain import components, matrix, protocol, ROOT
from .evidence import Evidence, identified
from .measurement import make_task, validate_summary


COST_FIELDS = ("service_ms", "run_core_ms", "boundary_core_ms", "set_ms", "run_ms", "get_ms",
               "set_core_ms", "get_core_ms")


def _identity():
    from .prepare import identity
    return identity()


def _save(root, state):
    write(Path(root) / "state.json", state, replace=True)


def _pin(root, state, *names):
    for name in names:
        path = Path(root) / name
        value = {"sha256": sha256(path), "bytes": path.stat().st_size}
        previous = state.setdefault("frozen_files", {}).get(name)
        if previous is not None and previous != value:
            raise ValueError("frozen campaign file changed: " + name)
        state["frozen_files"][name] = value
    _save(root, state)


def _publish(root, state, name, value):
    path = Path(root) / name
    if path.exists():
        if read(path) != value:
            raise ValueError("frozen campaign artifact disagrees: " + name)
    else:
        write(path, value)
    _pin(root, state, name)
    return value


def validate_state(root):
    root = Path(root).resolve()
    state = read(root / "state.json")
    if (state.get("experiment") != protocol()["experiment"]
            or state.get("controller_identity") != _identity()):
        raise ValueError("controller changed; use its source snapshot or create a new campaign")
    bundle = Path(state["bundle"])
    manifest = read(bundle / "manifest.json")
    if (manifest["bundle_id"] != state["bundle_id"]
            or manifest["controller_identity"] != state["controller_identity"]
            or sha256(bundle / "manifest.json") != state["bundle_manifest_sha256"]):
        raise ValueError("campaign and bundle identity mismatch")
    if read(root / "protocol.json") != protocol() or read(root / "matrix.json") != matrix():
        raise ValueError("frozen protocol or CPU configuration matrix changed")
    verify_inventory(root, state["frozen_files"])
    if (type(state["boot_index"]) is not int or not 0 <= state["boot_index"] <= 4
            or len(state["completed_boots"]) != state["boot_index"]
            or len(set(state["completed_boots"])) != state["boot_index"]
            or set(state["completed_boots"]) & set(state["excluded_boots"])):
        raise ValueError("invalid independent boot accounting")
    if state["status"] == "COMPLETE" and state["boot_index"] != 4:
        raise ValueError("incomplete campaign marked complete")
    return state


def create(bundle, root, connection, *, excluded_boot):
    from .prepare import validate
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    if root.exists():
        raise FileExistsError("run already exists; use board resume")
    manifest = validate(bundle)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", excluded_boot):
        raise ValueError("invalid preflight boot ID")
    root.mkdir(parents=True)
    state = {"experiment": protocol()["experiment"], "controller_identity": _identity(),
             "bundle": str(bundle), "bundle_id": manifest["bundle_id"],
             "bundle_manifest_sha256": sha256(bundle / "manifest.json"), "connection": connection,
             "created": now(), "status": "WAITING_NEW_BOOT", "boot_index": 0,
             "active_boot": None, "completed_boots": [], "excluded_boots": [excluded_boot],
             "abandoned_boots": [], "completed_rounds": [], "abandoned_rounds": [],
             "frozen_files": {}}
    write(root / "protocol.json", protocol())
    write(root / "matrix.json", matrix())
    _pin(root, state, "protocol.json", "matrix.json")
    return state


def _linked(root, path, receipt):
    return {**receipt["summary"], "_receipt": {"path": str(path.relative_to(root)), "sha256": sha256(path)}}


def measured(document, bundle, remote, root, boot, transport):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    if transport.boot_id() != boot:
        raise RuntimeError("board boot changed before measurement")
    document = identified(document, read(bundle / "manifest.json")["bundle_id"])
    path = root / "accepted" / boot / (document["task_id"] + ".json")
    evidence = Evidence(root, document["bundle_id"], bundle)
    if path.exists():
        value = _linked(root, path, read(path))
        evidence.summary(value, document, boot)
        _acknowledge(root, document, boot, transport)
        return value
    event(root, "session_start", boot_id=boot, task_id=document["task_id"], task_kind=document["kind"])
    try:
        result = Path(transport.measure(document, bundle, remote, root, boot)).resolve()
        if transport.boot_id() != boot:
            raise RuntimeError("board boot changed before result acceptance")
        if not result.is_relative_to(root / "raw" / boot / document["task_id"]):
            raise ValueError("native result is outside its task raw directory")
        summary = validate_summary(result, document, bundle)
        files = {str(result.relative_to(root) / name): value for name, value in inventory(result).items()}
        sync_tree(result)
        write(path, {"task": document, "boot_id": boot, "summary": summary, "files": files})
        value = _linked(root, path, read(path))
        evidence.summary(value, document, boot)
        event(root, "session_accepted", boot_id=boot, task_id=document["task_id"])
        _acknowledge(root, document, boot, transport)
        return value
    except Exception as error:
        event(root, "session_failed", boot_id=boot, task_id=document["task_id"], error=str(error))
        raise


def _acknowledge(root, document, boot, transport):
    if not hasattr(transport, "acknowledge"):
        return
    pending = root / "cleanup_pending" / boot / (document["task_id"] + ".json")
    try:
        transport.acknowledge(document, root, boot)
    except (OSError, RuntimeError) as error:
        write(pending, {"task": document, "boot_id": boot}, replace=True)
        event(root, "cleanup_pending", boot_id=boot, task_id=document["task_id"], error=str(error))
    else:
        pending.unlink(missing_ok=True)


def _retry_cleanup(root, state, transport):
    evidence = Evidence(root, state["bundle_id"], state["bundle"])
    for path in sorted((root / "cleanup_pending").glob("*/*.json")):
        item = read(path)
        receipt = root / "accepted" / item["boot_id"] / path.name
        value = _linked(root, receipt, read(receipt))
        evidence.summary(value, item["task"], item["boot_id"])
        _acknowledge(root, item["task"], item["boot_id"], transport)


class Recorder:
    def __init__(self, root, state, transport, remote):
        self.root, self.state, self.transport, self.remote = Path(root), state, transport, remote
        self.boot = state["active_boot"]
        self.evidence = Evidence(root, state["bundle_id"], state["bundle"])

    def saved(self, name, document):
        path = self.root / "records" / self.boot / (name + ".json")
        if path.exists():
            record = read(path)
            if record["document"] != document or record["boot_id"] != self.boot:
                raise ValueError("record belongs to another task or boot")
            value = record["summary"]
            self.evidence.summary(value, document, self.boot)
        else:
            if self.transport is None:
                raise ValueError("missing measurement record: " + name)
            print("[CPU配置] " + name, flush=True)
            value = measured(document, self.state["bundle"], self.remote,
                             self.root, self.boot, self.transport)
            self.evidence.summary(value, document, self.boot)
            write(path, {"document": document, "boot_id": self.boot, "summary": value})
        _pin(self.root, self.state, str(path.relative_to(self.root)), value["_receipt"]["path"])
        return value


def _task(kind, stages, state, *, warmup=0, scored=4, tail=0, indices=None, **labels):
    return make_task(kind, stages, warmup=warmup, scored=scored, tail=tail,
                     indices=indices if indices is not None else protocol()["qualification_indices"],
                     campaign_boot_index=state["boot_index"], **labels)


def _pipeline(kind, candidate, state, warmup, **labels):
    return _task(kind, candidate["stages"], state, warmup=warmup, scored=32, tail=4,
                 indices=[16], candidate_id=candidate["id"], **labels)


def calibrate(root, state, recorder):
    costs, blocks = {}, {}
    for component in components():
        stage = component["stage"]
        name = stage["signature"]
        document = _task("qualify_component", [stage], state, component_signature=name,
                         component_role=component["role"])
        recorder.saved("components/qualify/" + name, document)
        samples = []
        for block in range(component["blocks"]):
            document = _task("calibration", [stage], state, warmup=5, scored=32, indices=[0],
                             component_signature=name, component_role=component["role"],
                             component_block=block)
            value = recorder.saved(f"components/calibration/{name}/block{block}", document)
            samples.append({"document": document, "summary": value, "boot_id": state["active_boot"]})
        costs[name] = {key: statistics.mean(s["summary"][key] for s in samples) for key in COST_FIELDS}
        costs[name]["cpu_time_scope"] = "exclusive_process_cpu_time"
        blocks[name] = samples
    return _publish(root, state, "costs.json", {
        "source": "fresh_isolated_board_calibration", "boot_id": state["active_boot"],
        "bundle_id": state["bundle_id"], "costs": costs, "blocks": blocks,
    })


def profile_selection(cost_document):
    """Restricted load-balancing adaptation, frozen before any measured pipeline result."""
    if cost_document.get("source") != "fresh_isolated_board_calibration":
        raise ValueError("predictive selection requires current-board component measurements")
    costs, plans = cost_document["costs"], matrix()
    stages = [plans["configs"][key]["stages"] for key in plans["families"]["shared"]]
    chosen = []
    for index in (0, 2):
        options = {s[index]["signature"]: s[index] for s in stages}
        chosen.append(min(options, key=lambda key: (costs[key]["service_ms"], key)))
    local = next(key for key in plans["families"]["shared"]
                 if [plans["configs"][key]["stages"][i]["signature"] for i in (0, 2)] == chosen)
    scores = {key: max(costs[s["signature"]]["service_ms"] for s in plans["configs"][key]["stages"])
              for key in plans["families"]["disjoint"]}
    predicted = min(scores, key=lambda key: (scores[key], key))
    from .search import rank_dp
    ranking = rank_dp(cost_document)
    return {"source": "fresh_isolated_profiles_before_pipeline", "local_fastest": local,
            "profile_disjoint": predicted, "scores": scores,
            "dp_static": ranking["rankings"][0]["id"], "dp_ranking": ranking,
            "interpretation": "Pipe-it-inspired fixed-partition minmax adaptation; not its full algorithm",
            "model_scope": "isolated stage services; shared FPGA host interference is not explicitly predicted"}


def warmup_qualification(root, state, recorder):
    checks, use_extended = {}, False
    plans = matrix()
    for number, key in enumerate(plans["warmup_anchors"]):
        checks[key] = {}
        for count in (8, 24):
            doc = _pipeline("warmup_qualification", plans["configs"][key], state, count,
                            warmup_anchor=number, warmup_repeat=0)
            checks[key][str(count)] = recorder.saved(f"warmup/anchor{number}/warm{count}/repeat0", doc)
        use_extended |= abs(checks[key]["8"]["ii_ms"] / checks[key]["24"]["ii_ms"] - 1) > .05
    chosen = 24 if use_extended else 8
    if use_extended:
        for number, key in enumerate(plans["warmup_anchors"]):
            doc = _pipeline("warmup_qualification", plans["configs"][key], state, 24,
                            warmup_anchor=number, warmup_repeat=1)
            repeat = recorder.saved(f"warmup/anchor{number}/warm24/repeat1", doc)
            checks[key]["24_repeat"] = repeat
            if abs(repeat["ii_ms"] / checks[key]["24"]["ii_ms"] - 1) > .05:
                _publish(root, state, "warmup_failure.json", {"checks": checks, "boot_id": state["active_boot"]})
                raise RuntimeError("24-frame warmup remains unstable; stop and investigate, no selective extension")
    return _publish(root, state, "warmup.json", {"warmup": chosen, "checks": checks,
                                                "boot_id": state["active_boot"]})


def qualify_candidates(state, recorder, pool):
    for key, row in sorted(pool.items()):
        document = _task("qualify_pipeline", row["stages"], state, candidate_id=key)
        recorder.saved("pipeline/qualify/" + digest(key)[:20], document)


def control_pool(predicted):
    """Fixed references and mechanism controls supplement the static DP Top-5."""
    plans = matrix()
    wanted = {predicted["local_fastest"], predicted["profile_disjoint"],
              *plans["references"].values(), *plans["interventions"].values()}
    if len(wanted) > 8:
        raise ValueError("control pool exceeds the preregistered bound")
    return {key: plans["configs"][key] for key in sorted(wanted)}


def screening_pool(predicted):
    plans = matrix()
    keys = set(control_pool(predicted)) | {r["id"] for r in predicted["dp_ranking"]["rankings"][:5]}
    if len(keys) > 13:
        raise ValueError("screening pool exceeds the frozen budget")
    return {key: plans["configs"][key] for key in sorted(keys)}


def _round_directory(root, state, role):
    if state["boot_index"]:
        if role != "confirmation":
            raise ValueError("screening is forbidden in independent confirmation")
        return root / "confirmation" / f"boot{state['boot_index']}"
    if role != "screening":
        raise ValueError("invalid exploration measurement role")
    return root / "exploration" / state["active_boot"] / "screening"


def _round_labels(phase, role, number, attempt):
    return {"campaign_phase": phase, "measurement_role": role, "round": number, "attempt": attempt}


def run_rounds(root, state, recorder, pool, warmup):
    phase = "exploration" if state["boot_index"] == 0 else "confirmation"
    role = "screening" if phase == "exploration" else "confirmation"
    directory = _round_directory(root, state, role)
    directory.mkdir(parents=True, exist_ok=True)
    plans = matrix()
    anchor = plans["configs"][plans["references"][protocol()["anchor_reference"]]]
    result = []
    for number in range(2):
        path = directory / f"round{number}.json"
        if path.exists():
            record = read(path)
            if (record.get("phase") != phase or record.get("boot_index") != state["boot_index"]
                    or record.get("boot_id") != state["active_boot"] or record.get("round") != number
                    or record.get("role") != role):
                raise ValueError("round file belongs to a different replicate or round")
            _validate_round(root, state, record, pool, warmup)
        else:
            for attempt in range(protocol()["max_round_attempts"]):
                invalid = directory / f"invalid-round{number}-attempt{attempt}.json"
                if invalid.exists():
                    continue
                common = _round_labels(phase, role, number, attempt)
                before_doc = _pipeline("anchor", anchor, state, warmup, position="before", **common)
                prefix = f"{phase}/{role}/round{number}/attempt{attempt}"
                before = recorder.saved(prefix + "/before", before_doc)
                order = sorted(pool)
                random.Random(20261104 + state["boot_index"] * 101 + number * 17 + attempt * 1009).shuffle(order)
                values = {}
                for key in order:
                    doc = _pipeline("performance", pool[key], state, warmup, **common)
                    values[key] = recorder.saved(prefix + "/" + digest(key)[:20], doc)
                after_doc = _pipeline("anchor", anchor, state, warmup, position="after", **common)
                after = recorder.saved(prefix + "/after", after_doc)
                record = {"phase": phase, "boot_index": state["boot_index"], "boot_id": state["active_boot"],
                          "role": role, "round": number, "attempt": attempt, "order": order,
                          "before": before, "after": after, "measurements": values}
                if abs(after["ii_ms"] / before["ii_ms"] - 1) > protocol()["anchor_drift_limit"]:
                    write(invalid, {**record, "invalid_reason": "anchor_drift_exceeds_5pct"})
                    _pin(root, state, str(invalid.relative_to(root)))
                    continue
                write(path, record)
                _pin(root, state, str(path.relative_to(root)))
                break
            else:
                state["status"] = "STOPPED_ANCHOR_DRIFT"
                state["last_error"] = "anchor drift exhausted the single replacement"
                _save(root, state)
                event(root, "stopped_anchor_drift", boot_id=state["active_boot"], round=number)
                raise RuntimeError("anchor drift exhausted the single replacement; keep failed evidence and investigate")
        name = str(path.relative_to(root))
        if name not in state["completed_rounds"]:
            state["completed_rounds"].append(name)
            _save(root, state)
        result.append(record)
    return result


def _observed_mean(rounds, candidate):
    if len(rounds) != 2 or {r["round"] for r in rounds} != {0, 1}:
        raise ValueError("an observation requires exactly two complete rounds")
    return {key: statistics.mean(r["measurements"][candidate][key] for r in rounds)
            for key in ("ii_ms", "process_core_ms_per_completion", "average_cpu_cores")}


def selection_from_rounds(rounds, predicted):
    plans = matrix()
    rounds = [r for r in rounds if r["phase"] == "exploration" and r["role"] == "screening"]
    pool = screening_pool(predicted)
    if (len(rounds) != 2 or {r["round"] for r in rounds} != {0, 1}
            or len({r["boot_id"] for r in rounds}) != 1
            or any(r["boot_index"] != 0 or set(r["measurements"]) != set(pool) for r in rounds)):
        raise ValueError("selection requires two complete frozen DP screening rounds")
    means = {key: _observed_mean(rounds, key)["ii_ms"] for key in pool}
    top5 = [row["id"] for row in predicted["dp_ranking"]["rankings"][:5]]
    winner = min(top5, key=lambda key: (means[key], key))
    methods = {"local_fastest": predicted["local_fastest"], "profile_disjoint": predicted["profile_disjoint"],
               "dp_static": predicted["dp_static"], "dp_budget5": winner}
    wanted = set(methods.values()) | set(plans["references"].values()) | set(plans["interventions"].values())
    if len(wanted) > 10 or len(means) > 13:
        raise ValueError("confirmation pool exceeds frozen budget")
    best = min(means, key=lambda key: (means[key], key))
    if best not in wanted:
        raise ValueError("measured verification best is missing from confirmation pool")
    return {"methods": methods, "references": plans["references"], "interventions": plans["interventions"],
            "pool": {key: plans["configs"][key] for key in sorted(wanted)}, "exploration_best": best,
            "exploration_mean_ii_ms": means, "selection_boot_id": rounds[0]["boot_id"],
            "measured_exploration_configurations": len(means), "static_dp_top5": top5,
            "static_ranking_sha256": digest(predicted["dp_ranking"]),
            "ranking_uses_pipeline_feedback": False, "candidate_library_is_not_exhaustively_measured": True}


def freeze_selection(root, state, rounds, predicted):
    selection = selection_from_rounds(rounds, predicted)
    if selection["selection_boot_id"] != state["active_boot"]:
        raise ValueError("selection came from a different exploration boot")
    _publish(root, state, "selection.json", selection)
    state["selection"] = selection
    _save(root, state)
    return selection


def _validate_round(root, state, record, pool, warmup):
    phase, index, boot = record["phase"], record["boot_index"], record["boot_id"]
    expected_boot = (state["completed_boots"][index] if type(index) is int and 0 <= index < len(state["completed_boots"])
                     else state["active_boot"] if index == state["boot_index"] else None)
    if (type(index) is not int or not 0 <= index < 4 or boot != expected_boot
            or phase not in ("exploration", "confirmation") or (index == 0) != (phase == "exploration")
            or record["round"] not in (0, 1) or record["attempt"] not in (0, 1)
            or set(record["measurements"]) != set(pool)
            or sorted(record["order"]) != sorted(pool) or len(record["order"]) != len(pool)):
        raise ValueError("measurement round identity or configuration coverage changed")
    evidence = Evidence(root, state["bundle_id"], state["bundle"])
    temporary = {**state, "boot_index": index}
    role = "screening" if phase == "exploration" else "confirmation"
    if record["role"] != role:
        raise ValueError("round role differs from its phase")
    common = _round_labels(phase, role, record["round"], record["attempt"])
    plans = matrix()
    anchor = plans["configs"][plans["references"][protocol()["anchor_reference"]]]
    for position in ("before", "after"):
        doc = _pipeline("anchor", anchor, temporary, warmup, position=position, **common)
        evidence.summary(record[position], doc, boot)
    if abs(record["after"]["ii_ms"] / record["before"]["ii_ms"] - 1) > protocol()["anchor_drift_limit"]:
        raise ValueError("accepted round exceeds anchor drift limit")
    for key, value in record["measurements"].items():
        evidence.summary(value, _pipeline("performance", pool[key], temporary, warmup, **common), boot)


def validated_rounds(root, phase=None):
    root = Path(root).resolve()
    state = validate_state(root)
    evidence = Evidence(root, state["bundle_id"], state["bundle"])
    costs = read(root / "costs.json") if (root / "costs.json").exists() else None
    if costs:
        _validate_costs(state, costs, evidence)
        predicted = profile_selection(costs)
        if (root / "profile_selection.json").exists() and read(root / "profile_selection.json") != predicted:
            raise ValueError("profile selection disagrees with component costs")
    if not (root / "warmup.json").exists():
        return []
    if costs is None or not (root / "profile_selection.json").exists():
        raise ValueError("pipeline measurements exist without their prior static DP ranking")
    warmup = _validate_warmup(state, read(root / "warmup.json"), evidence, costs["boot_id"])
    selection = read(root / "selection.json") if (root / "selection.json").exists() else None
    if selection and selection != state.get("selection"):
        raise ValueError("frozen selection differs from state")
    rounds = []
    for name in state["completed_rounds"]:
        record = read(root / name)
        if record["boot_id"] not in state["completed_boots"] and record["boot_id"] != state["active_boot"]:
            raise ValueError("round boot is not part of this campaign")
        temporary = {**state, "boot_index": record["boot_index"], "active_boot": record["boot_id"]}
        expected_path = str((_round_directory(root, temporary, record["role"])
                             / f"round{record['round']}.json").relative_to(root))
        if name != expected_path:
            raise ValueError("round file path differs from its phase/boot identity")
        if record["phase"] == "confirmation":
            if selection is None:
                raise ValueError("confirmation exists before selection was frozen")
            pool = selection["pool"]
        else:
            pool = screening_pool(predicted)
        _validate_round(root, state, record, pool, warmup)
        rounds.append(record)
    if selection and selection != selection_from_rounds(rounds, predicted):
        raise ValueError("frozen selection disagrees with static ranking and screening measurements")
    return [r for r in rounds if phase is None or r["phase"] == phase]


def validated_ranking(root):
    """Replay the frozen DP against hash-verified isolated costs, never feedback."""
    root = Path(root).resolve()
    validated_rounds(root)
    if not (root / "profile_selection.json").exists():
        return None
    return read(root / "profile_selection.json")["dp_ranking"]


def _validate_costs(state, costs, evidence):
    expected = {c["stage"]["signature"]: c for c in components()}
    if (costs["source"] != "fresh_isolated_board_calibration" or costs["bundle_id"] != state["bundle_id"]
            or set(costs["blocks"]) != set(expected) or set(costs["costs"]) != set(expected)):
        raise ValueError("component table identity/coverage differs from the frozen protocol")
    temporary = {**state, "boot_index": 0}
    calculated = {}
    for name, component in expected.items():
        entries = costs["blocks"][name]
        if len(entries) != component["blocks"]:
            raise ValueError("missing component diagnostic blocks")
        for block, entry in enumerate(entries):
            document = _task("calibration", [component["stage"]], temporary, warmup=5, scored=32,
                             indices=[0], component_signature=name, component_role=component["role"],
                             component_block=block)
            if entry["document"] != document or entry["boot_id"] != costs["boot_id"]:
                raise ValueError("component block boot/task differs from the frozen profile")
            evidence.summary(entry["summary"], document, entry["boot_id"])
        calculated[name] = {key: statistics.mean(e["summary"][key] for e in entries) for key in COST_FIELDS}
        calculated[name]["cpu_time_scope"] = "exclusive_process_cpu_time"
    if costs["costs"] != calculated:
        raise ValueError("component costs disagree with accepted isolated native records")


def _validate_warmup(state, document, evidence, boot):
    plans, checks = matrix(), document["checks"]
    if document["boot_id"] != boot or set(checks) != set(plans["warmup_anchors"]):
        raise ValueError("warmup boot/configuration coverage differs")
    temporary = {**state, "boot_index": 0}
    extended = False
    for number, key in enumerate(plans["warmup_anchors"]):
        for count in (8, 24):
            task = _pipeline("warmup_qualification", plans["configs"][key], temporary, count,
                             warmup_anchor=number, warmup_repeat=0)
            evidence.summary(checks[key][str(count)], task, boot)
        extended |= abs(checks[key]["8"]["ii_ms"] / checks[key]["24"]["ii_ms"] - 1) > .05
    expected = 24 if extended else 8
    if document["warmup"] != expected:
        raise ValueError("warmup decision differs from the preregistered rule")
    for number, key in enumerate(plans["warmup_anchors"]):
        if set(checks[key]) != ({"8", "24", "24_repeat"} if extended else {"8", "24"}):
            raise ValueError("warmup stability check coverage changed")
        if extended:
            task = _pipeline("warmup_qualification", plans["configs"][key], temporary, 24,
                             warmup_anchor=number, warmup_repeat=1)
            evidence.summary(checks[key]["24_repeat"], task, boot)
            if abs(checks[key]["24_repeat"]["ii_ms"] / checks[key]["24"]["ii_ms"] - 1) > .05:
                raise ValueError("accepted extended warmup is unstable")
    return expected


def _abandon(root, state, old, transport):
    event(root, "incomplete_boot_abandoned", boot_id=old)
    if hasattr(transport, "recover_abandoned"):
        transport.recover_abandoned(root, old)
    state["abandoned_boots"].append(old)
    old_rounds = [name for name in state["completed_rounds"] if read(root / name)["boot_id"] == old]
    state["completed_rounds"] = [name for name in state["completed_rounds"] if name not in old_rounds]
    state["abandoned_rounds"].extend(old_rounds)
    if state["boot_index"] == 0:
        names = [n for n in ("costs.json", "profile_selection.json", "warmup.json", "warmup_failure.json", "selection.json")
                 if (root / n).exists()]
    else:
        prefix = f"confirmation/boot{state['boot_index']}"
        names = [prefix] if (root / prefix).exists() else []
    for name in names:
        destination = f"abandoned/{old}/{name}"
        (root / destination).parent.mkdir(parents=True, exist_ok=True)
        (root / name).rename(root / destination)
        for relative in list(state["frozen_files"]):
            if relative == name or relative.startswith(name + "/"):
                changed = destination + relative[len(name):]
                state["frozen_files"][changed] = state["frozen_files"].pop(relative)
        state["abandoned_rounds"] = [destination + n[len(name):] if n == name or n.startswith(name + "/") else n
                                     for n in state["abandoned_rounds"]]
    if state["boot_index"] == 0:
        state.pop("selection", None)
    state["active_boot"] = None
    _save(root, state)


def run_current(root, transport):
    from .prepare import validate
    root = Path(root).resolve()
    with locked(root):
        state = validate_state(root)
        validate(state["bundle"])
        if state["status"] == "COMPLETE":
            return state
        boot = transport.boot_id()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", boot):
            raise ValueError("invalid board boot identity")
        if boot in state["completed_boots"] or boot in state["excluded_boots"] or boot in state["abandoned_boots"]:
            state["status"] = "WAITING_NEW_BOOT"
            _save(root, state)
            return state
        if state["active_boot"] and state["active_boot"] != boot:
            _abandon(root, state, state["active_boot"], transport)
        if hasattr(transport, "wait_idle"):
            transport.wait_idle(boot)
        _retry_cleanup(root, state, transport)
        transport.mount_storage()
        remote = transport.deploy(state["bundle"], root)
        transport.verify_payload(remote, boot)
        transport.qualify_environment(remote, boot, root, state["bundle"])
        state["active_boot"], state["status"] = boot, "RUNNING"
        _save(root, state)
        recorder = Recorder(root, state, transport, remote)
        plans = matrix()
        if state["boot_index"] == 0:
            costs = calibrate(root, state, recorder)
            predicted = _publish(root, state, "profile_selection.json", profile_selection(costs))
            event(root, "profile_selection_frozen_before_pipeline", boot_id=boot,
                  profile_disjoint=predicted["profile_disjoint"], local_fastest=predicted["local_fastest"],
                  dp_static=predicted["dp_static"], ranking_sha256=digest(predicted["dp_ranking"]))
            pool = screening_pool(predicted)
            qualify_candidates(state, recorder, pool)
            warmup = warmup_qualification(root, state, recorder)["warmup"]
        else:
            selection = read(root / "selection.json")
            if selection != state["selection"]:
                raise ValueError("selection changed since exploration")
            pool, warmup = selection["pool"], read(root / "warmup.json")["warmup"]
            qualify_candidates(state, recorder, pool)
        rounds = run_rounds(root, state, recorder, pool, warmup)
        if state["boot_index"] == 0:
            freeze_selection(root, state, rounds, predicted)
        if transport.boot_id() != boot:
            raise RuntimeError("board rebooted before this replicate was accepted")
        transport.verify_payload(remote, boot)
        transport.idle_command("sync", boot=boot, timeout=120)
        state["completed_boots"].append(boot)
        state["boot_index"] += 1
        state["active_boot"] = None
        state["status"] = "COMPLETE" if state["boot_index"] == 4 else "WAITING_NEW_BOOT"
        _save(root, state)
        sync_tree(root)
        return state


def preflight(bundle, transport):
    from .prepare import validate
    bundle = Path(bundle).resolve()
    manifest = validate(bundle)
    boot = transport.boot_id()
    root = ROOT / "artifacts/qualification" / (manifest["bundle_id"] + "_" + boot)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root):
        state = {"bundle": str(bundle), "bundle_id": manifest["bundle_id"], "boot_index": -1,
                 "active_boot": boot, "frozen_files": {}, "status": "PREFLIGHT"}
        if (root / "state.json").exists():
            old = read(root / "state.json")
            if any(old[k] != state[k] for k in ("bundle", "bundle_id", "active_boot", "boot_index")):
                raise ValueError("preflight identity mismatch")
            state = old
            verify_inventory(root, state["frozen_files"])
        transport.wait_idle(boot)
        transport.mount_storage()
        remote = transport.deploy(bundle, root)
        transport.verify_payload(remote, boot)
        transport.qualify_environment(remote, boot, root, bundle)
        recorder = Recorder(root, state, transport, remote)
        plans = matrix()
        reports = []
        for key in plans["warmup_anchors"]:
            doc = _task("qualify_pipeline", plans["configs"][key]["stages"], state, candidate_id=key)
            reports.append(recorder.saved("pipeline/" + digest(key)[:20], doc))
        for item in components():
            stage = item["stage"]
            if item["role"] == "diagnostic" and stage["threads"] == 1:
                doc = _task("qualify_component", [stage], state, component_signature=stage["signature"])
                reports.append(recorder.saved("component/" + stage["signature"], doc))
        if transport.boot_id() != boot:
            raise RuntimeError("board rebooted during preflight")
        transport.verify_payload(remote, boot)
        transport.idle_command("sync", boot=boot, timeout=120)
        result = {"status": "PREFLIGHT_PASSED", "excluded_boot_id": boot,
                  "bundle_id": manifest["bundle_id"], "evidence_directory": str(root), "reports": reports}
        _publish(root, state, "PREFLIGHT.json", result)
        return result
