"""Durable one-boot campaign for the nine explicit resource contexts.

Imported component measurements retain their original boot/task identities.
Only the missing components, traffic probes and frozen pipeline pool are run.
Timing variation is descriptive; it never changes or repeats a valid window.
"""
from copy import deepcopy
from pathlib import Path
import random

from .common import identity, load_legacy, protocol
from .runtime import Evidence, make_task, validate_summary


COST_FIELDS = ("service_ms", "run_core_ms", "boundary_core_ms", "set_ms", "run_ms", "get_ms",
               "set_core_ms", "get_core_ms")
PHASES = ("CREATED", "QUALIFY", "COMPONENTS", "TRAFFIC", "HOST_RANK", "PIPELINE", "ANALYZE", "COMPLETE")
STOPPED = {"STOPPED_INVALID_RESULT", "STOPPED_BOOT_CHANGED", "STOPPED_IDENTITY", "PAUSED"}


def _common(bundle=None):
    load_legacy(bundle)
    from cpu_vta_partitioning import common
    return common


def _validate_bundle(bundle):
    from .prepare import validate_bundle
    return validate_bundle(bundle)


def _save(root, state):
    _common(state["bundle"]).write(Path(root) / "state.json", state, replace=True)


def _pin(root, state, *names):
    api = _common(state["bundle"])
    for name in names:
        path = api.relative_file(root, name)
        current = {"sha256": api.sha256(path), "bytes": path.stat().st_size}
        if name in state["frozen_files"] and state["frozen_files"][name] != current:
            raise ValueError("frozen open-resource artifact changed: " + name)
        state["frozen_files"][name] = current
    _save(root, state)


def _publish(root, state, name, value):
    api = _common(state["bundle"])
    path = api.relative_file(root, name)
    expected = api.digest(value)
    pending = state.setdefault("pending_publications", {})
    if name in pending and pending[name] != expected:
        raise ValueError("pending artifact differs: " + name)
    pending[name] = expected
    # Persist the intended content before creating it.  A controller crash
    # between JSON rename and inventory pinning is therefore recoverable.
    _save(root, state)
    if path.exists():
        if api.read(path) != value:
            raise ValueError("frozen open-resource artifact disagrees: " + name)
    else:
        api.write(path, value)
    _pin(root, state, name)
    state["pending_publications"].pop(name, None)
    _save(root, state)
    return value


def _set_phase(root, state, phase):
    if phase not in PHASES:
        raise ValueError("invalid open-resource phase")
    state.update(phase=phase, status=phase)
    state.pop("last_error", None)
    _save(root, state)


def create(bundle, root, connection):
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    manifest = _validate_bundle(bundle)
    if root.exists():
        raise FileExistsError("run already exists; use board resume")
    api = _common(bundle)
    root.mkdir(parents=True)
    state = {"experiment": protocol()["experiment"], "controller_identity": identity(),
             "bundle": str(bundle), "bundle_id": manifest["bundle_id"],
             "bundle_manifest_sha256": api.sha256(bundle / "manifest.json"),
             "connection": deepcopy(connection), "created": api.now(), "status": "CREATED",
             "phase": "CREATED", "active_boot": None, "boot_id": None, "completed_boots": [],
             "frozen_files": {}, "pending_publications": {},
             "completed_components": 0, "imported_components": 0,
             "source_pilot_components": 0, "reused_calibration_components": 0,
             "new_components": 0, "completed_traffic": 0,
             "imported_traffic": 0, "new_traffic": 0,
             "completed_pipeline_configurations": 0}
    api.write(root / "protocol.json", protocol())
    _pin(root, state, "protocol.json")
    return state


def validate_state(root, *, validate_payload=True):
    root = Path(root).resolve()
    api = _common()
    state = api.read(root / "state.json")
    if state.get("experiment") != protocol()["experiment"] or state.get("controller_identity") != identity():
        raise ValueError("open-resource controller identity changed; use its frozen snapshot")
    bundle = Path(state["bundle"])
    manifest = api.read(bundle / "manifest.json")
    if (manifest["bundle_id"] != state["bundle_id"]
            or manifest["controller_identity"] != state["controller_identity"]
            or api.sha256(bundle / "manifest.json") != state["bundle_manifest_sha256"]):
        raise ValueError("open-resource bundle identity differs")
    # Analysis/status can skip large ELF payloads, never the imported costs,
    # contexts or other small plans on which their conclusions depend.
    api.verify_inventory(bundle, {n: v for n, v in manifest.get("files", {}).items()
                                  if n.startswith("plans/")})
    if api.read(root / "protocol.json") != protocol():
        raise ValueError("open-resource protocol changed")
    api.verify_inventory(root, state["frozen_files"])
    for name, expected in state.get("pending_publications", {}).items():
        path = api.relative_file(root, name)
        if path.exists() and api.digest(api.read(path)) != expected:
            raise ValueError("pending publication content differs: " + name)
    if state.get("phase") not in PHASES or state.get("status") not in set(PHASES) | STOPPED | {"INTERRUPTED"}:
        raise ValueError("invalid open-resource campaign phase/status")
    boot = state.get("boot_id")
    if state.get("active_boot") not in (None, boot):
        raise ValueError("active boot identity differs")
    expected = [boot] if state["status"] == "COMPLETE" else []
    if state.get("completed_boots") != expected or (expected and not boot):
        raise ValueError("invalid one-boot accounting")
    if state["status"] == "COMPLETE":
        if state["phase"] != "COMPLETE" or state["active_boot"] is not None:
            raise ValueError("incomplete campaign marked complete")
        from .domain import requirements
        programs = api.read(bundle / "plans/programs.json")
        required = requirements(programs)
        records = load_costs(root, state, replay=False)
        if set(records) != set(required):
            raise ValueError("complete campaign lacks component coverage")
        if set(load_traffic(root, state, replay=False)) != set(traffic_requirements(programs)):
            raise ValueError("complete campaign lacks traffic coverage")
        for name in ("costs.json", "ddr.json", "rankings.json", "search_pool.json", "cpu_weight_sensitivity.json",
                     "screening.json", "screening_order.json"):
            if name not in state["frozen_files"]:
                raise ValueError("complete campaign lacks frozen artifact: " + name)
        from .search import select_pool
        ranked = api.read(root / "rankings.json")
        expected_k = protocol().get("top_k", 20)
        if (len(ranked.get("pure_static20", [])) != expected_k
                or len(ranked.get("diverse20", [])) != expected_k
                or set(ranked.get("methods", {})) != {"single_core", "prefix", "enhanced"}):
            raise ValueError("complete campaign ranking coverage differs")
        references_path = bundle / "plans/historical_references.json"
        references = api.read(references_path) if references_path.exists() else []
        if api.read(root / "search_pool.json") != select_pool(ranked, references=references):
            raise ValueError("complete campaign pool differs from frozen rankings")
        validated_screening(root, state)
    if validate_payload:
        _validate_bundle(bundle)
    return state


def _linked(root, path, receipt):
    api = _common()
    return {**receipt["summary"], "_receipt": {"path": str(path.relative_to(root)), "sha256": api.sha256(path)}}


class Recorder:
    """Accept a raw task once, recovering a receipt after controller interruption."""

    def __init__(self, root, state, transport, remote):
        self.root, self.state = Path(root).resolve(), state
        self.transport, self.remote, self.boot = transport, remote, state["boot_id"]
        self.evidence = Evidence(self.root, state["bundle"])

    def _acknowledge(self, document):
        if self.transport is None or not hasattr(self.transport, "acknowledge"):
            return
        api = _common(self.state["bundle"])
        path = self.root / "cleanup_pending" / self.boot / (document["task_id"] + ".json")
        try:
            self.transport.acknowledge(document, self.root, self.boot)
        except (OSError, RuntimeError) as error:
            api.write(path, {"document": document, "boot_id": self.boot}, replace=True)
            api.event(self.root, "cleanup_pending", task_id=document["task_id"], error=str(error))
        else:
            path.unlink(missing_ok=True)

    def saved(self, name, document):
        api = _common(self.state["bundle"])
        path = api.relative_file(self.root, "records/" + self.boot + "/" + name + ".json")
        receipt = self.root / "accepted" / self.boot / (document["task_id"] + ".json")
        if path.exists():
            record = api.read(path)
            if record["document"] != document or record["boot_id"] != self.boot:
                raise ValueError("record belongs to another task or boot")
            value = record["summary"]
            self.evidence.summary(value, document, self.boot)
        else:
            if receipt.exists():
                value = _linked(self.root, receipt, api.read(receipt))
                self.evidence.summary(value, document, self.boot)
            else:
                if self.transport is None:
                    raise ValueError("measurement missing: " + name)
                if self.transport.boot_id() != self.boot:
                    raise ValueError("board boot changed before measurement")
                print("[九种资源上下文] " + name, flush=True)
                api.event(self.root, "session_start", task_id=document["task_id"], task_kind=document["kind"], boot_id=self.boot)
                raw = Path(self.transport.measure(document, self.state["bundle"], self.remote,
                                                  self.root, self.boot)).resolve()
                if (self.transport.boot_id() != self.boot
                        or not raw.is_relative_to(self.root / "raw" / self.boot / document["task_id"])):
                    raise ValueError("result belongs to another boot or raw directory")
                summary = validate_summary(raw, document, self.state["bundle"])
                api.sync_tree(raw)
                files = {str(raw.relative_to(self.root) / n): item for n, item in api.inventory(raw).items()}
                api.write(receipt, {"task": document, "boot_id": self.boot, "summary": summary, "files": files})
                value = _linked(self.root, receipt, api.read(receipt))
                self.evidence.summary(value, document, self.boot)
                api.event(self.root, "session_accepted", task_id=document["task_id"], boot_id=self.boot)
            api.write(path, {"document": document, "boot_id": self.boot, "summary": value})
        _pin(self.root, self.state, str(path.relative_to(self.root)), str(receipt.relative_to(self.root)))
        self._acknowledge(document)
        return value


def _task(state, kind, config, **labels):
    return make_task(kind, config, state["bundle"], **labels)


def _imported(state):
    api = _common(state["bundle"])
    path = Path(state["bundle"]) / "plans/imported_components.json"
    result = api.read(path).get("costs", {}) if path.exists() else {}
    additional = _reused(state).get("costs", {})
    if set(result).intersection(additional):
        raise ValueError("reused calibration duplicates a source pilot component")
    return {**result, **additional}


def _reused(state):
    api = _common(state["bundle"])
    path = Path(state["bundle"]) / "plans/reused_calibration.json"
    return api.read(path) if path.is_file() else {}


def load_costs(root, state=None, *, replay=True):
    state = state or validate_state(root, validate_payload=False)
    api = _common(state["bundle"])
    from .domain import component_key
    result = deepcopy(_imported(state))
    evidence = Evidence(root, state["bundle"])
    for key, item in result.items():
        if item.get("kind") != "imported" or component_key(item["configuration"]) != key:
            raise ValueError("imported component provenance or identity differs")
    for path in sorted((Path(root) / "components").glob("*.json")):
        item = api.read(path)
        if item.get("kind") != "measured" or component_key(item["configuration"]) != path.stem or path.stem in result:
            raise ValueError("component provenance, duplicate or identity differs")
        entry, qualification = item["measurement"], item["qualification"]
        doc, qdoc = entry["document"], qualification["document"]
        expected = _task(state, "calibration", item["configuration"], warmup=2, scored=8, tail=0,
                         indices=[0], component_key=path.stem)
        qexpected = _task(state, "qualify_component", item["configuration"], component_key=path.stem)
        if (doc != expected or qdoc != qexpected or entry["boot_id"] != state["boot_id"]
                or qualification["boot_id"] != state["boot_id"]):
            raise ValueError("component measurement protocol or boot differs")
        if replay:
            evidence.summary(entry["summary"], doc, entry["boot_id"])
            evidence.summary(qualification["summary"], qdoc, qualification["boot_id"])
        if item["cost"] != {field: entry["summary"][field] for field in COST_FIELDS}:
            raise ValueError("component costs disagree with raw-backed summary")
        result[path.stem] = item
    return result


def traffic_requirements(programs):
    from .domain import requirements
    result = {}
    for config in requirements(programs).values():
        row = config["stages"][0]
        if row["device"] == "vta" and config["host_mask"] == [0, 1, 2, 3]:
            result[row["program"]] = config
    return dict(sorted(result.items()))


def load_traffic(root, state=None, *, replay=True):
    state = state or validate_state(root, validate_payload=False)
    api = _common(state["bundle"])
    evidence = Evidence(root, state["bundle"])
    # These records were replayed by the original frozen controller during
    # preparation, and the new manifest binds their complete evidence closure.
    # Their old task document must never be passed to the new task validator.
    result = deepcopy(_reused(state).get("traffic", {}))
    for program, item in result.items():
        if item.get("kind") != "imported" or item.get("program") != program:
            raise ValueError("reused traffic provenance or program differs")
        entry = item.get("measurement", {})
        summary = entry.get("summary", {})
        if not entry.get("boot_id") or not entry.get("document"):
            raise ValueError("reused traffic lacks original task or boot")
        if not summary.get("hooks_present") or summary.get("count") != 1:
            raise ValueError("reused traffic counters missing or ambiguous")
        for field in ("load_bytes", "store_bytes"):
            if type(summary.get(field)) not in (int, float) or summary[field] < 0 or item.get(field) != summary[field]:
                raise ValueError("reused traffic counter differs")
    for path in sorted((Path(root) / "traffic").glob("*.json")):
        item = api.read(path)
        program, config, entry = item["program"], item["configuration"], item["measurement"]
        if program in result:
            raise ValueError("new traffic duplicates a reused measurement")
        document = _task(state, "traffic", config, warmup=1, scored=1, tail=0,
                         indices=[0], program=program)
        if path.stem != program or entry["document"] != document or entry["boot_id"] != state["boot_id"]:
            raise ValueError("traffic task identity or boot differs")
        summary = entry["summary"]
        if not summary.get("hooks_present") or summary.get("count", 1) != 1:
            raise ValueError("traffic counters missing or ambiguous")
        for field in ("load_bytes", "store_bytes"):
            if type(summary.get(field)) not in (int, float) or summary[field] < 0 or item.get(field) != summary[field]:
                raise ValueError("traffic counter differs")
        if replay:
            evidence.summary(summary, document, entry["boot_id"])
        result[program] = item
    return result


def _calibrate(root, state, recorder, wanted):
    records = load_costs(root, state)
    if not set(records) <= set(wanted):
        raise ValueError("unexpected component in campaign")
    if any(records[k]["configuration"] != wanted[k] for k in records):
        raise ValueError("component identity collision")
    state["imported_components"] = sum(v["kind"] == "imported" for v in records.values())
    state["reused_calibration_components"] = len(_reused(state).get("costs", {}))
    state["source_pilot_components"] = state["imported_components"] - state["reused_calibration_components"]
    state["new_components"] = sum(v["kind"] == "measured" for v in records.values())
    for key, config in sorted(wanted.items()):
        if key in records:
            continue
        qdoc = _task(state, "qualify_component", config, component_key=key)
        qvalue = recorder.saved("components/" + key + "/qualify", qdoc)
        document = _task(state, "calibration", config, warmup=2, scored=8, tail=0,
                         indices=[0], component_key=key)
        value = recorder.saved("components/" + key + "/calibration", document)
        item = {"kind": "measured", "configuration": config,
                "cost": {field: value[field] for field in COST_FIELDS},
                "measurement": {"document": document, "summary": value, "boot_id": recorder.boot},
                "qualification": {"document": qdoc, "summary": qvalue, "boot_id": recorder.boot}}
        _publish(root, state, "components/" + key + ".json", item)
        records[key] = item
        state["completed_components"] = len(records)
        state["new_components"] = sum(v["kind"] == "measured" for v in records.values())
        _save(root, state)
    state["completed_components"] = len(records)
    _save(root, state)
    return records


def _traffic(root, state, recorder, wanted):
    records = load_traffic(root, state)
    if not set(records) <= set(wanted):
        raise ValueError("unexpected VTA traffic program")
    state["imported_traffic"] = sum(v.get("kind") == "imported" for v in records.values())
    state["new_traffic"] = len(records) - state["imported_traffic"]
    state["completed_traffic"] = len(records)
    _save(root, state)
    for program, config in sorted(wanted.items()):
        if program in records:
            if records[program]["configuration"] != config:
                raise ValueError("traffic configuration differs")
            continue
        document = _task(state, "traffic", config, warmup=1, scored=1, tail=0,
                         indices=[0], program=program)
        value = recorder.saved("traffic/" + program, document)
        if not value.get("hooks_present") or value.get("count", 1) != 1:
            raise ValueError("VTA profiler hooks/counters unavailable")
        item = {"program": program, "configuration": config,
                "load_bytes": value["load_bytes"], "store_bytes": value["store_bytes"],
                "measurement": {"document": document, "summary": value, "boot_id": recorder.boot}}
        _publish(root, state, "traffic/" + program + ".json", item)
        records[program] = item
        state["completed_traffic"] = len(records)
        state["new_traffic"] = len(records) - state["imported_traffic"]
        _save(root, state)
    return records


def rank(root, state=None):
    """Freeze component-only rankings; this function never accesses SSH or FPS."""
    root = Path(root).resolve()
    state = state or validate_state(root)
    api = _common(state["bundle"])
    from .domain import requirements
    from .search import rank_all, select_pool
    programs = api.read(Path(state["bundle"]) / "plans/programs.json")
    records = load_costs(root, state)
    if set(records) != set(requirements(programs)):
        raise ValueError("cannot rank before complete component coverage")
    if not (root / "ddr.json").exists():
        raise ValueError("cannot rank without the explicit DDR model")
    ddr = api.read(root / "ddr.json")
    if (root / "rankings.json").exists():
        if "rankings.json" not in state["frozen_files"]:
            pending = state.get("pending_publications", {}).get("rankings.json")
            if pending is None or api.digest(api.read(root / "rankings.json")) != pending:
                raise ValueError("unfrozen ranking file exists without a matching durable publication")
            _pin(root, state, "rankings.json")
            state["pending_publications"].pop("rankings.json", None)
            _save(root, state)
        rankings = api.read(root / "rankings.json")
    else:
        rankings = rank_all(programs, records, ddr=ddr, output_dir=root / "search",
                            unit_count=protocol().get("unit_count", 21),
                            max_vta_islands=protocol().get("maximum_vta_islands", 3))
        if len(rankings.get('pure_static20', [])) == 20:
            from .selection import select
            diversified, policy = select(rankings['pure_static20'],
                                         api.read(root / 'search/stratified_pool.json'))
            rankings['diverse20'] = diversified
            rankings['selection_protocol'] = policy
        _publish(root, state, "rankings.json", rankings)
    references_path = Path(state["bundle"]) / "plans/historical_references.json"
    references = api.read(references_path) if references_path.exists() else []
    pool = _publish(root, state, "search_pool.json", select_pool(rankings, references=references))
    if len(pool) > protocol().get("max_search_pool", 43):
        raise ValueError("frozen pipeline pool exceeds the agreed budget")
    order = sorted(pool)
    random.Random(protocol().get("seed", 20261107)).shuffle(order)
    _publish(root, state, "screening_order.json", order)
    from .sensitivity import evaluate
    _publish(root, state, "cpu_weight_sensitivity.json", evaluate(pool, records, ddr))
    return rankings, pool, order


def run_current(root, transport):
    root = Path(root).resolve()
    api = _common()
    from cpu_vta_partitioning.transport import TransportError
    from .domain import requirements
    with api.locked(root):
        state = validate_state(root)
        if state["status"] == "COMPLETE":
            return state
        if state["status"] in STOPPED:
            raise RuntimeError("campaign is stopped; do not silently resume " + state["status"])
        try:
            boot = transport.boot_id()
            if state["boot_id"] is not None and state["boot_id"] != boot:
                state.update(status="STOPPED_BOOT_CHANGED", last_error="one-boot campaign cannot mix another board boot")
                _save(root, state)
                raise RuntimeError(state["last_error"])
            state["boot_id"] = state["active_boot"] = boot
            _set_phase(root, state, "QUALIFY")
            transport.wait_idle(boot)
            transport.mount_storage()
            remote = transport.deploy(state["bundle"], root)
            transport.verify_payload(remote, boot)
            transport.qualify_environment(remote, boot, root, state["bundle"])
            environment = "environment-" + boot + ".json"
            if (root / environment).exists():
                _pin(root, state, environment)
            recorder = Recorder(root, state, transport, remote)
            for path in sorted((root / "cleanup_pending" / boot).glob("*.json")):
                pending = api.read(path)
                receipt = root / "accepted" / boot / (pending["document"]["task_id"] + ".json")
                recorder.evidence.summary(_linked(root, receipt, api.read(receipt)), pending["document"], boot)
                recorder._acknowledge(pending["document"])
            programs = api.read(Path(state["bundle"]) / "plans/programs.json")
            wanted = _publish(root, state, "component_requirements.json", requirements(programs))
            _set_phase(root, state, "COMPONENTS")
            records = _calibrate(root, state, recorder, wanted)
            _publish(root, state, "costs.json", {"source": "verified_imports_and_missing_components",
                     "boot_id": boot, "costs": records, "imported_components": state["imported_components"],
                     "new_components": state["new_components"],
                     "cpu_core_permutation": "explicit_homogeneous_core_approximation_not_each_mask_measured"})
            _set_phase(root, state, "TRAFFIC")
            traffic = _traffic(root, state, recorder, traffic_requirements(programs))
            from .ddr import build_ddr
            _publish(root, state, "ddr.json", build_ddr(programs, traffic, state["bundle"]))
            _set_phase(root, state, "HOST_RANK")
            _, pool, order = rank(root, state)
            _set_phase(root, state, "PIPELINE")
            measurements = {}
            for number, key in enumerate(order, 1):
                config = pool[key]
                qdoc = _task(state, "qualify_pipeline", config, candidate_id=key)
                recorder.saved("pipeline/" + key + "/qualify", qdoc)
                doc = _task(state, "performance", config, warmup=8, scored=16, tail=4,
                            indices=[0], candidate_id=key)
                value = recorder.saved("pipeline/" + key + "/performance", doc)
                measurements[key] = {"document": doc, "summary": value, "boot_id": boot}
                state["completed_pipeline_configurations"] = number
                _save(root, state)
            _publish(root, state, "screening.json", {"boot_id": boot, "rounds": 1,
                     "order": order, "measurements": measurements})
            if transport.boot_id() != boot:
                raise ValueError("board boot changed before final acceptance")
            transport.verify_payload(remote, boot)
            transport.idle_command("sync", boot=boot, timeout=120)
            _set_phase(root, state, "ANALYZE")
            from .analysis import analyze
            analyze(root)
            state["active_boot"] = None
            state["completed_boots"] = [boot]
            _set_phase(root, state, "COMPLETE")
            api.sync_tree(root)
            return state
        except TransportError as error:
            state.update(status="INTERRUPTED", last_error=str(error))
            _save(root, state)
            api.event(root, "transport_interrupted", error=str(error), boot_id=state.get("boot_id"))
            raise
        except Exception as error:
            if state["status"] not in STOPPED:
                state.update(status="STOPPED_INVALID_RESULT", last_error=str(error))
                _save(root, state)
            api.event(root, "open_resource_stopped", error=str(error), status=state["status"])
            raise


def status(root):
    state = validate_state(root, validate_payload=False)
    return {key: state.get(key) for key in ("status", "phase", "boot_id", "active_boot",
            "completed_components", "imported_components", "source_pilot_components", "reused_calibration_components",
            "new_components", "completed_traffic", "imported_traffic", "new_traffic",
            "completed_pipeline_configurations", "last_error")}


def validated_screening(root, state=None):
    state = state or validate_state(root, validate_payload=False)
    api = _common(state["bundle"])
    path = Path(root) / "screening.json"
    if not path.exists():
        return None
    screen, pool = api.read(path), api.read(Path(root) / "search_pool.json")
    order = api.read(Path(root) / "screening_order.json")
    if (screen["boot_id"] != state["boot_id"] or screen["rounds"] != 1
            or screen["order"] != order or set(screen["measurements"]) != set(pool)
            or len(order) != len(set(order)) or set(order) != set(pool)):
        raise ValueError("screening coverage, boot or order differs")
    recorder = Recorder(root, state, None, None)
    for key in order:
        config = pool[key]
        qdoc = _task(state, "qualify_pipeline", config, candidate_id=key)
        qpath = Path(root) / "records" / state["boot_id"] / "pipeline" / key / "qualify.json"
        qvalue = api.read(qpath)
        if qvalue["document"] != qdoc or qvalue["boot_id"] != state["boot_id"]:
            raise ValueError("pipeline qualification differs")
        recorder.evidence.summary(qvalue["summary"], qdoc, state["boot_id"])
        doc = _task(state, "performance", config, warmup=8, scored=16, tail=4,
                    indices=[0], candidate_id=key)
        value = screen["measurements"][key]
        if value["document"] != doc or value["boot_id"] != state["boot_id"]:
            raise ValueError("performance task differs")
        recorder.evidence.summary(value["summary"], doc, value["boot_id"])
    return screen
