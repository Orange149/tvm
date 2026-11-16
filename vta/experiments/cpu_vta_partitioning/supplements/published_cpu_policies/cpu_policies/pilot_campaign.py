"""One-boot, single-input component/DP/Top-20 pilot.

Performance variation is recorded, never used as an acceptance gate.  Numeric
outputs, execution identities and durable raw receipts remain mandatory.
"""
from copy import deepcopy
from pathlib import Path
import random

from .pilot_common import identity, load_legacy, protocol
from .pilot_runtime import Evidence, make_task, validate_summary


COST_FIELDS = ("service_ms", "run_core_ms", "boundary_core_ms", "set_ms", "run_ms", "get_ms",
               "set_core_ms", "get_core_ms")
PHASES = ("CREATED", "CALIBRATING", "RANKING", "SCREENING", "COMPLETE")
STOPPED = {"STOPPED_INVALID_RESULT", "STOPPED_BOOT_CHANGED", "STOPPED_IDENTITY", "PAUSED"}


def _common(bundle=None):
    load_legacy(bundle)
    from cpu_vta_partitioning import common
    return common


def _validate_bundle(bundle):
    from .pilot_prepare import validate_bundle
    return validate_bundle(bundle)


def _save(root, state):
    _common(state["bundle"]).write(Path(root) / "state.json", state, replace=True)


def _pin(root, state, *names):
    api = _common(state["bundle"])
    for name in names:
        path = api.relative_file(root, name)
        current = {"sha256": api.sha256(path), "bytes": path.stat().st_size}
        old = state["frozen_files"].get(name)
        if old is not None and old != current:
            raise ValueError("frozen pilot artifact changed: " + name)
        state["frozen_files"][name] = current
    _save(root, state)


def _publish(root, state, name, value):
    api = _common(state["bundle"])
    path = api.relative_file(root, name)
    if path.exists():
        if api.read(path) != value:
            raise ValueError("frozen pilot artifact disagrees: " + name)
    else:
        api.write(path, value)
    _pin(root, state, name)
    return value


def create(bundle, root, connection):
    """Create a new batch; preflight and measurement may share the same boot."""
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
             "frozen_files": {}, "completed_components": 0, "completed_pipeline_configurations": 0}
    api.write(root / "protocol.json", protocol())
    _pin(root, state, "protocol.json")
    return state


def validate_state(root, *, validate_payload=True):
    root = Path(root).resolve()
    # The bootstrap location is fixed by the read-only package, not by state.
    api = _common()
    state = api.read(root / "state.json")
    if state.get("experiment") != protocol()["experiment"] or state.get("controller_identity") != identity():
        raise ValueError("pilot controller identity changed; use the frozen snapshot")
    manifest = api.read(Path(state["bundle"]) / "manifest.json")
    if (manifest["bundle_id"] != state["bundle_id"]
            or manifest["controller_identity"] != state["controller_identity"]
            or api.sha256(Path(state["bundle"]) / "manifest.json") != state["bundle_manifest_sha256"]):
        raise ValueError("pilot bundle identity differs")
    if api.read(root / "protocol.json") != protocol():
        raise ValueError("pilot protocol changed")
    api.verify_inventory(root, state["frozen_files"])
    if state.get("phase") not in PHASES:
        raise ValueError("invalid pilot phase")
    if state.get("status") not in set(PHASES) | STOPPED | {"INTERRUPTED"}:
        raise ValueError("invalid pilot status")
    boot = state.get("boot_id")
    if state.get("active_boot") not in (None, boot):
        raise ValueError("pilot boot identity differs")
    expected = [boot] if state["status"] == "COMPLETE" else []
    if state.get("completed_boots") != expected or (expected and not boot):
        raise ValueError("invalid one-boot pilot accounting")
    if state["status"] == "COMPLETE" and state["phase"] != "COMPLETE":
        raise ValueError("incomplete pilot marked complete")
    if state["status"] == "COMPLETE":
        from .pilot_search import requirements, select_pool
        programs = api.read(Path(state["bundle"]) / "plans/programs.json")
        required = requirements(programs)
        records = load_costs(root, state, replay=False)
        if len(records) != protocol()["component_count"] or set(records) != set(required):
            raise ValueError("complete pilot lacks full component coverage")
        if any(records[key]["configuration"] != config for key, config in required.items()):
            raise ValueError("complete pilot component configurations differ")
        for name in ("rankings.json", "search_pool.json", "screening.json", "screening_order.json"):
            if name not in state["frozen_files"] or not (root / name).is_file():
                raise ValueError("complete pilot lacks frozen ranking/screening artifacts")
        ranked = api.read(root / "rankings.json")
        if (set(ranked["methods"]) != set(protocol()["methods"])
                or any(len(doc["rankings"]) != protocol()["top_k"] for doc in ranked["methods"].values())
                or api.read(root / "search_pool.json") != select_pool(ranked)):
            raise ValueError("complete pilot ranking/pool coverage differs")
        validated_screening(root, state)
    if validate_payload:
        _validate_bundle(state["bundle"])
    return state


def _linked(root, path, receipt):
    api = _common()
    return {**receipt["summary"], "_receipt": {"path": str(path.relative_to(root)), "sha256": api.sha256(path)}}


class Recorder:
    """Exactly-once acceptance even if collection or controller writing stops."""

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
                raise ValueError("pilot record belongs to another task or boot")
            value = record["summary"]
            self.evidence.summary(value, document, self.boot)
        else:
            if receipt.exists():
                value = _linked(self.root, receipt, api.read(receipt))
                self.evidence.summary(value, document, self.boot)
            else:
                if self.transport is None:
                    raise ValueError("pilot measurement missing: " + name)
                if self.transport.boot_id() != self.boot:
                    raise ValueError("board boot changed before pilot measurement")
                print("[单次试跑] " + name, flush=True)
                api.event(self.root, "session_start", task_id=document["task_id"], task_kind=document["kind"], boot_id=self.boot)
                raw = Path(self.transport.measure(document, self.state["bundle"], self.remote,
                                                  self.root, self.boot)).resolve()
                if (self.transport.boot_id() != self.boot
                        or not raw.is_relative_to(self.root / "raw" / self.boot / document["task_id"])):
                    raise ValueError("pilot result belongs to another boot or raw directory")
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


def load_costs(root, state=None, *, replay=True):
    state = state or validate_state(root, validate_payload=False)
    api = _common(state["bundle"])
    from cpu_strategy.policy.domain import component_key
    result = {}
    evidence = Evidence(root, state["bundle"])
    for path in sorted((Path(root) / "components").glob("*.json")):
        item = api.read(path)
        if item.get("kind") != "measured" or component_key(item["configuration"]) != path.stem:
            raise ValueError("pilot component provenance or identity differs")
        entry, qualification = item["measurement"], item["qualification"]
        doc = entry["document"]
        if (doc["configuration"] != item["configuration"] or doc["kind"] != "calibration"
                or doc["input_indices"] != [0] or (doc["warmup"], doc["scored"], doc["tail"]) != (2, 8, 0)
                or entry["boot_id"] != state["boot_id"]):
            raise ValueError("pilot component measurement protocol differs")
        qdoc = qualification["document"]
        if (qdoc["kind"] != "qualify_component" or qdoc["configuration"] != item["configuration"]
                or qdoc["input_indices"] != [0] or (qdoc["warmup"], qdoc["scored"], qdoc["tail"]) != (0, 1, 0)
                or qualification["boot_id"] != state["boot_id"]):
            raise ValueError("pilot component qualification differs")
        if replay:
            evidence.summary(entry["summary"], doc, entry["boot_id"])
            evidence.summary(qualification["summary"], qdoc, qualification["boot_id"])
        if item["cost"] != {field: entry["summary"][field] for field in COST_FIELDS}:
            raise ValueError("pilot component costs disagree with raw-backed summary")
        result[path.stem] = item
    return result


def _calibrate(root, state, recorder, wanted):
    records = load_costs(root, state)
    if not set(records) <= set(wanted):
        raise ValueError("unexpected component in pilot campaign")
    for number, (key, config) in enumerate(sorted(wanted.items()), 1):
        if key in records:
            if records[key]["configuration"] != config:
                raise ValueError("pilot component key collision")
            _pin(root, state, "components/" + key + ".json")
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
        state["completed_components"] = number
        _save(root, state)
        records[key] = item
    return records


def _set_phase(root, state, phase):
    state["phase"] = phase
    state["status"] = phase
    state.pop("last_error", None)
    _save(root, state)


def run_current(root, transport):
    """Run or resume the same frozen single-boot batch without performance gates."""
    root = Path(root).resolve()
    api = _common()
    from cpu_vta_partitioning.transport import TransportError
    from .pilot_search import requirements, rank_all, select_pool
    with api.locked(root):
        state = validate_state(root)
        if state["status"] == "COMPLETE":
            return state
        if state["status"] in STOPPED:
            raise RuntimeError("pilot is stopped; do not silently resume " + state["status"])
        try:
            boot = transport.boot_id()
            if state["boot_id"] is not None and state["boot_id"] != boot:
                state["status"] = "STOPPED_BOOT_CHANGED"
                state["last_error"] = "one-boot pilot cannot mix another board boot"
                _save(root, state)
                raise RuntimeError(state["last_error"])
            state["boot_id"] = state["active_boot"] = boot
            _save(root, state)
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
            wanted = requirements(programs)
            _publish(root, state, "component_requirements.json", wanted)
            _set_phase(root, state, "CALIBRATING")
            records = _calibrate(root, state, recorder, wanted)
            _publish(root, state, "costs.json", {"source": "fresh_single_boot_single_input_short_window",
                     "boot_id": boot, "costs": records,
                     "cpu_core_permutation": "explicit_homogeneous_core_approximation_not_each_mask_measured"})
            _set_phase(root, state, "RANKING")
            ranking_path = root / "rankings.json"
            if ranking_path.exists() and "rankings.json" in state["frozen_files"]:
                rankings = api.read(ranking_path)
            else:
                print("[单次试跑] 两种资源方案：完整DP及独立枚举Top-20核验", flush=True)
                rankings = rank_all(programs, records)
                _publish(root, state, "rankings.json", rankings)
            pool = _publish(root, state, "search_pool.json", select_pool(rankings))
            order = sorted(pool)
            random.Random(protocol().get("seed", 20261107)).shuffle(order)
            _publish(root, state, "screening_order.json", order)
            _set_phase(root, state, "SCREENING")
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
                raise ValueError("board boot changed before final pilot acceptance")
            transport.verify_payload(remote, boot)
            transport.idle_command("sync", boot=boot, timeout=120)
            state["active_boot"] = None
            state["completed_boots"] = [boot]
            _set_phase(root, state, "COMPLETE")
            api.sync_tree(root)
            return state
        except TransportError as error:
            state["status"] = "INTERRUPTED"
            state["last_error"] = str(error)
            _save(root, state)
            api.event(root, "transport_interrupted", error=str(error), boot_id=state.get("boot_id"))
            raise
        except Exception as error:
            if state["status"] not in STOPPED:
                state["status"] = "STOPPED_INVALID_RESULT"
                state["last_error"] = str(error)
                _save(root, state)
            api.event(root, "pilot_stopped", error=str(error), status=state["status"])
            raise


def status(root):
    state = validate_state(root, validate_payload=False)
    return {key: state.get(key) for key in ("status", "phase", "boot_id", "active_boot",
            "completed_components", "completed_pipeline_configurations", "last_error")}


def validated_screening(root, state=None):
    """Replay completed windows without rerunning measurements or ranking."""
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
        raise ValueError("pilot screening coverage/boot/order differs")
    recorder = Recorder(root, state, None, None)
    for key in order:
        config = pool[key]
        qdoc = _task(state, "qualify_pipeline", config, candidate_id=key)
        qpath = Path(root) / "records" / state["boot_id"] / "pipeline" / key / "qualify.json"
        qvalue = api.read(qpath)
        if qvalue["document"] != qdoc or qvalue["boot_id"] != state["boot_id"]:
            raise ValueError("pilot pipeline qualification differs")
        recorder.evidence.summary(qvalue["summary"], qdoc, state["boot_id"])
        doc = _task(state, "performance", config, warmup=8, scored=16, tail=4,
                    indices=[0], candidate_id=key)
        value = screen["measurements"][key]
        if value["document"] != doc or value["boot_id"] != state["boot_id"]:
            raise ValueError("pilot performance task differs")
        recorder.evidence.summary(value["summary"], doc, value["boot_id"])
    return screen
