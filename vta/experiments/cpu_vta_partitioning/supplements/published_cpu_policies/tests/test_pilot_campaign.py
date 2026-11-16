"""Recovery tests exercise durable receipts and the single-boot lifecycle."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from cpu_policies import pilot_campaign as campaign
from cpu_policies import pilot_analysis as analysis


@pytest.fixture
def harness(tmp_path, monkeypatch):
    api = campaign._common()
    monkeypatch.setattr(campaign, "_common", lambda bundle=None: api)
    monkeypatch.setattr(campaign, "identity", lambda: "test-controller")
    actual_protocol = campaign.protocol()
    monkeypatch.setattr(campaign, "protocol", lambda: {**actual_protocol, "component_count": 1, "top_k": 1})
    bundle = tmp_path / "bundle"
    api.write(bundle / "manifest.json", {"bundle_id": "test-bundle", "controller_identity": "test-controller"})
    api.write(bundle / "plans/programs.json", {})
    monkeypatch.setattr(campaign, "_validate_bundle", lambda value: api.read(Path(value) / "manifest.json"))

    from cpu_strategy.policy.domain import component, component_key, cut_candidate, stage
    component_config = component(stage("cpu", 0, 3, 1, (0,)), "pinned")
    component_id = component_key(component_config)
    pipeline = cut_candidate("current", (2, 2), ((0, 1), (2, 3)), "pinned", "resource_group")

    def make_task(kind, config, bundle, *, warmup=0, scored=1, tail=0, indices=None, **labels):
        value = {"kind": kind, "configuration": deepcopy(config), "stages": deepcopy(config["stages"]),
                 "warmup": warmup, "scored": scored, "tail": tail, "input_indices": indices or [0], **labels}
        return {**value, "task_id": api.digest(value)[:24]}

    monkeypatch.setattr(campaign, "make_task", make_task)

    class Evidence:
        def __init__(self, root, bundle):
            self.root = Path(root)

        def summary(self, value, document, boot):
            ref = value["_receipt"]
            path = self.root / ref["path"]
            if api.sha256(path) != ref["sha256"]:
                raise ValueError("receipt hash differs")
            receipt = api.read(path)
            if receipt["task"] != document or receipt["boot_id"] != boot:
                raise ValueError("wrong receipt task or boot")
            api.verify_inventory(self.root, receipt["files"])
            if receipt["summary"] != {k: v for k, v in value.items() if k != "_receipt"}:
                raise ValueError("summary differs")
            return receipt

    monkeypatch.setattr(campaign, "Evidence", Evidence)
    monkeypatch.setattr(campaign, "validate_summary", lambda raw, doc, bundle: api.read(Path(raw) / "summary.json"))
    from cpu_vta_partitioning.transport import TransportError

    class Transport:
        def __init__(self):
            self.boot = "boot-one"
            self.calls, self.acks = [], []
            self.failure = None
            self.verified = 0

        def boot_id(self):
            return self.boot

        def wait_idle(self, boot):
            assert boot == self.boot

        def mount_storage(self):
            pass

        def deploy(self, bundle, root):
            return "/remote/bundle"

        def verify_payload(self, remote, boot):
            assert boot == self.boot
            self.verified += 1

        def qualify_environment(self, remote, boot, root, bundle):
            path = Path(root) / ("environment-" + boot + ".json")
            if not path.exists():
                api.write(path, {"boot": boot})

        def idle_command(self, command, **kwargs):
            assert command == "sync"

        def acknowledge(self, document, root, boot):
            self.acks.append(document["task_id"])

        def measure(self, doc, bundle, remote, root, boot):
            self.calls.append(doc)
            if self.failure is not None:
                error, self.failure = self.failure, None
                raise error
            raw = Path(root) / "raw" / boot / doc["task_id"] / "collected"
            raw.mkdir(parents=True, exist_ok=True)
            if doc["kind"].startswith("qualify"):
                summary = {"correct": True}
            elif doc["kind"] == "calibration":
                summary = {field: 1.0 for field in campaign.COST_FIELDS}
            else:
                summary = {"ii_ms": 10.0, "p95_gap_ms": 30.0, "kept_timestamps": 16, "gaps": 15,
                           "process_core_ms_per_completion": 20., "average_cpu_cores": 2.}
            api.write(raw / "summary.json", summary)
            return raw

    from cpu_policies import pilot_search
    monkeypatch.setattr(pilot_search, "requirements", lambda programs: {component_id: component_config})
    ranked = {"methods": {method: {"rankings": [{**pipeline, "score_ms": 9.0}]}
                          for method in ("single_core", "prefix", "enhanced")}}
    monkeypatch.setattr(pilot_search, "rank_all", lambda programs, records: deepcopy(ranked))
    monkeypatch.setattr(pilot_search, "select_pool", lambda rankings: {pipeline["id"]: pipeline})
    root = tmp_path / "run"
    state = campaign.create(bundle, root, {"host": "test-only"})
    return {"api": api, "root": root, "state": state, "bundle": bundle, "transport": Transport(),
            "component": component_config, "pipeline": pipeline, "make_task": make_task,
            "TransportError": TransportError, "rankings": ranked}


def test_single_boot_runs_once_without_anchors_or_thresholds(harness):
    h = harness
    state = campaign.run_current(h["root"], h["transport"])
    assert state["status"] == "COMPLETE"
    assert state["completed_boots"] == ["boot-one"]
    assert [doc["kind"] for doc in h["transport"].calls] == [
        "qualify_component", "calibration", "qualify_pipeline", "performance"]
    cal, perf = h["transport"].calls[1], h["transport"].calls[3]
    assert (cal["warmup"], cal["scored"], cal["tail"], cal["input_indices"]) == (2, 8, 0, [0])
    assert (perf["warmup"], perf["scored"], perf["tail"], perf["input_indices"]) == (8, 16, 4, [0])
    # The p95 gap is 3x the average; it is accepted, not a performance gate.
    assert h["api"].read(h["root"] / "screening.json")["measurements"][h["pipeline"]["id"]]["summary"]["p95_gap_ms"] == 30.
    campaign.run_current(h["root"], h["transport"])
    assert len(h["transport"].calls) == 4


def test_disconnect_resumes_same_boot_and_reuses_accepted_window(harness):
    h = harness
    h["state"]["boot_id"] = h["state"]["active_boot"] = "boot-one"
    campaign._save(h["root"], h["state"])
    recorder = campaign.Recorder(h["root"], h["state"], h["transport"], "/remote")
    document = h["make_task"]("qualify_component", h["component"], h["bundle"])
    value = recorder.saved("components/accepted", document)
    record = h["root"] / "records/boot-one/components/accepted.json"
    record.unlink()  # Controller stopped after receipt acceptance, before record recovery.
    h["state"]["frozen_files"].pop(str(record.relative_to(h["root"])))
    campaign._save(h["root"], h["state"])
    again = recorder.saved("components/accepted", document)
    assert again == value
    assert len(h["transport"].calls) == 1


def test_transport_interruption_is_resumable_without_new_boot(harness):
    h = harness
    h["transport"].failure = h["TransportError"]("SSH disconnected")
    with pytest.raises(h["TransportError"], match="disconnected"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "INTERRUPTED"
    assert campaign.run_current(h["root"], h["transport"])["status"] == "COMPLETE"


def test_wrong_boot_stops_and_does_not_measure(harness):
    h = harness
    h["state"]["boot_id"] = h["state"]["active_boot"] = "boot-original"
    campaign._save(h["root"], h["state"])
    with pytest.raises(RuntimeError, match="cannot mix"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "STOPPED_BOOT_CHANGED"
    assert h["transport"].calls == []
    with pytest.raises(RuntimeError, match="silently resume"):
        campaign.run_current(h["root"], h["transport"])


def test_invalid_output_is_not_retried_or_counted(harness):
    h = harness
    h["transport"].failure = ValueError("incorrect full output")
    with pytest.raises(ValueError, match="incorrect full output"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "STOPPED_INVALID_RESULT"
    with pytest.raises(RuntimeError, match="silently resume"):
        campaign.run_current(h["root"], h["transport"])
    assert len(h["transport"].calls) == 1


def test_hash_tampering_is_rejected_on_resume(harness):
    h = harness
    campaign.run_current(h["root"], h["transport"])
    (h["root"] / "rankings.json").write_text("{}")
    with pytest.raises(ValueError, match="changed artifact"):
        campaign.run_current(h["root"], h["transport"])


def test_duplicate_record_wrong_document_is_rejected(harness):
    h = harness
    h["state"]["boot_id"] = h["state"]["active_boot"] = "boot-one"
    campaign._save(h["root"], h["state"])
    recorder = campaign.Recorder(h["root"], h["state"], h["transport"], "/remote")
    document = h["make_task"]("qualify_component", h["component"], h["bundle"])
    recorder.saved("reused_name", document)
    changed = h["make_task"]("calibration", h["component"], h["bundle"], warmup=2, scored=8)
    with pytest.raises(ValueError, match="another task"):
        recorder.saved("reused_name", changed)


def test_analysis_same_budget_and_zero_selection_difference(harness, monkeypatch):
    h = harness
    # Analysis reports a partial synthetic run without pretending it is complete.
    state = {**h["state"], "boot_id": "boot-one", "status": "SCREENING"}
    monkeypatch.setattr(analysis, "validate_state", lambda root, validate_payload=False: state)
    monkeypatch.setattr(analysis, "load_costs", lambda root, state: {})
    monkeypatch.setattr(analysis, "_common", lambda bundle=None: h["api"])
    monkeypatch.setattr(analysis, "_dispersion", lambda root, item: {"gap_min_ms": 5., "gap_max_ms": 20.})
    config = h["pipeline"]
    summary = {"ii_ms": 10., "process_core_ms_per_completion": 20., "average_cpu_cores": 2.,
               "p95_gap_ms": 20., "kept_timestamps": 16, "gaps": 15}
    monkeypatch.setattr(analysis, "validated_screening", lambda root, state: {
        "measurements": {config["id"]: {"summary": summary}}})
    h["api"].write(h["root"] / "rankings.json", h["rankings"])
    h["api"].write(h["root"] / "search_pool.json", {config["id"]: config})
    report = analysis.analyze(h["root"])
    assert report["status"] == "INCOMPLETE"
    assert report["cpu_time_scope"].endswith("15_intervals")
    assert report["performance_rejection_gates"] == []
    assert all(r["throughput_gain"] == 0. and r["same_configuration"] for r in report["comparisons_same_budget"].values())
    assert report["enhanced_best"]["fps"] == 100.
    assert (h["root"] / "analysis/top20.csv").is_file()


def test_configuration_report_keeps_host_and_control_explicit(harness):
    config = deepcopy(harness["pipeline"])
    config["host_mask"] = config["control_mask"] = [3]
    row = analysis._configuration(config)
    assert row["resource_policy"] == "reserved3"
    assert row["host_cores"] == row["control_cores"] == "3"


def test_complete_flag_cannot_bypass_missing_measurements(harness):
    h = harness
    h["state"].update(status="COMPLETE", phase="COMPLETE", boot_id="boot-one",
                       active_boot=None, completed_boots=["boot-one"])
    campaign._save(h["root"], h["state"])
    with pytest.raises(ValueError, match="component coverage"):
        campaign.validate_state(h["root"])
