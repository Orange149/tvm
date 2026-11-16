"""Recovery tests use raw-bound receipts, not fake performance thresholds."""
from copy import deepcopy
from pathlib import Path

import pytest

from cpu_policies.open_resources import campaign, analysis
from cpu_policies.open_resources import domain, search, ddr, sensitivity

ANALYZE = analysis.analyze


@pytest.fixture
def harness(tmp_path, monkeypatch):
    api = campaign._common()
    monkeypatch.setattr(campaign, "_common", lambda bundle=None: api)
    monkeypatch.setattr(analysis, "_common", lambda bundle=None: api)
    monkeypatch.setattr(campaign, "identity", lambda: "test-controller")
    settings = campaign.protocol()
    monkeypatch.setattr(campaign, "protocol", lambda: {**settings, "component_count": 2, "top_k": 1})
    bundle = tmp_path / "bundle"
    api.write(bundle / "manifest.json", {"bundle_id": "test-bundle", "controller_identity": "test-controller"})
    api.write(bundle / "plans/programs.json", {})
    monkeypatch.setattr(campaign, "_validate_bundle", lambda p: api.read(Path(p) / "manifest.json"))
    cpu = domain.component(domain.stage("cpu", 0, 3, 1, [0]))
    vta = domain.component(domain.stage("vta", 3, 18, 1, [0, 1, 2, 3]))
    ckey, vkey = domain.component_key(cpu), domain.component_key(vta)
    costs = {field: 1. for field in campaign.COST_FIELDS}
    imported = {ckey: {"kind": "imported", "configuration": cpu, "cost": costs,
                       "provenance": {"source_boot_id": "historical-boot", "source_task_id": "historical-task"}}}
    api.write(bundle / "plans/imported_components.json", {"costs": imported})
    monkeypatch.setattr(domain, "requirements", lambda programs: {ckey: cpu, vkey: vta})
    pipeline = domain.candidate([domain.stage("cpu", 0, 3, 2, [0, 1]),
                                 domain.stage("vta", 3, 18, 1, [0, 1, 2, 3]),
                                 domain.stage("cpu", 18, 21, 2, [2, 3])], gate="resource_group")

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
            self.boot = "new-boot"
            self.calls, self.acks = [], []
            self.failure = None
            self.verified = 0

        def boot_id(self): return self.boot
        def wait_idle(self, boot): assert boot == self.boot
        def mount_storage(self): pass
        def deploy(self, bundle, root): return "/remote/bundle"
        def verify_payload(self, remote, boot):
            assert boot == self.boot
            self.verified += 1
        def qualify_environment(self, remote, boot, root, bundle):
            p = Path(root) / ("environment-" + boot + ".json")
            if not p.exists(): api.write(p, {"boot": boot})
        def idle_command(self, command, **kwargs): assert command == "sync"
        def acknowledge(self, document, root, boot): self.acks.append(document["task_id"])
        def measure(self, doc, bundle, remote, root, boot):
            self.calls.append(doc)
            if self.failure is not None:
                error, self.failure = self.failure, None
                raise error
            raw = Path(root) / "raw" / boot / doc["task_id"] / "collected"
            raw.mkdir(parents=True, exist_ok=True)
            if doc["kind"].startswith("qualify"):
                value = {"correct": True}
            elif doc["kind"] == "calibration":
                value = dict(costs)
            elif doc["kind"] == "traffic":
                value = {"load_bytes": 32, "store_bytes": 16, "hooks_present": True, "count": 1}
            else:
                value = {"ii_ms": 10., "p95_gap_ms": 30., "kept_timestamps": 16, "gaps": 15,
                         "process_core_ms_per_completion": 20., "average_cpu_cores": 2.}
            api.write(raw / "summary.json", value)
            return raw

    row = {**pipeline, "score_ms": 9., "static_rank": 1}
    ranked = {"methods": {m: {"rankings": [row]} for m in ("single_core", "prefix", "enhanced")},
              "pure_static20": [row], "diverse20": [row], "per_context": {"shared4": {"count": 1}}}
    monkeypatch.setattr(search, "rank_all", lambda *args, **kwargs: deepcopy(ranked))
    monkeypatch.setattr(search, "select_pool", lambda ranked, references=(): {pipeline["id"]: pipeline})
    monkeypatch.setattr(ddr, "build_ddr", lambda *args: {"scope": "test-only"})
    monkeypatch.setattr(sensitivity, "evaluate", lambda pool, records, ddr: {
        "scope": "frozen_measurement_pool_only_not_global_domain_reranking", "uses_pipeline_fps": False,
        "changes_frozen_selection": False, "pool_order_changed": False,
        "maximum_absolute_relative_score_change": 0., "configurations": sorted(pool)})
    from cpu_policies.open_resources import explain
    monkeypatch.setattr(explain, "explain_pool", lambda pool, records, ddr, rankings: {
        "scope": "synthetic_campaign_fixture", "uses_pipeline_fps": False,
        "changes_selection": False, "configuration_count": len(pool)})
    monkeypatch.setattr(analysis, "analyze", lambda root: {"status": "COMPLETE"})
    root = tmp_path / "run"
    state = campaign.create(bundle, root, {"host": "test-only"})
    return {"api": api, "root": root, "state": state, "bundle": bundle, "transport": Transport(),
            "component": vta, "pipeline": pipeline, "make_task": make_task, "rankings": ranked,
            "TransportError": TransportError, "imported": imported, "actual_analysis": ANALYZE}


def test_imports_are_preserved_and_only_missing_components_run(harness):
    h = harness
    state = campaign.run_current(h["root"], h["transport"])
    assert state["status"] == "COMPLETE"
    assert (state["imported_components"], state["new_components"], state["completed_components"]) == (1, 1, 2)
    assert state["completed_traffic"] == state["completed_pipeline_configurations"] == 1
    assert state["completed_boots"] == ["new-boot"]
    assert "cpu_weight_sensitivity.json" in state["frozen_files"]
    assert [r["kind"] for r in h["transport"].calls] == [
        "qualify_component", "calibration", "traffic", "qualify_pipeline", "performance"]
    records = campaign.load_costs(h["root"])
    original = next(v for v in records.values() if v["kind"] == "imported")
    assert original["provenance"]["source_boot_id"] == "historical-boot"
    cal, traffic, perf = (h["transport"].calls[i] for i in (1, 2, 4))
    assert (cal["warmup"], cal["scored"], cal["tail"]) == (2, 8, 0)
    assert (traffic["warmup"], traffic["scored"], traffic["tail"]) == (1, 1, 0)
    assert (perf["warmup"], perf["scored"], perf["tail"]) == (8, 16, 4)
    # A 3x p95 is accepted and described; no noise-triggered extra measurement.
    campaign.run_current(h["root"], h["transport"])
    assert len(h["transport"].calls) == 5


def test_disconnect_resumes_and_does_not_repeat_accepted_receipt(harness):
    h = harness
    h["state"].update(boot_id="new-boot", active_boot="new-boot")
    campaign._save(h["root"], h["state"])
    recorder = campaign.Recorder(h["root"], h["state"], h["transport"], "/remote")
    doc = h["make_task"]("qualify_component", h["component"], h["bundle"])
    first = recorder.saved("before-interruption", doc)
    record = h["root"] / "records/new-boot/before-interruption.json"
    record.unlink()
    h["state"]["frozen_files"].pop(str(record.relative_to(h["root"])))
    campaign._save(h["root"], h["state"])
    assert recorder.saved("before-interruption", doc) == first
    assert len(h["transport"].calls) == 1


def test_transport_interruption_retains_and_resumes_same_boot(harness):
    h = harness
    h["transport"].failure = h["TransportError"]("network disconnected")
    with pytest.raises(h["TransportError"], match="disconnected"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "INTERRUPTED"
    assert campaign.run_current(h["root"], h["transport"])["status"] == "COMPLETE"


def test_wrong_boot_stops_without_new_measurement(harness):
    h = harness
    h["state"].update(boot_id="different-boot", active_boot="different-boot")
    campaign._save(h["root"], h["state"])
    with pytest.raises(RuntimeError, match="cannot mix"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "STOPPED_BOOT_CHANGED"
    assert not h["transport"].calls
    with pytest.raises(RuntimeError, match="silently resume"):
        campaign.run_current(h["root"], h["transport"])


def test_incorrect_output_is_retained_and_not_retried(harness):
    h = harness
    h["transport"].failure = ValueError("incorrect full output")
    with pytest.raises(ValueError, match="incorrect full output"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "STOPPED_INVALID_RESULT"
    assert len(h["transport"].calls) == 1


def test_duplicate_record_and_tampered_hash_rejected(harness):
    h = harness
    h["state"].update(boot_id="new-boot", active_boot="new-boot")
    campaign._save(h["root"], h["state"])
    recorder = campaign.Recorder(h["root"], h["state"], h["transport"], "/remote")
    doc = h["make_task"]("qualify_component", h["component"], h["bundle"])
    recorder.saved("same-name", doc)
    changed = h["make_task"]("calibration", h["component"], h["bundle"], warmup=2, scored=8)
    with pytest.raises(ValueError, match="another task"):
        recorder.saved("same-name", changed)
    receipt = h["root"] / "accepted/new-boot" / (doc["task_id"] + ".json")
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        recorder.saved("same-name", doc)


def test_complete_cannot_bypass_missing_components(harness):
    h = harness
    h["state"].update(status="COMPLETE", phase="COMPLETE", boot_id="new-boot", active_boot=None,
                       completed_boots=["new-boot"])
    campaign._save(h["root"], h["state"])
    with pytest.raises(ValueError, match="component coverage"):
        campaign.validate_state(h["root"])


def test_imported_plan_hash_is_required_even_for_status(harness):
    h = harness
    manifest = h["api"].read(h["bundle"] / "manifest.json")
    path = h["bundle"] / "plans/imported_components.json"
    manifest["files"] = {"plans/imported_components.json": {"sha256": h["api"].sha256(path), "bytes": path.stat().st_size}}
    h["api"].write(h["bundle"] / "manifest.json", manifest, replace=True)
    h["state"]["bundle_manifest_sha256"] = h["api"].sha256(h["bundle"] / "manifest.json")
    campaign._save(h["root"], h["state"])
    path.write_text("{}")
    with pytest.raises(ValueError): campaign.status(h["root"])


def test_missing_profiler_hooks_are_invalid_not_zero(harness, monkeypatch):
    h = harness
    original = h["transport"].measure
    def measure(doc, *args):
        raw = original(doc, *args)
        if doc["kind"] == "traffic":
            h["api"].write(raw / "summary.json", {"load_bytes": 0, "store_bytes": 0,
                          "hooks_present": False, "count": 1}, replace=True)
        return raw
    monkeypatch.setattr(h["transport"], "measure", measure)
    with pytest.raises(ValueError, match="hooks/counters unavailable"):
        campaign.run_current(h["root"], h["transport"])
    assert campaign.status(h["root"])["status"] == "STOPPED_INVALID_RESULT"
    assert not any(d["kind"] == "performance" for d in h["transport"].calls)


def test_source_boot_is_not_counted_as_new_boot(harness):
    h = harness
    campaign.run_current(h["root"], h["transport"])
    state = campaign.validate_state(h["root"])
    assert state["completed_boots"] == ["new-boot"]
    assert "historical-boot" not in state["completed_boots"]


@pytest.mark.parametrize("shared_source_boot", [False, True])
def test_report_unsampled_contexts_and_equal_budget_zero_difference(harness, monkeypatch, shared_source_boot):
    h = harness
    if shared_source_boot:
        h["transport"].boot = "historical-boot"
    campaign.run_current(h["root"], h["transport"])
    monkeypatch.setattr(analysis, "_dispersion", lambda root, item: {"gap_min_ms": 5., "gap_max_ms": 30.})
    report = h["actual_analysis"](h["root"])
    assert report["status"] == "COMPLETE"
    assert report["imported_components"] == report["new_components"] == 1
    assert report["measurement_boot_count"] == 1
    assert report["measurement_boot_shared_with_imports"] is shared_source_boot
    assert report["new_boot_count_relative_to_imports"] == (0 if shared_source_boot else 1)
    assert report["independent_confirmation_boots"] == 0
    assert "independent_new_boots" not in report
    assert report["control_thread_cpu_time_status"] == "not_separately_sampled_not_zero"
    assert report["performance_rejection_gates"] == []
    assert all(row["throughput_gain"] == 0 and row["same_configuration"]
               for row in report["comparisons_same_budget"].values())
    assert len(report["contexts"]) == 9
    unsampled = [r for r in report["contexts"] if not r["measured_count"]]
    assert len(unsampled) == 8
    assert all(r["best_fps"] is None and r["physical_performance_conclusion"] ==
               "not_measured_no_performance_claim" for r in unsampled)
    assert (h["root"] / "analysis/contexts.csv").is_file()


def test_pipeline_requires_explicit_board_execution_flag():
    from cpu_policies.open_resources.cli import main
    with pytest.raises(SystemExit) as error:
        main(["board", "resume", "--run-dir", "/not-accessed"])
    assert error.value.code == 2


def test_preparation_copies_both_raw_receipt_closures_and_rejects_tampering(tmp_path, monkeypatch):
    from cpu_policies.open_resources import prepare
    api = campaign._common()
    monkeypatch.setattr(prepare, "_api", lambda bundle=None: api)
    source, target = tmp_path / "source", tmp_path / "copied"
    api.write(source / "state.json", {"boot_id": "old-boot"})
    api.write(source / "protocol.json", {"indices": [0]})
    item = {"kind": "measured", "configuration": {"arbitrary": "the importer validates semantics"}}
    for kind, role in (("calibration", "measurement"), ("qualify_component", "qualification")):
        doc = {"kind": kind, "task_id": kind}
        name = f"raw/old-boot/{kind}/collected/result.jsonl"
        path = source / name
        path.parent.mkdir(parents=True)
        path.write_text('{"frame_id":0}\n')
        files = {name: {"sha256": api.sha256(path), "bytes": path.stat().st_size}}
        receipt = source / f"accepted/old-boot/{kind}.json"
        api.write(receipt, {"task": doc, "boot_id": "old-boot", "files": files, "summary": {"valid": True}})
        item[role] = {"document": doc, "boot_id": "old-boot", "summary": {
            "valid": True, "_receipt": {"path": str(receipt.relative_to(source)), "sha256": api.sha256(receipt)}}}
    api.write(source / "costs.json", {"costs": {"component": item}})
    result = prepare._copy_provenance(source, target)
    assert result["component_count"] == 1
    assert result["source_boot_ids"] == ["old-boot"]
    assert len([n for n in result["files"] if n.startswith("accepted/")]) == 2
    assert len([n for n in result["files"] if n.startswith("raw/")]) == 2
    api.verify_inventory(target, result["files"])
    (source / "raw/old-boot/calibration/collected/result.jsonl").write_text("tampered")
    with pytest.raises(ValueError):
        prepare._copy_provenance(source, tmp_path / "rejected")


def test_interrupted_rank_publication_recovers_without_research(harness, monkeypatch):
    h = harness
    campaign.run_current(h["root"], h["transport"])
    state = campaign.validate_state(h["root"])
    ranked = h["api"].read(h["root"] / "rankings.json")
    state.update(status="INTERRUPTED", phase="HOST_RANK", active_boot="new-boot", completed_boots=[])
    state["frozen_files"].pop("rankings.json")
    state["pending_publications"] = {"rankings.json": h["api"].digest(ranked)}
    campaign._save(h["root"], state)
    monkeypatch.setattr(search, "rank_all", lambda *a, **k: pytest.fail("accepted ranking must not be rerun"))
    result, pool, order = campaign.rank(h["root"])
    assert result == ranked
    restored = campaign.validate_state(h["root"])
    assert "rankings.json" in restored["frozen_files"]
    assert "rankings.json" not in restored["pending_publications"]
    assert set(order) == set(pool)


def test_offline_readiness_requires_completed_current_full_domain_verification(tmp_path, monkeypatch):
    from cpu_policies.open_resources import prepare
    api = campaign._common()
    monkeypatch.setattr(prepare, "_api", lambda bundle=None: api)
    root = tmp_path / "controller"
    monkeypatch.setattr(prepare, "ROOT", root)
    implementation = {}
    for name in ("search.py", "oracle.cc", "picojson.h"):
        path = root / "cpu_policies/open_resources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version-one")
        implementation[name] = api.sha256(path)
    report = root / "artifacts/open_resources_ns_full_domain_check/verification.json"
    with pytest.raises(ValueError, match="not complete"):
        prepare._full_domain_verification()
    value = {"passed": True, "synthetic_only": True, "counts": {"contexts": 9, "enhanced": 966240306},
             "implementation": implementation}
    api.write(report, value)
    assert prepare._full_domain_verification() == value
    (root / "cpu_policies/open_resources/oracle.cc").write_text("version-two")
    with pytest.raises(ValueError, match="implementation changed"):
        prepare._full_domain_verification()


@pytest.mark.parametrize("with_traffic", [False, True])
def test_reused_calibration_keeps_original_identity_and_skips_only_valid_work(harness, with_traffic):
    h = harness
    cfg = h["component"]
    key = domain.component_key(cfg)
    old = {"boot_id": "calibration-source-boot", "document": {
        "kind": "calibration", "task_id": "original-calibration-task", "bundle_id": "old-controller-bundle"},
        "summary": {field: 1. for field in campaign.COST_FIELDS}}
    additional = {key: {"kind": "imported", "configuration": cfg,
                        "cost": {field: 1. for field in campaign.COST_FIELDS}, "measurement": old,
                        "qualification": {"boot_id": "calibration-source-boot", "document": {
                            "kind": "qualify_component", "task_id": "original-qualification-task"}, "summary": {"correct": True}},
                        "provenance": {"source_task_id": "original-calibration-task"}}}
    traffic = {}
    if with_traffic:
        program = cfg["stages"][0]["program"]
        traffic[program] = {"kind": "imported", "program": program, "configuration": cfg,
                            "load_bytes": 32, "store_bytes": 16,
                            "measurement": {"boot_id": "calibration-source-boot", "document": {
                                "kind": "traffic", "task_id": "original-traffic-task", "bundle_id": "old-controller-bundle"},
                                "summary": {"hooks_present": True, "count": 1, "load_bytes": 32, "store_bytes": 16}}}
    h["api"].write(h["bundle"] / "plans/reused_calibration.json", {
        "costs": additional, "traffic": traffic, "source_boot_ids": ["calibration-source-boot"],
        "source_bundle_id": "old-controller-bundle"})
    state = campaign.run_current(h["root"], h["transport"])
    assert (state["imported_components"], state["new_components"]) == (2, 0)
    assert state["source_pilot_components"] == state["reused_calibration_components"] == 1
    assert (state["imported_traffic"], state["new_traffic"]) == ((1, 0) if with_traffic else (0, 1))
    assert [d["kind"] for d in h["transport"].calls] == (
        ["qualify_pipeline", "performance"] if with_traffic else ["traffic", "qualify_pipeline", "performance"])
    reread = campaign.load_costs(h["root"])[key]
    assert reread["measurement"] == old
    assert reread["measurement"]["document"]["bundle_id"] == "old-controller-bundle"
    assert state["completed_boots"] == ["new-boot"]
    if with_traffic:
        stored = campaign.load_traffic(h["root"])[program]
        assert stored["measurement"]["document"]["task_id"] == "original-traffic-task"
        assert stored["measurement"]["boot_id"] == "calibration-source-boot"
