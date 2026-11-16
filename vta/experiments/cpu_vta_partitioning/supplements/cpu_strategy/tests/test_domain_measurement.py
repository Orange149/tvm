import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / "src")]

from cpu_vta_partitioning.common import digest, inventory, read, sha256, write
from cpu_vta_partitioning.domain import Stage
from cpu_strategy.domain import (
    ENV, components, configuration, matrix, protocol, validate_candidate, validate_stage,
)
from cpu_strategy.evidence import Evidence, identified
from cpu_strategy.measurement import (
    isolated_components, make_task, validate_summary, whole_pipeline_window,
)


def candidate():
    m = matrix()
    return m["configs"][m["families"]["shared"][0]]


def native_rows(document):
    count = sum(document[k] for k in ("warmup", "scored", "tail"))
    rows = []
    for i in range(count):
        row = {"frame_id": i, "completion_index": i,
               "input_index": i % len(document["input_indices"]),
               "completion_ms": 10. + 4 * i, "completion_process_cpu_ms": 100. + 8 * i,
               "total_latency_ms": 20., "cpu_stage_gate_mode": "off",
               "stage_cpu_time_scope": "exclusive_process_cpu_time"
               if document["kind"] == "calibration" else "overlapping_process_cpu_time"}
        for j, stage in enumerate(document["stages"]):
            row.update({f"stage{j}_set_ms": .5, f"stage{j}_run_ms": 2., f"stage{j}_get_ms": .25,
                        f"stage{j}_set_process_cpu_ms": .25,
                        f"stage{j}_run_process_cpu_ms": 3., f"stage{j}_get_process_cpu_ms": .125})
        rows.append(row)
    return rows


def native_affinity(document):
    result = []
    for i, stage in enumerate(document["stages"]):
        tids = [1000 + 100 * i + j for j in range(max(stage["threads"], len(stage["mask"])))]
        result.append({"requested_cpus": stage["mask"], "thread_ids": tids,
                       "active_runtime_threads": stage["threads"],
                       "effective_affinities": [{"tid": tid, "cpus": stage["mask"],
                                                 "cpu_time_delta_ms": 10. if j == 0 else 0.}
                                                for j, tid in enumerate(tids)]})
    return {"stages": result}


def write_raw(directory, document):
    directory.mkdir(parents=True, exist_ok=True)
    rows = native_rows(document)
    (directory / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    write(directory / "affinity.json", native_affinity(document))
    return rows


def test_matrix_is_exact_and_families_disjoint():
    m = matrix()
    assert len(m["configs"]) == 32
    assert {k: len(v) for k, v in m["families"].items()} == {
        "shared": 16, "disjoint": 12, "overlap": 2, "references": 2}
    keys = [key for family in m["families"].values() for key in family]
    assert len(set(keys)) == len(keys)
    for row in m["configs"].values():
        stages = validate_candidate(row)
        assert stages[1] == Stage("vta", 3, 18, 1, (0, 1, 2, 3))
    for key in m["families"]["disjoint"]:
        stages = m["configs"][key]["stages"]
        assert not set(stages[0]["mask"]) & set(stages[2]["mask"])
        assert stages[0]["threads"] + stages[2]["threads"] <= 4
    assert m["interventions"]["separate_low"] in m["families"]["disjoint"]
    assert m["interventions"]["separate_high"] in m["families"]["disjoint"]
    assert len(set(m["warmup_anchors"])) == 3


def test_component_coverage_and_diagnostic_blocks():
    cs = components()
    assert len(cs) == len({c["stage"]["signature"] for c in cs}) == 34
    pipeline = [c for c in cs if c["role"] == "pipeline"]
    diagnostics = [c for c in cs if c["role"] == "diagnostic"]
    assert len(pipeline) == 22
    assert sum(c["stage"]["device"] == "cpu" for c in pipeline) == 21
    assert len(diagnostics) == 12 and {c["blocks"] for c in diagnostics} == {3}
    assert {c["stage"]["begin"] for c in diagnostics} == {0, 2, 20}
    assert {c["stage"]["program"] for c in cs} == {
        "cpu_00_02", "vta_03_17", "cpu_18_20", "cpu_00_00", "cpu_02_02", "cpu_20_20"}


def test_affinity_superset_supported_without_changing_old_validator():
    stage = Stage("cpu", 0, 3, 1, (0, 1, 2, 3))
    assert validate_stage(stage) == stage
    from cpu_vta_partitioning.domain import validate_candidate as old_validate
    values = validate_candidate(candidate())
    programs = {s.program: {"device": s.device, "begin": s.begin, "end": s.end} for s in values}
    validate_candidate(candidate(), programs)
    with pytest.raises(ValueError, match="CPU mask/thread mismatch"):
        old_validate(values, programs, 4)


@pytest.mark.parametrize("stage", [
    Stage("cpu", 0, 3, 3, (0, 1)), Stage("cpu", 0, 3, 0, (0,)),
    Stage("cpu", 0, 3, True, (0,)), Stage("cpu", 0, 3, 1, (0, 0)),
    Stage("cpu", 0, 3, 1, (1, 0)), Stage("cpu", 0, 3, 1, (4,)),
    Stage("cpu", 1, 2, 1, (0,)), Stage("vta", 3, 18, 1, (3,)),
])
def test_invalid_stage_rejected(stage):
    with pytest.raises(ValueError):
        validate_stage(stage)


def test_config_and_task_id_bind_threads_mask_and_context():
    a = configuration((1, 2), ((0, 1, 2, 3), (0, 1)))
    b = configuration((2, 2), ((0, 1, 2, 3), (0, 1)))
    c = configuration((1, 2), ((0, 1), (0, 1)))
    assert len({a["id"], b["id"], c["id"]}) == 3
    args = dict(warmup=1, scored=3, tail=1, indices=[16])
    x = make_task("screen", a["stages"], round=0, **args)
    y = make_task("screen", a["stages"], round=1, **args)
    assert x["task_id"] != y["task_id"]
    assert identified(x, "bundle-a")["task_id"] != identified(x, "bundle-b")["task_id"]
    with pytest.raises(ValueError, match="override"):
        make_task("screen", a["stages"], execution_mode="resource_group", **args)
    x["environment"]["host_cores"].append(99)
    assert ENV["host_cores"] == [0, 1, 2, 3]


def test_protocol_is_new_and_copies_environment():
    p = protocol()
    assert p["exploration_boots"] == 1 and p["confirmation_boots"] == 3
    assert p["expected"]["max_confirmation_configurations"] == 10
    assert p["formal_significance"] is False
    assert p["selection"]["profile_disjoint"].endswith("before_feedback")
    assert p["search"]["default_full_sweep"] is False
    assert p["search"]["screening_budget"] == 5
    assert p["expected"]["max_measured_pipeline_configurations"] == 13
    p["environment"]["cpu_cores"].clear()
    assert protocol()["environment"]["cpu_cores"] == [0, 1, 2, 3]


def test_whole_pipeline_core_time_uses_matching_completion_interval():
    document = make_task("screen", candidate()["stages"], warmup=1, scored=3, tail=1, indices=[16])
    rows = native_rows(document)
    for row in rows:
        for j in range(3):
            row[f"stage{j}_run_process_cpu_ms"] = 999999.
    value = whole_pipeline_window(rows, 1, 3, 1)
    assert value["ii_ms"] == 4.
    assert value["process_cpu_ms"] == 16.
    assert value["process_core_ms_per_completion"] == 8.
    assert value["average_cpu_cores"] == 2.
    assert value["gaps"] == 2
    assert value["cpu_time_scope"] == "whole_process_between_scored_completions"
    rows[2]["completion_process_cpu_ms"] = rows[1]["completion_process_cpu_ms"]
    whole_pipeline_window(rows, 1, 3, 1)  # equal adjacent clocks are legal


@pytest.mark.parametrize("mutation", ["decrease", "missing", "zero_window", "nan", "impossible_capacity"])
def test_invalid_whole_process_clocks_rejected(mutation):
    doc = make_task("screen", candidate()["stages"], warmup=1, scored=3, tail=1, indices=[16])
    rows = native_rows(doc)
    if mutation == "decrease": rows[2]["completion_process_cpu_ms"] = 0.
    elif mutation == "missing": del rows[2]["completion_process_cpu_ms"]
    elif mutation == "zero_window":
        for row in rows: row["completion_process_cpu_ms"] = 1.
    elif mutation == "nan": rows[2]["completion_process_cpu_ms"] = float("nan")
    else:
        for i, row in enumerate(rows): row["completion_process_cpu_ms"] = 100. + 100 * i
    with pytest.raises(ValueError): whole_pipeline_window(rows, 1, 3, 1)


def test_isolated_cpu_scope_and_boundary_single_accounting():
    stage = components()[0]["stage"]
    doc = make_task("calibration", [stage], warmup=1, scored=3, tail=0, indices=[0])
    rows = native_rows(doc)
    value = isolated_components(rows, 1, 3)
    assert value["service_ms"] == 2.75
    assert value["run_core_ms"] == 3.
    assert value["boundary_core_ms"] == .375
    assert value["set_ms"] == .5 and value["get_core_ms"] == .125
    rows[1]["stage_cpu_time_scope"] = "overlapping_process_cpu_time"
    with pytest.raises(ValueError, match="isolated"):
        isolated_components(rows, 1, 3)


def test_actual_activity_does_not_require_all_configured_threads_busy(tmp_path):
    doc = make_task("screen", candidate()["stages"], warmup=1, scored=3, tail=1, indices=[16])
    write_raw(tmp_path, doc)
    result = validate_summary(tmp_path, doc, None)
    assert result["thread_activity"][0]["configured_threads"] == 1
    assert result["thread_activity"][0]["observed_active_threads"] == 1
    value = read(tmp_path / "affinity.json")
    value["stages"][0]["effective_affinities"][0]["cpus"] = [99]
    write(tmp_path / "affinity.json", value, replace=True)
    with pytest.raises(ValueError, match="escapes"):
        validate_summary(tmp_path, doc, None)


def test_missing_thread_cpu_increment_rejected(tmp_path):
    doc = make_task("screen", candidate()["stages"], warmup=1, scored=3, tail=1, indices=[16])
    write_raw(tmp_path, doc)
    value = read(tmp_path / "affinity.json")
    del value["stages"][0]["effective_affinities"][0]["cpu_time_delta_ms"]
    write(tmp_path / "affinity.json", value, replace=True)
    with pytest.raises(ValueError, match="thread CPU"):
        validate_summary(tmp_path, doc, None)


def evidence_fixture(tmp_path, qualify=False):
    bundle = tmp_path / "bundle"
    write(bundle / "manifest.json", {"bundle_id": "bundle-a"})
    row = candidate()
    programs = {s["program"]: {"device": s["device"], "begin": s["begin"], "end": s["end"],
                                "output_node_ids": [0]} for s in row["stages"]}
    write(bundle / "plans/programs.json", programs)
    records = []
    for i in range(17):
        ref = bundle / "references" / f"{i}.npy"
        ref.parent.mkdir(parents=True, exist_ok=True)
        np.save(ref, np.array([1, 2], dtype="int8"))
        records.append({"index": i, "frontiers": {"0": {"npy": f"references/{i}.npy"}}})
    write(bundle / "plans/inputs.json", {"records": records})
    doc = make_task("qualify_pipeline" if qualify else "screen", row["stages"],
                    warmup=0 if qualify else 1, scored=2 if qualify else 3,
                    tail=0 if qualify else 1, indices=[0, 16] if qualify else [16])
    expected = identified(doc, "bundle-a")
    boot = "00000000-0000-0000-0000-000000000001"
    root = tmp_path / "run"
    raw = root / "raw" / boot / expected["task_id"] / "collected"
    rows = write_raw(raw, doc)
    if qualify:
        for r in rows:
            dumps = []
            for j in range(3):
                name = f"stage{j}.bin"
                path = raw / "outputs" / f"run_{r['completion_index']}" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes([1, 2]))
                dumps.append({"output_files": [{"path": "/remote/job/" + name,
                                                 "dtype": "int8", "shape": [2]}]})
            r["stage_raw_outputs"] = dumps
        (raw / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = validate_summary(raw, expected, bundle)
    files = {str((raw / k).relative_to(root)): v for k, v in inventory(raw).items()}
    receipt = {"task": expected, "boot_id": boot, "files": files, "summary": summary}
    name = f"accepted/{boot}/{expected['task_id']}.json"
    write(root / name, receipt)
    value = {**summary, "_receipt": {"path": name, "sha256": sha256(root / name)}}
    return root, bundle, doc, boot, raw, value


def test_evidence_recomputes_and_rechecks_raw_after_first_read(tmp_path):
    root, bundle, doc, boot, raw, value = evidence_fixture(tmp_path)
    e = Evidence(root, "bundle-a", bundle)
    e.summary(value, doc, boot)
    e.summary(value, doc, boot)
    with (raw / "result.jsonl").open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        e.summary(value, doc, boot)


def test_evidence_rejects_wrong_boot_bundle_and_forged_summary(tmp_path):
    root, bundle, doc, boot, raw, value = evidence_fixture(tmp_path)
    e = Evidence(root, "bundle-a", bundle)
    with pytest.raises(ValueError, match="boot/task"):
        e.summary(value, doc, boot[:-1] + "2")
    with pytest.raises(ValueError):
        Evidence(root, "bundle-b", bundle).summary(value, doc, boot)
    changed = copy.deepcopy(value); changed["ii_ms"] += 1
    with pytest.raises(ValueError, match="differs"):
        e.summary(changed, doc, boot)


def test_qualification_full_outputs_and_receipt_binding(tmp_path):
    root, bundle, doc, boot, raw, value = evidence_fixture(tmp_path, qualify=True)
    Evidence(root, "bundle-a", bundle).summary(value, doc, boot)
    receipt_path = root / value["_receipt"]["path"]
    receipt = read(receipt_path)
    omitted = next(name for name in receipt["files"] if name.endswith("stage0.bin"))
    receipt["files"].pop(omitted)
    write(receipt_path, receipt, replace=True)
    value["_receipt"]["sha256"] = sha256(receipt_path)
    with pytest.raises(ValueError, match="not bound"):
        Evidence(root, "bundle-a", bundle).summary(value, doc, boot)


def test_wrong_output_rejected_even_when_receipt_hash_is_updated(tmp_path):
    root, bundle, doc, boot, raw, value = evidence_fixture(tmp_path, qualify=True)
    output = raw / "outputs/run_0/stage0.bin"
    output.write_bytes(bytes([1, 3]))
    receipt_path = root / value["_receipt"]["path"]
    receipt = read(receipt_path)
    receipt["files"][str(output.relative_to(root))] = {"bytes": 2, "sha256": sha256(output)}
    write(receipt_path, receipt, replace=True)
    value["_receipt"]["sha256"] = sha256(receipt_path)
    with pytest.raises(ValueError, match="incorrect full output"):
        Evidence(root, "bundle-a", bundle).summary(value, doc, boot)


def test_duplicate_frame_and_wrong_input_sequence_are_rejected(tmp_path):
    doc = make_task("screen", candidate()["stages"], warmup=1, scored=3, tail=1, indices=[16])
    rows = write_raw(tmp_path, doc)
    rows[2]["frame_id"] = rows[1]["frame_id"]
    (tmp_path / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="duplicate/missing"):
        validate_summary(tmp_path, doc, None)
    rows = native_rows(doc)
    rows[2]["input_index"] = 1
    (tmp_path / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="input index"):
        validate_summary(tmp_path, doc, None)
