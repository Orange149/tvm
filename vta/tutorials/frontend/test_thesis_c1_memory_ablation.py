"""Protocol/adversarial fixtures only; these are never board experiment data."""

import copy
import io
import json
from pathlib import Path
import tarfile
import uuid

import numpy as np
import pytest

from thesis_c1_memory_ablation import (
    COMMON_ARTIFACTS, FACTORS, INPUT_SCHEMA, digest, evaluate_orders, file_hash,
    freeze, mask_name, ordered_support, score_order, seal, validate_contract,
)
from thesis_c1_memory_evidence import NumericMismatch, audit_stage_log, validate_performance, validate_qualification
from run_thesis_c1_memory_ablation import native_environment, safe_extract, session_argv, session_sources


def ref(path, content):
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return {"path": str(path), "sha256": file_hash(path)}


@pytest.fixture
def inventory(tmp_path):
    artifacts = {name: ref(tmp_path / name, name) for name in COMMON_ARTIFACTS}
    candidates = []
    for index in range(9):
        islands = index % 3 + 1
        stage_count = 2 * islands + 1
        cursor, stages = 0, []
        for stage in range(stage_count):
            end = (stage + 1) * 21 // stage_count + (index // 3 if stage == 0 else 0)
            device = "cpu" if stage % 2 == 0 else "vta"
            stages.append({"begin": cursor, "end": end, "device": device,
                           "cpu_affinity": [0, 1] if device == "cpu" else [],
                           "input_names": ["x"], "output_node_ids": [end],
                           "graph": ref(tmp_path / ("graph_{}_{}".format(index, stage)), "graph"),
                           "lib": ref(tmp_path / ("lib_{}_{}".format(index, stage)), "lib"),
                           "params": ref(tmp_path / ("params_{}_{}".format(index, stage)), "params"),
                           **({"final_tir": ref(tmp_path / ("tir_{}_{}".format(index, stage)), "tir")} if device == "vta" else {})})
            cursor = end
        slot_bytes = 1024 * (index % 4 + 1)
        plan = {"schema": "shared-buffer-plan-v1", "predicted_saved_bytes": 0,
                "runtime_sha256": artifacts["runtime"]["sha256"], "driver_sha256": artifacts["driver"]["sha256"],
                "stages": [{"device": s["device"], "input_names": s["input_names"],
                            **{field + "_sha256": s[field]["sha256"] for field in ("graph", "lib", "params")}} for s in stages],
                "edges": [{"producer_stage": n, "consumer_stage": n + 1, "source": "external",
                           "external_k2_bytes": slot_bytes if n == 0 else 0} for n in range(stage_count - 1)]}
        plan_ref = ref(tmp_path / ("plan_" + str(index)), json.dumps(plan))
        evidence = ref(tmp_path / ("feature_evidence_" + str(index)), "static-only fixture")
        candidates.append({"candidate_id": "base_" + str(index), "topology_id": "top_" + str(index),
                           "m1_score_ms": 100 + 0.1 * index, "vta_islands": islands, "static_legal": True,
                           "stages": stages, "thread_vector": [2 if s["device"] == "cpu" else 1 for s in stages],
                           "framework_materialization_bytes": 12000 - index * 512,
                           "shared_framework_materialization_bytes": 0,
                           "additional_slot_bytes": slot_bytes, "vta_dma_calls": 1000 - index * 10,
                           "shared_plan": plan_ref, "feature_sources": {f: [evidence] for f in "BCD"}})
    return {"schema": INPUT_SCHEMA, "model": "pretrained MXNet ResNet18_v1@224", "extra_quantize_calls": 0,
            "new_performance_labels_read": False, "prior_knowledge": ["synthetic protocol fixture; no board evidence"],
            "queue_capacity_bytes": {"instruction": 8192, "uop": 8192}, "artifacts": artifacts,
            "candidates": candidates}


def test_freeze_retains_pairs_and_all_orders(inventory):
    value = freeze(inventory)
    validate_contract(value)
    assert len(value["support"]) == 16
    assert len(value["orders"]) == 16
    assert len(value["random_orders"]) == 20
    for islands, count in ((1, 6), (2, 6), (3, 4)):
        assert sum(r["vta_islands"] == islands for r in value["support"]) == count
    ids = {r["configuration_id"] for r in value["support"]}
    for order in [*value["orders"].values(), *value["random_orders"].values()]:
        assert len(order) == 16 and set(order) == ids
        assert order[0] == value["common_first"]
    assert next(r for r in value["support"] if r["configuration_id"] == value["common_first"])["policy"] == "ordinary_copy"
    assert value["identifiability"]["ranking_windows_allowed"]


def test_source_input_not_mutated_and_best_threads_chosen(inventory):
    before = copy.deepcopy(inventory)
    inferior = copy.deepcopy(inventory["candidates"][0])
    inferior["candidate_id"] = "inferior_threads"
    inferior["m1_score_ms"] = 999
    inventory["candidates"].append(inferior)
    result = freeze(inventory)
    assert all(r["base_candidate_id"] != "inferior_threads" for r in result["support"])
    assert inventory["candidates"][:-1] == before["candidates"]


def test_topology_aliases_do_not_fill_distinct_topology_quotas(inventory):
    inventory["candidates"] = [r for r in inventory["candidates"] if r["vta_islands"] != 3 or r["candidate_id"] == "base_2"]
    alias = copy.deepcopy(next(r for r in inventory["candidates"] if r["vta_islands"] == 3))
    alias.update(candidate_id="alias", topology_id="different_name_for_same_partition")
    inventory["candidates"].append(alias)
    with pytest.raises(ValueError, match="insufficient 3-island"):
        freeze(inventory)


@pytest.mark.parametrize("field,value", [("vta_dma_calls", None), ("additional_slot_bytes", -1),
                                         ("shared_framework_materialization_bytes", None),
                                         ("framework_materialization_bytes", float("nan"))])
def test_unknown_or_invalid_features_rejected(inventory, field, value):
    inventory["candidates"][0][field] = value
    with pytest.raises(ValueError):
        freeze(inventory)


def test_incomplete_island_support_never_replaced(inventory):
    inventory["candidates"] = [r for r in inventory["candidates"] if r["vta_islands"] != 3]
    with pytest.raises(ValueError, match="3-island"):
        freeze(inventory)


def test_labels_and_source_drift_rejected(inventory):
    inventory["candidates"][0]["observed_ii_ms"] = 1
    with pytest.raises(ValueError, match="performance label"):
        freeze(inventory)
    del inventory["candidates"][0]["observed_ii_ms"]
    contract = freeze(inventory)
    Path(inventory["artifacts"]["driver"]["path"]).write_text("drift")
    with pytest.raises(ValueError, match="source drift"):
        validate_contract(contract)


def test_constant_factor_and_mask_independence(inventory):
    for r in inventory["candidates"]:
        r["vta_dma_calls"] = 100
    contract = freeze(inventory)
    assert contract["identifiability"]["factors"]["D"]["status"] == "not_identifiable"
    assert contract["orders"]["BCDI"] == contract["orders"]["BCI"]
    assert contract["orders"]["D"] == contract["orders"]["NONE"]
    changed = copy.deepcopy(contract["support"])
    for r in changed:
        r["vta_dma_calls"] = 999999 - r["vta_dma_calls"]
    assert ordered_support(changed, "BCI", contract["common_first"]) == contract["orders"]["BCI"]


def test_shared_passthrough_copy_bytes_are_not_assumed_zero(inventory):
    inventory["candidates"][0]["shared_framework_materialization_bytes"] = 256
    contract = freeze(inventory)
    pair = [r for r in contract["support"] if r["base_candidate_id"] == "base_0"]
    assert next(r for r in pair if r["policy"] != "ordinary_copy")["framework_materialization_bytes"] == 256


def test_no_difference_gate_is_a_real_stop(inventory):
    for i, r in enumerate(inventory["candidates"]):
        r["m1_score_ms"] = 100 + 10 * i
    contract = freeze(inventory)
    assert not contract["identifiability"]["ranking_windows_allowed"]
    assert contract["orders"]["NONE"][:5] == contract["orders"]["BCDI"][:5]


def test_invalid_candidates_and_capacity_rejections_consume_positions():
    ids = ["a", "b", "c", "d", "e", "f"]
    labels = {"a": None, "b": 80, "c": 90, "d": 100, "e": 100, "f": 100}
    rows = [{"configuration_id": k, "additional_slot_bytes": 200 if k == "b" else 0} for k in ids]
    result = score_order(ids, labels, capacity=100, rows=rows)
    assert result["first_hit"] == 3
    assert result["first_qualified"] == 3
    assert result["curve"][0]["regret"] is None
    assert result["curve"][1]["capacity_rejections"] == 1
    filtered = score_order(ids, labels, capacity=100, rows=rows, prefilter=True)
    assert filtered["first_hit"] == 2


def test_raw_window_uses_last_frame_and_299_gaps(inventory, tmp_path):
    contract = freeze(inventory)
    config = contract["support"][0]
    directory = tmp_path / "performance"
    directory.mkdir()
    (directory / "native.log").write_text("no diagnostics\n")
    rows = []
    for i in range(307):
        row = {"frame_id": i, "completion_index": i, "input_index": i % 64, "mode": "pipeline",
               "completion_ms": 10 + i * 100 + (50 if i == 306 else 0)}
        for stage, spec in enumerate(config["identity"]["stages"]):
            if spec["device"] == "vta":
                row["stage{}_vta_mutex_acquire_ms".format(stage)] = i * 100 + stage
                row["stage{}_vta_mutex_release_ms".format(stage)] = i * 100 + stage + 0.5
        rows.append(row)
    (directory / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    metric = validate_performance(directory, config)
    assert metric["gaps_scored"] == 299
    assert metric["ii_ms"] == pytest.approx(100 + 50 / 299)
    (directory / "native.log").write_text("[VTA_QUEUE] {}\n")
    with pytest.raises(ValueError, match="diagnostic"):
        validate_performance(directory, config)


def test_performance_argv_has_no_qualification_flags(inventory):
    contract = freeze(inventory)
    config = next(r for r in contract["support"] if r["policy"] != "ordinary_copy")
    args = session_argv(contract, config, "/ram/run", "inputs_real.txt", "/ram/run/perf", False)
    assert "--c1-memory-audit" not in args and "--vta-runtime-profile-dir" not in args
    assert args[args.index("--warmup-runs") + 1] == "0"
    assert args[args.index("--runs") + 1] == "307"
    assert "--shared-slot-budget-bytes" in args
    assert args[args.index("--stage0-lib") + 1].endswith(".so")
    assert args[args.index("--process-timeout-s") + 1] == "285"


def test_archive_traversal_and_links_rejected(tmp_path):
    for name, link in (("../escape", False), ("link", True)):
        archive = tmp_path / ("link.tar" if link else "escape.tar")
        with tarfile.open(archive, "w") as tar:
            item = tarfile.TarInfo(name)
            item.type = tarfile.SYMTYPE if link else tarfile.REGTYPE
            if link:
                item.linkname = "/etc/passwd"
            tar.addfile(item, None if link else io.BytesIO())
        with pytest.raises(ValueError, match="unsafe"):
            safe_extract(archive, tmp_path / "output")


def stage_log(config, queue, frames=2):
    lines = []
    for frame in range(frames):
        for stage, spec in enumerate(config["identity"]["stages"]):
            if spec["device"] != "vta":
                continue
            begin = {"event": "begin", "frame": frame, "stage": stage, "tid": 100 + stage,
                     "queue_id": "queue_" + str(stage), "time_ms": frame * 100 + stage, "queue_status": None}
            lines.append("[C1_MEMORY_STAGE] " + json.dumps(begin))
            lines += ["There are 1 instructions", "INSTRUCTION 0: FINISH", "\tl2g_queue = 0 g2l_queue = 0", "\ts2g_queue = 0 g2s_queue = 0"]
            lines.append('[C3_COMPLETION] ' + json.dumps({"after_clear_valid": 0, "return_valid": 1,
                          "return_done": 1, "timeout": 0, "insn_count": 1}))
            lines.append('[VTA_QUEUE] ' + json.dumps({"queue_id": begin["queue_id"], "insn_bytes": 16, "timeout": 0}))
            status = {"schema": "vta_queue_capacity_status_v1", "replay_policy": "disabled",
                      "insn_submit_threshold_configured": False, "insn_capacity_bytes": queue["instruction"],
                      "uop_capacity_bytes": queue["uop"], "insn_peak_bytes": 16, "uop_peak_bytes": 0}
            end = {**begin, "event": "end", "time_ms": begin["time_ms"] + 1, "queue_status": status}
            lines.append('[C1_MEMORY_STAGE] ' + json.dumps(end))
    return "\n".join(lines) + "\n"


def test_multiworker_audit_rejects_stale_completion_and_missing_calls(inventory, tmp_path):
    contract = freeze(inventory)
    config = next(r for r in contract["support"] if r["vta_islands"] == 3)
    path = tmp_path / "diagnostic.log"
    text = stage_log(config, contract["queue_capacity_bytes"])
    path.write_text(text)
    result = audit_stage_log(path, config, contract["queue_capacity_bytes"], expected_frames=2)
    assert len(result["workers"]) == 3 and result["calls"] == 6
    assert result["queue_model"] == "one_queue_per_worker"
    shared = text
    for stage, spec in enumerate(config["identity"]["stages"]):
        if spec["device"] == "vta":
            shared = shared.replace('"queue_{}"'.format(stage), '"one_process_queue"')
    path.write_text(shared)
    assert audit_stage_log(path, config, contract["queue_capacity_bytes"], expected_frames=2)["queue_model"] == "one_process_queue_shared_by_workers"
    partial = text.replace('"queue_3"', '"queue_1"')
    path.write_text(partial)
    with pytest.raises(ValueError, match="ownership model"):
        audit_stage_log(path, config, contract["queue_capacity_bytes"], expected_frames=2)
    path.write_text(text.replace('"after_clear_valid": 0', '"after_clear_valid": 1', 1))
    with pytest.raises(ValueError, match="fresh"):
        audit_stage_log(path, config, contract["queue_capacity_bytes"], expected_frames=2)
    path.write_text(text)
    with pytest.raises(ValueError, match="coverage"):
        audit_stage_log(path, config, contract["queue_capacity_bytes"], expected_frames=3)


def test_all_mask_evaluation_keeps_oracle_common(inventory):
    contract = freeze(inventory)
    labels = {r["configuration_id"]: 100 + i for i, r in enumerate(contract["support"])}
    result = evaluate_orders(contract, labels)
    assert len(result["0.02"]["methods"]) == 16
    assert {v["oracle_ii_ms"] for v in result["0.02"]["methods"].values()} == {100}
    assert not result["0.02"]["full_vs_M1_improved"]  # fixed first is already optimal
    assert len(result["0.02"]["marginal_factor_effects"]) == 32
    assert sum(result["0.02"]["random_summary"]["first_hit_distribution"].values()) == 20
    labels[contract["common_first"]] = None
    result = evaluate_orders(contract, labels)
    assert result["0.02"]["random_summary"]["curve"][0]["mean_regret"] is None
    assert len(result["0.02"]["methods"]["BCDI"]["curve"]) == 16


def test_missing_cost_journal_cannot_be_promoted_to_complete(tmp_path):
    from analyze_thesis_c1_memory_ablation import verify_costs
    from thesis_supplement_run import RunStore
    root = tmp_path / "empty_journal"
    RunStore.create(root, {"run_id": "fixture", "stage": "test", "campaign_id": "fixture"})
    with pytest.raises(ValueError, match="append-only journal"):
        verify_costs(root, {"cost_actions": []}, {})


def test_cost_audit_binds_script_reset_receipt_and_returned_archive(inventory, tmp_path):
    from analyze_thesis_c1_memory_ablation import verify_costs
    from run_thesis_c1_memory_ablation import native_session_script, reset_script, ssh_argv
    from thesis_c1_memory_ablation import write_new
    from thesis_supplement_run import RunStore
    contract = freeze(inventory)
    root = tmp_path / "fabricated_transport_fixture"
    store = RunStore.create(root, {"run_id": "fixture", "stage": "test",
                            "campaign_id": contract["artifact_sha256"], "boot_index": 0,
                            "board_host": "synthetic.fixture.invalid"})
    boot = {"boot_id": str(uuid.uuid4()), "boot_index": 0,
            "configurations": [{"qualification": "qual_fixture"}], "capacity_boundaries": []}
    def invented_action(name, script, stdout):
        # Construct a journal fixture; never invoke SSH or pretend it is evidence.
        path = root / "raw" / (name + ".sh")
        path.write_text("set -eu\n" + script + "\n")
        action_id = uuid.uuid4().hex
        argv = ssh_argv("synthetic.fixture.invalid") + ["sh", "-s"]
        parent = store.event("action_start", action_id=action_id, action=name, argv=argv,
                             cost_class="fixture", stdin_sha256=file_hash(path))
        out = root / "raw" / (name + ".stdout")
        err = root / "raw" / (name + ".stderr")
        out.write_bytes(stdout)
        err.write_bytes(b"")
        store.event("action_end", parent_event_id=parent, action_id=action_id, action=name,
                    argv=argv, cost_class="fixture", status="ok", returncode=0,
                    wall_seconds=1.0, begin_monotonic_ns=0, end_monotonic_ns=1000000000,
                    stdout=str(out.relative_to(root)), stderr=str(err.relative_to(root)),
                    stdout_sha256=file_hash(out), stderr_sha256=file_hash(err))
    invented_action("kernel_identity", "cat /proc/sys/kernel/random/boot_id", (boot["boot_id"] + "\n").encode())
    reset = reset_script(contract, "/ram/run", boot["boot_id"])
    marker = "C1_CONTEXT_RESET_READY " + boot["boot_id"] + " " + contract["artifacts"]["bitstream"]["sha256"]
    invented_action("qual_fixture_reset", reset, (marker + "\n").encode())
    directory = root / "raw/qual_fixture"
    directory.mkdir()
    args = session_argv(contract, contract["support"][0], "/ram/run", "inputs_all.txt", "/ram/run/qual_fixture", True)
    environment = native_environment(contract, "/ram/run", True)
    write_new(directory / "session_contract.json", {"argv": args, "environment": environment})
    (directory / "native.log").write_text("invented transport fixture; no inference")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        tar.add(directory / "native.log", arcname="./native.log")
    script = native_session_script(boot["boot_id"], "/ram/run/qual_fixture", args, environment)
    invented_action("qual_fixture", script, archive.getvalue())
    boot["cost_actions"] = [e for e in store.events() if e["event"] == "action_end"]
    assert verify_costs(root, boot, contract) == {"fixture": 3.0}
    path = root / "raw/qual_fixture_reset.sh"
    original = path.read_text()
    path.write_text("true\n")
    with pytest.raises(ValueError, match="executed reset/native script"):
        verify_costs(root, boot, contract)
    path.write_text(original)
    (directory / "native.log").write_text("substituted output after collection")
    with pytest.raises(ValueError, match="actual SSH archive"):
        verify_costs(root, boot, contract)


def test_tampered_order_is_rejected_even_if_resealed(inventory):
    contract = freeze(inventory)
    contract["orders"]["BCD"][1:3] = reversed(contract["orders"]["BCD"][1:3])
    with pytest.raises(ValueError, match="feature mask"):
        validate_contract(seal(contract))


def qualification_fixture(directory, config, contract, count=67):
    """Tiny invented raw files for auditor tests, never measurements or board proof."""
    directory.mkdir(parents=True)
    (directory / "outputs").mkdir()
    (directory / "profile").mkdir()
    stages = config["identity"]["stages"]
    frontiers, items = {}, []
    for index, stage in enumerate(stages):
        array = np.arange(4, dtype="float32" if index == len(stages) - 1 else "int8")
        np.save(directory / ("reference_{}.npy".format(index)), array)
        (directory / "outputs" / ("stage{}.bin".format(index))).write_bytes(array.tobytes())
        frontiers[str(stage["output_node_ids"][0])] = {
            "path": str(directory / ("reference_{}.npy".format(index)))}
        items.append({"path": "/ram/outputs/stage{}.bin".format(index), "dtype": str(array.dtype), "shape": list(array.shape)})
    rows = []
    for frame in range(count):
        row = {"frame_id": frame, "completion_index": frame, "input_index": frame,
               "mode": "pipeline", "output_mode": "raw_all_stages",
               "c1_framework_materialization_bytes": config["framework_materialization_bytes"],
               "raw_output_files": [items[-1]],
               "stage_raw_outputs": [{"stage_index": i, "output_files": [item]} for i, item in enumerate(items)]}
        if config["policy"] != "ordinary_copy":
            row.update(p8_managed_slot_count=2, p8_framework_materialization_bytes=config["framework_materialization_bytes"],
                       p8_boundaries=[{"edge_index": e, "slot_id": frame % 2, "generation": frame // 2 + 1,
                                       "tensor_bytes": [4], "boundary_bytes": 4,
                                       "physical_addresses": [hex(0x100000 + e * 65536 + (frame % 2) * 256)]}
                                      for e in range(len(stages) - 1)])
        rows.append(row)
    (directory / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (directory / "native.log").write_text(stage_log(config, contract["queue_capacity_bytes"], count))
    (directory / "profile/benchmark_totals_status.json").write_text(json.dumps({
        "load_buffer_2d_calls": count * config["vta_dma_calls"], "store_buffer_2d_calls": 0}))
    allocations = [{"phase": "before_graph_load", "initialized": 0, "capacity_bytes": 0,
                    "requested_bytes": 0, "high_water_bytes": 0, "physical_base": 0}]
    for phase in ("graph_and_params_loaded", "slots_allocated", "after_steady_state"):
        total = (1 << 20) + (0 if phase == "graph_and_params_loaded" else config["additional_slot_bytes"])
        allocations.append({"phase": phase, "initialized": 1, "capacity_bytes": 192 << 20,
                            "requested_bytes": total, "high_water_bytes": total, "physical_base": 0x100000})
    (directory / "allocations.jsonl").write_text("\n".join(json.dumps(r) for r in allocations))
    return [{"frontiers": frontiers} for _ in range(count)]


def test_full_qualification_rechecks_outputs_generations_dma_and_ranges(inventory, tmp_path):
    contract = freeze(inventory)
    config = next(r for r in contract["support"] if r["policy"] != "ordinary_copy" and r["vta_islands"] == 3)
    directory = tmp_path / "raw_qualification"
    records = qualification_fixture(directory, config, contract)
    assert validate_qualification(contract, config, directory, records)["calls"] == 201
    target = directory / "outputs/stage0.bin"
    original = target.read_bytes()
    target.write_bytes(bytes([99]) + original[1:])
    with pytest.raises(NumericMismatch, match="frontier"):
        validate_qualification(contract, config, directory, records)
    target.write_bytes(original)
    path = directory / "result.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[2]["p8_boundaries"][0]["generation"] = 1
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="generation"):
        validate_qualification(contract, config, directory, records)
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["p8_boundaries"][0]["physical_addresses"] = ["0x100"]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="physical range"):
        validate_qualification(contract, config, directory, records)
    path.write_text(original)
    profile = directory / "profile/benchmark_totals_status.json"
    profile.write_text(json.dumps({"load_buffer_2d_calls": 0, "store_buffer_2d_calls": 0}))
    with pytest.raises(ValueError, match="DMA"):
        validate_qualification(contract, config, directory, records)


def test_instruction_dump_must_match_actual_queue_bytes(inventory, tmp_path):
    contract = freeze(inventory)
    config = contract["support"][0]
    text = stage_log(config, contract["queue_capacity_bytes"])
    text = text.replace('"insn_bytes": 16', '"insn_bytes": 32').replace('"insn_count": 1', '"insn_count": 2')
    path = tmp_path / "inconsistent_insns.log"
    path.write_text(text)
    with pytest.raises(ValueError, match="dump differs"):
        audit_stage_log(path, config, contract["queue_capacity_bytes"], expected_frames=2)


def test_analyzer_recomputes_three_boots_and_rejects_duplicate_identity(inventory, tmp_path, monkeypatch):
    import analyze_thesis_c1_memory_ablation as analyzer
    from thesis_c1_memory_ablation import BOOT_SCHEMA, write_new
    contract = freeze(inventory)
    by_id = {r["configuration_id"]: r for r in contract["support"]}
    # Only the image decoder and transport/cost auditor are isolated here. Raw
    # frontiers, completion windows, bindings and pair comparisons are real code.
    monkeypatch.setattr(analyzer, "inputs_manifest", lambda _: [])
    monkeypatch.setattr(analyzer, "verify_costs", lambda *_: {})
    paths = []
    for boot_index in range(3):
        root = tmp_path / ("boot_" + str(boot_index))
        (root / "raw").mkdir(parents=True)
        boot = {"schema": BOOT_SCHEMA, "status": "complete", "boot_id": str(uuid.uuid4()), "boot_index": boot_index,
                "selection_contract_sha256": contract["artifact_sha256"], "configurations": [], "capacity_boundaries": []}
        def bind(directory, config, diagnostic, budget=None, boundary=False):
            args = session_argv(contract, config, "/ram/run", "inputs_real.txt" if boundary or not diagnostic else "inputs_all.txt",
                                "/ram/run/" + directory.name, diagnostic, budget, boundary)
            write_new(directory / "session_contract.json", {"boot_id": boot["boot_id"], "boot_index": boot_index,
                      "configuration_id": config["configuration_id"], "selection_contract_sha256": contract["artifact_sha256"],
                      "diagnostic": diagnostic, "capacity_bytes": budget, "argv": args,
                      "environment": native_environment(contract, "/ram/run", diagnostic)})
        records = None
        for key in contract["measurement_orders"][str(boot_index)]:
            config = by_id[key]
            qual = root / "raw" / ("qual_" + key)
            current = qualification_fixture(qual, config, contract)
            if records is None:
                records = [{"frontiers": {}} for _ in range(67)]
            for index in range(67):
                records[index]["frontiers"].update(current[index]["frontiers"])
            bind(qual, config, True)
            perf = root / "raw" / ("perf_" + key)
            perf.mkdir()
            (perf / "native.log").write_text("diagnostics disabled")
            rows = []
            for i in range(307):
                row = {"frame_id": i, "completion_index": i, "input_index": i % 64, "mode": "pipeline",
                       "completion_ms": 1 + i * (100 + boot_index)}
                for stage, spec in enumerate(config["identity"]["stages"]):
                    if spec["device"] == "vta":
                        row["stage{}_vta_mutex_acquire_ms".format(stage)] = i * 100 + stage
                        row["stage{}_vta_mutex_release_ms".format(stage)] = i * 100 + stage + 0.5
                rows.append(row)
            (perf / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            bind(perf, config, False)
            boot["configurations"].append({"configuration_id": key, "status": "qualified",
                     "qualification": qual.name, "qualification_files": session_sources(qual),
                     "performance": perf.name, "performance_files": session_sources(perf),
                     "performance_metrics": {"ii_ms": -999}})  # summaries must be ignored
        for key in contract["capacity_boundary_ids"]:
            config = by_id[key]
            for budget in (config["additional_slot_bytes"] - 256, config["additional_slot_bytes"]):
                directory = root / "raw" / ("capacity_{}_{}".format(key, budget))
                if budget < config["additional_slot_bytes"]:
                    directory.mkdir()
                    (directory / "native.log").write_text("[C1_SLOT_CAPACITY_REJECTED]")
                else:
                    current = qualification_fixture(directory, config, contract, 64)
                    for index in range(64):
                        records[index]["frontiers"].update(current[index]["frontiers"])
                bind(directory, config, True, budget, True)
                boot["capacity_boundaries"].append({"configuration_id": key, "budget_bytes": budget,
                            "session": directory.name, "files": session_sources(directory)})
        # Nodes in this tiny fixture can occur at several cuts; last unit is
        # float and all earlier nodes are the same integer tensor by design.
        monkeypatch.setattr(analyzer, "inputs_manifest", lambda _, rows=records: rows)
        path = root / "boot.json"
        write_new(path, seal(boot))
        paths.append(path)
    result = analyzer.analyze(contract, paths)
    assert set(result["candidate_median_ii_ms"].values()) == {101}
    assert result["completion_rows"] == 14496 and result["performance_windows"] == 48
    analyzer.artifacts(tmp_path / "invented_analysis_outputs", result)
    assert (tmp_path / "invented_analysis_outputs/quality_by_budget.pdf").is_file()
    assert (tmp_path / "invented_analysis_outputs/capacity_coverage.png").is_file()
    assert len((tmp_path / "invented_analysis_outputs/all_masks.csv").read_text().splitlines()) == 33
    value = json.loads(paths[1].read_text())
    value["boot_id"] = json.loads(paths[0].read_text())["boot_id"]
    paths[1].write_text(json.dumps(seal(value)))
    with pytest.raises(ValueError, match="distinct kernel"):
        analyzer.analyze(contract, paths)
