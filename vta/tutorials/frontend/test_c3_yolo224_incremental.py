"""Local protocol tests: never imports the board backend or opens SSH."""
import copy
from pathlib import Path

import pytest

import c3_yolo224_incremental as p
import run_vta_c3_yolo224_incremental_graph as runner


def rows(boot=1, times=(429., 426., 420.)):
    output = []
    for block, order in zip(("A", "B"), p.ORDERS[boot]):
        seeds = p.GRAPH_SEEDS[:5] if block == "A" else p.GRAPH_SEEDS[5:]
        for variant in order:
            for index, seed in enumerate(seeds):
                output.append(dict(phase="timing", variant=variant, block=block,
                    index=index, seed=seed, correct=True, diagnostic=False,
                    boot_id=str(boot), contract_sha256="c", build_ledger="l", attempt_id="a",
                    latency_ms=times[int(variant[1])]))
    return output


def candidate_rows():
    result = []
    for node, target in p.TARGETS.items():
        for key, mode in (("input_program", 1), ("resident_program", 11)):
            result.append(dict(program_id=target[key], execution_task_id=target["task_id"],
                implementation_mode=mode, visible_features=dict(zip(p.TILE_KEYS, target["tile"])),
                local_qualification_passed=True, graph_call_sites=[dict(node_id=node)],
                complete_config_entity={"entity": list(target["tile"])}))
    return result


def sample_graph():
    # Node 0 has TWO entries. Node 1 is entry 2, not entry 1.
    return dict(nodes=[{"op": "null", "name": "x"},
        {"op": "tvm_op", "attrs": {"func_name": "same", "num_outputs": "1"}, "inputs": [[0, 1, 0]]},
        {"op": "tvm_op", "attrs": {"func_name": "same", "num_outputs": "1"}, "inputs": [[1, 0, 0]]}],
        node_row_ptr=[0, 2, 3, 4], heads=[[2, 0, 0]], attrs={
            "shape": ["list_shape", [[2, 3]] * 4], "dltype": ["list_str", ["int8"] * 4],
            "device_index": ["list_int", [12] * 4], "storage_id": ["list_int", [12, 13, 23, 12]]})


def test_frozen_identities_and_threads():
    assert len(p.bind_candidates(candidate_rows())) == 2
    for field, value in (("oc_nthread", 2), ("tile_h", 99)):
        data = candidate_rows()
        data[0]["visible_features"][field] = value
        with pytest.raises(ValueError):
            p.bind_candidates(data)


def test_complete_config_and_call_site():
    data = candidate_rows()
    data[1]["complete_config_entity"] = {}
    with pytest.raises(ValueError):
        p.bind_candidates(data)
    data = candidate_rows()
    data[0]["graph_call_sites"] *= 2
    with pytest.raises(ValueError):
        p.bind_candidates(data)


def test_entry_mapping_and_repeated_function():
    inv = p.graph_inventory(sample_graph())
    assert [r["node"] for r in inv] == [1, 2]
    assert inv[0]["outputs"][0]["entry_id"] == 2
    assert inv[0]["inputs"][0]["entry_id"] == 1
    assert inv[0]["outputs"][0]["strides_elements"] == [3, 1]
    with pytest.raises(ValueError):
        p.tensor_contract(sample_graph(), 1, 1)


def test_prefix_does_not_rewrite_lifetimes_or_original():
    graph = sample_graph()
    snapshot = copy.deepcopy(graph)
    prefix = p.prefix_graph(graph, 1, [[1, 0, 0]])
    assert prefix["nodes"][2]["attrs"]["func_name"] == "__nop"
    assert prefix["attrs"] == graph["attrs"]
    assert graph == snapshot
    assert prefix["attrs"]["storage_id"][1][-1] == 12


def test_exact_order_counts_formula_and_gate():
    for boot in (1, 2):
        data = rows(boot)
        result = p.evaluation(data, boot)
        assert len(data) == 30 and result["request_second_boot"]
        assert result["statistics"]["G2"]["equivalent_fps"] == 1000 / 420
        assert result["gains_pct"]["G0"] == pytest.approx(100 * (1 - 420 / 429))
    assert not p.evaluation(rows(times=(429, 427, 426)), 1)["request_second_boot"]
    assert not p.evaluation(rows(times=(429, 419, 420)), 1)["request_second_boot"]


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), True])
def test_invalid_latency_is_never_zero_imputed(value):
    data = rows()
    data[0]["latency_ms"] = value
    with pytest.raises(ValueError):
        p.evaluation(data, 1)


@pytest.mark.parametrize("kind", ["diagnostic", "incorrect", "missing", "duplicate", "order", "seed"])
def test_invalid_sessions_fail_closed(kind):
    data = rows()
    if kind == "diagnostic": data[0]["diagnostic"] = True
    if kind == "incorrect": data[0]["correct"] = False
    if kind == "missing": data.pop()
    if kind == "duplicate": data.append(data[0])
    if kind == "order": data.reverse()
    if kind == "seed": data[0]["seed"] = 42
    with pytest.raises(ValueError):
        p.evaluation(data, 1)


def test_cross_boot_requires_identities_and_validity():
    first = dict(boot_id="a", contract_sha256="x", build_ledger="y", status="GRAPH_COMPLETE",
                 runtime_contract_sha256="r", helper_sha256="h",
                 all_correct=True, evaluation=p.evaluation(rows(), 1))
    second = {**first, "boot_id": "b", "evaluation": p.evaluation(rows(2), 2)}
    assert p.cross_boot(first, second)["practical_residency_increment"]
    for field, value in (("boot_id", "a"), ("build_ledger", "z"), ("contract_sha256", "z"),
                         ("all_correct", False), ("status", "INVALID")):
        with pytest.raises(ValueError):
            p.cross_boot(first, {**second, field: value})


def test_sealed_artifacts_are_immutable(tmp_path):
    p.write_new(tmp_path / "x.json", {"a": 1})
    with pytest.raises(FileExistsError):
        p.write_new(tmp_path / "x.json", {})
    p.seal(tmp_path)
    assert p.verify(tmp_path)
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ValueError):
        p.verify(tmp_path)


def test_board_paused_before_import_or_contact(tmp_path):
    with pytest.raises(SystemExit, match="BOARD_PAUSED"):
        runner.main(["profile", "--prepared", str(tmp_path), "--runtime-contract", "missing",
                     "--helper-build-dir", "missing", "--completion-dir", "missing",
                     "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_observer_has_no_sync_or_debug_executor_api():
    source = (Path(__file__).parent / "c3_yolo224_graph_observer.cc").read_text()
    for forbidden in ("VTASynchronize", "VTADeviceRun", "printf(", "CopyTo(", "time_evaluator"):
        assert forbidden not in source
    assert "original.CallPacked(args, rv)" in source
    backend = (Path(__file__).parent / "c3_yolo224_incremental_board.py").read_text()
    for forbidden in ("debug_executor", "debug_get_output", "run_individual", "time_evaluator"):
        assert forbidden not in backend


def test_mixed_attempts_cannot_be_spliced():
    for field in ("boot_id", "contract_sha256", "build_ledger", "attempt_id"):
        data = rows()
        data[-1][field] = "other"
        with pytest.raises(ValueError, match="identity"):
            p.evaluation(data, 1)


def test_nested_historical_binary_manifest(tmp_path):
    (tmp_path / "binaries").mkdir()
    (tmp_path / "binaries/a.so").write_bytes(b"test")
    p.write_new(tmp_path / "artifact_hashes.json", {"artifacts": {"binaries": {"a.so": p.sha(tmp_path / "binaries/a.so")}}})
    assert p.verify(tmp_path)


def test_actual_archived_full_fusion_semantics():
    directory = runner.GROUPED / "20260926_c3_mode11_graph_context_contract_v61_run03/incumbent"
    if not directory.exists():
        pytest.skip("archived fusion fixture not available")
    graph = p.read(directory / "graph.json")
    abi = next(row for row in p.graph_inventory(graph) if row["node"] == 61)
    text = (directory / "relay_after_fuse_ops_00.txt").read_text()
    result = p.fusion_semantics([text], abi)
    assert result["shift"] == 6 and result["clip"] == [-127, 127]
    assert result["add_dtype"] == "int32"
    assert result["add_broadcast_indices"] == ["output_axis_1", "0", "0", "0", "output_axis_5"]
    for bad in (text.replace("right_shift(", "divide("), text.replace('out_dtype="int32"', 'out_dtype="float32"')):
        with pytest.raises(ValueError):
            p.fusion_semantics([bad], abi)


def test_static_address_boundaries_and_unknown_evidence():
    from c3_thread_sync_audit import bounded, CAPACITY
    for domain, capacity in CAPACITY.items():
        bounded(0, capacity, capacity, domain)
        with pytest.raises(ValueError): bounded(1, capacity, capacity, domain)
        with pytest.raises(ValueError): bounded(-1, 1, capacity, domain)


def test_output_sentinel_proves_all_cells_changed():
    import numpy as np
    from c3_yolo224_incremental_board import output_sentinel, equal
    reference = np.array([-127, -113, 0, 112, 127], dtype="int8")
    sentinel = output_sentinel(reference)
    assert np.all(sentinel != reference) and sentinel.dtype == reference.dtype
    assert equal([reference], [reference.copy()])
    assert not equal([reference], [reference.astype("int32")])
    assert not equal([reference], [])


@pytest.mark.parametrize("boot", [1, 2])
def test_board_graph_loop_with_fake_transport_only(boot):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from c3_yolo224_incremental_board import graph

    class FakeRunner:
        def __init__(self):
            self.args = SimpleNamespace(boot_index=boot)
            self.rows, self.sessions = [], []

        @contextmanager
        def session(self, variant):
            yield None
            self.sessions.append({"helper_counts": {"graph_calls": 5}})

        def call(self, ctx, seed, phase, variant, block=None, index=0, **kwargs):
            self.rows.append(dict(phase=phase, variant=variant, block=block, index=index,
                seed=seed, correct=True, diagnostic=False, latency_ms=420 - int(variant[1]) if phase == "timing" else None,
                boot_id=str(boot), contract_sha256="c", build_ledger="l", attempt_id="a"))

        def references(self):
            for seed in p.SEEDS:
                self.rows.append(dict(phase="qualification"))
            return {s: None for s in p.SEEDS}

        def finish(self, status, **kwargs):
            self.result = dict(status=status, **kwargs)

    fake = FakeRunner()
    graph(fake)
    assert fake.result["status"] == "GRAPH_COMPLETE"
    assert len(fake.rows) == 48
    assert fake.result["phase_counts"] == {"timing": 30, "warmup": 6, "qualification": 12}
