"""Local regression guards for the label-blind mechanism protocol."""
import json
from pathlib import Path
import subprocess
import sys
import copy

import pytest
import tvm
from tvm import tir
import vta

from c3_reuse_semantics import extract_requests, union_length, capture_lower
from run_vta_c3_reuse_supplement import reject_labels, write, seal, verify, dual_effect, diversity, matched_reload_control


def synthetic_dma(padding=(0, 0, 0, 0), conditional=True):
    i = tir.Var("i", "int32")
    data = tir.Var("data", "handle")
    handle = tir.Var("command", "handle")
    call = tir.Evaluate(tir.call_extern("int32", "VTALoadBuffer2D", handle, data,
                                      i * 4, 2, 2, 2, *padding, 0, 2))
    body = tir.IfThenElse(i % 2 == 0, call, tir.Evaluate(0)) if conditional else call
    loop = tir.For(i, 0, 4, tir.ForKind.SERIAL, body)
    return tvm.IRModule({"main": tir.PrimFunc([data, handle], loop)})


def test_region_union_is_overlap_safe():
    assert union_length([(0, 4), (2, 8), (12, 16), (3, 5)]) == 12
    assert union_length([]) == 0
    with pytest.raises(ValueError):
        union_length([(4, 1)])


def test_conditional_loads_are_evaluated_not_multiplied_blindly():
    result = extract_requests(synthetic_dma(), vta.get_env())
    assert result["totals"]["load_buffer_2d_calls"] == 2
    assert result["totals"]["load_buffer_2d_inp_bytes"] == 128
    assert result["unique_source_region_bytes"]["load_inp"] == 128
    assert result["load_store_sites"][0]["conditions"]


def test_padding_is_not_counted_as_ddr_source_bytes():
    result = extract_requests(synthetic_dma((1, 1, 1, 1), False), vta.get_env())
    assert result["totals"]["load_buffer_2d_inp_bytes"] == 256
    assert result["request_groups"][0]["padding"] == [1, 1, 1, 1]


def test_else_branch_and_let_bound_condition():
    module = synthetic_dma()
    function = module["main"]
    loop = function.body
    inner = loop.body
    body = tir.IfThenElse(inner.condition, inner.then_case, inner.then_case)
    result = extract_requests(tvm.IRModule({"main": function.with_body(tir.For(loop.loop_var, 0, 4, tir.ForKind.SERIAL, body))}), vta.get_env())
    assert result["totals"]["load_buffer_2d_calls"] == 4


def test_let_substitution_and_invariant_loop_multiplicity():
    module = synthetic_dma(conditional=False)
    function, j = module["main"], tir.Var("j", "int32")
    folded = tir.For(j, 0, 3, tir.ForKind.SERIAL, function.body)
    result = extract_requests(tvm.IRModule({"main": function.with_body(folded)}), vta.get_env())
    assert result["totals"]["load_buffer_2d_calls"] == 12
    assert result["unique_source_region_bytes"]["load_inp"] == 256
    assert result["reload_ratios"]["load_inp"] == 3
    assert all(row["multiplicity"] == 3 for row in result["request_groups"])
    x = tir.Var("x", "int32")
    loop = function.body
    inner = tir.LetStmt(x, loop.loop_var % 2, tir.IfThenElse(x == 0, loop.body, tir.Evaluate(0)))
    bound = tir.For(loop.loop_var, 0, 4, tir.ForKind.SERIAL, inner)
    assert extract_requests(tvm.IRModule({"main": function.with_body(bound)}), vta.get_env())["totals"]["load_buffer_2d_calls"] == 2


def test_dynamic_condition_fails_closed():
    module = synthetic_dma(conditional=False)
    function, flag = module["main"], tir.Var("flag", "int32")
    guarded = tir.IfThenElse(flag > 0, function.body, tir.Evaluate(0))
    with pytest.raises(ValueError, match="non-static"):
        extract_requests(tvm.IRModule({"main": function.with_body(guarded)}), vta.get_env())


def test_changed_sync_or_compute_scope_rejects_control():
    resident = {"status": "passed", "sync_multiplicity": [1, 1], "semantics": {
        "compute_sites": [{"loop_scope": [{"var": "k", "extent": 2}]}],
        "totals": {"load_buffer_2d_inp_bytes": 10, "load_buffer_2d_inp_calls": 1,
                   "store_buffer_2d_out_bytes": 10, "store_buffer_2d_out_calls": 1,
                   "load_buffer_2d_wgt_bytes": 10}}}
    control = json.loads(json.dumps(resident))
    control["semantics"]["totals"]["load_buffer_2d_wgt_bytes"] = 20
    assert matched_reload_control(resident, control)
    control["sync_multiplicity"] = [2, 1]
    assert not matched_reload_control(resident, control)
    control["sync_multiplicity"] = [1, 1]
    control["semantics"]["compute_sites"][0]["loop_scope"][0]["extent"] = 3
    assert not matched_reload_control(resident, control)


@pytest.mark.parametrize("key", ["latency_ms", "performance_label", "oracle_latency", "fpga_correct"])
def test_future_labels_are_rejected_recursively(key):
    with pytest.raises(ValueError, match="future label"):
        reject_labels({"nested": [{key: 1}]})
    reject_labels({key: None})


def test_artifact_mutation_and_extra_files_are_rejected(tmp_path):
    write(tmp_path / "contract.json", {"phase": "prepare"})
    seal(tmp_path)
    assert verify(tmp_path)["phase"] == "prepare"
    (tmp_path / "extra").write_text("extra")
    with pytest.raises(ValueError, match="unbound"):
        verify(tmp_path)
    (tmp_path / "extra").unlink()
    write(tmp_path / "contract.json", {"phase": "changed"})
    with pytest.raises(ValueError, match="mismatch"):
        verify(tmp_path)


def test_same_flags_with_unchanged_operand_is_not_dual_reuse():
    modes = ["original", "input_stationary", "weight_resident_barrier", "input_weight_resident_barrier"]
    rows = {mode: {"status": "passed", "semantics": {
        "reload_ratios": {"load_inp": 2, "load_wgt": 2},
        "totals": {"load_buffer_2d_inp_bytes": 200, "load_buffer_2d_wgt_bytes": 200}}} for mode in modes}
    assert dual_effect(rows) is False
    combined = rows[modes[-1]]["semantics"]["totals"]
    combined["load_buffer_2d_inp_bytes"] = 100
    assert dual_effect(rows) is False
    combined["load_buffer_2d_wgt_bytes"] = 100
    assert dual_effect(rows) is True
    rows[modes[-1]]["status"] = "not_applicable"
    assert dual_effect(rows) is None


def test_capture_is_observational_and_reload_controls_preserve_sync():
    from prepare_vta_c3_resnet18_literature_candidates import make_workload, create_task
    from run_vta_c3_resnet18_local_qualification import sync_multiplicity
    import c3_reuse_experimental_template  # noqa: F401
    import vta.top.vta_conv2d_residency  # noqa: F401
    env = vta.get_env()
    spec = dict(ci=64, co=64, height=56, width=56, kernel=3, stride=1)
    workload = make_workload(spec, env)
    for mode in (0, 1, 4, 5):
        task = create_task(workload, mode, env)
        with task.target:
            schedule, tensors = task.instantiate(task.config_space.get(252))
        captured, _, arenas = capture_lower(schedule, tensors)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            original = tvm.lower(schedule, tensors, name="main")
        assert tvm.ir.structural_equal(captured, original)
        assert arenas["allocations"]
        assert any(item["scope"] == env.wgt_scope for item in arenas["allocations"])
        if mode in (4, 5):
            control = tvm.autotvm.task.create("conv2d_packed_reuse_control.vta", args=tuple(workload[1:]) + (mode,),
                                             target=env.target, target_host=env.target_host)
            with control.target:
                control_schedule, control_tensors = control.instantiate(control.config_space.get(252))
            control_module, _, _ = capture_lower(control_schedule, control_tensors)
            assert sync_multiplicity(control_module) == sync_multiplicity(captured)
            resident_bytes = extract_requests(captured, env)["totals"]["load_buffer_2d_wgt_bytes"]
            reload_bytes = extract_requests(control_module, env)["totals"]["load_buffer_2d_wgt_bytes"]
            assert reload_bytes > resident_bytes


@pytest.mark.parametrize("mode", ["original", "input_stationary", "weight_resident_barrier", "input_weight_resident_barrier"])
def test_real_fsim_three_seeds_dma_and_warm_command_signatures(mode, tmp_path):
    from prepare_vta_c3_resnet18_literature_candidates import make_workload, candidate_row, hardware_fingerprint
    from c3_candidate_identity import normalize_config_entity
    from run_vta_c3_reuse_supplement import MODES, ROOT, instantiate, lower_candidate, fsim
    env = vta.get_env()
    spec = dict(ci=64, co=64, height=28, width=28, kernel=3, stride=1)
    workload = make_workload(spec, env)
    from prepare_vta_c3_resnet18_literature_candidates import create_task
    task = create_task(workload, 0, env)
    entity = normalize_config_entity(task.config_space.get(17).to_json_dict())
    candidate = candidate_row("test-M1", spec, workload, entity, 17, "test-M1D0017", mode,
                              "test-only-no-board", hardware_fingerprint(env), env)
    candidate["identity"] = {"workload": workload, "complete_config_entity": entity}
    static = lower_candidate(candidate, tmp_path)
    assert static["status"] == "passed"
    assert all(row["ddr_bounds_checked"] for row in static["semantics"]["request_groups"])
    result = fsim(candidate, static, tmp_path, str(ROOT / "build/libvta_fsim.so"), 60)
    assert result["status"] == "passed", result.get("failure")
    assert result["warmup"]["status"] == "passed"
    assert result["function_invocations_attempted"] == result["function_invocations_completed"] == 4
    assert len(result["seeds"]) == 3
    assert len(set(result["command_features"]["per_seed_sha256"].values())) == 1


@pytest.mark.parametrize("execute", [False, True])
def test_board_phase_is_explicitly_unavailable_without_closed_protocol(tmp_path, execute):
    import run_vta_c3_reuse_supplement as runner
    destination = tmp_path / "must_not_exist"
    command = [sys.executable, runner.__file__, "--phase", "board", "--output-dir", str(destination)]
    if execute:
        command.append("--execute-board")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "not implemented" in result.stderr
    assert not destination.exists()


def test_output_refuses_overwrite_before_any_implementation_work(tmp_path):
    import run_vta_c3_reuse_supplement as runner
    marker = tmp_path / "user_evidence"
    marker.write_text("unchanged")
    result = subprocess.run([sys.executable, runner.__file__, "--phase", "prepare", "--output-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0 and marker.read_text() == "unchanged"
    assert not (tmp_path / "artifact_hashes.json").exists()


def test_sdk_environment_cleanup_is_scoped(monkeypatch):
    from recover_vta_c3_reuse_cross import sdk_environment
    monkeypatch.setenv("LD_LIBRARY_PATH", "/task/test-library")
    import os
    clean = sdk_environment()
    assert "LD_LIBRARY_PATH" not in clean
    assert os.environ["LD_LIBRARY_PATH"] == "/task/test-library"


def test_eight_latin_orders_balance_every_mode_position():
    from freeze_vta_c3_reuse_confirmation import latin_orders
    from run_vta_c3_reuse_supplement import MODES
    orders = latin_orders()
    assert len(orders) == 8
    assert all(set(order) == set(MODES) and len(order) == 4 for order in orders)
    for position in range(4):
        assert all(sum(order[position] == mode for order in orders) == 2 for mode in MODES)


def test_changed_source_binding_rejects_execution(tmp_path):
    from run_vta_c3_reuse_supplement import check_bindings, file_hash
    source = tmp_path / "test_source.py"
    source.write_text("first version")
    bindings = {str(source): file_hash(source)}
    check_bindings(bindings)
    source.write_text("changed version")
    with pytest.raises(ValueError, match="implementation changed"):
        check_bindings(bindings)


@pytest.mark.parametrize("tamper", ["missing_dma", "wrong_output", "missing_seed", "wrong_tir", "missing_arena"])
def test_independent_verifier_rejects_missing_or_inconsistent_evidence(tamper):
    from analyze_vta_c3_reuse_local import check_candidate
    path = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20260915_c3_reuse_node_b_cross_recovery_run01/candidates.json"
    row = copy.deepcopy(json.loads(path.read_text())[0])
    assert check_candidate(row)
    if tamper == "missing_dma":
        del row["fsim_qualification"]["seeds"][0]["runtime_logical_dma"]["load_buffer_2d_inp_bytes"]
    elif tamper == "wrong_output":
        row["fsim_qualification"]["seeds"][0]["actual_sha256"] = "wrong"
    elif tamper == "missing_seed":
        row["fsim_qualification"]["seeds"].pop()
    elif tamper == "wrong_tir":
        row["cross_qualification"]["relowered_tir_sha256"] = "wrong"
    else:
        row["local_qualification"]["storage_arenas"]["allocations"] = []
    with pytest.raises(ValueError):
        check_candidate(row)
