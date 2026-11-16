"""Read-only tests of the new fixed program; never connects to FPGA."""

import pytest
from pathlib import Path
import tvm
from tvm import tir

from c3_five_tradeoff_queue import certify_stream_window
from c3_five_tradeoff_prefetch import Node24TwoSlotPrefetch
HERE = Path(__file__).resolve().parent


def test_all_stream_interleavings_and_overlap_are_explicit():
    result = certify_stream_window()
    assert result["all_interleavings_no_RAW_WAR_or_circular_wait"]
    assert [13, 12] in result["possible_independent_load_mac_pairs"]
    assert [14, 13] in result["possible_independent_load_mac_pairs"]


def test_single_slot_cannot_be_mislabelled_as_prefetch():
    with pytest.raises(ValueError, match="overwrite|overwritten|active"):
        certify_stream_window(mapping={g: 12 for g in range(12, 16)})


def test_wrong_release_dependency_detects_circular_wait():
    with pytest.raises(ValueError, match="FIFO circular wait"):
        certify_stream_window(release_groups=(14, 15))


def anchor():
    path = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261012_c3_residency_instruction_tradeoff_run01/local/R12-G/node24_gemm.tir.json"
    return tvm.ir.load_json(path.read_text())


def test_missing_actual_allocation_certificate_fails_closed():
    with pytest.raises(ValueError, match="12 resident"):
        Node24TwoSlotPrefetch()(tvm.IRModule({"main": anchor()}))


def test_fixed_transform_preserves_actual_compute_order_and_loads_ahead():
    old = anchor().with_attr("c3_prefetch_allocation_bytes", 14 * 18432)
    new = Node24TwoSlotPrefetch()(tvm.IRModule({"main": old}))["main"]
    groups = []
    tir.stmt_functor.post_order_visit(new.body, lambda n: groups.append(n)
        if isinstance(n, tir.For) and str(n.loop_var) == "c_o" else None)
    assert len(groups) == 1 and int(groups[0].extent) == 12
    text = new.script()
    assert "VTALoadBuffer2D" in text
    # Exact mathematical/address equivalence is checked by the complete
    # final address audit, not asserted from this textual smoke test.


def test_changed_geometry_is_rejected():
    old = anchor().with_attr("c3_prefetch_allocation_bytes", 14 * 18432)
    def post(node):
        if isinstance(node, tir.For) and str(node.loop_var) == "c_o":
            return tir.For(node.loop_var, node.min, 15, node.kind, node.body)
    changed = old.with_body(tir.stmt_functor.ir_transform(old.body, None, post))
    with pytest.raises(ValueError, match="geometry"):
        Node24TwoSlotPrefetch()(tvm.IRModule({"main": changed}))


def test_native_queue_circular_wait_is_rejected():
    from c3_five_tradeoff_commands import completion_dag
    records = [{"stage": stage, "bits": bits} for stage, bits in (
        (1, (0, 1, 0, 0)), (1, (0, 0, 0, 1)),
        (2, (1, 0, 0, 0)), (2, (0, 0, 1, 0)),
    )]
    with pytest.raises(ValueError, match="circular wait"):
        completion_dag(records)


def test_unmatched_native_tokens_are_not_ignored():
    from c3_five_tradeoff_commands import completion_dag
    with pytest.raises(ValueError, match="unmatched"):
        completion_dag([{"stage": 1, "bits": (0, 0, 0, 1)}])


def test_actual_native_prefetch_all_four_invocations():
    from c3_five_tradeoff_commands import certify_prefetch_dump
    path = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261012_c3_five_tradeoff_selection_run01/local/prefetch_command_diagnostic02.log"
    result = certify_prefetch_dump(path.read_text())
    assert len(result["invocations"]) == 4
    for proof in result["invocations"]:
        assert len(proof["independent_windows"]) == 21
        assert len(proof["overwrite_edges"]) == 26
        assert proof["INP_ACC_spatial_lifetimes_and_Compute_FIFO_UOP_protection"]


def test_normal_mode_check_rejects_unknown_and_force_serial():
    from audit_c3_five_tradeoff_local import require_normal_debug_modes
    require_normal_debug_modes({"debug_mode_transitions": []})
    for bad in ({}, {"debug_mode_transitions": [32, 0]}):
        with pytest.raises(ValueError, match="mode transitions"):
            require_normal_debug_modes(bad)
