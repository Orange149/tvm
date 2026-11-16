"""Local-only guards for the bounded paper-pattern adaptation."""

from pathlib import Path

import pytest
import tvm
from tvm import tir

from c3_banerjee_local import (author_namespace, complete_compute, BanerjeeNode24Reuse,
                              SpecializeNode24Contexts, private_author_schedule)

ROOT = Path(__file__).parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261012_c3_banerjee_double_buffer_local_run01"


def snapshot(name):
    return tvm.ir.load_json((ROOT / "local/ordinary_attempt03" / name).read_text())


def test_author_schedule_loaded_privately_and_configuration_exact():
    from vta.top import vta_conv2d
    original = vta_conv2d._schedule_conv2d_packed_impl
    namespace = author_namespace(ROOT / "author_sources")
    _, tensors, contract = complete_compute(namespace)
    assert original is vta_conv2d._schedule_conv2d_packed_impl
    assert len(tensors) == 4 and contract["buffers"][2]["shape"] == [16, 1, 1, 1, 16]
    assert contract["tile"] == (1, 2, 14, 1, 1) and contract["oc_nthread"] == 2


def test_source_identity_drift_is_rejected(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        author_namespace(tmp_path)


def test_common_hazard_adaptation_is_explicit_and_not_the_default():
    ordinary = author_namespace(ROOT / "author_sources")
    safe = author_namespace(ROOT / "author_sources", safe_uop_order=True)
    assert not ordinary["shared_safe_uop_order"] and safe["shared_safe_uop_order"]
    _, _, contract = complete_compute(safe)
    assert contract["shared_safe_uop_order"]
    assert contract["tile"] == (1, 2, 14, 1, 1)


def test_paper_pattern_applies_to_exact_two_contexts():
    old = snapshot("011_after_tir_InjectVirtualThread.json")
    new = BanerjeeNode24Reuse()(old)
    function = next(iter(new.functions.values()))
    assert int(function.attrs["c3_banerjee_functional_adaptation"]) == 1
    calls = []
    tir.stmt_functor.post_order_visit(function.body, lambda n: calls.append(n)
        if isinstance(n, tir.Call) and n.op.name == "tir.call_extern" and n.args[0].value == "VTALoadBuffer2D" else None)
    assert sum(int(c.args[-1]) == 2 for c in calls) == 1
    assert sum(int(c.args[-1]) == 1 for c in calls) == 2
    assert sum(int(c.args[-1]) == 3 for c in calls) == 2
    assert not any("VTASetDebugMode" in str(c) or "VTASynchronize" in str(c) for c in calls)


def test_inapplicable_geometry_is_not_silently_changed():
    module = snapshot("011_after_tir_InjectVirtualThread.json")
    function = next(iter(module.functions.values()))
    def post(node):
        if isinstance(node, tir.For) and str(node.loop_var) == "k_o_outer":
            return tir.For(node.loop_var, node.min, 4, node.kind, node.body)
    changed = function.with_body(tir.stmt_functor.ir_transform(function.body, None, post))
    with pytest.raises((ValueError, tvm.error.TVMError), match="reduction changed"):
        BanerjeeNode24Reuse()(tvm.IRModule({"main": changed}))


def test_repeated_adaptation_is_not_silently_reapplied():
    once = BanerjeeNode24Reuse()(snapshot("011_after_tir_InjectVirtualThread.json"))
    with pytest.raises((ValueError, tvm.error.TVMError)):
        BanerjeeNode24Reuse()(once)


def test_context_specialisation_is_shared_without_changing_loads():
    old = snapshot("011_after_tir_InjectVirtualThread.json")
    common = SpecializeNode24Contexts()(old)
    function = next(iter(common.functions.values()))
    loops, calls = [], []
    tir.stmt_functor.post_order_visit(function.body, lambda n: loops.append(n) if isinstance(n, tir.For) else None)
    assert not any(str(n.loop_var).startswith("cthread") for n in loops)
    tir.stmt_functor.post_order_visit(function.body, lambda n: calls.append(n)
        if isinstance(n, tir.Call) and n.op.name == "tir.call_extern" and n.args[0].value == "VTALoadBuffer2D" else None)
    assert sum(int(c.args[-1]) == 2 for c in calls) == 2
    reuse = BanerjeeNode24Reuse()(common)
    assert int(next(iter(reuse.functions.values())).attrs["c3_common_context_specialisation"]) > 0


def test_private_worker_hook_is_restored_on_exception():
    from vta.top import vta_conv2d
    original = vta_conv2d._schedule_conv2d_packed_impl
    with pytest.raises(RuntimeError, match="intentional"):
        with private_author_schedule(author_namespace(ROOT / "author_sources")):
            assert vta_conv2d._schedule_conv2d_packed_impl is not original
            raise RuntimeError("intentional")
    assert vta_conv2d._schedule_conv2d_packed_impl is original


def native_part():
    import re
    text = (ROOT / "local/reuse_relay_fsim02/native.log").read_text()
    return re.search(r"C3_NATIVE_CALL_BEGIN 1\n(.*?)C3_NATIVE_CALL_END 1\n", text, re.S)[1]


def test_actual_native_lifetimes_have_shared_input_and_safe_overwrites():
    from audit_c3_banerjee_local import native_certificate
    proof = native_certificate(native_part(), True)
    assert proof["native_counts"]["INP_LOAD"] == 448
    assert proof["proved_overwrite_edges"] == {"INP": 446, "WGT": 894}
    assert proof["same_input_shared_by_two_MACs"]


def test_changed_native_slot_is_rejected():
    from audit_c3_banerjee_local import native_certificate
    part = native_part().replace("SRAM:0x00000040", "SRAM:0x00000000", 1)
    # Dump's address width is stable but verify mutation before testing it.
    if part == native_part():
        import re
        part = re.sub(r"(SRAM:0x)0*40\b", r"\g<1>00000000", native_part(), count=1)
    assert part != native_part()
    with pytest.raises(ValueError, match="physical INP slot"):
        native_certificate(part, True)


def test_invalid_native_token_does_not_pass_lifetime_audit():
    from audit_c3_banerjee_local import native_certificate
    part = native_part().replace("dep - pop prev: 1", "dep - pop prev: 0", 1)
    assert part != native_part()
    with pytest.raises(ValueError):
        native_certificate(part, True)


def test_manifest_checks_json_values_not_python_tuple_representation():
    from seal_c3_banerjee_local import canonical_payload
    assert canonical_payload({"ranges": [(0, 28)], "bytes": 1832}) == {"ranges": [[0, 28]], "bytes": 1832}
    assert canonical_payload({"ranges": [(0, 29)], "bytes": 1832}) != {"ranges": [[0, 28]], "bytes": 1832}


def test_manifest_normalises_only_directory_representation(tmp_path):
    from seal_c3_banerjee_local import canonical_payload
    a = {"programs": {"x": {"directory": str(tmp_path / "a" / ".." / "b"), "bytes": 1832}}}
    b = {"programs": {"x": {"directory": str(tmp_path / "b"), "bytes": 1832}}}
    assert canonical_payload(a) == canonical_payload(b)
    b["programs"]["x"]["bytes"] += 1
    assert canonical_payload(a) != canonical_payload(b)
