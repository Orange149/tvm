"""Semantic regression for call closure and author sharing adaptation."""
from pathlib import Path
import re

import pytest
import tvm
from tvm import tir

from c3_banerjee_baseline import GROUPED
from c3_banerjee_completion import CloseCallBoundary, ImprovedDoubleBuffer
from c3_banerjee_local import BanerjeeNode24Reuse

V1 = GROUPED / "20261013_c3_banerjee_baseline_completion_run01"
OLD = GROUPED / "20261012_c3_banerjee_double_buffer_local_run01"


def calls(module):
    result = []
    for f in module.functions.values():
        tir.stmt_functor.post_order_visit(f.body, lambda n: result.append(n) if isinstance(n, tir.Call) else None)
    return result


def test_boundary_retains_all_math_and_tokens_and_moves_one_terminal_pop():
    source = V1 / "tps_single_thread_attempt03/015_after_tir_vta_CPUAccessRewrite.json"
    old = tvm.ir.load_json(source.read_text())
    new = CloseCallBoundary()(old)
    a, b = calls(old), calls(new)
    assert len(a) == len(b)
    assert sorted(map(str, a)) == sorted(map(str, b))
    seq_a = [c for c in a if c.op.name.startswith("tir.vta.coproc")]
    seq_b = [c for c in b if c.op.name.startswith("tir.vta.coproc")]
    assert seq_a[-2].op.name.endswith("coproc_sync")
    assert seq_a[-1].op.name.endswith("coproc_dep_pop")
    assert seq_b[-2].op.name.endswith("coproc_dep_pop")
    assert seq_b[-1].op.name.endswith("coproc_sync")
    # Idempotent body, only bookkeeping attr changes 1 -> 0 on a second call.
    assert tvm.ir.structural_equal(next(iter(new.functions.values())).body,
        next(iter(CloseCallBoundary()(new).functions.values())).body, map_free_vars=True)


def test_nonterminal_post_sync_pop_is_rejected():
    sync = tir.Evaluate(tir.call_intrin("int32", "tir.vta.coproc_sync"))
    pop = tir.Evaluate(tir.call_intrin("int32", "tir.vta.coproc_dep_pop", 3, 2))
    f = tir.PrimFunc([], tir.SeqStmt([sync, pop, tir.Evaluate(42)]))
    with pytest.raises((ValueError, tvm.error.TVMError), match="terminal"):
        CloseCallBoundary()(tvm.IRModule({"main": f}))


def test_general_input_reuse_reproduces_old_fixed_node24_body():
    source = OLD / "local/ordinary_attempt03/011_after_tir_InjectVirtualThread.json"
    old = tvm.ir.load_json(source.read_text())
    fixed, general = BanerjeeNode24Reuse()(old), ImprovedDoubleBuffer()(old)
    # The old walker retains expressions such as 0*bank; compare simplified
    # bodies, without discarding instructions, addresses or dependencies.
    fixed = tir.transform.Simplify()(fixed)
    general = tir.transform.Simplify()(general)
    assert tvm.ir.structural_equal(next(iter(fixed.functions.values())).body,
                                  next(iter(general.functions.values())).body, map_free_vars=True)
    assert int(next(iter(general.functions.values())).attrs["c3_shared_domain"]) == 2
    assert int(next(iter(general.functions.values())).attrs["c3_shared_bank_elements"]) == 1024


def test_spatial_threading_shares_weight_not_input():
    root = GROUPED / "20261013_c3_banerjee_baseline_completion_run02"
    source = root / "local/TPS-h-reuse/011_after_tir_InjectVirtualThread.json"
    f = next(iter(ImprovedDoubleBuffer()(tvm.ir.load_json(source.read_text())).functions.values()))
    assert int(f.attrs["c3_shared_domain"]) == 1
    assert int(f.attrs["c3_shared_bank_elements"]) == 18432
    actual = calls(tvm.IRModule({"main": f}))
    loads = [c for c in actual if c.op.name == "tir.call_extern" and c.args[0].value == "VTALoadBuffer2D"]
    assert sum(int(c.args[-1]) == 1 for c in loads) == 1
    assert sum(int(c.args[-1]) == 2 for c in loads) == 2
