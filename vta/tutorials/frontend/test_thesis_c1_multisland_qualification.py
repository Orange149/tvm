"""Synthetic static-counter checks; no board or candidate performance evidence."""

import pytest

from prepare_thesis_c1_multisland_qualification import PROBES, dma_calls


def test_probes_cover_the_same_graph_with_two_and_three_distinct_workers():
    for islands, ranges in zip((2, 3), PROBES.values()):
        assert ranges[0][0] == 0 and ranges[-1][1] == 21
        assert all(a[1] == b[0] and a[2] != b[2] for a, b in zip(ranges, ranges[1:]))
        assert sum(s[2] == "vta" for s in ranges) == islands
        assert all(3 <= begin < end <= 20 for begin, end, device in ranges if device == "vta")


def test_final_tir_counts_host_dma_loops_and_rejects_conditional_or_dynamic_dma():
    tvm = pytest.importorskip("tvm")
    tir = tvm.tir
    i, j = tir.Var("i", "int32"), tir.Var("j", "int32")
    calls = tir.SeqStmt([tir.Evaluate(tir.call_extern("int32", "VTALoadBuffer2D", 0)),
                        tir.Evaluate(tir.call_extern("int32", "VTAStoreBuffer2D", 0)),
                        tir.Evaluate(tir.call_extern("int32", "VTAUopLoopBegin", 99))])
    body = tir.For(i, 0, 2, 0, tir.For(j, 0, 3, 0, calls))
    module = tvm.IRModule({"main": tir.PrimFunc([], body)})
    assert dma_calls(module) == {"main": {"VTALoadBuffer2D": 6, "VTAStoreBuffer2D": 6}}
    conditional = tvm.IRModule({"main": tir.PrimFunc([], tir.IfThenElse(i > 0, calls, None))})
    with pytest.raises(ValueError, match="conditional"):
        dma_calls(conditional)
    dynamic = tvm.IRModule({"main": tir.PrimFunc([j], tir.For(i, 0, j, 0, calls))})
    with pytest.raises(ValueError, match="nonconstant"):
        dma_calls(dynamic)
