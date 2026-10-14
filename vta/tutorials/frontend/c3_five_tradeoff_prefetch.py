"""Single fixed experiment: node24 R12 plus two alternating streaming slots.

Not installed in production build_config. Consumes the exact post-FoldUopLoop
merged node24 program. Leaves the mathematical order, input/ACC/output work,
resident first-use policy and spatial boundary bridge unchanged.
"""

from __future__ import annotations

import tvm
from tvm import tir
from vta.top.vta_staged_weight import _integer


def _loops(stmt):
    rows = []
    tir.stmt_functor.post_order_visit(stmt, lambda n: rows.append(n) if isinstance(n, tir.For) else None)
    return rows


def _strip_dependencies(stmt):
    def post(node):
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) \
                and node.value.op.name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
            return tir.Evaluate(0)
    return tir.stmt_functor.ir_transform(stmt, None, post)


def _dep(kind, source, target):
    return tir.Evaluate(tir.call_intrin("int32", "tir.vta.coproc_dep_" + kind, source, target))


def _stage(block, before=(), after=()):
    if not isinstance(block, tir.AttrStmt) or block.attr_key != "coproc_scope":
        raise ValueError("fixed prefetch requires explicit coprocessor scope")
    return tir.AttrStmt(block.node, block.attr_key, block.value,
                        tir.SeqStmt([*before, block.body, *after]), block.span)


def Node24TwoSlotPrefetch(expected_symbol=None):
    """12 fixed slots, slots 12/13 stream groups 12/13/14/15.

    L12,L13 precede MAC12. MAC12 completion releases slot12 for L14;
    MAC13 completion releases slot13 for L15. The original spatial bridge
    closes all last-tile compute before next input LOAD, hence before reuse
    of both streaming slots in the next spatial tile. No extra drain/flush.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.Node24TwoSlotPrefetch")
    def transform(function, _module, _context):
        if expected_symbol is not None and str(function.attrs["global_symbol"]) != expected_symbol:
            return function
        groups = [n for n in _loops(function.body) if str(n.loop_var) == "c_o"]
        spatial = [n for n in _loops(function.body) if str(n.loop_var) == "ax2_outer"]
        if len(groups) != 1 or len(spatial) != 1 or \
                _integer(groups[0].extent, "CO") != 16 or _integer(spatial[0].extent, "spatial") != 7:
            raise ValueError("fixed prefetch geometry changed")
        group = groups[0]
        if not isinstance(group.body, tir.SeqStmt) or len(group.body.seq) != 4:
            raise ValueError("expected reset/group LOAD/merged MAC/final pop")
        pieces = list(group.body.seq)
        # EarlyRewrite has already converted pointers to SRAM element indices
        # and removed Allocate. The marker is produced only after inspecting
        # the actual byte arena at the immediately preceding slots stage.
        if function.attrs is None or "c3_prefetch_allocation_bytes" not in function.attrs or \
                int(function.attrs["c3_prefetch_allocation_bytes"]) != 14 * 18432:
            raise ValueError("prefetch needs exactly 12 resident + 2 stream slots")

        stream = {}
        for g in range(12, 16):
            chosen = [tir.stmt_functor.substitute(x, {group.loop_var: tir.IntImm("int32", g)})
                      for x in pieces[:3]]
            counts = {"load": 0, "uop": 0}
            delta = ((g - 12) % 2) * 72

            def remap(node):
                if not isinstance(node, tir.Call):
                    return None
                if node.op.name == "tir.call_extern" and isinstance(node.args[0], tir.StringImm) \
                        and node.args[0].value == "VTALoadBuffer2D":
                    args = list(node.args)
                    if _integer(args[3], "stream WGT source") != g * 72 or \
                            tuple(_integer(args[i], "group DMA") for i in (4, 5, 6)) != (9, 8, 9) or \
                            _integer(args[11], "old stream destination") != 12 * 72:
                        raise ValueError("not fixed R12 group DMA")
                    args[11] = args[11] + delta
                    counts["load"] += 1
                    return tir.Call(node.dtype, node.op, args, node.span)
                if node.op.name == "tir.vta.uop_push" and _integer(node.args[0], "mode") == 0 \
                        and _integer(node.args[1], "reset") == 0:
                    args = list(node.args)
                    if not 12 * 72 <= _integer(args[4], "WGT UOP base") < 13 * 72:
                        raise ValueError("prefetch requires the frozen stream UOP range")
                    counts["uop"] += 1
                    args[4] = args[4] + delta
                    return tir.Call(node.dtype, node.op, args, node.span)
                return None

            stream[g] = [tir.stmt_functor.ir_transform(_strip_dependencies(x), None, remap)
                         for x in chosen]
            if counts != {"load": 1, "uop": 8}:
                raise ValueError("stream kernel/LOAD shape changed: " + str(counts))

        resident = tir.For(group.loop_var, group.min, 12, group.kind, group.body,
                           group.thread_binding, group.annotations, group.span)
        def reset(g):
            return stream[g][0]
        def load(g, reused=False):
            return _stage(stream[g][1], before=[_dep("pop", 2, 1)] if reused else [],
                          after=[_dep("push", 1, 2)])
        def mac(g, release=False):
            return _stage(stream[g][2], before=[_dep("pop", 1, 2)],
                          after=[_dep("push", 2, 1)] if release else [])

        replacement = tir.SeqStmt([
            resident,
            reset(12), load(12), load(13), mac(12, release=True),
            reset(13), load(14, reused=True), mac(13, release=True),
            reset(14), load(15, reused=True), mac(14), reset(15), mac(15),
        ])
        body = tir.stmt_functor.ir_transform(function.body, None,
            lambda n: replacement if isinstance(n, tir.For) and n.same_as(group) else None)
        return function.with_body(body)

    return transform


def make_probe():
    """Reuse hash-bound full fusion builder, without modifying its source."""
    import run_vta_c3_node24_partial_local as base
    from vta.top import vta_node24_partial as passes
    from c3_residency_tradeoff import digest
    from c3_residency_tradeoff_local import (BASE_PASS_SHA, BASE_BUILDER_SHA,
                                            compile_factory)
    if digest(base.__file__) != BASE_BUILDER_SHA or digest(passes.__file__) != BASE_PASS_SHA:
        raise ValueError("historical builder/pass binding drift")
    base_slots = compile_factory(passes.Node24WeightSlots, [
        ("arena_bytes = (resident_groups + 1) * group_bytes",
         "arena_bytes = (resident_groups + 2) * group_bytes"),
    ], dict(passes.__dict__))
    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.CertifyNode24PrefetchArena")
    def mark_arena(function, _module, _context):
        arenas = []
        tir.stmt_functor.post_order_visit(function.body, lambda n: arenas.append(n)
            if isinstance(n, tir.Allocate) and
            str(getattr(n.buffer_var.type_annotation, "storage_scope", "")) == "local.wgt_buffer" else None)
        if len(arenas) != 1 or [_integer(x, "WGT arena") for x in arenas[0].extents] != [14 * 18432]:
            raise ValueError("actual allocated WGT arena is not 12 + 2 slots")
        return function.with_attr("c3_prefetch_allocation_bytes", 14 * 18432)
    def slots(*args, **kwargs):
        return tvm.transform.Sequential([base_slots(*args, **kwargs), mark_arena])
    dma = compile_factory(passes.Node24GroupWeightDMA, [
        ("slot = tir.Select(group.loop_var < 13, group.loop_var, tir.IntImm(\"int32\", 13))",
         "slot = tir.Select(group.loop_var < 12, group.loop_var, tir.IntImm(\"int32\", 12))"),
    ], dict(passes.__dict__))
    def merged(expected_symbol=None):
        return tvm.transform.Sequential([passes.Node24GroupGEMM(expected_symbol),
                                         Node24TwoSlotPrefetch(expected_symbol)])
    namespace = dict(base.__dict__)
    namespace.update(Node24WeightSlots=slots, Node24GroupWeightDMA=dma, Node24GroupGEMM=merged)
    return compile_factory(base.probe, [
        ("resident_groups != 13", "resident_groups != 12"),
        ("actual = runner.get_output(0).numpy()", "actual = runner.get_output(0).numpy()\n"
         "                device_unchanged = all(np.array_equal(runner.get_input(n).numpy(), v) "
         "for n, v in zip(('p015', 'p17', 'p26'), (x, w, add)))"),
        ("\"arguments_unchanged\": before == [sha(value.tobytes()) for value in (x, w, add)]",
         "\"arguments_unchanged\": device_unchanged and "
         "before == [sha(value.tobytes()) for value in (x, w, add)]"),
    ], namespace)
