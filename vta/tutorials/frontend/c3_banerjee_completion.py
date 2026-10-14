"""Experimental Banerjee baseline adapters and call-boundary certificates.

Production passes and historical artifacts remain untouched. The boundary
adapter consumes the compiler's loop-exit token in its own submission, before
the existing full sync, instead of enqueueing it after that sync.
"""

import tvm
from tvm import tir


def _intrinsic(stmt, name):
    return (isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call)
            and stmt.value.op.name == "tir.vta." + name)


def CloseCallBoundary():
    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.CloseCallBoundary")
    def transform(f, _mod, _ctx):
        changes = []
        def post(node):
            if not isinstance(node, tir.SeqStmt):
                return None
            seq = list(node.seq)
            for i in range(len(seq) - 1):
                if _intrinsic(seq[i], "coproc_sync") and _intrinsic(seq[i + 1], "coproc_dep_pop"):
                    pop = seq[i + 1]
                    if [int(a) for a in pop.value.args] != [3, 2]:
                        raise ValueError("unexpected post-sync dependence; cannot apply bounded repair")
                    if i + 2 != len(seq):
                        raise ValueError("post-sync dependence is not a terminal loop-exit pop")
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
                    changes.append([3, 2])
            return tir.SeqStmt(seq)
        body = tir.stmt_functor.ir_transform(f.body, None, post, ["tir.SeqStmt"])
        return f.with_body(body).with_attr("c3_boundary_pop_before_sync", len(changes))
    return transform


def SplitPaddedInputChannels():
    """Expose each packed CI slice as a legal padded 2-D DMA, same order."""
    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.SplitPaddedInputChannels")
    def transform(f, _mod, _ctx):
        matched = []
        def pre(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "pragma_dma_copy" and isinstance(node.body, tir.For):
                loop = node.body
                if str(loop.loop_var) == "i1" and int(loop.extent) in (2, 4, 8):
                    if matched:
                        raise ValueError("padded input channel-copy scope is ambiguous")
                    matched.append(int(loop.extent))
                    inside = tir.AttrStmt(node.node, node.attr_key, node.value, loop.body, node.span)
                    return tir.For(loop.loop_var, loop.min, loop.extent, loop.kind, inside,
                                   loop.thread_binding, loop.annotations, loop.span)
        body = tir.stmt_functor.ir_transform(f.body, pre, None)
        return f.with_body(body).with_attr("c3_padded_channel_split", matched[0] if matched else 0)
    return transform


def ImprovedDoubleBuffer():
    """Reconstruct author's dthread sharing for two lanes, using local TIR.

    Detect which LOAD is identical across lanes, remove the second copy, and
    remap both MACs to a reduction-index alternating bank. Supports output
    channel (shared INP) and spatial-height (shared WGT) threading. ACC and
    original queue dependencies are preserved. No GEMM kernels are merged.
    """
    analyzer = tvm.arith.Analyzer()

    def simplify(expr):
        return analyzer.simplify(expr)

    def load_calls(stmt):
        found = []
        tir.stmt_functor.post_order_visit(stmt, lambda n: found.append(n)
            if isinstance(n, tir.Call) and n.op.name == "tir.call_extern"
            and n.args[0].value == "VTALoadBuffer2D" and int(n.args[-1]) in (1, 2) else None)
        return found

    @tir.transform.prim_func_pass(opt_level=0, name="tir.c3.ImprovedDoubleBuffer")
    def transform(f, _mod, _ctx):
        reductions, threads = [], []
        tir.stmt_functor.post_order_visit(f.body, lambda n: reductions.append(n)
            if isinstance(n, tir.For) and str(n.loop_var) == "k_o_outer" else None)
        tir.stmt_functor.post_order_visit(f.body, lambda n: threads.append(n)
            if isinstance(n, tir.For) and str(n.loop_var).startswith("cthread") else None)
        if len(reductions) != 1 or not threads:
            raise ValueError("expected one frozen convolution reduction and two-context loops")
        reduction = reductions[0].loop_var
        shared = []
        for thread in threads:
            if int(thread.extent) != 2 or int(thread.min) != 0:
                raise ValueError("double buffering requires two virtual contexts")
        def unroll(node):
            if isinstance(node, tir.For) and str(node.loop_var).startswith("cthread"):
                return tir.SeqStmt([tir.stmt_functor.substitute(node.body,
                    {node.loop_var: tir.IntImm("int32", i)}) for i in (0, 1)])
        # Some DMA contexts remain loops when their CI copy has a nested loop.
        body = tir.stmt_functor.ir_transform(f.body, None, unroll, ["tir.For"])
        # Stock injection already expands the DMA lanes, while MAC/ALU host
        # contexts remain loops. Pair actual DMA calls by memory domain.
        loads = load_calls(body)
        for domain in (1, 2):
            pair = [call for call in loads if int(call.args[-1]) == domain]
            if len(pair) != 2:
                raise ValueError("expected exactly two static DMA lanes per operand")
            a, b = pair
            # Dest is a banked pointer, all other DMA fields must agree.
            if all(tvm.ir.structural_equal(simplify(x), simplify(y), map_free_vars=False)
                   for i, (x, y) in enumerate(zip(a.args, b.args)) if i != 11):
                pa, pb = a.args[11], b.args[11]
                if not (isinstance(pa, tir.Call) and isinstance(pb, tir.Call)
                        and pa.op.name == pb.op.name == "tir.tvm_access_ptr"):
                    raise ValueError("sharing must precede device storage lowering")
                delta = simplify(pb.args[2] - pa.args[2])
                base = simplify(pa.args[2])
                if not isinstance(delta, tir.IntImm) or int(delta) <= 0:
                    raise ValueError("not two contiguous banks")
                shared.append((int(a.args[-1]), pa.args[1], int(delta)))
        unique = {(d, str(var), span) for d, var, span in shared}
        if len(unique) != 1:
            raise ValueError("exactly one input or weight domain must be shared")
        domain, buffer_var, bank_span = shared[0]
        changes = {"kept": 0, "deleted": 0, "remapped_mac": 0}
        def post(node):
            if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
                return None
            call = node.value
            if call.op.name == "tir.call_extern" and call.args[0].value == "VTALoadBuffer2D" and int(call.args[-1]) == domain:
                ptr = call.args[11]
                if not ptr.args[1].same_as(buffer_var):
                    raise ValueError("shared DMA buffer changed")
                offset = simplify(ptr.args[2])
                # Packed CI DMA slices carry a within-bank offset. Bound it
                # using the existing channel-copy loop, before bank remapping.
                ci_vars = []
                tir.stmt_functor.post_order_visit(offset, lambda n: ci_vars.append(n)
                    if isinstance(n, tir.Var) and str(n) == "i1" else None)
                normalized = simplify(tir.stmt_functor.substitute(offset,
                    {v: tir.IntImm("int32", 0) for v in ci_vars}))
                if not isinstance(normalized, tir.IntImm) or int(normalized) not in (0, bank_span):
                    raise ValueError("shared LOAD has unexpected bank base")
                if int(normalized) == bank_span:
                    changes["deleted"] += 1
                    return tir.Evaluate(0)
                args, pointer = list(call.args), list(ptr.args)
                pointer[2] = tir.indexmod(reduction, 2) * bank_span + tir.indexmod(pointer[2], bank_span)
                args[11] = tir.Call(ptr.dtype, ptr.op, pointer, ptr.span)
                changes["kept"] += 1
                return tir.Evaluate(tir.Call(call.dtype, call.op, args, call.span))
            if call.op.name == "tir.vta.uop_push" and [int(call.args[i]) for i in (0, 1)] == [0, 0]:
                arg_index = 3 if domain == 2 else 4
                ptr = call.args[arg_index]
                if not isinstance(ptr, tir.Call) or ptr.op.name != "tir.tvm_access_ptr" or not ptr.args[1].same_as(buffer_var):
                    raise ValueError("shared MAC operand buffer changed")
                args, pointer = list(call.args), list(ptr.args)
                pointer[2] = tir.indexmod(reduction, 2) * bank_span + tir.indexmod(pointer[2], bank_span)
                args[arg_index] = tir.Call(ptr.dtype, ptr.op, pointer, ptr.span)
                changes["remapped_mac"] += 1
                return tir.Evaluate(tir.Call(call.dtype, call.op, args, call.span))
        body = tir.stmt_functor.ir_transform(body, None, post, ["tir.Evaluate"])
        if changes != {"kept": 1, "deleted": 1, "remapped_mac": 2}:
            raise ValueError("unexpected sharing pattern: " + str(changes))
        return f.with_body(body).with_attr("c3_shared_domain", domain).with_attr("c3_shared_bank_elements", bank_span)
    return transform
