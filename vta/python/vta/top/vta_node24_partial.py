"""Fail-closed, experiment-only normalization for node24 grouped Input DMA.

The grouped TE schedule keeps eight CI slices across output-channel groups.
Its padded input copy is emitted as one four-dimensional DMA, while the VTA
copy injector accepts padding only on the outer two DMA dimensions.  Moving
the existing CI loop outside the pragma produces eight independent padded
two-dimensional copies without changing the memory indices or arithmetic.
This pass is not installed by production ``vta.build_config``.
"""

import tvm
from tvm import tir

from .vta_staged_weight import _access_ptr_with_offset, _calls, _integer, _weight_load_info


def _matches_symbol(function, expected_symbol):
    if expected_symbol is None:
        return True
    attrs = function.attrs
    return attrs is not None and "global_symbol" in attrs and \
        str(attrs["global_symbol"]) == expected_symbol


def SplitNode24InputDMA(expected_symbol=None):
    """Move the exact node24 CI loop outside its padded DMA pragma."""

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.SplitNode24InputDMA")
    def transform(function, _module, _context):
        if not _matches_symbol(function, expected_symbol):
            return function
        matches = []

        def pre(node):
            if not isinstance(node, tir.AttrStmt) or node.attr_key != "pragma_dma_copy":
                return None
            loop = node.body
            if not isinstance(loop, tir.For):
                return None
            if str(loop.loop_var) != "i1" or int(loop.extent) not in (4, 8):
                return None
            if len(matches) != 0:
                raise ValueError("node24 input DMA is not unique")
            matches.append(True)
            narrowed = tir.AttrStmt(node.node, node.attr_key, node.value, loop.body, node.span)
            return tir.For(
                loop.loop_var,
                loop.min,
                loop.extent,
                loop.kind,
                narrowed,
                loop.thread_binding,
                loop.annotations,
                loop.span,
            )

        body = tir.stmt_functor.ir_transform(function.body, pre, None)
        if len(matches) != 1:
            raise ValueError("expected exactly one node24 grouped padded input DMA")
        return function.with_body(body)

    return transform


def SplitNode24AltOutputDMA(expected_symbol=None):
    """Keep the frozen h7w7ci4co2 Input control's OUT stores 2-D.

    Moving only its two-way CO loop outside the DMA pragma preserves store
    order and addresses.  No primary A/F candidate uses this adapter.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.SplitNode24AltOutputDMA")
    def transform(function, _module, _context):
        if not _matches_symbol(function, expected_symbol):
            return function
        matches = []

        def pre(node):
            if not isinstance(node, tir.AttrStmt) or node.attr_key != "pragma_dma_copy":
                return None
            loop = node.body
            if not isinstance(loop, tir.For) or str(loop.loop_var) != "ax1_inner" \
                    or _integer(loop.extent, "alternative OUT CO extent") != 2:
                return None
            matches.append(True)
            narrowed = tir.AttrStmt(node.node, node.attr_key, node.value, loop.body, node.span)
            return tir.For(
                loop.loop_var, loop.min, loop.extent, loop.kind, narrowed,
                loop.thread_binding, loop.annotations, loop.span,
            )

        body = tir.stmt_functor.ir_transform(function.body, pre, None)
        if len(matches) != 1:
            raise ValueError("alternative Input Tile has no unique two-way OUT DMA")
        return function.with_body(body)

    return transform


def Node24WeightSlots(resident_groups, keep_resident, expected_symbol=None,
                      group_count=16):
    """Assign fixed/streaming slots, optionally omit resident re-LOADs.

    Consumes only the exact post-EarlyRewrite node24 grouped loop nest.  The
    matched k-loop has one WGT LOAD followed by one MAC block.  Suppressed
    LOADs suppress their four matching load/compute dependency-token ops as
    well; the GEMM remains and the co-loop completion token is unchanged.
    """
    resident_groups = int(resident_groups)
    group_count = int(group_count)
    if resident_groups not in (12, 13) or group_count not in (14, 15, 16) or \
            resident_groups >= group_count:
        raise ValueError("unsupported node24 partial-residency geometry")
    keep_resident = bool(keep_resident)
    block_bytes = 2304
    group_bytes = 8 * block_bytes
    arena_bytes = (resident_groups + 1) * group_bytes

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.Node24WeightSlots")
    def transform(function, _module, _context):
        if not _matches_symbol(function, expected_symbol):
            return function
        loops = {}

        def find(node):
            if isinstance(node, tir.For):
                name = str(node.loop_var)
                if name in ("ax2_outer", "c_o", "k_o_outer"):
                    loops.setdefault(name, []).append(node)

        tir.stmt_functor.post_order_visit(function.body, find)
        if [len(loops.get(name, [])) for name in ("ax2_outer", "c_o", "k_o_outer")] != [1, 1, 1]:
            raise ValueError("node24 weight slots require one spatial/co/reduction loop")
        spatial, co, reduction = (loops[name][0] for name in ("ax2_outer", "c_o", "k_o_outer"))
        if any(_integer(loop.min, "loop min") != 0 for loop in (spatial, co, reduction)):
            raise ValueError("node24 weight slots require zero-based loops")
        if [_integer(loop.extent, "loop extent") for loop in (spatial, co, reduction)] != [7, group_count, 8]:
            raise ValueError("node24 weight slots found changed loop geometry")
        if not any(node.same_as(reduction) for node in _loop_nodes(co.body)):
            raise ValueError("reduction loop is not nested within output group")
        if not any(node.same_as(co) for node in _loop_nodes(spatial.body)):
            raise ValueError("output group is not nested within spatial loop")

        loads = _calls(reduction.body, lambda call: _weight_load_info(call) is not None)
        macs = _calls(
            reduction.body,
            lambda call: isinstance(call, tir.Call)
            and call.op.name == "tir.vta.uop_push"
            and _integer(call.args[0], "uop mode") == 0
            and _integer(call.args[1], "uop reset") == 0,
        )
        if len(loads) != 1 or len(macs) != 1:
            raise ValueError("node24 requires one WGT load and one symbolic MAC")
        load = loads[0]
        info = _weight_load_info(load)
        if (info["x_size"], info["y_size"], info["stride"]) != (9, 1, 9):
            raise ValueError("node24 WGT DMA shape changed")
        if _integer(load.args[11].args[2], "old WGT destination") != 0:
            raise ValueError("node24 expected compact WGT destination")
        if _integer(load.args[11].args[3], "old WGT DMA extent") != block_bytes:
            raise ValueError("node24 WGT DMA extent changed")
        if _integer(load.args[3] - (co.loop_var * group_bytes + reduction.loop_var * block_bytes) // 256,
                    "WGT source offset") != 0:
            raise ValueError("node24 source WGT group layout changed")
        slot = tir.Select(co.loop_var < resident_groups, co.loop_var, tir.IntImm("int32", resident_groups))
        offset = slot * group_bytes + reduction.loop_var * block_bytes
        load_required = tir.Or(co.loop_var >= resident_groups, spatial.loop_var == 0)
        count = {"load": 0, "mac": 0, "tokens": []}

        def post(node):
            if isinstance(node, tir.Call) and _weight_load_info(node) is not None:
                args = list(node.args)
                args[11] = _access_ptr_with_offset(args[11], offset)
                count["load"] += 1
                return tir.Call(node.dtype, node.op, args, node.span)
            if isinstance(node, tir.Call) and node.op.name == "tir.vta.uop_push":
                if _integer(node.args[0], "uop mode") == 0 and _integer(node.args[1], "uop reset") == 0:
                    args = list(node.args)
                    args[4] = _access_ptr_with_offset(args[4], args[4].args[2] + offset)
                    count["mac"] += 1
                    return tir.Call(node.dtype, node.op, args, node.span)
            if not keep_resident or not isinstance(node, tir.Evaluate):
                return None
            value = node.value
            if not isinstance(value, tir.Call):
                return None
            if _weight_load_info(value) is not None:
                return tir.IfThenElse(load_required, node, tir.Evaluate(0))
            if value.op.name in ("tir.vta.coproc_dep_pop", "tir.vta.coproc_dep_push"):
                lane_pair = tuple(_integer(arg, "token lane") for arg in value.args)
                if lane_pair not in ((2, 1), (1, 2)):
                    raise ValueError("node24 reduction token relationship changed")
                count["tokens"].append((value.op.name, lane_pair))
                return tir.IfThenElse(load_required, node, tir.Evaluate(0))
            return None

        new_reduction_body = tir.stmt_functor.ir_transform(reduction.body, None, post)
        if count["load"] != 1 or count["mac"] != 1:
            raise ValueError("node24 weight rewrite did not match exactly one load/MAC")
        if keep_resident and len(count["tokens"]) != 4:
            raise ValueError("node24 did not find exactly four reduction token operations")
        # The VTA runtime accepts only one dynamic integer in a cached UOP
        # kernel.  Specialize k=0..7 here so CO is the sole dynamic key.
        rewritten_reduction = tir.SeqStmt([
            tir.stmt_functor.substitute(
                new_reduction_body, {reduction.loop_var: tir.IntImm("int32", k)}
            ) for k in range(8)
        ])
        body = tir.stmt_functor.ir_transform(
            function.body,
            None,
            lambda node: rewritten_reduction
            if isinstance(node, tir.For) and node.same_as(reduction) else None,
        )
        allocations = []

        def grow(node):
            if not isinstance(node, tir.Allocate):
                return None
            scope = str(getattr(node.buffer_var.type_annotation, "storage_scope", "") or "")
            if scope != "local.wgt_buffer":
                return None
            if [_integer(v, "WGT arena extent") for v in node.extents] != [block_bytes]:
                raise ValueError("node24 compact WGT arena changed")
            allocations.append(True)
            return tir.Allocate(
                node.buffer_var, node.dtype, [tir.IntImm("int32", arena_bytes)],
                node.condition, node.body, node.annotations, node.span,
            )

        body = tir.stmt_functor.ir_transform(body, None, grow)
        if len(allocations) != 1:
            raise ValueError("node24 expected one WGT arena")
        return function.with_body(body)

    return transform


def _loop_nodes(stmt):
    loops = []
    tir.stmt_functor.post_order_visit(
        stmt, lambda node: loops.append(node) if isinstance(node, tir.For) else None
    )
    return loops


def Node24GroupWeightDMA(expected_symbol=None):
    """Experiment-only eight-to-one WGT DMA on the exact node24 group nest.

    The grouped schedule finishes all eight reduction blocks before moving
    to another output group.  A single 9-by-8 DMA therefore fills the same
    18 KiB group address range.  Retain one Load→Compute handshake, with the
    final MAC carrying the Compute→Load completion token.  The next group's
    reset GEMM is FIFO-ordered after that MAC and precedes its WGT overwrite.
    No prefetch, extra stream slot, or explicit drain is introduced.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.Node24GroupWeightDMA")
    def transform(function, _module, _context):
        if not _matches_symbol(function, expected_symbol):
            return function
        groups = [loop for loop in _loop_nodes(function.body)
                  if str(loop.loop_var) == "c_o"]
        spatial = [loop for loop in _loop_nodes(function.body)
                   if str(loop.loop_var) == "ax2_outer"]
        if len(groups) != 1 or len(spatial) != 1 or \
                _integer(groups[0].extent, "group extent") != 16 or \
                _integer(spatial[0].extent, "spatial extent") != 7:
            raise ValueError("node24 group DMA geometry changed")
        group = groups[0]
        if not isinstance(group.body, tir.SeqStmt) or len(group.body.seq) != 18:
            raise ValueError("node24 group DMA requires reset, eight load/MAC pairs, final pop")
        items = list(group.body.seq)
        slot = tir.Select(group.loop_var < 13, group.loop_var, tir.IntImm("int32", 13))
        loads, macs = [], []
        for k in range(8):
            load = _calls(items[1 + 2 * k], lambda call: _weight_load_info(call) is not None)
            mac = _calls(items[2 + 2 * k], lambda call: isinstance(call, tir.Call) and
                         call.op.name == "tir.vta.uop_push" and
                         _integer(call.args[0], "GEMM mode") == 0 and
                         _integer(call.args[1], "GEMM reset") == 0)
            if len(load) != 1 or len(mac) != 1:
                raise ValueError("node24 group DMA lost one-to-one load/MAC order")
            info = _weight_load_info(load[0])
            if (info["x_size"], info["y_size"], info["stride"]) != (9, 1, 9):
                raise ValueError("node24 small WGT DMA shape changed")
            if _integer(load[0].args[3] - (group.loop_var * 72 + k * 9),
                        "WGT source element") != 0:
                raise ValueError("node24 WGT source is not one consecutive group")
            target = load[0].args[11]
            if isinstance(target, tir.Call) and target.op.name == "tir.tvm_access_ptr":
                if _integer(target.args[2] - (slot * 18432 + k * 2304),
                            "WGT destination byte") != 0 or \
                        _integer(target.args[3], "small WGT extent") != 2304:
                    raise ValueError("node24 SRAM subblock layout changed")
            elif _integer(target - (slot * 72 + k * 9), "WGT destination element") != 0:
                raise ValueError("node24 SRAM subblock layout changed")
            loads.append(load[0])
            macs.append(mac[0])

        old_first = loads[0]
        rewritten = {"loads": 0}

        def batch_first(node):
            if not isinstance(node, tir.Call) or not node.same_as(old_first):
                return None
            args = list(node.args)
            args[4], args[5], args[6] = (tir.IntImm("int32", 9),
                                         tir.IntImm("int32", 8), tir.IntImm("int32", 9))
            if isinstance(args[11], tir.Call):
                access = list(args[11].args)
                access[2] = slot * 18432
                access[3] = tir.IntImm("int32", 18432)
                args[11] = tir.Call(args[11].dtype, args[11].op, access, args[11].span)
            else:
                args[11] = slot * 72
            rewritten["loads"] += 1
            return tir.Call(node.dtype, node.op, args, node.span)

        group_load = tir.stmt_functor.ir_transform(items[1], batch_first, None)
        if rewritten["loads"] != 1:
            raise ValueError("node24 group DMA did not replace exactly one first LOAD")

        def remove_token(block, lane_pairs):
            remaining = list(lane_pairs)

            def rewrite(node):
                if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
                    return None
                call = node.value
                if call.op.name not in ("tir.vta.coproc_dep_pop", "tir.vta.coproc_dep_push"):
                    return None
                key = (call.op.name, tuple(_integer(arg, "token lane") for arg in call.args))
                if key not in remaining:
                    return None
                remaining.remove(key)
                return tir.Evaluate(0)

            updated = tir.stmt_functor.ir_transform(block, None, rewrite)
            if remaining:
                raise ValueError("node24 group DMA token sites changed: {}".format(remaining))
            return updated

        pop_load = ("tir.vta.coproc_dep_pop", (1, 2))
        push_compute = ("tir.vta.coproc_dep_push", (2, 1))
        output = [items[0], group_load]
        for k in range(8):
            remove = []
            if k != 0:
                remove.append(pop_load)
            if k != 7:
                remove.append(push_compute)
            output.append(remove_token(items[2 + 2 * k], remove) if remove else items[2 + 2 * k])
        output.append(items[17])
        new_group = tir.For(group.loop_var, group.min, group.extent, group.kind,
                            tir.SeqStmt(output), group.thread_binding,
                            group.annotations, group.span)
        body = tir.stmt_functor.ir_transform(
            function.body, None,
            lambda node: new_group if isinstance(node, tir.For) and node.same_as(group) else None,
        )
        return function.with_body(body)

    return transform


def Node24GroupGEMM(expected_symbol=None):
    """Experiment-only merge of the eight already-folded node24 MAC kernels.

    This is deliberately applied after FoldUopLoop.  It preserves the two
    hardware loops and every base micro-op, but issues one GEMM command for
    the eight reduction slices of a group.  No DMA or dependency is removed.
    The fixed node24 int8 convolution has at most 9*128*128*128 < 2**31 in
    its accumulator, so changing the reduction order cannot overflow int32.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.Node24GroupGEMM")
    def transform(function, _module, _context):
        if not _matches_symbol(function, expected_symbol):
            return function
        groups = [loop for loop in _loop_nodes(function.body)
                  if str(loop.loop_var) == "c_o"]
        spatial = [loop for loop in _loop_nodes(function.body)
                   if str(loop.loop_var) == "ax2_outer"]
        if len(groups) != 1 or len(spatial) != 1 or \
                _integer(groups[0].extent, "group extent") != 16 or \
                _integer(spatial[0].extent, "spatial extent") != 7:
            raise ValueError("node24 GEMM merge geometry changed")
        group = groups[0]
        if not isinstance(group.body, tir.SeqStmt) or len(group.body.seq) != 11:
            raise ValueError("node24 GEMM merge requires reset, group LOAD, eight MACs, pop")
        items = list(group.body.seq)
        kernels, bodies, wrappers = [], [], []
        for k, block in enumerate(items[2:10]):
            if not isinstance(block, tir.AttrStmt) or block.attr_key != "coproc_scope" or \
                    _integer(block.value, "MAC coprocessor lane") != 2:
                raise ValueError("node24 MAC block changed at slice {}".format(k))
            stmts = list(block.body.seq) if isinstance(block.body, tir.SeqStmt) else [block.body]
            scoped = [stmt for stmt in stmts if isinstance(stmt, tir.AttrStmt) and
                      stmt.attr_key == "coproc_uop_scope"]
            if len(scoped) != 1 or len(stmts) != (2 if k in (0, 7) else 1):
                raise ValueError("node24 MAC/token structure changed at slice {}".format(k))
            kernel = scoped[0]
            if not isinstance(kernel.value, tir.StringImm) or \
                    kernel.value.value != "VTAPushGEMMOp" or \
                    not isinstance(kernel.body, tir.SeqStmt) or len(kernel.body.seq) != 5:
                raise ValueError("node24 folded GEMM kernel changed at slice {}".format(k))
            parts = list(kernel.body.seq)
            if not isinstance(parts[2], tir.For):
                raise ValueError("node24 GEMM base UOP loop missing")
            pushes = _calls(parts[2], lambda call: isinstance(call, tir.Call) and
                            call.op.name == "tir.vta.uop_push")
            if len(pushes) != 1 or \
                    _integer(pushes[0].args[0], "GEMM mode") != 0 or \
                    _integer(pushes[0].args[1], "GEMM reset") != 0:
                raise ValueError("node24 GEMM base UOP changed at slice {}".format(k))
            loops = _loop_nodes(parts[2])
            if sorted(_integer(loop.extent, "base UOP extent") for loop in loops) != [2, 14]:
                raise ValueError("node24 GEMM base UOP count changed")
            if k and any(not tvm.ir.structural_equal(parts[i], kernels[0][i])
                         for i in (0, 1, 3, 4)):
                raise ValueError("node24 GEMM hardware loop headers differ")
            kernels.append(parts)
            bodies.append(parts[2])
            wrappers.append((block, kernel, stmts))
        # Preserve the initial Load->Compute pop and final Compute->Load push.
        first, first_kernel, first_stmts = wrappers[0]
        last_stmts = wrappers[-1][2]
        if first_stmts[0].same_as(first_kernel) or \
                last_stmts[-1].same_as(wrappers[-1][1]):
            raise ValueError("node24 group boundary token changed")
        merged_body = tir.SeqStmt([kernels[0][0], kernels[0][1], *bodies,
                                   kernels[0][3], kernels[0][4]])
        merged_kernel = tir.AttrStmt(first_kernel.node, first_kernel.attr_key,
                                     first_kernel.value, merged_body, first_kernel.span)
        merged_mac = tir.AttrStmt(first.node, first.attr_key, first.value,
                                  tir.SeqStmt([first_stmts[0], merged_kernel,
                                               last_stmts[-1]]), first.span)
        new_group = tir.For(group.loop_var, group.min, group.extent, group.kind,
                            tir.SeqStmt([items[0], items[1], merged_mac, items[10]]),
                            group.thread_binding, group.annotations, group.span)
        body = tir.stmt_functor.ir_transform(
            function.body, None,
            lambda node: new_group if isinstance(node, tir.For) and node.same_as(group) else None,
        )
        return function.with_body(body)

    return transform


def Node24SpatialBoundaryBridge(expected_symbol=None):
    """Replace unsupported STORE→LOAD edges with STORE→Compute→LOAD.

    Only the grouped node24 program has this four-site dependency pattern.
    A zero-work Compute command carries the completion token; the final
    function synchronization stays intact.  This pass is shared by A and F.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.Node24SpatialBoundaryBridge")
    def transform(function, _module, _context):
        if not _matches_symbol(function, expected_symbol):
            return function
        spatial = [loop for loop in _loop_nodes(function.body)
                   if str(loop.loop_var) == "ax2_outer"]
        if len(spatial) != 1 or _integer(spatial[0].extent, "spatial extent") != 7:
            raise ValueError("node24 spatial boundary loop changed")
        spatial = spatial[0]
        all_sites = []
        inside_sites = []

        def collect(stmt, found):
            def visit(node):
                if not isinstance(node, tir.Evaluate) or not isinstance(node.value, tir.Call):
                    return
                call = node.value
                if call.op.name not in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
                    return
                lanes = tuple(_integer(arg, "boundary lane") for arg in call.args)
                if lanes == (3, 1):
                    found.append(node)
            tir.stmt_functor.post_order_visit(stmt, visit)

        collect(function.body, all_sites)
        collect(spatial.body, inside_sites)
        if len(all_sites) != 4 or len(inside_sites) != 2:
            raise ValueError("node24 expected four STORE→LOAD sites, two inside spatial loop")
        outside = [node for node in all_sites
                   if not any(node.same_as(inner) for inner in inside_sites)]
        if len(outside) != 2:
            raise ValueError("node24 boundary classification failed")
        in_pop = next((n for n in inside_sites if n.value.op.name.endswith("pop")), None)
        in_push = next((n for n in inside_sites if n.value.op.name.endswith("push")), None)
        out_push = next((n for n in outside if n.value.op.name.endswith("push")), None)
        out_pop = next((n for n in outside if n.value.op.name.endswith("pop")), None)
        if any(node is None for node in (in_pop, in_push, out_push, out_pop)):
            raise ValueError("node24 boundary token direction changed")

        def dep(kind, src, dst):
            return tir.Evaluate(tvm.tir.call_intrin(
                "int32", "tir.vta.coproc_dep_" + kind, src, dst))

        def rewrite(node):
            if not isinstance(node, tir.Evaluate):
                return None
            if node.same_as(out_push) or node.same_as(out_pop):
                return tir.Evaluate(0)
            if node.same_as(in_pop):
                return tir.IfThenElse(spatial.loop_var > 0, dep("pop", 2, 1), tir.Evaluate(0))
            if node.same_as(in_push):
                bridge = tir.SeqStmt([
                    dep("push", 3, 2), dep("pop", 3, 2), dep("push", 2, 1)
                ])
                return tir.IfThenElse(spatial.loop_var < 6, bridge, tir.Evaluate(0))
            return None

        body = tir.stmt_functor.ir_transform(function.body, None, rewrite)
        return function.with_body(body)

    return transform
