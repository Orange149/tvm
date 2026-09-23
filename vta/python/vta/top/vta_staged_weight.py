# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Experiment-only first-use loading for two-vthread resident weights.

This pass is deliberately narrower than a general residency transform.  It
consumes mode 8 of ``vta_conv2d`` after VTA dependency injection and early
StorageRewrite.  The input program uses the proven O2 load/compute order, a
statically unrolled reduction loop, and a compact per-thread WGT slot.  The
pass derives the exact weight-block geometry from the lowered DMA, gives every
``(cthread, reduction-block)`` pair a non-overwriting slot, and executes each
WGT DMA only in the first spatial tile.

Production schedules and the frozen modes 6/7 never install this pass.
"""

import tvm
from tvm import tir


_WGT_VECTOR_BYTES = 256


def _extern_name(call):
    if not isinstance(call, tir.Call) or call.op.name != "tir.call_extern":
        return None
    if not call.args or not isinstance(call.args[0], tir.StringImm):
        return None
    return call.args[0].value


def _access_ptr_with_offset(access, offset):
    if not isinstance(access, tir.Call) or access.op.name != "tir.tvm_access_ptr":
        raise ValueError("staged WGT rewrite expected tvm_access_ptr")
    args = list(access.args)
    if len(args) != 5:
        raise ValueError("staged WGT rewrite found unsupported access pointer")
    args[2] = offset
    return tir.Call(access.dtype, access.op, args, access.span)


def _integer(expr, label):
    value = tvm.arith.Analyzer().simplify(expr)
    if not isinstance(value, tir.IntImm):
        raise ValueError("staged WGT rewrite requires static {}: {}".format(label, value))
    return int(value)


def _weight_load_info(call):
    if _extern_name(call) != "VTALoadBuffer2D":
        return None
    args = list(call.args)
    if len(args) != 13 or _integer(args[12], "memory type") != 1:
        return None
    x_size = _integer(args[4], "WGT x size")
    y_size = _integer(args[5], "WGT y size")
    stride = _integer(args[6], "WGT stride")
    if x_size <= 0 or y_size <= 0 or stride < x_size:
        raise ValueError(
            "staged WGT rewrite found an invalid WGT DMA {}x{}/{}".format(
                x_size, y_size, stride
            )
        )
    return {"call": call, "x_size": x_size, "y_size": y_size, "stride": stride}


def _calls(node, predicate):
    found = []

    def visit(value):
        if isinstance(value, tir.Call) and predicate(value):
            found.append(value)

    tir.stmt_functor.post_order_visit(node, visit)
    return found


def StagedWeightResidencyRewrite():
    """Return the exact post-StorageRewrite transform used by mode 8."""

    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.StagedWeightResidencyRewrite")
    def transform(function, _module, _context):
        height_loops = []
        width_loops = []

        def find_spatial(node):
            if isinstance(node, tir.For) and str(node.loop_var).startswith("i2_outer"):
                height_loops.append(node)
            if isinstance(node, tir.For) and str(node.loop_var).startswith("i3_outer"):
                width_loops.append(node)

        tir.stmt_functor.post_order_visit(function.body, find_spatial)
        if len(height_loops) != 1 or len(width_loops) > 1:
            raise ValueError(
                "staged WGT rewrite requires one i2_outer and at most one i3_outer loop, "
                "found {}/{}".format(len(height_loops), len(width_loops))
            )
        spatial = height_loops[0]
        active_spatial_loops = [spatial] + width_loops
        first_spatial = None
        for loop in active_spatial_loops:
            if _integer(loop.min, "spatial loop minimum") != 0:
                raise ValueError("staged WGT rewrite requires zero-based spatial loops")
            at_first = loop.loop_var == 0
            first_spatial = at_first if first_spatial is None else tir.And(first_spatial, at_first)

        reduction_loops = []

        def find_reduction(node):
            if isinstance(node, tir.For) and str(node.loop_var).startswith("k_o_outer"):
                reduction_loops.append(node)

        tir.stmt_functor.post_order_visit(spatial.body, find_reduction)
        if len(reduction_loops) > 1:
            raise ValueError(
                "staged WGT rewrite requires at most one k_o_outer loop, found {}".format(
                    len(reduction_loops)
                )
            )
        if not reduction_loops:
            source_loads = _calls(spatial.body, lambda call: _weight_load_info(call) is not None)
            if len(source_loads) != 1:
                raise ValueError(
                    "single-group staged WGT rewrite requires one symbolic WGT load, found {}".format(
                        len(source_loads)
                    )
                )
            load_info = _weight_load_info(source_loads[0])
            reduction_block_bytes = (
                load_info["x_size"] * load_info["y_size"] * _WGT_VECTOR_BYTES
            )
            source_offset = tvm.arith.Analyzer().simplify(source_loads[0].args[3])
            source_vars = []

            def collect_single_group_source_var(value):
                if isinstance(value, tir.Var) and all(
                    not value.same_as(loop.loop_var) for loop in active_spatial_loops
                ):
                    source_vars.append(value)

            tir.stmt_functor.post_order_visit(source_offset, collect_single_group_source_var)
            unique_source_vars = []
            for value in source_vars:
                if all(not value.same_as(existing) for existing in unique_source_vars):
                    unique_source_vars.append(value)
            if len(unique_source_vars) > 1:
                raise ValueError(
                    "single-group staged WGT rewrite expected at most one output cthread "
                    "in the source offset, found {}".format(len(unique_source_vars))
                )
            if unique_source_vars:
                coefficients = tvm.arith.detect_linear_equation(source_offset, unique_source_vars)
                source_layout_ok = bool(coefficients) and [
                    _integer(value, "single-group WGT source coefficient")
                    for value in coefficients
                ] == [reduction_block_bytes // _WGT_VECTOR_BYTES, 0]
            else:
                source_layout_ok = _integer(source_offset, "single-group WGT source offset") == 0
            if not source_layout_ok:
                raise ValueError(
                    "single-group staged WGT rewrite found an unexpected source layout: {}".format(
                        source_offset
                    )
                )

            macs = _calls(
                spatial.body,
                lambda call: call.op.name == "tir.vta.uop_push"
                and _integer(call.args[0], "uop mode") == 0
                and _integer(call.args[1], "uop reset") == 0,
            )
            if len(macs) != 1:
                raise ValueError(
                    "single-group staged WGT rewrite expected one symbolic MAC, found {}".format(
                        len(macs)
                    )
                )

            rewritten_loads = []

            def guard_single_group_load(node):
                if isinstance(node, tir.Evaluate) and _weight_load_info(node.value) is not None:
                    rewritten_loads.append(True)
                    return tir.IfThenElse(first_spatial, node, tir.Evaluate(0))
                return None

            body = tir.stmt_functor.ir_transform(function.body, None, guard_single_group_load)
            if len(rewritten_loads) != 1:
                raise ValueError(
                    "single-group staged WGT rewrite expected to guard one load, got {}".format(
                        len(rewritten_loads)
                    )
                )
            return function.with_body(body)

        reduction = reduction_loops[0]
        if not isinstance(spatial.body, tir.SeqStmt):
            raise ValueError("multi-group staged WGT rewrite requires a sequential spatial body")
        reduction_extent = _integer(reduction.extent, "reduction loop extent")
        if (
            _integer(reduction.min, "reduction loop minimum") != 0
            or reduction_extent <= 0
            or reduction.kind != tir.ForKind.UNROLLED
        ):
            raise ValueError("staged WGT rewrite requires a non-empty zero-based unrolled reduction")

        source_loads = _calls(reduction.body, lambda call: _weight_load_info(call) is not None)
        if len(source_loads) != 1:
            raise ValueError(
                "staged WGT rewrite requires one symbolic WGT load, found {}".format(
                    len(source_loads)
                )
            )
        load_info = _weight_load_info(source_loads[0])
        reduction_block_bytes = (
            load_info["x_size"] * load_info["y_size"] * _WGT_VECTOR_BYTES
        )
        resident_thread_bytes = reduction_block_bytes * reduction_extent
        source_offset = tvm.arith.Analyzer().simplify(source_loads[0].args[3])
        source_vars = []

        def collect_source_var(value):
            if isinstance(value, tir.Var) and not value.same_as(reduction.loop_var):
                source_vars.append(value)

        tir.stmt_functor.post_order_visit(source_offset, collect_source_var)
        unique_source_vars = []
        for value in source_vars:
            if all(not value.same_as(existing) for existing in unique_source_vars):
                unique_source_vars.append(value)
        if len(unique_source_vars) != 1:
            raise ValueError(
                "staged WGT rewrite expected one output cthread in the source offset, found {}".format(
                    len(unique_source_vars)
                )
            )
        coefficients = tvm.arith.detect_linear_equation(
            source_offset, [unique_source_vars[0], reduction.loop_var]
        )
        if not coefficients or [
            _integer(value, "WGT source coefficient") for value in coefficients
        ] != [resident_thread_bytes // _WGT_VECTOR_BYTES, load_info["x_size"], 0]:
            raise ValueError(
                "staged WGT rewrite found an unexpected source layout: {}".format(source_offset)
            )

        rewritten_loads = []
        rewritten_macs = []

        def post(node):
            if isinstance(node, tir.Call):
                if _weight_load_info(node) is not None:
                    args = list(node.args)
                    old_access = args[11]
                    old_offset = old_access.args[2]
                    if _integer(old_offset, "compact WGT destination") != 0:
                        raise ValueError("staged WGT rewrite expected a zero-based compact WGT slot")
                    args[11] = _access_ptr_with_offset(
                        old_access, reduction.loop_var * reduction_block_bytes
                    )
                    rewritten_loads.append(True)
                    return tir.Call(node.dtype, node.op, args, node.span)

                if node.op.name == "tir.vta.uop_push":
                    args = list(node.args)
                    mode = _integer(args[0], "uop mode")
                    reset = _integer(args[1], "uop reset")
                    if mode == 0 and reset == 0:
                        old_access = args[4]
                        if (
                            not isinstance(old_access, tir.Call)
                            or old_access.op.name != "tir.tvm_access_ptr"
                        ):
                            raise ValueError("staged MAC expected a WGT access pointer")
                        args[4] = _access_ptr_with_offset(
                            old_access,
                            old_access.args[2] + reduction.loop_var * reduction_block_bytes,
                        )
                        rewritten_macs.append(True)
                        return tir.Call(node.dtype, node.op, args, node.span)

            if isinstance(node, tir.Evaluate) and _weight_load_info(node.value) is not None:
                return tir.IfThenElse(first_spatial, node, tir.Evaluate(0))
            return None

        rewritten_reduction_body = tir.stmt_functor.ir_transform(reduction.body, None, post)
        if len(rewritten_loads) != 1 or len(rewritten_macs) != 1:
            raise ValueError(
                "staged WGT rewrite expected one symbolic load/MAC, got {}/{}".format(
                    len(rewritten_loads), len(rewritten_macs)
                )
            )
        rewritten_reduction = tir.For(
            reduction.loop_var,
            reduction.min,
            reduction.extent,
            reduction.kind,
            rewritten_reduction_body,
            reduction.thread_binding,
            reduction.annotations,
            reduction.span,
        )
        rewritten_spatial_body = tir.stmt_functor.ir_transform(
            spatial.body,
            None,
            lambda node: rewritten_reduction
            if isinstance(node, tir.For) and node.same_as(reduction)
            else None,
        )
        rewritten_spatial = tir.For(
            spatial.loop_var,
            spatial.min,
            spatial.extent,
            spatial.kind,
            rewritten_spatial_body,
            spatial.thread_binding,
            spatial.annotations,
            spatial.span,
        )
        replaced_spatial = tir.stmt_functor.ir_transform(
            function.body,
            None,
            lambda node: rewritten_spatial
            if isinstance(node, tir.For) and node.same_as(spatial)
            else None,
        )

        allocations = []

        def expand(node):
            if not isinstance(node, tir.Allocate):
                return None
            scope = str(getattr(node.buffer_var.type_annotation, "storage_scope", "") or "")
            if scope != "local.wgt_buffer":
                return None
            extents = [_integer(value, "WGT allocation extent") for value in node.extents]
            if extents != [reduction_block_bytes]:
                raise ValueError(
                    "staged WGT rewrite expected a {}-byte per-thread WGT arena, got {}".format(
                        reduction_block_bytes, extents
                    )
                )
            allocations.append(str(node.buffer_var))
            return tir.Allocate(
                node.buffer_var,
                node.dtype,
                [tir.IntImm("int32", resident_thread_bytes)],
                node.condition,
                node.body,
                node.annotations,
                node.span,
            )

        body = tir.stmt_functor.ir_transform(replaced_spatial, None, expand)
        if len(allocations) != 1:
            raise ValueError(
                "staged WGT rewrite expected one WGT allocation, found {}".format(len(allocations))
            )
        return function.with_body(body)

    return transform


def staged_weight_build_config(debug_flag=0, **kwargs):
    """Return the stock VTA build pipeline with the isolated staged pass installed."""
    from .. import build_module  # pylint: disable=import-outside-toplevel

    base = build_module.build_config(debug_flag=debug_flag)
    passes = list(base.config["tir.add_lower_pass"])
    early = next(
        index
        for index, (_, item) in enumerate(passes)
        if str(item.info.name).endswith("EarlyRewrite")
    )
    passes.insert(early + 1, (1, StagedWeightResidencyRewrite()))
    return tvm.transform.PassContext(config={"tir.add_lower_pass": passes}, **kwargs)
