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
"""Experiment-only P1/P2 partial weight residency for R18-H1 T8.

The mode-8 schedule supplies the proven two-vthread O2 traversal, an
unrolled four-group reduction, and one compact WGT slot per virtual thread.
This post-StorageRewrite pass retains the first one or two reduction groups
in fixed slots and maps every remaining group to one shared streaming slot.
Resident loads execute only in the first spatial tile; streaming loads keep
the original per-spatial-tile cadence.  No synchronization is inserted.

This is deliberately not a general cache or eviction policy.  Every shape,
loop, source-layout, and allocation assumption is checked fail-closed.
"""

import tvm
from tvm import tir

from .vta_staged_weight import (
    _access_ptr_with_offset,
    _calls,
    _integer,
    _weight_load_info,
)


_WGT_VECTOR_BYTES = 256
_EXPECTED_REDUCTION_GROUPS = 4


def PartialWeightResidencyRewrite(resident_groups):
    """Return the exact P1/P2 transform used by the local pilot."""
    resident_groups = int(resident_groups)
    if resident_groups not in (1, 2):
        raise ValueError("partial WGT residency supports exactly P1 or P2")

    @tir.transform.prim_func_pass(
        opt_level=0,
        name="tir.vta.PartialWeightResidencyRewriteP{}".format(resident_groups),
    )
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
                "partial WGT residency requires one i2_outer and at most one "
                "i3_outer loop, found {}/{}".format(len(height_loops), len(width_loops))
            )
        spatial = height_loops[0]
        active_spatial_loops = [spatial] + width_loops
        first_spatial = None
        for loop in active_spatial_loops:
            if _integer(loop.min, "spatial loop minimum") != 0:
                raise ValueError("partial WGT residency requires zero-based spatial loops")
            at_first = loop.loop_var == 0
            first_spatial = at_first if first_spatial is None else tir.And(first_spatial, at_first)

        reduction_loops = []

        def find_reduction(node):
            if isinstance(node, tir.For) and str(node.loop_var).startswith("k_o_outer"):
                reduction_loops.append(node)

        tir.stmt_functor.post_order_visit(spatial.body, find_reduction)
        if len(reduction_loops) != 1:
            raise ValueError(
                "partial WGT residency requires one k_o_outer loop, found {}".format(
                    len(reduction_loops)
                )
            )
        reduction = reduction_loops[0]
        reduction_extent = _integer(reduction.extent, "reduction loop extent")
        if (
            _integer(reduction.min, "reduction loop minimum") != 0
            or reduction_extent != _EXPECTED_REDUCTION_GROUPS
            or reduction.kind != tir.ForKind.UNROLLED
        ):
            raise ValueError(
                "partial WGT residency requires a zero-based unrolled four-group reduction"
            )

        source_loads = _calls(reduction.body, lambda call: _weight_load_info(call) is not None)
        if len(source_loads) != 1:
            raise ValueError(
                "partial WGT residency requires one symbolic WGT load, found {}".format(
                    len(source_loads)
                )
            )
        load_info = _weight_load_info(source_loads[0])
        reduction_block_bytes = (
            load_info["x_size"] * load_info["y_size"] * _WGT_VECTOR_BYTES
        )
        full_thread_bytes = reduction_block_bytes * reduction_extent
        partial_thread_bytes = reduction_block_bytes * (resident_groups + 1)

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
                "partial WGT residency expected one output cthread in the source "
                "offset, found {}".format(len(unique_source_vars))
            )
        coefficients = tvm.arith.detect_linear_equation(
            source_offset, [unique_source_vars[0], reduction.loop_var]
        )
        if not coefficients or [
            _integer(value, "WGT source coefficient") for value in coefficients
        ] != [full_thread_bytes // _WGT_VECTOR_BYTES, load_info["x_size"], 0]:
            raise ValueError(
                "partial WGT residency found an unexpected source layout: {}".format(
                    source_offset
                )
            )

        # Groups [0, resident_groups) get fixed slots.  Every later group uses
        # the one streaming slot immediately after the resident region.
        destination_group = tir.Select(
            reduction.loop_var < resident_groups,
            reduction.loop_var,
            tir.IntImm("int32", resident_groups),
        )
        destination_offset = destination_group * reduction_block_bytes
        load_required = tir.Or(reduction.loop_var >= resident_groups, first_spatial)
        rewritten_loads = []
        rewritten_macs = []

        def post(node):
            if isinstance(node, tir.Call):
                if _weight_load_info(node) is not None:
                    args = list(node.args)
                    old_access = args[11]
                    if _integer(old_access.args[2], "compact WGT destination") != 0:
                        raise ValueError(
                            "partial WGT residency expected a zero-based compact WGT slot"
                        )
                    args[11] = _access_ptr_with_offset(old_access, destination_offset)
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
                            raise ValueError("partial-resident MAC expected a WGT access pointer")
                        args[4] = _access_ptr_with_offset(
                            old_access, old_access.args[2] + destination_offset
                        )
                        rewritten_macs.append(True)
                        return tir.Call(node.dtype, node.op, args, node.span)

            if isinstance(node, tir.Evaluate) and _weight_load_info(node.value) is not None:
                return tir.IfThenElse(load_required, node, tir.Evaluate(0))
            return None

        rewritten_reduction_body = tir.stmt_functor.ir_transform(
            reduction.body, None, post
        )
        if len(rewritten_loads) != 1 or len(rewritten_macs) != 1:
            raise ValueError(
                "partial WGT residency expected one symbolic load/MAC, got {}/{}".format(
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
                    "partial WGT residency expected a {}-byte compact per-thread "
                    "arena, got {}".format(reduction_block_bytes, extents)
                )
            allocations.append(str(node.buffer_var))
            return tir.Allocate(
                node.buffer_var,
                node.dtype,
                [tir.IntImm("int32", partial_thread_bytes)],
                node.condition,
                node.body,
                node.annotations,
                node.span,
            )

        body = tir.stmt_functor.ir_transform(replaced_spatial, None, expand)
        if len(allocations) != 1:
            raise ValueError(
                "partial WGT residency expected one WGT allocation, found {}".format(
                    len(allocations)
                )
            )
        return function.with_body(body)

    return transform


def partial_weight_build_config(resident_groups, debug_flag=0, **kwargs):
    """Return the stock VTA build pipeline with one isolated P1/P2 pass."""
    from .. import build_module  # pylint: disable=import-outside-toplevel

    base = build_module.build_config(debug_flag=debug_flag)
    passes = list(base.config["tir.add_lower_pass"])
    early = next(
        index
        for index, (_, item) in enumerate(passes)
        if str(item.info.name).endswith("EarlyRewrite")
    )
    passes.insert(early + 1, (1, PartialWeightResidencyRewrite(resident_groups)))
    return tvm.transform.PassContext(config={"tir.add_lower_pass": passes}, **kwargs)
