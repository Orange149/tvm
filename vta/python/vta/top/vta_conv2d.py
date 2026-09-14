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
"""Conv2D operator declaration and schedule registration for VTA."""

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
from tvm.autotvm.task.space import OtherOptionEntity, SplitEntity

from .utils import is_packed_layout
from .residency_dispatch import resolve_schedule_route
from ..environment import get_env


def _set_safe_fallback(cfg, b, c_o, x_i, x_j, c_i):
    """Install a deterministic fallback config for packed conv2d.

    For AXU5EVB we bias the fallback toward the best legal configs observed from
    direct-RPC sweeps on the dominant ResNet-18 conv workloads. Other targets
    keep the original conservative single-thread behavior.
    """

    def _extent(axis):
        return topi.utils.get_const_int(axis.dom.extent)

    def _split_by_inner(extent, inner):
        inner = max(1, min(extent, inner))
        assert extent % inner == 0
        return SplitEntity([extent // inner, inner])

    b_ext = _extent(b)
    co_ext = _extent(c_o)
    h_ext = _extent(x_i)
    w_ext = _extent(x_j)
    ci_ext = _extent(c_i)

    cfg["tile_b"] = SplitEntity([b_ext, 1])
    cfg["tile_h"] = SplitEntity([h_ext, 1])
    cfg["tile_w"] = SplitEntity([w_ext, 1])
    cfg["tile_ci"] = SplitEntity([ci_ext, 1])
    cfg["tile_co"] = SplitEntity([co_ext, 1])
    cfg["oc_nthread"] = OtherOptionEntity(1)
    cfg["h_nthread"] = OtherOptionEntity(1)

    env = get_env()
    if env.TARGET != "axu5evb":
        return

    # AXU5EVB heuristics from measured legal configs:
    # - C2: 56x56 ic64 oc64 k3s1      -> tile_h=56, tile_w=1, oc_nthread=2
    # - C5: 28x28 ic128 oc128 k3s1    -> tile_h=28, tile_w=2, oc_nthread=1
    # - P3: 14x14 ic256 oc512 k1s2    -> output 7x7, tile_h=7, tile_w=7, oc_nthread=2
    cfg["tile_h"] = _split_by_inner(h_ext, h_ext)

    if (h_ext, w_ext, ci_ext, co_ext) == (56, 56, 4, 4):
        cfg["tile_w"] = _split_by_inner(w_ext, 1)
        cfg["oc_nthread"] = OtherOptionEntity(2)
    elif (h_ext, w_ext, ci_ext, co_ext) == (28, 28, 8, 8):
        cfg["tile_w"] = _split_by_inner(w_ext, 2)
        cfg["oc_nthread"] = OtherOptionEntity(1)
    elif (h_ext, w_ext, ci_ext, co_ext) == (7, 7, 16, 32):
        cfg["tile_w"] = _split_by_inner(w_ext, 7)
        cfg["oc_nthread"] = OtherOptionEntity(2)
    else:
        # Generic AXU5EVB fallback: maximize row reuse, keep width conservative,
        # and only enable channel threading when the outer channel tile is large
        # enough to split safely.
        cfg["tile_w"] = _split_by_inner(w_ext, 1)
        cfg["oc_nthread"] = OtherOptionEntity(2 if co_ext >= 2 else 1)


@autotvm.register_topi_compute("conv2d_packed.vta")
def conv2d_packed(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    """Packed conv2d function."""
    if not is_packed_layout(layout):
        raise topi.InvalidShapeError()
    assert dilation == (1, 1)

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, data.shape[4], kernel.shape[4])

    ishape = topi.utils.get_const_tuple(data.shape)
    kshape = topi.utils.get_const_tuple(kernel.shape)
    d_i = te.reduce_axis((0, kshape[2]), name="d_i")
    d_j = te.reduce_axis((0, kshape[3]), name="d_j")
    k_o = te.reduce_axis((0, ishape[1]), name="k_o")
    k_i = te.reduce_axis((0, ishape[-1]), name="k_i")
    hstride, wstride = strides
    res = te.compute(
        oshape,
        lambda b_o, c_o, i, j, b_i, c_i: te.sum(
            pad_data[b_o, k_o, i * hstride + d_i, j * wstride + d_j, b_i, k_i].astype(out_dtype)
            * kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i],
        ),
        name="res",
        tag="conv2d_dense",
    )

    cfg.add_flop(
        2
        * np.prod(topi.utils.get_const_tuple(oshape))
        * kshape[2]
        * kshape[3]
        * ishape[1]
        * ishape[-1]
    )

    return res


RESIDENCY_MODES = {
    0: "original",
    1: "input_stationary",
    2: "weight_stationary",
    3: "paper_inspired_hybrid",
    4: "weight_stationary_sync_probe",
    5: "input_weight_resident_barrier",
}


def _normalize_residency_mode(mode):
    """Return the canonical name for an experimental residency mode."""
    if isinstance(mode, str):
        if mode not in RESIDENCY_MODES.values():
            raise ValueError("Unknown VTA residency mode: {}".format(mode))
        return mode
    try:
        return RESIDENCY_MODES[int(mode)]
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError("Unknown VTA residency mode: {}".format(mode)) from err


def _schedule_conv2d_packed_impl(cfg, outs, residency_mode="original"):
    """Build the packed-conv schedule, optionally using an experimental residency mode.

    The registered ``conv2d_packed.vta`` entry always selects ``original``.  The other
    modes are deliberately reached only through the separate experimental template in
    ``vta_conv2d_residency.py`` so existing TopHub records and ConfigSpace indices keep
    their original meaning.
    """
    residency_mode = _normalize_residency_mode(residency_mode)
    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert "int" in output.op.input_tensors[0].dtype

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "conv2d_dense"
            conv2d_res.append(op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)
    s = te.create_schedule(output.op)

    ##### space definition begin #####
    b, c_o, x_i, x_j, _, _ = s[conv2d_stage].op.axis
    c_i, _, _, _ = s[conv2d_stage].op.reduce_axis
    cfg.define_split("tile_b", b, num_outputs=2)
    cfg.define_split("tile_h", x_i, num_outputs=2)
    cfg.define_split("tile_w", x_j, num_outputs=2)
    cfg.define_split("tile_ci", c_i, num_outputs=2)
    cfg.define_split("tile_co", c_o, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    cfg.define_knob("h_nthread", [1, 2])
    ###### space definition end ######

    if cfg.is_fallback:
        _set_safe_fallback(cfg, b, c_o, x_i, x_j, c_i)

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None

    env = get_env()

    # setup pad
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.inp_scope)
    else:
        cdata = s.cache_read(data, env.inp_scope, [conv2d_stage])
    ckernel = s.cache_read(kernel, env.wgt_scope, [conv2d_stage])
    s[conv2d_stage].set_scope(env.acc_scope)

    # cache read input
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(s.cache_read(tensor, env.acc_scope, [consumer]))

    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)

    for op in const_ops:
        s[op].compute_inline()

    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg["tile_co"].apply(s, output, x_co)
    x_i0, x_i1 = cfg["tile_h"].apply(s, output, x_i)
    x_j0, x_j1 = cfg["tile_w"].apply(s, output, x_j)
    weight_residency_pt = None
    sync_axis = None
    full_weight_residency = False
    hybrid_oc_vthread_axis = None
    if residency_mode == "original":
        s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
        store_pt = x_j0
    elif residency_mode == "input_stationary":
        # Compute one wider conv tile below x_j0.  The existing cdata placement at the
        # reduction tile then covers all output-channel tiles in that conv region; this
        # also preserves the VTA padded-DMA shape required by the original template.
        s[output].reorder(x_bo, x_i0, x_j0, x_co0, x_co1, x_i1, x_j1, x_bi, x_ci)
        store_pt = x_j0
    elif residency_mode in ("weight_stationary", "weight_stationary_sync_probe"):
        # Group adjacent outer width tiles.  Mode 2 remains a feasibility-only shape;
        # mode 4 promotes ckernel and adds an explicit drain, realizing bounded weight
        # reuse for identities that pass lowering, FSim, and FPGA qualification.
        w_extent = topi.utils.get_const_int(s[output].op.axis[3].dom.extent)
        w_outer_extent = w_extent // cfg["tile_w"].size[-1]
        w_reuse_factor = 2 if w_outer_extent % 2 == 0 else 1
        x_jg, x_jr = s[output].split(x_j0, factor=w_reuse_factor)
        s[output].reorder(
            x_bo, x_co0, x_i0, x_jg, x_jr, x_co1, x_i1, x_j1, x_bi, x_ci
        )
        store_pt = x_jg
        if residency_mode == "weight_stationary_sync_probe":
            # Keep one weight tile across all spatial groups of x_co0 and
            # request a full command-queue drain at the end of that residency scope.
            # This mode is excluded from the production candidate identity vocabulary.
            weight_residency_pt = x_co0
            # A pragma on x_i0 wraps the complete spatial loop and appends the sync
            # inside each x_co0 iteration, after its final STORE and before the next
            # x_co0 iteration begins with a new weight LOAD.
            sync_axis = x_i0
    elif residency_mode == "paper_inspired_hybrid":
        # A local, paper-inspired bounded composition.  This is not claimed to be an
        # exact reproduction.  The safe version realizes bounded input reuse; weight
        # reuse remains a hypothesis because promoting ckernel across this group forms
        # an unsupported STORE->LOAD dependency in the current VTA task graph.
        co_extent = topi.utils.get_const_int(s[output].op.axis[1].dom.extent)
        w_extent = topi.utils.get_const_int(s[output].op.axis[3].dom.extent)
        co_outer_extent = co_extent // cfg["tile_co"].size[-1]
        w_outer_extent = w_extent // cfg["tile_w"].size[-1]
        co_reuse_factor = 2 if co_outer_extent % 2 == 0 else 1
        w_reuse_factor = 2 if w_outer_extent % 2 == 0 else 1
        x_cog, x_cor = s[output].split(x_co0, factor=co_reuse_factor)
        # Keep the bounded-reuse group inside one virtual context.  Contexts are
        # distributed across groups, not across x_cor inside a group; splitting the
        # latter interleaves writes to the same accumulator indices and violates the
        # VTA two-cycle UOP destination dependency rule.
        hybrid_oc_vthread_axis = x_cog
        x_jg, x_jr = s[output].split(x_j0, factor=w_reuse_factor)
        s[output].reorder(
            x_bo,
            x_i0,
            x_cog,
            x_jg,
            x_jr,
            x_cor,
            x_co1,
            x_i1,
            x_j1,
            x_bi,
            x_ci,
        )
        store_pt = x_jg
    elif residency_mode == "input_weight_resident_barrier":
        # Functional reimplementation of Cheng Scheme 4 for the current VTA tree.
        # Input-prioritized traversal keeps x_j0 outside x_co0.  To make weight
        # reuse compatible with that traversal, retain the *whole* layer kernel
        # below the batch axis instead of retaining only one x_co0 tile.  This is
        # intentionally different from Cheng's unpublished non-overwriting-address
        # runtime patch and is exposed only through the experimental template.
        full_weight_bytes = int(np.prod(topi.utils.get_const_tuple(kernel.shape)))
        if full_weight_bytes > env.WGT_BUFF_SIZE:
            raise ValueError(
                "input_weight_resident_barrier not_applicable: full kernel {} exceeds "
                "weight SRAM {}".format(full_weight_bytes, env.WGT_BUFF_SIZE)
            )
        s[output].reorder(x_bo, x_i0, x_j0, x_co0, x_co1, x_i1, x_j1, x_bi, x_ci)
        store_pt = x_j0
        weight_residency_pt = x_bo
        sync_axis = x_bo
        full_weight_residency = True

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    # Virtual-thread support is deliberately narrow for the P7R joint experiment.
    # Input-stationary schedules may retain output-channel cthreads so the compiler
    # can certify the real context-replicated SRAM footprint.  Other experimental
    # modes still use the older mechanism-isolation boundary, and spatial cthreads
    # remain excluded because they introduce a second replication dimension.
    if residency_mode != "original" and cfg["h_nthread"].val > 1:
        raise ValueError("Experimental residency schedules require h_nthread = 1")
    if residency_mode not in (
        "original", "input_stationary", "paper_inspired_hybrid"
    ) and cfg["oc_nthread"].val > 1:
        raise ValueError(
            "Only input-stationary and bounded-hybrid residency currently support "
            "oc_nthread > 1"
        )

    if cfg["oc_nthread"].val > 1:
        oc_vthread_axis = (
            hybrid_oc_vthread_axis
            if residency_mode == "paper_inspired_hybrid"
            else x_co0
        )
        _, v_t = s[output].split(oc_vthread_axis, factor=cfg["oc_nthread"].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, te.thread_axis("cthread"))

    # virtual threading along spatial rows
    if cfg["h_nthread"].val > 1:
        _, v_t = s[output].split(x_i0, factor=cfg["h_nthread"].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, te.thread_axis("cthread"))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

    k_o, _ = cfg["tile_ci"].apply(s, conv2d_stage, k_o)
    s[cdata].compute_at(s[conv2d_stage], k_o)
    if residency_mode == "weight_stationary_sync_probe":
        s[ckernel].compute_at(s[output], weight_residency_pt)
        s[output].pragma(sync_axis, "coproc_sync")
    elif full_weight_residency:
        s[ckernel].compute_at(s[output], weight_residency_pt)
        s[output].pragma(sync_axis, "coproc_sync")
    else:
        # Keep both DMA producers inside the conv region.  Promoting ckernel to an
        # output loop without an explicit drain creates an unsupported STORE->LOAD
        # dependency in the VTA runtime.  Modes 0--3 retain that verified boundary.
        s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    s[conv2d_stage].tensorize(x_bi, env.gemm)
    s[output].pragma(x_co1, env.dma_copy)

    return s


@autotvm.register_topi_schedule("conv2d_packed.vta")
def schedule_conv2d_packed(cfg, outs):
    """Schedule packed conv2d, honoring only an explicit fail-closed route."""
    residency_mode = resolve_schedule_route(cfg, outs)
    return _schedule_conv2d_packed_impl(cfg, outs, residency_mode=residency_mode)


def schedule_conv2d_packed_residency(cfg, outs, residency_mode):
    """Experimental packed-conv schedule with an explicit integer/string mode."""
    return _schedule_conv2d_packed_impl(cfg, outs, residency_mode=residency_mode)
