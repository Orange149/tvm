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
"""Isolated AutoTVM entry for local VTA residency-schedule experiments."""

import tvm
from tvm import autotvm, te, topi

from .vta_conv2d import conv2d_packed, schedule_conv2d_packed_residency


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def _clip(x, lower, upper):
    lower = tvm.tir.const(lower, x.dtype)
    upper = tvm.tir.const(upper, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), upper), name="residency_clip_max")
    return te.compute(x.shape, lambda *i: tvm.te.max(x(*i), lower), name="residency_clip_min")


@autotvm.template("conv2d_packed_residency.vta")
def conv2d_packed_residency_template(
    data, kernel, strides, padding, dilation, layout, out_dtype, residency_mode
):
    """Build an explicitly selected, experimental residency schedule.

    ``residency_mode`` is part of the task workload rather than the original VTA
    ConfigSpace: 0=original, 1=input-stationary, 2=weight-stationary, and
    3=paper-inspired-hybrid.  Mode 4 is the barrier-delimited weight-residency
    implementation. Mode 5 combines input-prioritized traversal with full-layer
    weight residency when the kernel fits weight SRAM. Keeping the mode out of the original ConfigSpace preserves every
    historical TopHub index; production use requires an explicit qualified route.
    """
    cfg = autotvm.get_config()
    if isinstance(padding, (tuple, list)) and len(padding) == 4:
        top, left, bottom, right = padding
        if top != bottom or left != right:
            raise ValueError("Asymmetric padding is not supported")
        padding = (top, left)

    with tvm.target.vta():
        result = conv2d_packed(data, kernel, strides, padding, dilation, layout, out_dtype)
        result = topi.right_shift(result, 8)
        result = _clip(result, 0, 127)
        result = topi.cast(result, "int8")

    if tvm.target.Target.current().device_name == "vta":
        schedule = schedule_conv2d_packed_residency(cfg, [result], residency_mode)
    else:
        schedule = te.create_schedule([result.op])
    return schedule, [data, kernel, result]
