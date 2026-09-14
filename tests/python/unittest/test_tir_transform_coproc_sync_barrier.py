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
"""Pure-TIR tests for full-sync dependency segmentation."""

from collections import Counter

import tvm
from tvm import te, tir


for suffix in ("coproc_sync", "coproc_dep_push", "coproc_dep_pop"):
    tvm.ir.register_op_attr("tir.barrier_test." + suffix, "TGlobalSymbol", suffix)


def _scope_store(builder, axis, queue_id, buffer, value):
    with builder.new_scope():
        builder.scope_attr(axis, "coproc_scope", queue_id)
        buffer[0] = value


def _make_module(queue_ids, barrier_after=None):
    builder = tir.ir_builder.create()
    axis = te.thread_axis((0, 1), "barrier_test")
    buffer = builder.allocate("int32", 1, name="buffer")
    for index, queue_id in enumerate(queue_ids):
        _scope_store(builder, axis, queue_id, buffer, index)
        if barrier_after is not None and index == barrier_after:
            builder.emit(
                tir.Evaluate(
                    tir.Call(
                        "int32", tvm.ir.Op.get("tir.barrier_test.coproc_sync"), []
                    )
                )
            )
    return tvm.IRModule.from_expr(tir.PrimFunc([], builder.get()))


def _pairs(module):
    result = []

    def visit(node):
        if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name in (
            "tir.barrier_test.coproc_dep_push",
            "tir.barrier_test.coproc_dep_pop",
        ):
            result.append(
                (node.op.name.rsplit("_", 1)[-1], int(node.args[0]), int(node.args[1]))
            )

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    return result


def _assert_balanced(pairs):
    pushes = Counter((source, target) for kind, source, target in pairs if kind == "push")
    pops = Counter((source, target) for kind, source, target in pairs if kind == "pop")
    assert pushes == pops
    return pushes


def test_without_full_sync_retains_direct_cross_queue_dependency():
    lowered = tvm.tir.transform.CoProcSync()(_make_module([3, 1]))
    pushes = _assert_balanced(_pairs(lowered))
    assert pushes[(3, 1)] > 0


def test_full_sync_closes_segments_without_cross_barrier_dependency():
    lowered = tvm.tir.transform.CoProcSync()(_make_module([1, 2, 2, 3], barrier_after=1))
    pushes = _assert_balanced(_pairs(lowered))
    assert pushes[(1, 2)] > 0
    assert pushes[(2, 3)] > 0
    assert not any({source, target} == {1, 3} for source, target in pushes)


def test_full_sync_between_store_and_load_removes_only_crossing_edge():
    without = _pairs(tvm.tir.transform.CoProcSync()(_make_module([3, 1])))
    with_barrier = _pairs(
        tvm.tir.transform.CoProcSync()(_make_module([3, 1], barrier_after=0))
    )
    assert ("push", 3, 1) in without and ("pop", 3, 1) in without
    assert ("push", 3, 1) not in with_barrier and ("pop", 3, 1) not in with_barrier
    _assert_balanced(with_barrier)


if __name__ == "__main__":
    test_without_full_sync_retains_direct_cross_queue_dependency()
    test_full_sync_closes_segments_without_cross_barrier_dependency()
    test_full_sync_between_store_and_load_removes_only_crossing_edge()
