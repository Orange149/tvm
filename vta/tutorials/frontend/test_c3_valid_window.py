"""Offline support-domain examples, using small TIR fixtures only."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import tvm
from tvm import tir
import vta

from c3_valid_window import (AggregateWindows, certify_trace, expanded_uops,
                             statement_sequence, trace_function)


def fixture(segments=6, width=4, repetitions=1, internal=None, loops=False):
    x = tir.decl_buffer((2048, 16), "int8", name="fixture_x")
    w = tir.decl_buffer((1024, 16, 16), "int8", name="fixture_w")
    add = tir.decl_buffer((16,), "int32", name="fixture_add")
    y = tir.decl_buffer((width, 16), "int8", name="fixture_y")
    params = [tir.Var("arg%d" % i, "handle") for i in range(4)]
    lane = tir.IterVar(None, tir.Var("fixture_lane", "int32"), tir.IterVar.ThreadIndex, "vta")
    handle = tir.call_intrin("handle", "tir.vta.command_handle")

    def dep(kind, src, dst):
        return tir.Evaluate(tir.call_intrin("int32", "tir.vta.coproc_dep_" + kind, src, dst))

    def load(buffer, domain, count, offset=0):
        return tir.Evaluate(tir.call_extern("int32", "VTALoadBuffer2D", handle, buffer.data,
            offset, count, 1, count, 0, 0, 0, 0, 0, domain))

    def kernel(reset, slice_id, hardware=False):
        var = tir.Var("base_index_%d_%d" % (reset, slice_id), "int32")
        push = tir.Evaluate(tir.call_intrin("int32", "tir.vta.uop_push", 0, reset,
            var, var, slice_id, 0, 0, 0))
        body = tir.For(var, 0, width, tir.ForKind.SERIAL, push)
        if repetitions > 1 and not reset:
            rep = tir.Var("repeat_%d" % slice_id, "int32")
            body = tir.For(rep, 0, repetitions, tir.ForKind.SERIAL, body)
        if hardware:
            body = statement_sequence([
                tir.Evaluate(tir.call_extern("int32", "VTAUopLoopBegin", 1, 0, 0, 0)),
                body, tir.Evaluate(tir.call_extern("int32", "VTAUopLoopEnd"))])
        return tir.AttrStmt(lane, "coproc_uop_scope", tir.StringImm("VTAPushGEMMOp"), body)

    def scope(stage, body):
        return tir.AttrStmt(lane, "coproc_scope", stage, statement_sequence(body))

    items = [scope(2, [kernel(1, 0)]), scope(1, [load(x, 2, width),
        load(w, 1, segments), dep("push", 1, 2)])]
    for part in range(segments):
        body = ([dep("pop", 1, 2)] if part == 0 else [])
        body.append(kernel(0, part, hardware=loops and part % 2 == 0))
        if internal == "notify" and part == 1:
            body.extend([dep("push", 2, 1)])
        if part == segments - 1:
            body.append(dep("push", 2, 3))
        items.append(scope(2, body))
        if internal == "notify" and part == 1:
            items.append(scope(1, [dep("pop", 2, 1), dep("push", 1, 2)]))
            items.append(dep("pop", 1, 2))
        if internal == "reset" and part == 1:
            items.append(scope(2, [kernel(1, 0)]))
        if internal == "version" and part == 1:
            # A correct ordered LOAD changes all same-address WGT versions.
            items.extend([dep("push", 2, 1), scope(1, [dep("pop", 2, 1),
                load(w, 1, segments, offset=segments), dep("push", 1, 2)]), dep("pop", 1, 2)])
    store = tir.Evaluate(tir.call_extern("int32", "VTAStoreBuffer2D", handle, 0,
        4, y.data, 0, width, 1, width))
    items.extend([scope(3, [dep("pop", 2, 3), store]),
                  tir.Evaluate(tir.call_intrin("int32", "tir.vta.coproc_sync"))])
    return tir.PrimFunc(params, statement_sequence(items), buffer_map=dict(zip(params, (x, w, add, y)))) \
        .with_attr("tir.noalias", True).with_attr("global_symbol", "offline_fixture")


def macs(function):
    return [event for event in trace_function(function) if event["kind"] == "kernel" and event["mode_reset"] == [0, 0]]


def apply(function, q):
    decisions = []
    result = AggregateWindows(q, decisions)(tvm.IRModule({"main": function}))["main"]
    return result, decisions


class WindowExamples(unittest.TestCase):
    def test_renaming_and_non_eight_window(self):
        f = fixture(6)
        original = certify_trace(trace_function(f))
        result, decisions = apply(f, 8)
        self.assertEqual([len(k["uops"]) for k in macs(result)], [24])
        self.assertEqual(decisions[-1]["segments"], 6)
        self.assertEqual(original["output_product_postprocess_sha256"],
                         certify_trace(trace_function(result))["output_product_postprocess_sha256"])
        variables = []
        tir.stmt_functor.post_order_visit(f.body, lambda node: variables.append(node.loop_var)
                                         if isinstance(node, tir.For) else None)
        names = {var: tir.Var("unrelated_%d" % i, var.dtype) for i, var in enumerate(variables)}
        def rename(node):
            if isinstance(node, tir.For):
                var = names[node.loop_var]
                return tir.For(var, node.min, node.extent, node.kind,
                    tir.stmt_functor.substitute(node.body, {node.loop_var: var}),
                    node.thread_binding, node.annotations, node.span)
        renamed = tir.stmt_functor.ir_transform(f.body, None, rename, ["tir.For"])
        renamed_result, _ = apply(f.with_body(renamed).with_attr("global_symbol", "no_node_number"), 8)
        self.assertEqual([len(k["uops"]) for k in macs(renamed_result)], [24])

    def test_same_address_different_loaded_version_cuts(self):
        f = fixture(6, internal="version")
        result, _ = apply(f, 8)
        self.assertEqual([len(k["uops"]) for k in macs(result)], [8, 16])
        self.assertEqual(certify_trace(trace_function(f))["output_product_postprocess_sha256"],
                         certify_trace(trace_function(result))["output_product_postprocess_sha256"])

    def test_internal_necessary_notification_cuts(self):
        result, _ = apply(fixture(6, internal="notify"), 8)
        self.assertEqual([len(k["uops"]) for k in macs(result)], [8, 16])

    def test_reset_boundary(self):
        result, _ = apply(fixture(6, internal="reset"), 8)
        self.assertEqual([len(k["uops"]) for k in macs(result)], [8, 16])

    def test_incompatible_hardware_loop_cuts(self):
        result, decisions = apply(fixture(3, loops=True), 8)
        self.assertEqual(len(macs(result)), 3)
        self.assertIn("INCOMPATIBLE_HARDWARE_LOOP", [d["stop"] for d in decisions])

    def test_capacity_uses_real_8192(self):
        result, decisions = apply(fixture(5, width=2048), 8)
        self.assertEqual([len(k["uops"]) for k in macs(result)], [8192, 2048])
        self.assertIn("UOP_CAPACITY", [d["stop"] for d in decisions])

    def test_single_kernel_exceeds_capacity_rejected(self):
        with self.assertRaisesRegex(Exception, "UOP_KERNEL_CAPACITY"):
            trace_function(fixture(1, width=2048, repetitions=5))

    def test_missing_first_load_rejected(self):
        events = trace_function(fixture())
        events = [e for e in events if e["kind"] != "load" or e["domain"] != 1]
        with self.assertRaisesRegex(ValueError, "MISSING_FIRST_USE_LOAD"):
            certify_trace(events)

    def test_unknown_call_rejected(self):
        f = fixture()
        unknown = tir.Evaluate(tir.call_extern("int32", "UnknownModuleNotification"))
        with self.assertRaisesRegex(Exception, "UNKNOWN_CALL"):
            apply(f.with_body(statement_sequence([unknown, f.body])), 8)

    def test_overwrite_without_read_completion_rejected(self):
        events = trace_function(fixture(6, internal="version"))
        # Remove the old-reader completion handshake. Host adjacency is not
        # an asynchronous ordering proof.
        events = [e for e in events if not (e["kind"] in ("push", "pop") and e.get("pair") == [2, 1])]
        with self.assertRaisesRegex(ValueError, "OVERWRITE_BEFORE_LAST_READ"):
            certify_trace(events)

    def test_postprocess_then_mac_rejected(self):
        events = trace_function(fixture())
        first = next(i for i, e in enumerate(events) if e["kind"] == "kernel" and e["mode_reset"] == [0, 0])
        alu = deepcopy(events[first])
        alu["mode_reset"] = [1, 0]
        for uop in alu["uops"]:
            uop["args"] = [1, 0, uop["args"][2], uop["args"][2], 0, 3, 1, 7]
        events.insert(first + 1, alu)
        with self.assertRaisesRegex(ValueError, "MAC_AFTER_POSTPROCESS"):
            certify_trace(events)

    def test_runtime_hardware_loops_are_cumulative(self):
        f = fixture(1, loops=True)
        kernels = []
        tir.stmt_functor.post_order_visit(f.body, lambda n: kernels.append(n)
            if isinstance(n, tir.AttrStmt) and n.attr_key == "coproc_uop_scope" else None)
        target = kernels[-1]
        def clone_loop(loop):
            if isinstance(loop, tir.For):
                var = tir.Var("second_body_" + str(loop.loop_var), loop.loop_var.dtype)
                return tir.For(var, loop.min, loop.extent, loop.kind,
                    tir.stmt_functor.substitute(loop.body, {loop.loop_var: var}))
        second = tir.stmt_functor.ir_transform(target.body, None, clone_loop, ["tir.For"])
        def duplicate(node):
            if isinstance(node, tir.AttrStmt) and node.same_as(target):
                return tir.AttrStmt(node.node, node.attr_key, node.value,
                    statement_sequence([node.body, second]))
        duplicated = tir.stmt_functor.ir_transform(f.body, None, duplicate, ["tir.AttrStmt"])
        with self.assertRaisesRegex(ValueError, "HARDWARE_LOOP_ARITY"):
            trace_function(f.with_body(duplicated))


if __name__ == "__main__":
    unittest.main()
