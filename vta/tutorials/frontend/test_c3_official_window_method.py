"""Offline counterexamples for official-baseline aggregation, no target calls."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import tvm
from tvm import tir

from c3_valid_window import expanded_uops, statement_sequence
from c3_official_window_method import (AuditFailure, FROZEN_BITSTREAM_SHA256,
    _limits, _trace, aggregate_function, audit_function, structure_statistics)
from test_c3_valid_window import fixture


def macs(audit):
    return [e for e in audit["events"] if e["kind"] == "kernel" and e["mode_reset"] == [0, 0]]


class OfficialExamples(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="c3_official_method_offline_")
        directory = Path(self.temp.name)
        source = directory / "offline_hardware_fixture.txt"
        source.write_text("OFFLINE logical fixture only; no actual bitstream interval claim\n")
        binding = {"status": "BUILD_AND_INTERVAL_VERIFIED", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "bitstream_sha256": FROZEN_BITSTREAM_SHA256, "verify_dep_interval": 2,
                   "scope": "OFFLINE_FIXTURE_NOT_A_PRODUCTION_HARDWARE_CERTIFICATE"}
        path = directory / "offline_binding.json"
        path.write_text(json.dumps(binding, sort_keys=True))
        self.hw = {**binding, "status": "VERIFIED_DEP_INTERVAL", "source_path": str(source),
                   "verified_build_binding_path": str(path),
                   "verified_build_binding_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_hardware_preserves_real_opportunity(self):
        f = fixture(3)
        a, r = aggregate_function(f)
        self.assertTrue(a.same_as(f))
        self.assertEqual(r["status"], "SUPPORT_UNRESOLVED_BASELINE_PRESERVED")
        self.assertEqual(r["B"]["counts"]["expanded_uop_steps"], 16)
        self.assertEqual(r["A"]["counts"]["expanded_uop_steps"], 16)
        self.assertGreater(r["B"]["statistics"]["counts"]["natural_adjacent_seams"], 0)
        self.assertTrue(r["native_fifo_operand_certificate_required"])

    def test_bound_fixture_joins_all_non_eight_segments(self):
        a, r = aggregate_function(fixture(3), self.hw)
        self.assertEqual(r["joins"], 2)
        self.assertEqual([len(k["uops"]) for k in macs(r["A"])], [12])
        self.assertEqual(r["B"]["semantic"]["packed_product_count"], 12)
        self.assertEqual(r["A"]["semantic"]["packed_product_count"], 12)
        self.assertEqual(r["A"]["semantic"]["output_product_postprocess_sha256"],
                         r["B"]["semantic"]["output_product_postprocess_sha256"])

    def test_seam_failure_keeps_longest_legal_prefix(self):
        f = fixture(3)
        kernels = []
        tir.stmt_functor.post_order_visit(f.body, lambda n: kernels.append(n)
            if isinstance(n, tir.AttrStmt) and n.attr_key == "coproc_uop_scope" else None)
        target = kernels[-1]
        def shift_output(call):
            if isinstance(call, tir.Call) and call.op.name == "tir.vta.uop_push":
                args = list(call.args)
                args[2] = (args[2] + 2) % 4
                return tir.call_intrin("int32", call.op, *args)
        def alter(node):
            if isinstance(node, tir.AttrStmt) and node.same_as(target):
                body = tir.stmt_functor.ir_transform(node.body, None, shift_output, ["tir.Call"])
                return tir.AttrStmt(node.node, node.attr_key, node.value, body)
        f = f.with_body(tir.stmt_functor.ir_transform(f.body, None, alter, ["tir.AttrStmt"]))
        _, r = aggregate_function(f, self.hw)
        self.assertEqual([len(k["uops"]) for k in macs(r["A"])], [8, 4])
        self.assertIn("RUNTIME_VERIFYDEP_JOIN_SEAM", [d["stop"] for d in r["decisions"]])

    def test_multiple_kernels_inside_one_compute_scope(self):
        f = fixture(3)
        parts = list(f.body.seq)
        combined = []
        for block in parts[2:5]:
            combined.extend(list(block.body.seq) if isinstance(block.body, tir.SeqStmt) else [block.body])
        outer = parts[2]
        all_compute = tir.AttrStmt(outer.node, outer.attr_key, outer.value, statement_sequence(combined))
        f = f.with_body(statement_sequence([*parts[:2], all_compute, *parts[5:]]))
        self.assertEqual(structure_statistics(f)["counts"]["compute_scopes_with_multiple_direct_kernels"], 1)
        _, r = aggregate_function(f, self.hw)
        self.assertEqual([len(k["uops"]) for k in macs(r["A"])], [12])

    def test_identical_hardware_loops_reorder_but_products_match(self):
        f = fixture(2, width=8)
        def add_loop(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_uop_scope":
                calls = []
                tir.stmt_functor.post_order_visit(node.body, lambda x: calls.append(x)
                    if isinstance(x, tir.Call) and x.op.name == "tir.vta.uop_push" else None)
                if calls and int(calls[0].args[1]) == 0:
                    body = statement_sequence([tir.Evaluate(tir.call_extern("int32", "VTAUopLoopBegin", 2, 0, 0, 0)),
                                               node.body, tir.Evaluate(tir.call_extern("int32", "VTAUopLoopEnd"))])
                    return tir.AttrStmt(node.node, node.attr_key, node.value, body)
        f = f.with_body(tir.stmt_functor.ir_transform(f.body, None, add_loop, ["tir.AttrStmt"]))
        _, r = aggregate_function(f, self.hw)
        before = [v for k in macs(r["B"]) for v in expanded_uops(k)]
        after = [v for k in macs(r["A"]) for v in expanded_uops(k)]
        self.assertNotEqual(before, after)
        self.assertEqual(r["B"]["counts"]["expanded_uop_steps"], 40)
        self.assertEqual(r["A"]["counts"]["expanded_uop_steps"], 40)
        self.assertEqual(r["B"]["semantic"]["output_product_postprocess_sha256"], r["A"]["semantic"]["output_product_postprocess_sha256"])

    def test_tensor_versions_and_notification_boundaries_preserved(self):
        for boundary in ("version", "notify", "reset"):
            _, r = aggregate_function(fixture(6, internal=boundary), self.hw)
            self.assertEqual([len(k["uops"]) for k in macs(r["A"])], [8, 16])
            self.assertEqual(r["B"]["semantic"]["tensor_requests_sha256"], r["A"]["semantic"]["tensor_requests_sha256"])
            self.assertEqual(r["B"]["semantic"]["control_trace_sha256"], r["A"]["semantic"]["control_trace_sha256"])

    def test_capacity_is_longest_8192_prefix(self):
        _, r = aggregate_function(fixture(5, width=2048), self.hw)
        self.assertEqual([len(k["uops"]) for k in macs(r["A"])], [8192, 2048])
        self.assertIn("UOP_CAPACITY", [d["stop"] for d in r["decisions"]])

    def test_host_iterations_are_never_merged(self):
        f = fixture(2)
        parts = list(f.body.seq)
        wrapped = []
        for index, block in enumerate(parts[2:4]):
            iv = tir.Var("host_%d" % index, "int32")
            body = list(block.body.seq) if isinstance(block.body, tir.SeqStmt) else [block.body]
            # The producer/consumer token remains once around the entire loop;
            # duplicating it each iteration would be a different invalid input.
            body = [tir.For(iv, 0, 2, tir.ForKind.SERIAL, child)
                    if isinstance(child, tir.AttrStmt) and child.attr_key == "coproc_uop_scope" else child
                    for child in body]
            wrapped.append(tir.AttrStmt(block.node, block.attr_key, block.value, statement_sequence(body)))
        f = f.with_body(statement_sequence([*parts[:2], *wrapped, *parts[4:]]))
        a, r = aggregate_function(f, self.hw)
        self.assertTrue(a.same_as(f))
        self.assertEqual(r["joins"], 0)
        self.assertEqual(len(macs(r["A"])), 4)
        self.assertTrue(r["host_loop_iterations_preserved"])

    def test_unknown_attributes_and_parallel_for_rejected(self):
        f = fixture(2)
        for body in (tir.AttrStmt(tir.Var("opaque", "handle"), "UnknownOrderingEffect", 1, f.body),
                     tir.For(tir.Var("parallel", "int32"), 0, 2, tir.ForKind.PARALLEL, f.body)):
            with self.assertRaises(AuditFailure) as error:
                audit_function(f.with_body(body))
            self.assertEqual(error.exception.category, "UNSUPPORTED_SEMANTICS")

    def test_hardware_hashes_without_build_binding_not_verified(self):
        hw = deepcopy(self.hw)
        hw.pop("verified_build_binding_path")
        a, r = aggregate_function(fixture(2), hw)
        self.assertEqual(r["status"], "SUPPORT_UNRESOLVED_BASELINE_PRESERVED")
        self.assertFalse(r["hardware_contract_valid_for_join"])

    def test_exact_five_million_boundary_not_ast_visits(self):
        f = fixture(1, width=400)
        reset = f.body.seq[0].body
        def loop_kernel(extent):
            body = statement_sequence([tir.Evaluate(tir.call_extern("int32", "VTAUopLoopBegin", extent, 0, 0, 0)),
                                       reset.body, tir.Evaluate(tir.call_extern("int32", "VTAUopLoopEnd"))])
            k = tir.AttrStmt(reset.node, reset.attr_key, reset.value, body)
            outer = f.body.seq[0]
            return f.with_body(tir.AttrStmt(outer.node, outer.attr_key, outer.value, k))
        events, count = _trace(loop_kernel(12500), _limits(None))
        self.assertEqual(count["expanded_uop_steps"], 5_000_000)
        self.assertLess(count["tir_walk_visits"], 2000)
        with self.assertRaises(AuditFailure) as error:
            _trace(loop_kernel(12501), _limits(None))
        self.assertEqual(error.exception.category, "AUDIT_STEP_LIMIT")
        self.assertEqual(error.exception.details["expanded_uop_steps"], 5_000_400)

    def test_mixed_modes_and_missing_operand_load_are_rejected(self):
        f = fixture(2)
        parts = list(f.body.seq)
        load_scope = parts[1]
        # Both MAC kernels still reference WGT, but this removes its producer.
        calls = list(load_scope.body.seq)
        parts[1] = tir.AttrStmt(load_scope.node, load_scope.attr_key, load_scope.value,
                               statement_sequence([calls[0], *calls[2:]]))
        with self.assertRaisesRegex(AuditFailure, "MISSING_FIRST_USE_LOAD"):
            audit_function(f.with_body(statement_sequence(parts)))

    def _signature_fixture(self, count, dtype="int32"):
        f = fixture(1)
        co = tir.Var("captured_co", dtype)
        parts = list(f.body.seq)
        load = parts[1]
        def widen(call):
            if isinstance(call, tir.Call) and call.op.name == "tir.call_extern" and call.args[0].value == "VTALoadBuffer2D" and int(call.args[-1]) == 1:
                args = list(call.args[1:])
                args[3], args[5] = tir.IntImm("int32", count), tir.IntImm("int32", count)
                return tir.call_extern("int32", "VTALoadBuffer2D", *args)
        parts[1] = tir.AttrStmt(load.node, load.attr_key, load.value,
            tir.stmt_functor.ir_transform(load.body, None, widen, ["tir.Call"]))
        block = parts[2]
        body = list(block.body.seq)
        for at, child in enumerate(body):
            if isinstance(child, tir.AttrStmt) and child.attr_key == "coproc_uop_scope":
                def capture(call):
                    if isinstance(call, tir.Call) and call.op.name == "tir.vta.uop_push":
                        args = list(call.args)
                        args[4] = co
                        return tir.call_intrin("int32", call.op, *args)
                changed = tir.AttrStmt(child.node, child.attr_key, child.value,
                    tir.stmt_functor.ir_transform(child.body, None, capture, ["tir.Call"]))
                body[at] = tir.For(co, 0, count, tir.ForKind.SERIAL, changed)
        parts[2] = tir.AttrStmt(block.node, block.attr_key, block.value, statement_sequence(body))
        return f.with_body(statement_sequence(parts))

    def test_signature_accepts_zero_through_98_and_rejects_99(self):
        a = audit_function(self._signature_fixture(99))
        self.assertEqual([k["signature"] for k in macs(a)], [(i,) for i in range(99)])
        self.assertEqual(a["counts"]["distinct_cache_identities"], 100)
        with self.assertRaisesRegex(AuditFailure, "CACHE_SIGNATURE_ENCODING_RANGE"):
            audit_function(self._signature_fixture(100))
        with self.assertRaisesRegex(AuditFailure, "CACHE_SIGNATURE_ARITY_OR_DTYPE"):
            audit_function(self._signature_fixture(1, "int64"))

    def test_merged_two_int_signature_rejected_before_publication(self):
        f = fixture(2)
        co, k = tir.Var("captured_co", "int32"), tir.Var("captured_k", "int32")
        parts = list(f.body.seq)
        for offset, variable in enumerate((co, k)):
            block = parts[2+offset]
            def alter(node):
                if isinstance(node, tir.Call) and node.op.name == "tir.vta.uop_push":
                    args = list(node.args)
                    args[4] = variable+offset
                    return tir.call_intrin("int32", node.op, *args)
            parts[2+offset] = tir.AttrStmt(block.node, block.attr_key, block.value,
                tir.stmt_functor.ir_transform(block.body, None, alter, ["tir.Call"]))
        f = f.with_body(tir.For(co, 0, 1, tir.ForKind.SERIAL,
            tir.For(k, 0, 1, tir.ForKind.SERIAL, statement_sequence(parts))))
        a, report = aggregate_function(f, self.hw)
        self.assertTrue(a.same_as(f))
        self.assertEqual(report["joins"], 0)
        self.assertIn("MERGED_CACHE_SIGNATURE_ARITY_OR_DTYPE", [d["stop"] for d in report["decisions"]])
        self.assertEqual(report["status"], "SUPPORT_UNRESOLVED_BASELINE_PRESERVED")
        decision = next(d for d in report["decisions"] if d["stop"] == "MERGED_CACHE_SIGNATURE_ARITY_OR_DTYPE")
        self.assertEqual(decision["category"], "UNSUPPORTED_SEMANTICS")

    def _ambiguous_source_fixture(self, necessary_barrier=False):
        # Two distinct empty-signature cache sites intentionally share the same
        # initializer body. A third has a different body. The current bounded
        # source-hash witness map has multiplicity 2,2,1, so it cannot infer the
        # dynamic correspondence at the third seam. This is valid B dataflow,
        # rather than an injected audit exception or an illegal operand load.
        f = fixture(3)
        parts = list(f.body.seq)
        first = next(x.body for x in parts[2].body.seq
                     if isinstance(x, tir.AttrStmt) and x.attr_key == "coproc_uop_scope")
        # Fresh definitions keep the full merged body SSA, while identical
        # names/content intentionally produce the same scope-local source SHA.
        second_initializer = tvm.ir.load_json(tvm.ir.save_json(first))
        second = parts[3]
        body = [tir.AttrStmt(x.node, x.attr_key, x.value, second_initializer)
                if isinstance(x, tir.AttrStmt) and x.attr_key == "coproc_uop_scope" else x
                for x in (list(second.body.seq) if isinstance(second.body, tir.SeqStmt) else [second.body])]
        parts[3] = tir.AttrStmt(second.node, second.attr_key, second.value, statement_sequence(body))
        if necessary_barrier:
            # Keep the first duplicate site in a separate notification window,
            # leaving only the ambiguous second/third seam available to join.
            lane = parts[2].node
            def dep(kind, src, dst):
                return tir.Evaluate(tir.call_intrin("int32", "tir.vta.coproc_dep_" + kind, src, dst))
            first_block = parts[2]
            first_parts = list(first_block.body.seq)
            parts[2] = tir.AttrStmt(lane, first_block.attr_key, first_block.value,
                                   statement_sequence([*first_parts, dep("push", 2, 1)]))
            parts[3:3] = [tir.AttrStmt(lane, "coproc_scope", 1,
                                      statement_sequence([dep("pop", 2, 1), dep("push", 1, 2)])),
                            dep("pop", 1, 2)]
        return f.with_body(statement_sequence(parts))

    def test_ambiguous_dynamic_source_preserves_boundary_and_reports_unresolved(self):
        f = self._ambiguous_source_fixture()
        a, report = aggregate_function(f, self.hw)
        self.assertEqual(report["status"], "SUPPORT_UNRESOLVED_PARTIAL_PROJECTED_NATIVE_PENDING")
        self.assertEqual(report["joins"], 1)
        self.assertEqual([len(k["uops"]) for k in macs(report["A"])], [8, 4])
        decision = next(d for d in report["decisions"] if d["stop"] == "JOIN_DYNAMIC_SCOPE_PROVENANCE_UNRESOLVED")
        self.assertEqual(decision["category"], "UNSUPPORTED_SEMANTICS")
        self.assertEqual(report["B"]["semantic"]["output_product_postprocess_sha256"],
                         report["A"]["semantic"]["output_product_postprocess_sha256"])

    def test_unresolved_source_without_legal_prefix_is_not_no_join(self):
        f = self._ambiguous_source_fixture(necessary_barrier=True)
        a, report = aggregate_function(f, self.hw)
        self.assertTrue(a.same_as(f))
        self.assertEqual(report["joins"], 0)
        self.assertEqual(report["status"], "SUPPORT_UNRESOLVED_BASELINE_PRESERVED")
        unknown = next(d for d in report["decisions"] if d["stop"] == "JOIN_DYNAMIC_SCOPE_PROVENANCE_UNRESOLVED")
        self.assertEqual(unknown["category"], "UNSUPPORTED_SEMANTICS")
        self.assertEqual([len(k["uops"]) for k in macs(report["A"])], [4, 4, 4])

    def test_dynamic_initializer_extent_preserves_exact_B_and_is_unresolved(self):
        f = fixture(2)
        co = tir.Var("dynamic_initializer_count", "int32")
        parts = list(f.body.seq)
        kernels = []
        for block in parts[2:4]:
            child = next(x for x in block.body.seq if isinstance(x, tir.AttrStmt)
                         and x.attr_key == "coproc_uop_scope")
            original = child.body
            dynamic = tir.For(original.loop_var, original.min, co+3, original.kind,
                              original.body, original.thread_binding, original.annotations)
            kernels.append(tir.AttrStmt(block.node, block.attr_key, block.value,
                                       tir.AttrStmt(child.node, child.attr_key, child.value, dynamic)))
        # Notifications occur once around the original two host iterations;
        # the variable remains a legitimate one-int kernel cache signature.
        before = parts[2].body.seq[0]
        after = parts[3].body.seq[-1]
        loop = tir.For(co, 0, 2, tir.ForKind.SERIAL, statement_sequence(kernels))
        f = f.with_body(statement_sequence([*parts[:2], before, loop, after, *parts[4:]]))
        a, report = aggregate_function(f, self.hw)
        self.assertTrue(a.same_as(f))
        self.assertEqual(report["joins"], 0)
        self.assertEqual(report["status"], "SUPPORT_UNRESOLVED_BASELINE_PRESERVED")
        self.assertEqual(report["B"]["semantic"]["packed_product_count"], 14)
        self.assertEqual([k["signature"] for k in macs(report["B"])], [(0,), (0,), (1,), (1,)])
        self.assertTrue(any(d.get("category") == "UNSUPPORTED_SEMANTICS" and
                            d["stop"].startswith("SUPPORT_UNRESOLVED_KERNEL_CANDIDATE:")
                            for d in report["decisions"]))

    def test_extern_uop_push_candidate_alias_is_unresolved(self):
        f = fixture(2)
        def alias(call):
            if isinstance(call, tir.Call) and call.op.name == "tir.vta.uop_push" and int(call.args[1]) == 0:
                return tir.call_extern("int32", "VTAUopPush", *call.args)
        f = f.with_body(tir.stmt_functor.ir_transform(f.body, None, alias, ["tir.Call"]))
        a, report = aggregate_function(f, self.hw)
        self.assertTrue(a.same_as(f))
        self.assertEqual(report["joins"], 0)
        self.assertEqual(report["status"], "SUPPORT_UNRESOLVED_BASELINE_PRESERVED")
        self.assertEqual(report["B"]["semantic"]["packed_product_count"], 8)
        self.assertIn("SUPPORT_UNRESOLVED_KERNEL_CANDIDATE:UNKNOWN_KERNEL_EFFECT", [d["stop"] for d in report["decisions"]])

    def test_empty_signatures_different_sites_keep_distinct_identity(self):
        f = fixture(2)
        parts = list(f.body.seq)
        first_body = next(x.body for x in parts[2].body.seq if isinstance(x, tir.AttrStmt) and x.attr_key == "coproc_uop_scope")
        second = parts[3]
        body = list(second.body.seq)
        body = [tir.AttrStmt(x.node, x.attr_key, x.value, first_body)
                if isinstance(x, tir.AttrStmt) and x.attr_key == "coproc_uop_scope" else x for x in body]
        parts[3] = tir.AttrStmt(second.node, second.attr_key, second.value, statement_sequence(body))
        audit = audit_function(f.with_body(statement_sequence(parts)))
        kernels = macs(audit)
        self.assertEqual(kernels[0]["uops"], kernels[1]["uops"])
        self.assertEqual(kernels[0]["signature"], ())
        self.assertNotEqual(kernels[0]["cache_identity"], kernels[1]["cache_identity"])
        self.assertEqual(audit["counts"]["distinct_cache_identities"], 3)

    def test_incompatible_hardware_loop_is_a_real_boundary(self):
        f = fixture(3, loops=True)
        _, report = aggregate_function(f, self.hw)
        self.assertEqual(len(macs(report["A"])), 3)
        self.assertIn("INCOMPATIBLE_HARDWARE_LOOP", [d["stop"] for d in report["decisions"]])


if __name__ == "__main__":
    unittest.main()
