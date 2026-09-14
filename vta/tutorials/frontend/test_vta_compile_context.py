"""Checks for the compile-context serializer, independent of board execution."""
import unittest

import tvm
from tvm import relay
from audit_vta_compile_context import signature
from compare_vta_segment_tir import normalize
from verify_vta_representative_contexts import cpu_accesses
from summarize_vta_archived_dma import uop_scope_calls


def typed(function):
    return relay.transform.InferType()(tvm.IRModule.from_expr(function))["main"]


class SignatureTests(unittest.TestCase):
    def function(self, name, shift=3, width=4):
        x = relay.var(name, shape=(1, width), dtype="int32")
        return typed(relay.Function([x], relay.right_shift(x, relay.const(shift, "int32"))))

    def test_alpha_rename_invariant(self):
        self.assertEqual(signature(self.function("data"))[0], signature(self.function("renamed"))[0])

    def test_quant_scalar_preserved(self):
        self.assertNotEqual(signature(self.function("data", 3))[0], signature(self.function("data", 4))[0])

    def test_tensor_shape_preserved(self):
        self.assertNotEqual(signature(self.function("data", width=4))[0], signature(self.function("data", width=8))[0])

    def test_generated_function_hash_ignored(self):
        fn = self.function("data")
        self.assertEqual(signature(fn.with_attr("hash", "abc"))[0], signature(fn.with_attr("hash", "def"))[0])

    def test_argument_relationship_preserved(self):
        x = relay.var("x", shape=(1, 4), dtype="int32")
        y = relay.var("y", shape=(1, 4), dtype="int32")
        left = typed(relay.Function([x, y], x + x))
        right = typed(relay.Function([x, y], x + y))
        self.assertNotEqual(signature(left)[0], signature(right)[0])


class TirComparisonTests(unittest.TestCase):
    def function(self, name, extent=4):
        var = tvm.tir.Var(name, "int32")
        body = tvm.tir.For(var, 0, extent, tvm.tir.ForKind.SERIAL,
                          tvm.tir.Evaluate(var))
        return tvm.tir.PrimFunc([], body).with_attr("global_symbol", name)

    def test_generated_symbol_and_loop_names_ignored(self):
        self.assertTrue(tvm.ir.structural_equal(normalize(self.function("left")),
                        normalize(self.function("right")), map_free_vars=True))

    def test_loop_extent_preserved(self):
        self.assertFalse(tvm.ir.structural_equal(normalize(self.function("left", 4)),
                         normalize(self.function("right", 8)), map_free_vars=True))


class CpuAccessTests(unittest.TestCase):
    def test_uop_scope_host_loop_only(self):
        i=tvm.tir.Var("i","int32")
        inner=tvm.tir.For(tvm.tir.Var("j","int32"),0,100,tvm.tir.ForKind.SERIAL,tvm.tir.Evaluate(0))
        scope=tvm.tir.AttrStmt(i,"coproc_uop_scope",tvm.tir.StringImm("VTAPushALUOp"),inner)
        body=tvm.tir.For(i,0,4,tvm.tir.ForKind.SERIAL,scope)
        self.assertEqual(uop_scope_calls(tvm.tir.PrimFunc([],body)),{"VTAPushALUOp":4})

    def test_static_loop_and_dtype_bytes(self):
        data=tvm.tir.decl_buffer((4,),"float32")
        out=tvm.tir.decl_buffer((4,),"float32")
        i=tvm.tir.Var("i","int32")
        body=tvm.tir.For(i,0,4,tvm.tir.ForKind.SERIAL,
                        tvm.tir.BufferStore(out,tvm.tir.BufferLoad(data,[i]),[i]))
        self.assertEqual(cpu_accesses(tvm.tir.PrimFunc([],body)),{"read_bytes":16,"write_bytes":16})

    def test_vector_lanes_counted(self):
        data=tvm.tir.decl_buffer((4,),"int8")
        index=tvm.tir.Ramp(0,1,4)
        body=tvm.tir.Evaluate(tvm.tir.BufferLoad(data,[index]))
        self.assertEqual(cpu_accesses(tvm.tir.PrimFunc([],body)),{"read_bytes":4})

    def test_condition_rejected(self):
        data=tvm.tir.decl_buffer((1,),"int8")
        body=tvm.tir.IfThenElse(tvm.tir.Var("c","bool"),
                               tvm.tir.Evaluate(tvm.tir.BufferLoad(data,[0])),None)
        with self.assertRaisesRegex(ValueError,"Conditional"):
            cpu_accesses(tvm.tir.PrimFunc([],body))

    def test_static_predicate_with_else(self):
        data=tvm.tir.decl_buffer((8,),"int8")
        out=tvm.tir.decl_buffer((8,),"int32")
        i=tvm.tir.Var("i","int32")
        branch=tvm.tir.IfThenElse(i<3,tvm.tir.Evaluate(tvm.tir.BufferLoad(data,[i])),
                                 tvm.tir.BufferStore(out,tvm.tir.const(0,"int32"),[i]))
        body=tvm.tir.For(i,0,8,tvm.tir.ForKind.SERIAL,branch)
        self.assertEqual(cpu_accesses(tvm.tir.PrimFunc([],body)),{"read_bytes":3,"write_bytes":20})


if __name__ == "__main__":
    unittest.main()
