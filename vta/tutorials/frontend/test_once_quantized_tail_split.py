"""Host-only tests of the dependency-closed once-quantized cut."""
import unittest
import numpy as np
import tvm
from tvm import relay

from qualify_once_quantized_tail_split import split,tail_edges,typed


class SplitTests(unittest.TestCase):
    def graph(self):
        a=relay.var("a",shape=(1,4),dtype="int8")
        b=relay.var("b",shape=(1,4),dtype="int8")
        left=relay.annotation.stop_fusion(a)
        right=relay.annotation.stop_fusion(b)
        total=relay.nn.relu(relay.cast(left,"int32")+relay.cast(right,"int32"))
        out=relay.annotation.stop_fusion(relay.cast(total,"int8"))
        return typed(relay.cast(out,"float32")*relay.const(0.0625,"float32"),[a,b])

    def test_typed_recomposition(self):
        mod=self.graph()
        producer,consumer=split(mod,tail_edges(mod))
        self.assertEqual(len(producer["main"].body.fields),2)
        self.assertEqual(len(consumer["main"].params),2)
        self.assertTrue(all(str(p.checked_type.dtype)=="int8" for p in consumer["main"].params))

    def test_float_transport_contract(self):
        mod=self.graph()
        _,consumer=split(mod,tail_edges(mod),True)
        self.assertTrue(all(str(p.checked_type.dtype)=="float32" for p in consumer["main"].params))

    def test_all_int8_values_have_exact_binary_scale_roundtrip(self):
        values=np.arange(-128,128,dtype="int16").astype("int8")
        restored=(values.astype("float32")*np.float32(0.0625)*np.float32(16)).astype("int8")
        self.assertTrue(np.array_equal(values,restored))

    def test_duplicate_cut_rejected(self):
        mod=self.graph()
        edge=tail_edges(mod)[0]
        with self.assertRaises(AssertionError):
            split(mod,[edge,edge])

    def test_wrong_terminal_graph_rejected(self):
        x=relay.var("x",shape=(4,),dtype="float32")
        with self.assertRaises(AssertionError):
            tail_edges(typed(relay.nn.relu(x),[x]))


if __name__=="__main__":
    unittest.main()
