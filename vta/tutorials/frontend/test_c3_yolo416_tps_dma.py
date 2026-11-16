"""Real failed TPS IR regression; zero compiled target calls."""
import unittest
import tvm
import vta
from tvm import tir
from c3_tps_sync_release_local import HOLD
from c3_yolo416_tps_dma import SplitStridedOutputChannels, strip_dma

FIXTURE=HOLD/'20261017_c3_yolo416_tps_dma_toolfix01/before_copy.json'


class DMATests(unittest.TestCase):
    def setUp(self):
        self.mod=tvm.ir.load_json(FIXTURE.read_text())

    def adapted(self,mod=None):
        return SplitStridedOutputChannels()(mod or self.mod)

    def copy_pass(self):
        return vta.transform.InjectDMAIntrin()

    def output_copy(self,mod):
        nodes=[]
        def find(n):
            if not isinstance(n,tir.AttrStmt) or n.attr_key!='pragma_dma_copy':return
            loops=[];body=n.body
            while isinstance(body,tir.For):loops.append(body);body=body.body
            if isinstance(body,tir.BufferStore) and body.buffer.scope()=='global':nodes.append((n,loops,body))
        tir.stmt_functor.post_order_visit(list(mod.functions.values())[0].body,find)
        self.assertEqual(len(nodes),1)
        return nodes[0]

    def replace_output(self,change):
        target,loops,store=self.output_copy(self.mod)
        body=tir.stmt_functor.ir_transform(list(self.mod.functions.values())[0].body,None,
             lambda n:change(n,loops,store) if n.same_as(target) else None,['tir.AttrStmt'])
        fn=list(self.mod.functions.values())[0].with_body(body)
        return tvm.IRModule({'main':fn})

    def test_real_original_copy_rejected(self):
        with self.assertRaisesRegex(Exception,'cannot detect 2d pattern'):
            self.copy_pass()(self.mod)

    def test_real_split_copy_lowers(self):
        result=self.copy_pass()(self.adapted())
        calls=[]
        tir.stmt_functor.post_order_visit(list(result.functions.values())[0].body,
            lambda n:calls.append(n) if isinstance(n,tir.Call) and n.op.name=='tir.call_extern'
            and n.args[0].value=='VTAStoreBuffer2D' else None)
        self.assertEqual(len(calls),1) # one static call, under the original CO loop
        call=calls[0]
        self.assertEqual([int(x) for x in call.args[6:9]],[52,16,208])
        co=self.output_copy(self.mod)[1][0].loop_var
        analyzer=tvm.arith.Analyzer()
        for expression,step in ((call.args[2].args[2],13312),(call.args[5],43264)):
            delta=tir.stmt_functor.substitute(expression,{co:co+1})-expression
            self.assertEqual(int(analyzer.simplify(delta)),step)
        _,loops,_=self.output_copy(self.adapted())
        self.assertEqual([int(x.extent) for x in loops],[16,52,16])

    def test_scalar_order_and_addresses_exact(self):
        a=list(self.mod.functions.values())[0];b=list(self.adapted().functions.values())[0]
        self.assertTrue(tvm.ir.structural_equal(strip_dma(a.body),strip_dma(b.body)))
        self.assertEqual(int(b.attrs['c3_output_channel_dma_split']),1)

    def test_idempotent(self):
        a=self.adapted();b=self.adapted(a)
        self.assertTrue(tvm.ir.structural_equal(list(a.functions.values())[0].body,
                                               list(b.functions.values())[0].body))

    def test_nonserial_copy_rejected(self):
        def edit(n,loops,_):
            q=loops[0]
            return tir.AttrStmt(n.node,n.attr_key,n.value,
                tir.For(q.loop_var,q.min,q.extent,tir.ForKind.PARALLEL,q.body))
        with self.assertRaisesRegex(Exception,'unsupported output copy loop semantics'):
            self.adapted(self.replace_output(edit))

    def test_overlapping_destination_rejected(self):
        def edit(n,loops,store):
            index=store.indices[0]-loops[0].loop_var*690000
            new=tir.BufferStore(store.buffer,store.value,[index])
            return tir.stmt_functor.ir_transform(n,None,
                lambda s:new if s.same_as(store) else None,['tir.BufferStore'])
        with self.assertRaisesRegex(Exception,'overlapping output copy slices'):
            self.adapted(self.replace_output(edit))

    def test_already_two_dimensional_copy_retained(self):
        def edit(n,loops,store):
            index=store.indices[0]-loops[1].loop_var*(3328-832)
            new=tir.BufferStore(store.buffer,store.value,[index])
            return tir.stmt_functor.ir_transform(n,None,
                lambda s:new if s.same_as(store) else None,['tir.BufferStore'])
        compact=self.replace_output(edit)
        self.assertTrue(tvm.ir.structural_equal(list(compact.functions.values())[0].body,
                                               list(self.adapted(compact).functions.values())[0].body))


if __name__=='__main__':unittest.main()
