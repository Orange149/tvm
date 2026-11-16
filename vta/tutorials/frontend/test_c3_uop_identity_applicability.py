"""Read-only equivalence and scope/domain boundary regressions."""
from copy import deepcopy
import unittest
from c3_uop_identity_applicability import classify,inspect_function


def event(k,bank=None,scope=0):
    return {'scope':scope,'signature':[] if k is None else [k],'mode':'VTAPushGEMMOp',
            'uops':[{'args':[0,0,0,k%3 if bank is None else bank,0,0,0,0],
                     'loops':[[4,1,1,0]]}]}


class ContentTests(unittest.TestCase):
    def test_inventory_uses_final_requalification_index(self):
        from c3_uop_identity_applicability import inventory
        row=next(r for r in inventory() if r['id']=='yolov3_tiny_416_node014')
        self.assertIn('/local_requalification01/',row['native'])

    def test_discovers_period_three_without_names_or_bank_count(self):
        r=classify([event(k) for k in range(12)])[0]
        self.assertEqual((r['signatures'],r['content_classes'],r['redundant_identities']),(12,3,9))
        self.assertEqual(r['smallest_period_on_complete_frozen_domain'],3)

    def test_aperiodic_classes_use_original_representative(self):
        r=classify([event(k,b) for k,b in zip(range(4,8),(1,2,2,3))])[0]
        self.assertEqual([c['representative'] for c in r['classes']],[[4],[5],[7]])
        self.assertIsNone(r['smallest_period_on_complete_frozen_domain'])

    def test_full_loops_and_alu_immediates_are_not_ignored(self):
        a=event(0,0);b=event(1,0);b['uops'][0]['loops'][0][0]=5
        self.assertEqual(classify([a,b])[0]['content_classes'],2)
        for e,imm in ((a,7),(b,8)):
            e['mode']='VTAPushALUOp';e['uops'][0]['args']=[1,0,0,0,0,3,1,imm]
            e['uops'][0]['loops']=[[4,1,1,0]]
        self.assertEqual(classify([a,b])[0]['content_classes'],2)

    def test_separate_static_handles_stay_separate(self):
        rows=classify([event(0,0,0),event(1,0,1)])
        self.assertEqual(sum(r['redundant_identities'] for r in rows),0)

    def test_same_identity_different_content_and_bad_key_rejected(self):
        for sequence in ([event(0,0),event(0,1)],[event(99,0)]):
            with self.assertRaises(ValueError):classify(sequence)

    def test_empty_signature_has_no_compressible_domain(self):
        self.assertEqual(classify([event(None,0)])[0]['redundant_identities'],0)

    def test_unknown_conditional_is_unresolved(self):
        import tvm,vta
        from tvm import tir
        from c3_uop_identity_applicability import HOLD
        from c3_tps_sync_release_prepare import complete_function
        source=HOLD/'20261017_c3_node34_tps_local_run02/prepare/R/parent.json'
        mod=tvm.ir.load_json(source.read_text());fn=mod[list(mod.functions)[0]]
        unknown=fn.with_body(tir.IfThenElse(tir.IntImm('bool',1),fn.body,None))
        result=inspect_function(unknown)
        self.assertEqual(result['status'],'SUPPORT_UNRESOLVED')
        self.assertTrue(result['unresolved'])

    def test_two_free_integers_are_unresolved(self):
        import tvm,vta
        from tvm import tir
        from c3_uop_identity_applicability import HOLD
        mod=tvm.ir.load_json((HOLD/'20261017_c3_node34_tps_local_run02/prepare/R/parent.json').read_text())
        fn=mod[list(mod.functions)[0]];extra=tir.Var('independent_unknown','int32')
        def edit(node):
            if isinstance(node,tir.AttrStmt) and node.attr_key=='coproc_uop_scope' and len(tir.analysis.undefined_vars(node.body))==1:
                def call(x):
                    if isinstance(x,tir.Call) and x.op.name=='tir.vta.uop_push':
                        args=list(x.args);args[2]=args[2]+extra
                        return tir.Call(x.dtype,x.op,args)
                body=tir.stmt_functor.ir_transform(node.body,None,call,['tir.Call'])
                return tir.AttrStmt(node.node,node.attr_key,node.value,body)
        changed=fn.with_body(tir.stmt_functor.ir_transform(fn.body,edit,None,['tir.AttrStmt']))
        r=inspect_function(changed)
        self.assertEqual(r['status'],'SUPPORT_UNRESOLVED')
        self.assertTrue(any('CLOSURE_ARITY' in x['reason'] for x in r['unresolved']))


if __name__=='__main__':unittest.main()
