import unittest
from c3_yolo224_two_node_native import frames, strict_material_order, control_only, submit_signature


class NativeTests(unittest.TestCase):
    def test_frame_order_and_missing_end(self):
        text='[C3_GRAPH_BEGIN] foo\nbody\n[C3_GRAPH_END] foo\n'
        self.assertEqual(frames(text,['foo']), [('foo','body\n')])
        with self.assertRaises(ValueError): frames(text,['bar'])
        with self.assertRaises(ValueError): frames(text.replace('[C3_GRAPH_END]','END'),['foo'])

    def test_serial_order_is_stronger_than_unrelated_material(self):
        strict_material_order({'material_map':[{'tir_event':0},{'tir_event':2}], 'ancestors':[0,0,1]})
        with self.assertRaisesRegex(ValueError,'SERIAL_MATERIAL_ORDER'):
            strict_material_order({'material_map':[{'tir_event':0},{'tir_event':2}], 'ancestors':[0,0,0]})

    def test_empty_sync_cannot_hide_material_or_uops(self):
        records=[{'index':0,'stage':2,'bits':(0,0,0,0),'description':'FINISH'}]
        control_only(records,{'serialized_uop_bytes':0})
        with self.assertRaises(ValueError): control_only(records,{'serialized_uop_bytes':4})
        with self.assertRaises(ValueError): control_only([{**records[0],'description':'GEMM'}],{'serialized_uop_bytes':0})

    def test_seventeen_submissions_and_overflow_rejection(self):
        rows=[{'ordinal':i,'completion':True,'timeout':0,'reason':'explicit_sync',
               'insns':1,'insn_bytes':16,'serialized_uop_bytes':0} for i in range(1,18)]
        ledger={'schema':'c3_official_window_submit_ledger_v1','mode':'minimal','overflow':False,
                'attempted_submits':17,'recorded_submits':17,'submits':rows}
        self.assertEqual(len(submit_signature(ledger,'minimal')),17)
        with self.assertRaises(ValueError): submit_signature({**ledger,'overflow':True},'minimal')


if __name__=='__main__':unittest.main()
