"""Pure integer model checks; no compiled target calls."""
from copy import deepcopy
import unittest
from c3_banerjee_baseline import Tile
from c3_node34_tps_applicability import enumerate_shape
from c3_yolo416_tps_comparison_scope import read,BASE,group_calls
import c3_yolo416_tps_model as m

CAP={'INP':32768,'WGT':262144,'ACC':131072}


class ModelTests(unittest.TestCase):
    def test_all_existing_3x3_rows_equal(self):
        calls=read(BASE/'baseline_identity.json')['callsites'];graph=read(BASE/'B_raw/graph.json')
        for group in group_calls(calls,graph):
            w=group[0]['workload']
            if w[2][1][2:4]==[3,3]:
                dims,_,_=m.geometry(w)
                self.assertEqual(m.enumerate_workload(w,CAP),enumerate_shape(dims,CAP))

    def pointwise(self):
        return ['conv2d_packed.vta',['TENSOR',[1,2,4,4,1,16],'int8'],
                ['TENSOR',[2,2,1,1,16,16],'int8'],[1,1],[0,0,0,0],[1,1],'NCHW1n16c','int32']

    def test_pointwise_independent_hand_calculation(self):
        r=m.cost(Tile(1,2,2,1,1,1,1),self.pointwise(),CAP)
        self.assertEqual(r['paper_allocation_bytes'],{'INP':64,'WGT':256,'ACC':320})
        self.assertEqual(r['paper_payload_bytes'],{'INP':1024,'WGT':4096,'ACC_bias_proxy':512})
        self.assertEqual(r['paper_score_bytes'],5632)

    def test_pointwise_no_3x3_padding_or_weight_factor(self):
        a=self.pointwise();b=deepcopy(a);b[2][1][2:4]=[3,3];b[4]=[1,1,1,1]
        x=m.cost(Tile(1,2,2,1,1,1,1),a,CAP);y=m.cost(Tile(1,2,2,1,1,1,1),b,CAP)
        self.assertEqual(y['paper_allocation_bytes']['WGT'],9*x['paper_allocation_bytes']['WGT'])
        self.assertEqual(y['paper_allocation_bytes']['INP'],4*x['paper_allocation_bytes']['INP'])

    def test_thread_and_capacity_rejections(self):
        r=m.cost(Tile(1,2,2,1,1,2,2),self.pointwise(),CAP)
        self.assertIn('paper_both_threads_two',r['rejections'])
        r=m.cost(Tile(1,4,4,2,2,2,1),self.pointwise(),CAP)
        self.assertIn('local_exact_outer_thread_divisibility',r['rejections'])
        small=dict(CAP,WGT=1)
        self.assertIn('WGT_paper_capacity',m.cost(Tile(1,2,2,1,1,1,1),self.pointwise(),small)['rejections'])

    def test_unsupported_geometry_rejected(self):
        for field,value in ((3,[2,2]),(4,[2,2,2,2]),(5,[2,2])):
            w=self.pointwise();w[field]=value
            with self.assertRaises(ValueError):m.select(w,CAP)

    def test_all_11_functions_select_without_timing(self):
        calls=read(BASE/'baseline_identity.json')['callsites'];graph=read(BASE/'B_raw/graph.json')
        rows=[m.select(g[0]['workload'],CAP) for g in group_calls(calls,graph)]
        self.assertEqual(len(rows),11)
        self.assertTrue(all(not r['performance_read'] and not r['hardware_qualified'] for r in rows))
        self.assertTrue(all(r['selected']['necessary_conditions_passed'] for r in rows))


if __name__=='__main__':unittest.main()
