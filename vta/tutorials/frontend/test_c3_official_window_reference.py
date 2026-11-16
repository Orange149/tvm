"""Independent reference tests with explicit arithmetic, not target executions."""

from copy import deepcopy
from itertools import product
import unittest
import numpy as np

import c3_official_window_reference as ref


def fixture(h=3,w=4,kh=3,kw=2,stride=(1,1),padding=(1,0,0,1),dilation=(1,1),residual=False):
    xshape=(1,1,h,w,1,16)
    wshape=(1,1,kh,kw,16,16)
    oh=(h+padding[0]+padding[2]-(kh-1)*dilation[0]-1)//stride[0]+1
    ow=(w+padding[1]+padding[3]-(kw-1)*dilation[1]-1)//stride[1]+1
    yshape=(1,1,oh,ow,1,16)
    descriptors=[(xshape,"int8"),(wshape,"int8"),((1,1,1,1,16),"int32")]
    if residual:
        descriptors.append((yshape,"int32"))
    descriptors.append((yshape,"int8"))
    abi=[{"ordinal":i,"kind":"buffer","shape":list(shape),"dtype":dtype,"strides":[],"elem_offset":"0"}
         for i,(shape,dtype) in enumerate(descriptors)]
    workload=["conv2d_packed.vta",["TENSOR",list(xshape),"int8"],["TENSOR",list(wshape),"int8"],
              list(stride),list(padding),list(dilation),"NCHW1n16c","int32"]
    tail=[{"op":"add","source":{"kind":"buffer","param":2}}]
    if residual:
        tail.append({"op":"add","source":{"kind":"buffer","param":3}})
    tail += [{"op":"max","source":{"kind":"imm","value":0}},
             {"op":"add","source":{"kind":"imm","value":4}},
             {"op":"shr","source":{"kind":"imm","value":3}},
             {"op":"min","source":{"kind":"imm","value":127}},
             {"op":"max","source":{"kind":"imm","value":-127}}]
    return workload,abi,tail


def scalar_convolution(spec,values):
    result=np.zeros(spec.output_shape,dtype=np.int32)
    for n,co,y,x,b,o in product(*(range(v) for v in spec.output_shape)):
        total=0
        for ci,ky,kx,i in product(range(spec.xshape[1]),range(spec.wshape[2]),range(spec.wshape[3]),range(16)):
            ih=y*spec.strides[0]+ky*spec.dilation[0]-spec.padding[0]
            iw=x*spec.strides[1]+kx*spec.dilation[1]-spec.padding[1]
            if 0<=ih<spec.xshape[2] and 0<=iw<spec.xshape[3]:
                total += int(values[0][n,ci,ih,iw,b,i])*int(values[1][co,ci,ky,kx,o,i])
        result[n,co,y,x,b,o]=total
    return result


class ReferenceTests(unittest.TestCase):
    def test_matrix_layout_strided_dilated_asymmetric_padding(self):
        spec=ref.from_workload(*fixture(h=5,w=6,kh=2,kw=2,stride=(2,1),dilation=(2,2)))
        values=spec.arrays(17)
        actual=spec.raw_convolution(values)
        self.assertTrue(np.array_equal(actual,scalar_convolution(spec,values)))

    def test_bias_residual_round_shift_and_relu_follow_actual_tail(self):
        spec=ref.from_workload(*fixture(residual=True))
        values=spec.arrays(19)
        raw=scalar_convolution(spec,values)
        expected=np.add(raw,np.broadcast_to(values[2],spec.output_shape),dtype=np.int32)
        expected=np.add(expected,values[3],dtype=np.int32)
        expected=np.maximum(expected,0)
        expected=np.right_shift(np.add(expected,np.int32(4),dtype=np.int32),3)
        expected=np.maximum(np.minimum(expected,127),-127).astype(np.int8)
        self.assertTrue(np.array_equal(spec.reference(values),expected))
        self.assertTrue(np.array_equal(spec.reference(values[:-1]),expected))
        self.assertFalse(spec.contract()["ADD_overflow_bound_claimed"])

    def test_int32_add_wrap_preserved_not_claimed_no_overflow(self):
        workload,abi,_tail=fixture(kh=1,kw=1,padding=(0,0,0,0))
        tail=[{"op":"add","source":{"kind":"buffer","param":2}},
              {"op":"max","source":{"kind":"imm","value":0}},
              {"op":"min","source":{"kind":"imm","value":127}}]
        spec=ref.from_workload(workload,abi,tail)
        values=spec.arrays(1)
        values[0].fill(1); values[1].fill(1); values[2].fill(np.iinfo(np.int32).max)
        expected=np.int8(0)
        self.assertTrue(np.all(spec.reference(values)==expected))

    def test_qualification_cases_observable_ABA_and_sentinel(self):
        spec=ref.from_workload(*fixture(residual=True))
        cases=spec.qualification_cases()
        self.assertEqual(len(cases),7)
        self.assertEqual([row["kind"] for row in cases[3:6]], ["weight_ABA_W0","weight_ABA_W1","weight_ABA_W0_return"])
        self.assertTrue(np.array_equal(cases[3]["expected"],cases[5]["expected"]))
        self.assertFalse(np.array_equal(cases[3]["expected"],cases[4]["expected"]))
        self.assertGreater(cases[6]["nonzero_outputs"],0)
        for row in cases:
            self.assertEqual(len(row["inputs"]),len(spec.abi)-1)
            self.assertTrue(np.all(row["arrays"][-1] != row["expected"]))
        self.assertTrue(np.all(ref.output_sentinel(np.zeros((16,),dtype=np.int8))==1))

    def test_unobservable_weight_switch_refuses_not_allpositive(self):
        workload,abi,_=fixture()
        spec=ref.from_workload(workload,abi,[{"op":"min","source":{"kind":"imm","value":0}},
                                         {"op":"max","source":{"kind":"imm","value":0}}])
        with self.assertRaisesRegex(ref.SupportUnresolved,"observably distinct"):
            spec.qualification_cases()

    def test_uniform_tail_broadcast_maps_and_reject_wrong_source(self):
        workload,abi,tail=fixture(residual=True)
        spec=ref.from_workload(workload,abi,tail)
        outputs=[]
        count=np.prod(spec.output_shape[:-1])
        for index in range(count):
            phases=[[2,0,["tensor",[2,0]]],[2,0,["tensor",[3,index]]],
                    [1,1,["imm",0]],[2,1,["imm",4]],[3,1,["imm",3]],
                    [0,1,["imm",127]],[1,1,["imm",-127]]]
            outputs.append([[len(abi)-1,index],[],phases])
        self.assertEqual(ref.postprocess_from_output_terms(outputs,abi),tail)
        wrong=deepcopy(outputs); wrong[0][2][1][2][1][1]=1
        with self.assertRaisesRegex(ref.SupportUnresolved,"broadcast"):
            ref.postprocess_from_output_terms(wrong,abi)
        wrong=deepcopy(outputs); wrong[-1][2][3][2][1]=5
        with self.assertRaisesRegex(ref.SupportUnresolved,"different postprocess"):
            ref.postprocess_from_output_terms(wrong,abi)

    def test_exact_product_multisets_padding_and_duplicate_rejected(self):
        spec=ref.from_workload(*fixture(h=1,w=1,kh=1,kw=1,padding=(1,1,1,1)))
        outputs=[]
        for index in range(9):
            origin=[0,0] if index==4 else ["zero"]
            phases=[[2,0,["tensor",[2,0]]],[1,1,["imm",0]],[2,1,["imm",4]],
                    [3,1,["imm",3]],[0,1,["imm",127]],[1,1,["imm",-127]]]
            outputs.append([[3,index],[[origin,[1,0],1]],phases])
        self.assertEqual(ref.semantic_gates(spec,outputs)["packed_outputs"],9)
        wrong=deepcopy(outputs); wrong[0][1][0][2]=2
        with self.assertRaisesRegex(ref.SupportUnresolved,"duplicated"):
            ref.validate_product_terms(spec,wrong)
        wrong=deepcopy(outputs); wrong[4][1][0][0]=["zero"]
        with self.assertRaises(ref.SupportUnresolved):
            ref.validate_product_terms(spec,wrong)

    def test_unknown_layout_stride_alias_view_and_tail_rejected(self):
        workload,abi,tail=fixture()
        for mutate in (lambda w,a,t:w.__setitem__(6,"NCHW"),
                       lambda w,a,t:a[0].update(elem_offset="16"),
                       lambda w,a,t:a[0].update(strides=["1"]),
                       lambda w,a,t:t[0]["source"].update(param=0),
                       lambda w,a,t:t.append({"op":"shr","source":{"kind":"imm","value":16}}),
                       lambda w,a,t:t.append({"op":"mul","source":{"kind":"imm","value":2}})):
            w,a,t=deepcopy((workload,abi,tail)); mutate(w,a,t)
            with self.assertRaises(ref.SupportUnresolved):
                ref.from_workload(w,a,t)


if __name__ == "__main__":
    unittest.main()
