"""Real CPU TIR gates and CPU fake callbacks; zero DNN target calls."""
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
import tvm,vta
from vta.testing import simulator
from c3_tps_sync_release_local import HOLD,read
from c3_yolo416_graph_cpu_reference import CPUCallback,cpu_instruction_gate


SOURCE=HOLD/'20261017_c3_yolo416_tps_prepare_run02/P'


def abi():
    return [{'kind':'buffer','dtype':'int32','shape':[2]} for _ in range(3)]


class FakeCPU:
    def __init__(self,mutate=False):self.calls=0;self.mutate=mutate
    def get_function(self,symbol,_query):
        if symbol!='add_cpu':return None
        def invoke(x,w,y):
            self.calls+=1
            if self.mutate:x.copyfrom(x.numpy()+1)
            y.copyfrom(x.numpy()+w.numpy())
        return invoke


class CPUReferenceTests(unittest.TestCase):
    def callback(self,mutate=False):
        fake=FakeCPU(mutate)
        return CPUCallback(fake,{'add_cpu':{'abi':abi(),'instruction_gate':{
            'status':'NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION'}}}),fake

    def test_all_real_CPU_functions_do_not_enqueue(self):
        rows=read(SOURCE/'all_final.json');count=0
        for row in rows.values():
            if row['VTA']:continue
            m=tvm.ir.load_json(Path(row['file']).read_text())
            self.assertEqual(cpu_instruction_gate(list(m.functions.values())[0])['status'],
                             'NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION');count+=1
        self.assertEqual(count,21)

    def test_real_VTA_function_rejected(self):
        row=read(SOURCE/'functions.json')[0]
        m=tvm.ir.load_json(Path(row['final']).read_text())
        with self.assertRaisesRegex(ValueError,'WOULD_ENQUEUE'):
            cpu_instruction_gate(list(m.functions.values())[0])

    def test_one_fake_CPU_call_and_flattened_ABI(self):
        callback,fake=self.callback()
        x=np.array([[1,3]],dtype='int32');w=np.array([[2,4]],dtype='int32')
        out=callback('add_cpu',[x,w],[{'shape':[1,2],'dtype':'int32'}],True)
        np.testing.assert_array_equal(out[0],[[3,7]])
        self.assertEqual(fake.calls,1);self.assertEqual(callback.calls[0]['VTA_submissions'],0)

    def test_readonly_mutation_and_unknown_symbol_rejected(self):
        x=np.array([1,3],dtype='int32');w=np.array([2,4],dtype='int32')
        callback,_=self.callback(True)
        with self.assertRaisesRegex(ValueError,'READONLY'):
            callback('add_cpu',[x,w],[{'shape':[2],'dtype':'int32'}],False)
        with self.assertRaisesRegex(ValueError,'UNBOUND'):
            callback('not_CPU',[x,w],[{'shape':[2],'dtype':'int32'}],False)


if __name__=='__main__':unittest.main()
