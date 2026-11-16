"""Small CPU-only graph fixtures, never a compiled DNN invocation."""
from copy import deepcopy
import unittest
import numpy as np
from c3_yolo416_graph_reference import evaluate


def fixture():
    return {'nodes':[{'op':'null','name':'x'},{'op':'null','name':'p'},
       {'op':'tvm_op','name':'a','inputs':[[0,0,0],[1,0,0]],
        'attrs':{'func_name':'add_cpu','num_inputs':'2','num_outputs':'1','flatten_data':'0'}},
       {'op':'tvm_op','name':'r','inputs':[[2,0,0]],
        'attrs':{'func_name':'__nop','num_inputs':'1','num_outputs':'1','flatten_data':'0'}}],
       'node_row_ptr':[0,1,2,3,4],
       'attrs':{'shape':['list_shape',[[2],[2],[2],[1,2]]],
                'dltype':['list_str',['int32']*4]},'heads':[[2,0,0],[3,0,0]]}


class ReferenceTests(unittest.TestCase):
    def run_graph(self,g=None,cpu=None,convs=None):
        def add(symbol,inputs,outputs,flatten):
            self.assertEqual(symbol,'add_cpu');return [inputs[0]+inputs[1]]
        return evaluate(g or fixture(),{'p':np.array([2,4],dtype='int32')},
                        {'x':np.array([1,3],dtype='int32')},convs or {},cpu or add)

    def test_cpu_path_and_multiple_heads(self):
        heads,audit=self.run_graph()
        np.testing.assert_array_equal(heads[0],[3,7])
        np.testing.assert_array_equal(heads[1],[[3,7]])
        self.assertEqual(audit['GraphExecutor_target_calls'],0)
        self.assertEqual(len(audit['CPU_calls']),1)

    def test_cpu_readonly_input_mutation_rejected(self):
        def bad(symbol,inputs,outputs,flatten):
            inputs[0][:]=0;return [inputs[0]+inputs[1]]
        with self.assertRaisesRegex(ValueError,'READONLY'):self.run_graph(cpu=bad)

    def test_bad_input_version_and_shape_rejected(self):
        g=deepcopy(fixture());g['nodes'][2]['inputs'][0][2]=1
        with self.assertRaisesRegex(ValueError,'VERSION'):self.run_graph(g)
        g=deepcopy(fixture());g['attrs']['shape'][1][0]=[3]
        with self.assertRaisesRegex(ValueError,'ABI'):self.run_graph(g)

    def test_missing_convolution_is_not_zero_opportunity(self):
        with self.assertRaisesRegex(ValueError,'VTA_FUNCTION_COVERAGE'):
            self.run_graph(convs={'unknown_symbol':None})

    def test_shared_symbol_uses_each_calls_real_inputs(self):
        class FakeSpec:
            abi=[{}, {}, {}]
            def reference(self,inputs):return inputs[0]+inputs[1]
        g=deepcopy(fixture())
        g['attrs']['shape'][1][-1]=[2]
        for i in (2,3):
            g['nodes'][i]['attrs'].update(func_name='conv_fake',num_inputs='2')
        g['nodes'][3]['inputs']=[[0,0,0],[2,0,0]]
        def forbidden(*args):raise AssertionError('convolution dispatched to CPU callback')
        heads,audit=self.run_graph(g,forbidden,{'conv_fake':FakeSpec()})
        np.testing.assert_array_equal(heads[0],[3,7])
        np.testing.assert_array_equal(heads[1],[4,10])
        self.assertEqual(len(audit['convolution_calls']),2)
        self.assertEqual(audit['CPU_calls'],[])


if __name__=='__main__':unittest.main()
