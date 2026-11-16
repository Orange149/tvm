import unittest
from types import SimpleNamespace
import numpy as np
import tvm,vta
from vta.testing import simulator
from c3_yolo416_graph_remote_reference import RemoteCPUCallback


class RemoteReferenceTests(unittest.TestCase):
    def test_explicit_device_ledger_and_one_CPU_function(self):
        class Module:
            def __init__(self):self.calls=0
            def get_function(self,symbol,imports):
                if symbol!='cpu':return None
                def f(x,y):self.calls+=1;y.copyfrom(x.numpy()+3)
                return f
        module=Module();lookups=[]
        def ledger():lookups.append(1);return '{"attempted_submits":0}'
        abi=[{'kind':'buffer','shape':[2],'dtype':'int32'}]*2
        callback=RemoteCPUCallback(module,{'cpu':{'abi':abi,'instruction_gate':{
            'status':'NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION'}}},tvm.device('ext_dev',0),ledger)
        x=np.array([1,2],dtype='int32')
        output=callback('cpu',[x],[{'dtype':'int32','shape':[2]}],False)
        np.testing.assert_array_equal(output[0],[4,5])
        self.assertEqual(module.calls,1);self.assertEqual(len(lookups),2)
        self.assertEqual(callback.calls,[{'symbol':'cpu','trusted_CPU_function_calls':1,'VTA_submissions':0}])

    def test_changed_submission_ledger_rejects(self):
        class Module:
            def get_function(self,*args):return lambda x,y:y.copyfrom(x.numpy())
        seq=iter(['{"attempted_submits":0}','{"attempted_submits":1}'])
        callback=RemoteCPUCallback(Module(),{'cpu':{'abi':[
            {'kind':'buffer','shape':[1],'dtype':'int32'}]*2,'instruction_gate':{
            'status':'NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION'}}},tvm.device('ext_dev',0),lambda:next(seq))
        with self.assertRaisesRegex(ValueError,'CHANGED_SUBMISSION_LEDGER'):
            callback('cpu',[np.array([1],dtype='int32')],[{'dtype':'int32','shape':[1]}],False)

    def test_wrong_device_or_missing_ledger_reject(self):
        for device,ledger in [(SimpleNamespace(device_type=1),lambda:'{}'),
                              (SimpleNamespace(device_type=140),None)]:
            with self.assertRaises(ValueError):RemoteCPUCallback(None,{},device,ledger)


if __name__=='__main__':unittest.main()
