"""Local CPU fake/RPC representation checks; no graph or accelerator calls."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tvm
from tvm.contrib import cc
from c3_tps_sync_release_local import REPO,HERE
from test_c3_yolo416_graph_observer import FAKE


class RpcObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        (root/'fake.cc').write_text(FAKE)
        cc.create_shared(str(root/'helper.so'),[str(HERE/'c3_yolo416_graph_observer.cc'),
            str(HERE/'c3_yolo416_rpc_observer.cc'),str(root/'fake.cc')],
            options=['-std=c++17','-O2','-ldl',*['-I'+str(REPO/p) for p in
                ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]])
        cls.lib=tvm.runtime.load_module(str(root/'helper.so'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_flat_metadata_and_individual_snapshots(self):
        fake=tvm.get_global_func('c3.yolo416.observer_test_fake')()
        proxy=tvm.get_global_func('c3.yolo416.rpc_observe_module')(fake,'9 fake_conv 3 1\n57 fake_conv 3 1')
        x,w,a,y=[tvm.nd.array(np.array([v,v+1],dtype=d)) for v,d in
                 ((1,'int8'),(3,'int8'),(5,'int32'),(0,'int8'))]
        proxy['c3_yolo416_observer_begin']()
        self.assertEqual(proxy['fake_conv'](x,w,a,y),42)
        w.copyfrom(np.array([7,8],dtype='int8'));self.assertEqual(proxy['fake_conv'](x,w,a,y),42)
        self.assertEqual(proxy['c3_yolo416_observer_end'](),2)
        rows=json.loads(proxy['c3_yolo416_observer_metadata']())
        self.assertEqual([int(r['node']) for r in rows],[9,57])
        self.assertEqual([r['symbol'] for r in rows],['fake_conv','fake_conv'])
        self.assertEqual(fake['fake_calls'](),2)
        self.assertEqual(rows[0]['inputs_before_count'],3);self.assertEqual(rows[0]['outputs_count'],1)
        self.assertEqual(rows[0]['data_handles'],rows[1]['data_handles'])
        first=proxy['c3_yolo416_observer_snapshot'](0,0,1).numpy()
        second=proxy['c3_yolo416_observer_snapshot'](1,0,1).numpy()
        np.testing.assert_array_equal(first,[3,4]);np.testing.assert_array_equal(second,[7,8])
        with self.assertRaises(Exception):proxy['c3_yolo416_observer_snapshot'](2,0,0)

    def test_local_RPC_module_JSON_and_NDArray_wire_types(self):
        # LocalSession uses RPC's argument/return encoders without a TCP
        # listener, remote process, graph run or accelerator instruction.
        from tvm import rpc
        session=rpc.LocalSession()
        fake=session.get_function('c3.yolo416.observer_test_fake')()
        proxy=session.get_function('c3.yolo416.rpc_observe_module')(fake,'9 fake_conv 3 1')
        arrays=[tvm.nd.array(np.array(v,dtype=d),session.cpu(0)) for v,d in
                (([1,3],'int8'),([2,4],'int8'),([1,1],'int32'),([0,0],'int8'))]
        proxy['c3_yolo416_observer_begin']()
        self.assertEqual(proxy['fake_conv'](*arrays),42)
        self.assertEqual(proxy['c3_yolo416_observer_end'](),1)
        row=json.loads(proxy['c3_yolo416_observer_metadata']())[0]
        self.assertEqual(row['symbol'],'fake_conv')
        np.testing.assert_array_equal(proxy['c3_yolo416_observer_snapshot'](0,2,0).numpy(),[4,8])
        self.assertEqual(fake['fake_calls'](),1)


if __name__=='__main__':unittest.main()
