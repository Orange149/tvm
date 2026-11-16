"""CPU fake functions only; exercise the real diagnostic proxy."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tvm
from tvm.contrib import cc
from c3_tps_sync_release_local import HERE,REPO


FAKE=r'''
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
namespace tvm { namespace runtime {
class FakeConv final : public ModuleNode {
 public:
  int calls{0};
  const char* type_key() const final { return "c3_yolo416_observer_fake"; }
  PackedFunc GetFunction(const String& name,const ObjectPtr<Object>& self) final {
    if (name=="fake_calls") return PackedFunc([this,self](TVMArgs,TVMRetValue* rv){*rv=calls;});
    if (name!="fake_conv") return PackedFunc();
    return PackedFunc([this,self](TVMArgs args,TVMRetValue* rv) {
      ICHECK_EQ(args.size(),4); ++calls;
      DLTensor *x=args[0],*w=args[1],*a=args[2],*y=args[3];
      for (int i=0;i<2;++i) static_cast<int8_t*>(y->data)[i]=
        static_cast<int8_t*>(x->data)[i]+static_cast<int8_t*>(w->data)[i]+static_cast<int32_t*>(a->data)[i];
      *rv=42;
    });
  }
};
TVM_REGISTER_GLOBAL("c3.yolo416.observer_test_fake").set_body_typed([](){return Module(make_object<FakeConv>());});
}}
'''


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        source=root/'fake.cc';source.write_text(FAKE)
        includes=[REPO/'include',REPO/'3rdparty/dmlc-core/include',REPO/'3rdparty/dlpack/include']
        cc.create_shared(str(root/'observer.so'),[str(HERE/'c3_yolo416_graph_observer.cc'),str(source)],
                         options=['-std=c++17','-O2',*['-I'+str(p) for p in includes]])
        cls.lib=tvm.runtime.load_module(str(root/'observer.so'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def wrap(self,spec='9 fake_conv 3 1\n57 fake_conv 3 1'):
        inner=tvm.get_global_func('c3.yolo416.observer_test_fake')()
        return inner,tvm.get_global_func('c3.yolo416.observe_module')(inner,spec)

    def arrays(self):
        return [tvm.nd.array(np.array(v,dtype=d)) for v,d in
                (([1,3],'int8'),([2,4],'int8'),([1,1],'int32'),([0,0],'int8'))]

    def test_same_symbol_distinct_calls_snapshots_and_return(self):
        inner,proxy=self.wrap();args=self.arrays()
        proxy['c3_yolo416_observer_begin']()
        self.assertEqual(proxy['fake_conv'](*args),42)
        args[1].copyfrom(np.array([5,6],dtype='int8'))
        proxy['fake_conv'](*args)
        rows=proxy['c3_yolo416_observer_end']()
        self.assertEqual(inner['fake_calls'](),2)
        self.assertEqual([int(str(r['node'])) for r in rows],[9,57])
        np.testing.assert_array_equal(rows[0]['outputs'][0].numpy(),[4,8])
        np.testing.assert_array_equal(rows[1]['outputs'][0].numpy(),[7,10])
        for row in rows:
            for a,b in zip(row['inputs_before'],row['inputs_after']):
                np.testing.assert_array_equal(a.numpy(),b.numpy())
        self.assertEqual([str(x) for x in rows[0]['data_handles']],
                         [str(x) for x in rows[1]['data_handles']])

    def test_missing_call_rejected_and_partial_readable(self):
        inner,proxy=self.wrap();proxy['c3_yolo416_observer_begin']()
        proxy['fake_conv'](*self.arrays())
        with self.assertRaises(Exception):proxy['c3_yolo416_observer_end']()
        self.assertEqual(len(proxy['c3_yolo416_observer_partial']()),1)
        self.assertEqual(inner['fake_calls'](),1)

    def test_unknown_symbol_and_unsupported_ABI_rejected(self):
        for spec in ('9 absent 3 1','9 fake_conv 2 1','9 fake_conv 3 1\n8 fake_conv 3 1'):
            with self.assertRaises(Exception):self.wrap(spec)

    def test_inactive_proxy_forwards_without_capture(self):
        inner,proxy=self.wrap();proxy['fake_conv'](*self.arrays())
        self.assertEqual(inner['fake_calls'](),1)
        self.assertEqual(len(proxy['c3_yolo416_observer_partial']()),0)


if __name__=='__main__':unittest.main()
