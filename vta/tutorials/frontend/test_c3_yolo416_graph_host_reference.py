"""CPU-only adapter fixtures and real CPU IR gates; zero DNN executions."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tvm,vta
from tvm.contrib import cc
from c3_tps_sync_release_local import REPO,HERE,HOLD,read
from c3_yolo416_graph_host_reference import HostCPUCallback,raw_pointer_gate,raw_pointer_llvm_gate

FAKE=r'''
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
namespace tvm { namespace runtime {
class FakeCPU final:public ModuleNode {
 public:
  int calls{0};const char* type_key() const final {return "c3_yolo416_cpu_reference_test";}
  PackedFunc GetFunction(const String& name,const ObjectPtr<Object>& self) final {
    if(name=="calls")return PackedFunc([this,self](TVMArgs,TVMRetValue* r){*r=calls;});
    if(name!="cpu" && name!="fail")return PackedFunc();
    return PackedFunc([this,self,name](TVMArgs args,TVMRetValue*){
      ICHECK_EQ(args.size(),2);DLTensor* x=args[0];DLTensor* y=args[1];
      ICHECK_EQ(x->device.device_type,kDLExtDev);ICHECK_EQ(y->device.device_type,kDLExtDev);
      ++calls;if(name=="fail")LOG(FATAL)<<"deliberate CPU reference fixture failure";
      for(int i=0;i<2;++i)reinterpret_cast<int32_t*>(y->data)[i]=reinterpret_cast<int32_t*>(x->data)[i]+4;
    });
  }
};
TVM_REGISTER_GLOBAL("c3.yolo416.test_cpu_reference").set_body_typed([](){return Module(make_object<FakeCPU>());});
}}
'''


class HostReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        (root/'fake.cc').write_text(FAKE)
        cc.create_shared(str(root/'helper.so'),[str(HERE/'c3_yolo416_cpu_reference.cc'),str(root/'fake.cc')],
            options=['-std=c++17','-O2','-ldl',*['-I'+str(REPO/p) for p in
                ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]])
        cls.lib=tvm.runtime.load_module(str(root/'helper.so'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def fake(self):return tvm.get_global_func('c3.yolo416.test_cpu_reference')()

    def test_same_CPU_storage_descriptor_view_and_one_call(self):
        module=self.fake();x=tvm.nd.array(np.array([1,3],dtype='int32'));y=tvm.nd.empty([2],'int32')
        pointers=[int(a.handle.contents.data) for a in (x,y)]
        tvm.get_global_func('c3.yolo416.cpu_reference_call')(module,'cpu',x,y)
        np.testing.assert_array_equal(y.numpy(),[5,7]);self.assertEqual(module['calls'](),1)
        self.assertEqual([a.device.device_type for a in (x,y)],[1,1])
        self.assertEqual([int(a.handle.contents.data) for a in (x,y)],pointers)

    def test_exception_does_not_mutate_original_descriptor(self):
        module=self.fake();x=tvm.nd.array(np.array([1,3],dtype='int32'));y=tvm.nd.empty([2],'int32')
        with self.assertRaises(Exception):tvm.get_global_func('c3.yolo416.cpu_reference_call')(module,'fail',x,y)
        self.assertEqual(x.device.device_type,1);self.assertEqual(y.device.device_type,1)
        self.assertEqual(module['calls'](),1)

    def test_host_callback_guards_and_exact_output(self):
        module=self.fake();abi=[{'kind':'buffer','dtype':'int32','shape':[2]}]*2
        manifest={'cpu':{'abi':abi,'instruction_gate':{'status':'NO_VTA_INSTRUCTION_ENQUEUE_IN_CPU_FUNCTION'},
                        'raw_pointer_gate':{'status':'CPU_RAW_POINTER_REFERENCE_SUPPORTED'}}}
        callback=HostCPUCallback(module,manifest,tvm.cpu(0),lambda:'{"attempted_submits":0}',
                                 tvm.get_global_func('c3.yolo416.cpu_reference_call'))
        result=callback('cpu',[np.array([2,5],dtype='int32')],[{'shape':[2],'dtype':'int32'}],False)
        np.testing.assert_array_equal(result[0],[6,9]);self.assertEqual(len(callback.calls),1)
        with self.assertRaisesRegex(ValueError,'UNBOUND'):callback('unknown',[],[],False)
        with self.assertRaises(ValueError):HostCPUCallback(module,manifest,tvm.device('ext_dev',0),lambda:'{}',lambda:None)

    def test_real_21_CPU_functions_support_raw_pointer_gate(self):
        rows=read(HOLD/'20261017_c3_yolo416_tps_prepare_run02/P/all_final.json');count=0
        for row in rows.values():
            if row['VTA']:continue
            fn=list(tvm.ir.load_json(Path(row['file']).read_text()).functions.values())[0]
            self.assertEqual(raw_pointer_gate(fn)['status'],'CPU_RAW_POINTER_REFERENCE_SUPPORTED');count+=1
        self.assertEqual(count,21)

    def test_DMA_read_barrier_rejected(self):
        from tvm import tir
        fn=tir.PrimFunc([],tir.Evaluate(tir.call_extern('int32','VTAReadBarrier',0,0,8,0,1)))
        with self.assertRaisesRegex(ValueError,'REQUIRES_REAL_DMA'):raw_pointer_gate(fn)

    def test_local_RPC_CPU_views_keep_wire_device_and_storage(self):
        from tvm import rpc
        session=rpc.LocalSession();module=session.get_function('c3.yolo416.test_cpu_reference')()
        x=tvm.nd.array(np.array([1,3],dtype='int32'),session.cpu(0))
        y=tvm.nd.empty([2],'int32',session.cpu(0))
        session.get_function('c3.yolo416.cpu_reference_call')(module,'cpu',x,y)
        np.testing.assert_array_equal(y.numpy(),[5,7]);np.testing.assert_array_equal(x.numpy(),[1,3])
        self.assertEqual(x.device.device_type%128,1);self.assertEqual(y.device.device_type%128,1)
        self.assertEqual(module['calls'](),1)


class ActualLLVMGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=HOLD/'20261017_c3_yolo416_tps_prepare_run02/P'
        cls.rows=read(source/'all_final.json');cls.llvm=(source/'module.ll').read_text()

    def test_all_actual_CPU_llvm_bodies_support_raw_pointer_reference(self):
        count=0
        for symbol,row in self.rows.items():
            if row['VTA']:continue
            self.assertEqual(raw_pointer_llvm_gate(self.llvm,symbol)['status'],'CPU_RAW_POINTER_LLVM_CONFIRMED')
            count+=1
        self.assertEqual(count,21)

    def test_actual_VTA_llvm_rejected(self):
        symbol=next(s for s,r in self.rows.items() if r['VTA'])
        with self.assertRaisesRegex(ValueError,'CPU_LLVM_REQUIRES'):raw_pointer_llvm_gate(self.llvm,symbol)


if __name__=='__main__':unittest.main()
