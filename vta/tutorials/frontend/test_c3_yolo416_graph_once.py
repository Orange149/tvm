"""CPU fake module only: never executes a DNN or a VTA command."""
import json
from pathlib import Path
import tempfile
import unittest
import tvm
from tvm.contrib import cc
from c3_tps_sync_release_local import REPO,HERE


FAKE=r'''
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
namespace tvm { namespace runtime {
class FakeGraph final : public ModuleNode {
 public:
  int calls{0}; bool fail{false};
  const char* type_key() const final { return "c3_graph_once_test_fake"; }
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& self) final {
    if (name=="run") return PackedFunc([this,self](TVMArgs args,TVMRetValue*) {
      ICHECK_EQ(args.size(),0); ++calls;
      if (fail) LOG(FATAL)<<"deliberate fake graph failure";
    });
    if (name=="fake_calls") return PackedFunc([this,self](TVMArgs,TVMRetValue* rv) {*rv=calls;});
    return PackedFunc();
  }
};
TVM_REGISTER_GLOBAL("c3.yolo416.test_fake_graph").set_body_typed([](bool fail) {
  auto m=make_object<FakeGraph>();m->fail=fail;return Module(m);
});
}}
'''


class GraphHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        fixture=root/'fake.cc';fixture.write_text(FAKE)
        includes=[REPO/'include',REPO/'3rdparty/dmlc-core/include',REPO/'3rdparty/dlpack/include']
        cc.create_shared(str(root/'helper.so'),[str(HERE/'c3_yolo416_graph_once.cc'),str(fixture)],
                         options=['-std=c++17','-O2','-ldl',*['-I'+str(p) for p in includes]])
        cls.lib=tvm.runtime.load_module(str(root/'helper.so'))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):tvm.get_global_func('c3.yolo416.graph_reset_counts')()

    def metadata(self):return json.loads(tvm.get_global_func('c3.yolo416.graph_metadata')())

    def fake(self,fail=False):return tvm.get_global_func('c3.yolo416.test_fake_graph')(fail)

    def test_exactly_one_zero_argument_run(self):
        graph=self.fake();timer=tvm.get_global_func('c3.yolo416.graph_timed_once')
        timer(graph)
        self.assertEqual(graph['fake_calls'](),1)
        m=self.metadata();self.assertEqual((m['started_calls'],m['completed_calls']),(1,1))
        self.assertEqual(m['packed_calls_per_invocation'],1)
        self.assertFalse(m['adds_synchronize']);self.assertEqual(m['parameter_count'],0)

    def test_no_implicit_warmup_or_number_growth(self):
        graph=self.fake();timer=tvm.get_global_func('c3.yolo416.graph_timed_once')
        for _ in range(3):timer(graph)
        self.assertEqual(graph['fake_calls'](),3)
        self.assertEqual(self.metadata()['completed_calls'],3)

    def test_throwing_run_remains_started_without_completion(self):
        graph=self.fake(True)
        with self.assertRaises(Exception):tvm.get_global_func('c3.yolo416.graph_timed_once')(graph)
        self.assertEqual(graph['fake_calls'](),1)
        m=self.metadata();self.assertEqual((m['started_calls'],m['completed_calls']),(1,0))
        # Failure releases the helper guard; only the experiment controller stops.
        tvm.get_global_func('c3.yolo416.graph_timed_once')(self.fake())
        self.assertEqual(self.metadata()['completed_calls'],1)

    def test_extra_argument_rejected_before_run(self):
        graph=self.fake()
        with self.assertRaises(Exception):tvm.get_global_func('c3.yolo416.graph_timed_once')(graph,1)
        self.assertEqual(graph['fake_calls'](),0)
        self.assertEqual(self.metadata()['started_calls'],0)


if __name__=='__main__':unittest.main()
