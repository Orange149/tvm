// Diagnostic-only VTA call snapshots. No clocks, added sync, or graph run.
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace tvm { namespace runtime {
class Yolo416Observer final : public ModuleNode {
 public:
  struct Entry { int node; std::string symbol; int inputs, outputs; };
  Module inner;
  std::vector<Entry> expected;
  Array<Map<String,ObjectRef>> rows;
  size_t cursor{0}; bool active{false};
  const char* type_key() const final { return "c3_yolo416_graph_observer"; }

  Array<NDArray> Snapshot(TVMArgs args, int begin, int end) {
    Array<NDArray> arrays;
    for (int i=begin; i<end; ++i) {
      ICHECK(args.type_codes[i]==kTVMDLTensorHandle || args.type_codes[i]==kTVMNDArrayHandle);
      DLTensor* tensor=args[i];
      ICHECK(tensor->strides==nullptr) << "only compact graph entries are supported";
      arrays.push_back(NDArray::NewFromDLTensor(tensor,DLDevice{kDLCPU,0}));
    }
    return arrays;
  }

  PackedFunc GetFunction(const String& name,const ObjectPtr<Object>& self) final {
    if (name=="c3_yolo416_observer_begin") return PackedFunc([this,self](TVMArgs args,TVMRetValue*) {
      ICHECK_EQ(args.size(),0); ICHECK(!active);
      cursor=0; rows=Array<Map<String,ObjectRef>>(); active=true;
    });
    if (name=="c3_yolo416_observer_end") return PackedFunc([this,self](TVMArgs args,TVMRetValue* rv) {
      ICHECK_EQ(args.size(),0); ICHECK(active); ICHECK_EQ(cursor,expected.size());
      active=false; *rv=rows;
    });
    if (name=="c3_yolo416_observer_partial") return PackedFunc([this,self](TVMArgs args,TVMRetValue* rv) {
      ICHECK_EQ(args.size(),0); *rv=rows; // read-only failure diagnosis, never reset
    });
    PackedFunc original=inner.GetFunction(name,true);
    if (original==nullptr) return PackedFunc();
    bool watched=false;
    for (const auto& e:expected) watched |= e.symbol==std::string(name);
    if (!watched) return original;
    return PackedFunc([this,self,original,name](TVMArgs args,TVMRetValue* rv) {
      if (!active) { original.CallPacked(args,rv); return; }
      ICHECK_LT(cursor,expected.size()); const auto& e=expected[cursor];
      ICHECK_EQ(e.symbol,std::string(name)); ICHECK_EQ(args.size(),e.inputs+e.outputs);
      Array<String> handles,offsets;
      for (int i=0;i<args.size();++i) {
        DLTensor* tensor=args[i];
        handles.push_back(String(std::to_string(reinterpret_cast<uintptr_t>(tensor->data))));
        offsets.push_back(String(std::to_string(tensor->byte_offset)));
      }
      auto before=Snapshot(args,0,e.inputs);
      original.CallPacked(args,rv); // unchanged arguments, result and exceptions
      auto after=Snapshot(args,0,e.inputs);
      auto outputs=Snapshot(args,e.inputs,args.size());
      Map<String,ObjectRef> row;
      row.Set("node",String(std::to_string(e.node))); row.Set("symbol",name);
      row.Set("inputs_before",before); row.Set("inputs_after",after); row.Set("outputs",outputs);
      row.Set("data_handles",handles); row.Set("byte_offsets",offsets);
      rows.push_back(row); ++cursor;
    });
  }
};

TVM_REGISTER_GLOBAL("c3.yolo416.observe_module").set_body([](TVMArgs args,TVMRetValue* rv) {
  ICHECK_EQ(args.size(),2);
  auto observer=make_object<Yolo416Observer>();observer->inner=args[0].operator Module();
  std::istringstream lines(args[1].operator std::string());std::string line;
  int last=-1;
  while (std::getline(lines,line)) {
    std::istringstream fields(line);Yolo416Observer::Entry e;std::string extra;
    ICHECK(static_cast<bool>(fields>>e.node>>e.symbol>>e.inputs>>e.outputs));
    ICHECK(!(fields>>extra));ICHECK_GT(e.node,last);
    ICHECK_EQ(e.inputs,3);ICHECK_EQ(e.outputs,1); // frozen YOLO416 four-buffer ABI
    ICHECK(observer->inner.GetFunction(e.symbol,true)!=nullptr);
    observer->expected.push_back(e);last=e.node;
  }
  ICHECK(!observer->expected.empty());*rv=Module(observer);
});
}}  // namespace tvm::runtime
