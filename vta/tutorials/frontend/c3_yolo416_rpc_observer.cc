// RPC view of the already tested diagnostic observer. TVM RPC transports
// strings and individual NDArrays, but not Array<Map<ObjectRef>> containers.
// Snapshots stay in the original observer; this adapter only retrieves them.
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <sstream>

namespace tvm { namespace runtime {
class RpcObserver final:public ModuleNode {
 public:
  Module observer;
  Array<Map<String,ObjectRef>> rows;
  const char* type_key() const final {return "c3_yolo416_rpc_observer";}
  std::string Quote(const String& s) {
    std::ostringstream out;out<<'"';
    for(char c:std::string(s)) {
      if(c=='"' || c=='\\')out<<'\\'<<c;
      else {ICHECK_GE(static_cast<unsigned char>(c),32);out<<c;}
    }
    out<<'"';return out.str();
  }
  std::string Metadata() {
    std::ostringstream out;out<<'[';
    for(size_t i=0;i<rows.size();++i) {
      if(i)out<<',';const auto& r=rows[i];
      out<<"{\"node\":"<<Quote(Downcast<String>(r.at("node")))
         <<",\"symbol\":"<<Quote(Downcast<String>(r.at("symbol")));
      for(const auto* key:{"data_handles","byte_offsets"}) {
        out<<",\""<<key<<"\":[";auto values=Downcast<Array<String>>(r.at(key));
        for(size_t j=0;j<values.size();++j){if(j)out<<',';out<<Quote(values[j]);}out<<']';
      }
      for(const auto* key:{"inputs_before","inputs_after","outputs"})
        out<<",\""<<key<<"_count\":"<<Downcast<Array<NDArray>>(r.at(key)).size();
      out<<'}';
    }
    out<<']';return out.str();
  }
  PackedFunc GetFunction(const String& name,const ObjectPtr<Object>& self) final {
    if(name=="c3_yolo416_observer_begin")return PackedFunc([this,self](TVMArgs args,TVMRetValue*) {
      ICHECK_EQ(args.size(),0);rows=Array<Map<String,ObjectRef>>();
      observer.GetFunction("c3_yolo416_observer_begin",false)();
    });
    if(name=="c3_yolo416_observer_end" || name=="c3_yolo416_observer_partial")
      return PackedFunc([this,self,name](TVMArgs args,TVMRetValue* rv) {
        ICHECK_EQ(args.size(),0);rows=observer.GetFunction(name,false)();*rv=static_cast<int64_t>(rows.size());
      });
    if(name=="c3_yolo416_observer_metadata")return PackedFunc([this,self](TVMArgs args,TVMRetValue* rv) {
      ICHECK_EQ(args.size(),0);*rv=Metadata();
    });
    if(name=="c3_yolo416_observer_snapshot")return PackedFunc([this,self](TVMArgs args,TVMRetValue* rv) {
      ICHECK_EQ(args.size(),3);int i=args[0];int kind=args[1];int j=args[2];
      ICHECK_GE(i,0);ICHECK_LT(static_cast<size_t>(i),rows.size());ICHECK_GE(kind,0);ICHECK_LT(kind,3);
      const char* key=kind==0?"inputs_before":kind==1?"inputs_after":"outputs";
      auto arrays=Downcast<Array<NDArray>>(rows[i].at(key));
      ICHECK_GE(j,0);ICHECK_LT(static_cast<size_t>(j),arrays.size());*rv=arrays[j];
    });
    return observer.GetFunction(name,true);
  }
};
TVM_REGISTER_GLOBAL("c3.yolo416.rpc_observe_module").set_body([](TVMArgs args,TVMRetValue* rv) {
  ICHECK_EQ(args.size(),2);
  const PackedFunc* create=Registry::Get("c3.yolo416.observe_module");ICHECK(create!=nullptr);
  auto m=make_object<RpcObserver>();m->observer=(*create)(args[0],args[1]);*rv=Module(m);
});
}}  // namespace tvm::runtime
