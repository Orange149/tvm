// CPU-only test module. Not included in the board helper.
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

namespace tvm { namespace runtime {
class ObserverFixture final : public ModuleNode {
 public:
  int calls{0};
  const char* type_key() const final { return "c3_observer_test_fixture"; }
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& self) final {
    if (name == "calls") return PackedFunc([this, self](TVMArgs, TVMRetValue* rv) { *rv = calls; });
    if (name == "a" || name == "b") return PackedFunc([this, self](TVMArgs args, TVMRetValue* rv) {
      ++calls;
      *rv = args[0].operator int() + 17;
    });
    return PackedFunc();
  }
};
TVM_REGISTER_GLOBAL("c3.incremental.test_fixture").set_body([](TVMArgs, TVMRetValue* rv) {
  *rv = Module(make_object<ObserverFixture>());
});
}}
