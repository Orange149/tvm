// Resolve only two graph symbols to sealed replacement modules at graph creation.
// The formal route returns the original PackedFunc directly: no per-call wrapper.
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm { namespace runtime {
class Yolo224TwoNodeRoute final : public ModuleNode {
 public:
  Module base, node34, node46;
  std::string symbol34, symbol46, replacement;
  std::unordered_set<std::string> watched;
  bool diagnostic{false};
  const char* type_key() const final { return "c3_yolo224_two_node_route"; }
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& self) final {
    std::string symbol(name);
    PackedFunc function = symbol == symbol34 ? node34.GetFunction(replacement, true)
                        : symbol == symbol46 ? node46.GetFunction(replacement, true)
                                             : base.GetFunction(name, true);
    if (function == nullptr || !diagnostic || !watched.count(symbol)) return function;
    return PackedFunc([self, function, symbol](TVMArgs args, TVMRetValue* result) {
      std::printf("[C3_GRAPH_BEGIN] %s\n", symbol.c_str());
      function.CallPacked(args, result);
      std::printf("[C3_GRAPH_END] %s\n", symbol.c_str());
    });
  }
};

TVM_REGISTER_GLOBAL("c3.yolo224.route_module").set_body([](TVMArgs args, TVMRetValue* result) {
  ICHECK_EQ(args.size(), 8);
  auto route = make_object<Yolo224TwoNodeRoute>();
  route->base = args[0].operator Module();
  route->node34 = args[1].operator Module();
  route->node46 = args[2].operator Module();
  route->symbol34 = args[3].operator std::string();
  route->symbol46 = args[4].operator std::string();
  route->replacement = args[5].operator std::string();
  std::istringstream names(args[6].operator std::string());
  std::string name;
  while (names >> name) route->watched.insert(name);
  route->diagnostic = args[7].operator bool();
  ICHECK_NE(route->symbol34, route->symbol46);
  ICHECK(route->watched.count(route->symbol34) && route->watched.count(route->symbol46));
  ICHECK(route->base.GetFunction(route->symbol34, true) != nullptr);
  ICHECK(route->base.GetFunction(route->symbol46, true) != nullptr);
  ICHECK(route->node34.GetFunction(route->replacement, true) != nullptr);
  ICHECK(route->node46.GetFunction(route->replacement, true) != nullptr);
  *result = Module(route);
});
}}  // namespace tvm::runtime
