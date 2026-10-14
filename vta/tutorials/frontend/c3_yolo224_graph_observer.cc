// Diagnostic-only module proxy. Never link a second VTA runtime or synchronize.
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm { namespace runtime {
class C3GraphObserver final : public ModuleNode {
 public:
  Module inner;
  std::vector<std::pair<int, std::string>> expected;
  std::vector<double> elapsed;
  size_t cursor{0};
  bool active{false};
  const char* type_key() const final { return "c3_graph_observer"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& self) final {
    if (name == "c3_observer_begin") {
      return PackedFunc([this, self](TVMArgs, TVMRetValue*) {
        ICHECK(!active) << "unfinished observed invocation";
        cursor = 0;
        active = true;
        std::fill(elapsed.begin(), elapsed.end(), 0.0);
      });
    }
    if (name == "c3_observer_end") {
      return PackedFunc([this, self](TVMArgs, TVMRetValue* rv) {
        ICHECK(active && cursor == expected.size()) << "observed graph call coverage mismatch";
        active = false;
        std::ostringstream out;
        out.precision(17);
        out << "[";
        for (size_t i = 0; i < cursor; ++i) {
          if (i) out << ",";
          out << "{\"node\":" << expected[i].first << ",\"host_call_ms\":" << elapsed[i] << "}";
        }
        out << "]";
        *rv = out.str();
      });
    }
    PackedFunc original = inner.GetFunction(name, true);
    if (original == nullptr) return PackedFunc();
    bool watched = false;
    for (const auto& item : expected) watched |= item.second == std::string(name);
    if (!watched) return original;  // Linked parameter lookup is not a graph invocation.
    return PackedFunc([this, self, original, name](TVMArgs args, TVMRetValue* rv) {
      if (!active) { original.CallPacked(args, rv); return; }
      ICHECK_LT(cursor, expected.size());
      ICHECK_EQ(expected[cursor].second, std::string(name)) << "graph invocation order changed";
      const auto start = std::chrono::steady_clock::now();
      original.CallPacked(args, rv);  // Exact original ABI, return value, and exception behavior.
      const auto end = std::chrono::steady_clock::now();
      elapsed[cursor++] = std::chrono::duration<double, std::milli>(end - start).count();
    });
  }
};

TVM_REGISTER_GLOBAL("c3.incremental.observe_module")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      auto observer = make_object<C3GraphObserver>();
      observer->inner = args[0].operator Module();
      std::istringstream lines(args[1].operator std::string());
      std::string line;
      int last_node = -1;
      while (std::getline(lines, line)) {
        std::istringstream fields(line);
        int node;
        std::string symbol, extra;
        ICHECK(static_cast<bool>(fields >> node >> symbol));
        ICHECK(!(fields >> extra));
        ICHECK_GT(node, last_node);
        ICHECK(symbol != "__copy" && symbol != "__nop");
        ICHECK(observer->inner.GetFunction(symbol, true) != nullptr);
        observer->expected.emplace_back(node, symbol);
        last_node = node;
      }
      ICHECK(!observer->expected.empty());
      observer->elapsed.resize(observer->expected.size());
      *rv = Module(observer);
    });
}}  // namespace tvm::runtime
