// Experiment-only four-argument full-fusion timer. No VTA runtime is linked.
// One invocation performs precisely one packed-function call and no extra sync.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

namespace {
std::atomic<uint64_t> calls{0};
}  // namespace

TVM_REGISTER_GLOBAL("c3.node34.reset_counts")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) { calls.store(0); });

TVM_REGISTER_GLOBAL("c3.node34.counts")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue* rv) {
      *rv = static_cast<int64_t>(calls.load());
    });

TVM_REGISTER_GLOBAL("c3.node34.fused_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      tvm::runtime::Module module = args[0];
      std::string symbol = args[1];
      auto function = module.GetFunction(symbol, true);
      ICHECK(function != nullptr) << "missing fused symbol " << symbol;
      const auto begin = std::chrono::steady_clock::now();
      function(args[2], args[3], args[4], args[5]);
      const auto end = std::chrono::steady_clock::now();
      ++calls;
      *rv = std::chrono::duration<double, std::milli>(end - begin).count();
    });
