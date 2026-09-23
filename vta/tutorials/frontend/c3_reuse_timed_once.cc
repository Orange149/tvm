// Experimental RPC helper: exactly one call, timed on the board, no warmup.
// Does not change VTA runtime or the frozen operator modules.
#include <chrono>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

TVM_REGISTER_GLOBAL("c3.reuse.timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      tvm::runtime::Module module = args[0];
      auto function = module.GetFunction("main", true);
      ICHECK(function != nullptr);
      const auto start = std::chrono::steady_clock::now();
      function(args[1], args[2], args[3]);
      const auto end = std::chrono::steady_clock::now();
      *rv = std::chrono::duration<double, std::milli>(end - start).count();
    });
