// Experiment-only VTA operator-boundary controls.  No timing is performed.
#include <dlfcn.h>

#include <tvm/runtime/registry.h>

namespace {

using CommandHandle = void*;
using GetHandle = CommandHandle (*)();
using Synchronize = void (*)(CommandHandle, unsigned int);
using Shutdown = void (*)();

template <typename T>
T Resolve(const char* name) {
  T function = reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
  ICHECK(function != nullptr) << "missing runtime symbol " << name;
  return function;
}

}  // namespace

TVM_REGISTER_GLOBAL("c3.transition.empty_sync")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) {
      auto handle = Resolve<GetHandle>("VTATLSCommandHandle");
      auto synchronize = Resolve<Synchronize>("VTASynchronize");
      synchronize(handle(), 1U << 31);
    });

TVM_REGISTER_GLOBAL("c3.transition.shutdown_queue")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) {
      Resolve<Shutdown>("VTARuntimeShutdown")();
    });
