// Experiment plugins register callbacks and return custom ModuleNode objects.
// Their implementation must remain mapped through callback/object destruction,
// including RPC and interpreter teardown. Only code is retained: all ordinary
// ModuleNode/NDArray destructors still run. No target/runtime instruction changes.
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

#ifndef RTLD_NODELETE
#error "This experiment requires Linux RTLD_NODELETE support"
#endif

namespace {
__attribute__((constructor)) void PinPluginImplementation() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&PinPluginImplementation), &info) || !info.dli_fname) {
    std::fputs("C3_PLUGIN_LIFETIME_DLADDR_FAILED\n", stderr);
    std::abort();
  }
  // Global registry callbacks can outlive a Python/RPC library handle. The
  // process owns this mapping; never dlclose it while callback code can exist.
  void* owner = dlopen(info.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (!owner) {
    std::fprintf(stderr, "C3_PLUGIN_LIFETIME_PIN_FAILED: %s\n", dlerror());
    std::abort();
  }
}
}  // namespace
