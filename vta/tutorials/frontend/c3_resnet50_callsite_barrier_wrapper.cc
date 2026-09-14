/*
 * Exact-graph experiment: expose one renamed ResNet50 VTA kernel from a
 * separately loaded weight-resident graph module.  The wrapper resolves the
 * original symbol from the barrier DSO itself, avoiding ELF interposition with
 * the incumbent graph module.  Both DSOs must be loaded by TVM first so their
 * module contexts are initialized.
 */
#include <dlfcn.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>

#include <mutex>
#include <string>

namespace {

constexpr const char* kBarrierFile = "c3_r50_barrier_graphlib.so";
constexpr const char* kSourceSymbol =
    "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_3";

using Kernel = TVMBackendPackedCFunc;

Kernel ResolveBarrierKernel() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&ResolveBarrierKernel), &info) == 0 ||
      info.dli_fname == nullptr) {
    TVMAPISetLastError("call-site wrapper could not resolve its own DSO path");
    return nullptr;
  }
  std::string path(info.dli_fname);
  const size_t slash = path.find_last_of('/');
  path = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + kBarrierFile;
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    TVMAPISetLastError(dlerror());
    return nullptr;
  }
  void* symbol = dlsym(handle, kSourceSymbol);
  if (symbol == nullptr) {
    TVMAPISetLastError(dlerror());
    return nullptr;
  }
  return reinterpret_cast<Kernel>(symbol);
}

}  // namespace

extern "C" TVM_DLL int tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_3_c3_callsite0(
    TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
    int* out_ret_tcode, void* resource_handle) {
  static std::once_flag once;
  static Kernel kernel = nullptr;
  std::call_once(once, []() { kernel = ResolveBarrierKernel(); });
  if (kernel == nullptr) return -1;
  return kernel(args, type_codes, num_args, out_ret_value, out_ret_tcode, resource_handle);
}
