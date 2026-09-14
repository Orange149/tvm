/* Route three exact ResNet50 graph aliases to two input-stationary kernels. */
#include <dlfcn.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>

#include <mutex>
#include <string>

namespace {

constexpr const char* kResidencyFile = "c3_r50a_input_graphlib.so";
using Kernel = TVMBackendPackedCFunc;

Kernel ResolveKernel(const char* source_symbol) {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&ResolveKernel), &info) == 0 ||
      info.dli_fname == nullptr) {
    TVMAPISetLastError("R50A call-site wrapper could not resolve its own DSO path");
    return nullptr;
  }
  std::string path(info.dli_fname);
  const size_t slash = path.find_last_of('/');
  path = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) +
         kResidencyFile;
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    TVMAPISetLastError(dlerror());
    return nullptr;
  }
  void* symbol = dlsym(handle, source_symbol);
  if (symbol == nullptr) {
    TVMAPISetLastError(dlerror());
    return nullptr;
  }
  return reinterpret_cast<Kernel>(symbol);
}

}  // namespace

#define C3_DEFINE_ROUTE(alias_name, source_name)                                      \
  extern "C" TVM_DLL int alias_name(                                                 \
      TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,         \
      int* out_ret_tcode, void* resource_handle) {                                    \
    static std::once_flag once;                                                        \
    static Kernel kernel = nullptr;                                                   \
    std::call_once(once, []() { kernel = ResolveKernel(source_name); });               \
    if (kernel == nullptr) return -1;                                                  \
    return kernel(args, type_codes, num_args, out_ret_value, out_ret_tcode,            \
                  resource_handle);                                                    \
  }

C3_DEFINE_ROUTE(
    tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_7_c3_node67,
    "tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_7")
C3_DEFINE_ROUTE(
    tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_9_c3_node80,
    "tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_9")
C3_DEFINE_ROUTE(
    tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_9_c3_node93,
    "tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_9")

#undef C3_DEFINE_ROUTE
