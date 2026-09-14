// Read-only metadata helper for E3 RPC experiments; no allocator/runtime changes.
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <vta/driver.h>
#include <cstdint>
#include <sstream>

TVM_REGISTER_GLOBAL("vta.e3.describe_cpu_view")
    .set_body_typed([](DLTensor* arr) {
      using namespace tvm::runtime;
      int dtype = arr->device.device_type % kRPCSessMask;
      ICHECK_EQ(dtype, kDLCPU);
      ICHECK_EQ(arr->byte_offset, 0U);
      ICHECK(arr->strides == nullptr);
      uint64_t elements = 1;
      for (int i = 0; i < arr->ndim; ++i) elements *= arr->shape[i];
      uint64_t bytes = elements * ((arr->dtype.bits * arr->dtype.lanes + 7) / 8);
      auto address = reinterpret_cast<uintptr_t>(arr->data);
      auto physical = static_cast<uint64_t>(VTAMemGetPhyAddr(arr->data));
      std::ostringstream os;
      os << "{\"virtual_address\":" << address << ",\"physical_address\":" << physical
         << ",\"bytes\":" << bytes << ",\"contiguous\":true,\"byte_offset\":0}";
      return os.str();  // RPC supports packed C strings, not runtime.String objects.
    });

// RPC serializes NDArray arguments as DLTensor handles. The original ext_dev
// owner must stay alive in the caller until all views/consumers have finished.
TVM_REGISTER_GLOBAL("vta.e3.shared_cpu_view").set_body_typed([](DLTensor* arr) {
  auto external = tvm::runtime::NDArray::FromExternalDLTensor(*arr);
  auto* make_view = tvm::runtime::Registry::Get("vta.runtime.ndarray_shared_cpu_view");
  ICHECK(make_view != nullptr);
  tvm::runtime::NDArray result = (*make_view)(external);
  return result;
});

TVM_REGISTER_GLOBAL("vta.e3.sync_host_read").set_body_typed([](DLTensor* arr) {
  auto external = tvm::runtime::NDArray::FromExternalDLTensor(*arr);
  auto* sync = tvm::runtime::Registry::Get("vta.runtime.ndarray_sync_host_read");
  ICHECK(sync != nullptr);
  (*sync)(external);
});
