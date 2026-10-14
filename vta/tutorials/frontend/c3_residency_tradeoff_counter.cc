// CPU-only four-argument helper regression. It never references VTA runtime.
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>

extern "C" TVM_DLL int c3_four_arg_counter(TVMValue* args, int* type_codes, int num_args,
                                          TVMValue* out_ret, int* out_type) {
  if (num_args != 4) return -1;
  DLTensor* tensors[4];
  for (int i = 0; i < 4; ++i) {
    if (type_codes[i] != kTVMDLTensorHandle && type_codes[i] != kTVMNDArrayHandle) return -2;
    tensors[i] = static_cast<DLTensor*>(args[i].v_handle);
    if (tensors[i]->ndim != 1 || tensors[i]->shape[0] != 1 ||
        tensors[i]->dtype.code != kDLInt || tensors[i]->dtype.bits != 32 ||
        tensors[i]->dtype.lanes != 1 || tensors[i]->device.device_type != kDLCPU) return -3;
  }
  auto value = [](DLTensor* tensor) -> int32_t& {
    return *reinterpret_cast<int32_t*>(static_cast<char*>(tensor->data) + tensor->byte_offset);
  };
  value(tensors[3]) += 100 * value(tensors[0]) + 10 * value(tensors[1]) + value(tensors[2]);
  *out_type = kTVMNullptr;
  out_ret->v_int64 = 0;
  return 0;
}
