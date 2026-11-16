// CPU-reference-only adapter. Original compiled CPU functions stay unchanged.
// Frozen raw-pointer contract: no VTA enqueue/read/write barrier; CPUPtr accepts
// ordinary CPU pointers. Views retain CPU-owned storage; no DMA tensor copies.
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <vector>

namespace tvm { namespace runtime {
TVM_REGISTER_GLOBAL("c3.yolo416.cpu_reference_call").set_body([](TVMArgs args,TVMRetValue*) {
  ICHECK_GE(args.size(),3);
  Module module=args[0]; String symbol=args[1];
  PackedFunc function=module.GetFunction(symbol,true); ICHECK(function!=nullptr);
  const int count=args.size()-2;
  std::vector<DLTensor> views(count);
  std::vector<TVMValue> values(count);
  std::vector<int> codes(count,kTVMDLTensorHandle);
  for (int i=0;i<count;++i) {
    ICHECK(args.type_codes[i+2]==kTVMNDArrayHandle || args.type_codes[i+2]==kTVMDLTensorHandle);
    DLTensor* source=args[i+2];
    ICHECK_EQ(source->device.device_type,kDLCPU);
    ICHECK_EQ(source->device.device_id,0);
    ICHECK(source->strides==nullptr);
    ICHECK_EQ(source->byte_offset,0);
    views[i]=*source;
    // Only a temporary descriptor changes. The actual NDArray's ownership,
    // device metadata, shape and storage remain intact on success or throw.
    views[i].device=DLDevice{kDLExtDev,0};
    values[i].v_handle=&views[i];
  }
  TVMRetValue ignored;
  function.CallPacked(TVMArgs(values.data(),codes.data(),count),&ignored);
});
}}  // namespace tvm::runtime
