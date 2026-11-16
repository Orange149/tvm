// Out-of-timer storage identity. No target, copy, synchronize or VTA API call.
#include <cstdint>
#include <sstream>
#include <tvm/runtime/registry.h>

TVM_REGISTER_GLOBAL("c3.tps_window.buffer_storage")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      std::ostringstream out;
      out << "[";
      for (int i = 0; i < args.size(); ++i) {
        ICHECK(args.type_codes[i] == kTVMDLTensorHandle ||
               args.type_codes[i] == kTVMNDArrayHandle);
        DLTensor* tensor = args[i];
        ICHECK(tensor->data != nullptr);
        if (i) out << ",";
        out << "{\"data_address\":" << reinterpret_cast<uintptr_t>(tensor->data)
            << ",\"byte_offset\":" << tensor->byte_offset
            << ",\"device_type\":" << int(tensor->device.device_type)
            << ",\"device_id\":" << tensor->device.device_id << "}";
      }
      out << "]";
      *rv = out.str();
    });
