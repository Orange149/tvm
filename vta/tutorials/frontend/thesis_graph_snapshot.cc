// Single-input diagnostic only. Read the executor's backing twice on the board.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

TVM_REGISTER_GLOBAL("thesis.graph_snapshot_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      using namespace tvm::runtime;
      using Clock = std::chrono::steady_clock;
      auto ns = []() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count();
      };
      Module executor = args[0];
      int output_index = args[1];
      DLTensor* immediate = args[2];
      DLTensor* delayed = args[3];
      std::string identity = args[4];
      ICHECK(IsContiguous(*immediate) && IsContiguous(*delayed));
      ICHECK_EQ(immediate->device.device_type, kDLCPU);
      ICHECK_EQ(delayed->device.device_type, kDLCPU);
      ICHECK_EQ(GetDataSize(*immediate), GetDataSize(*delayed));
      auto run = executor.GetFunction("run");
      auto get = executor.GetFunction("get_output");
      ICHECK(run != nullptr && get != nullptr);
      std::setvbuf(stdout, nullptr, _IONBF, 0);
      std::fprintf(stderr, "[THESIS_CALL_BEGIN] %s pid=%d ns=%lld\n",
                   identity.c_str(), getpid(), static_cast<long long>(ns()));
      std::fflush(stderr);
      run();
      const auto returned = ns();
      NDArray backing = get(output_index);
      ICHECK_EQ(GetDataSize(*backing.operator->()), GetDataSize(*immediate));
      backing.CopyTo(immediate);
      const auto first = ns();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      // Fetch the actual executor output again; do not reread a detached copy.
      NDArray second_backing = get(output_index);
      second_backing.CopyTo(delayed);
      const auto second = ns();
      const bool stable = std::memcmp(immediate->data, delayed->data, GetDataSize(*immediate)) == 0;
      std::ostringstream result;
      result << "{\"pid\":" << getpid() << ",\"returned_ns\":" << returned
             << ",\"immediate_read_ns\":" << first << ",\"delayed_read_ns\":" << second
             << ",\"stable\":" << (stable ? "true" : "false") << "}";
      std::fprintf(stderr, "[THESIS_CALL_END] %s %s\n", identity.c_str(), result.str().c_str());
      std::fflush(stderr);
      *rv = result.str();
    });
