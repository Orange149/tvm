// Stage01: keep all reads and the 10ms stability delay on the board.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

// RPC passes borrowed DLTensor handles. The RPC session retains the underlying
// buffers throughout this synchronous call; this container owns metadata only.
static tvm::runtime::NDArray Borrow(const tvm::runtime::TVMArgValue& argument) {
  using namespace tvm::runtime;
  DLTensor* tensor = argument;
  ICHECK(tvm::runtime::IsContiguous(*tensor));
  ICHECK_EQ(tensor->byte_offset, 0);
  auto* container = new NDArray::Container(
      tensor->data, ShapeTuple(tensor->shape, tensor->shape + tensor->ndim),
      tensor->dtype, tensor->device);
  container->SetDeleter([](Object* object) { delete static_cast<NDArray::Container*>(object); });
  return NDArray(GetObjectPtr<Object>(container));
}

TVM_REGISTER_GLOBAL("thesis.completion_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      using tvm::runtime::NDArray;
      using Clock = std::chrono::steady_clock;
      // Keep instruction dumps in order with submission and call markers.
      std::setvbuf(stdout, nullptr, _IONBF, 0);
      auto ns = []() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
      };
      tvm::runtime::Module module = args[0];
      NDArray input = Borrow(args[1]), weight = Borrow(args[2]), output = Borrow(args[3]);
      NDArray expected = Borrow(args[4]), immediate = Borrow(args[5]), delayed = Borrow(args[6]), sentinel = Borrow(args[7]);
      std::string identity = args[8];
      const auto bytes = tvm::runtime::GetDataSize(*expected.operator->());
      ICHECK_EQ(bytes, tvm::runtime::GetDataSize(*output.operator->()));
      auto function = module.GetFunction("main", true);
      ICHECK(function != nullptr);
      const auto* get_view = tvm::runtime::Registry::Get("vta.runtime.ndarray_shared_cpu_view");
      const auto* sync_read = tvm::runtime::Registry::Get("vta.runtime.ndarray_sync_host_read");
      ICHECK(get_view != nullptr && sync_read != nullptr);
      NDArray view = (*get_view)(output);
      ICHECK_EQ(view->device.device_type, kDLCPU);
      const auto start = ns();
      // The output NDArray is the actual argument backing, not a detached host tensor.
      output.CopyFrom(sentinel);
      (*sync_read)(output);
      std::memcpy(immediate->data, view->data, bytes);
      const bool sentinel_visible = std::memcmp(immediate->data, sentinel->data, bytes) == 0;
      ICHECK(sentinel_visible) << "Output sentinel did not become device-visible";
      const auto* a = static_cast<const uint8_t*>(expected->data);
      const auto* b = static_cast<const uint8_t*>(sentinel->data);
      for (size_t i = 0; i < bytes; ++i) ICHECK_NE(a[i], b[i]);
      std::fprintf(stderr, "[THESIS_CALL_BEGIN] %s pid=%d ns=%lld output_handle=%p shared_cpu_pointer=%p bytes=%zu sentinel_verified=true\n",
                   identity.c_str(), getpid(), static_cast<long long>(ns()), output->data, view->data, bytes);
      std::fflush(stderr);
      const auto invoke_start = ns();
      function(input, weight, output);
      const auto returned = ns();
      (*sync_read)(output);
      std::memcpy(immediate->data, view->data, bytes);
      const auto first_read = ns();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      (*sync_read)(output);
      std::memcpy(delayed->data, view->data, bytes);
      const auto second_read = ns();
      const bool first_correct = std::memcmp(immediate->data, expected->data, bytes) == 0;
      const bool second_correct = std::memcmp(delayed->data, expected->data, bytes) == 0;
      const bool stable = std::memcmp(immediate->data, delayed->data, bytes) == 0;
      std::ostringstream value;
      value << "{\"pid\":" << getpid() << ",\"start_ns\":" << start
            << ",\"invoke_start_ns\":" << invoke_start << ",\"returned_ns\":" << returned
            << ",\"immediate_read_ns\":" << first_read << ",\"delayed_read_ns\":" << second_read
            << ",\"sentinel_visible\":true,\"first_correct\":" << (first_correct ? "true" : "false")
            << ",\"delayed_correct\":" << (second_correct ? "true" : "false")
            << ",\"stable\":" << (stable ? "true" : "false") << "}";
      std::fprintf(stderr, "[THESIS_CALL_END] %s %s\n", identity.c_str(), value.str().c_str());
      std::fflush(stderr);
      *rv = value.str();
    });
