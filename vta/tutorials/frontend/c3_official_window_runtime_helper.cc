// Independent generic helper: exactly one target call; no added VTA sync.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

namespace {
std::atomic<uint64_t> started{0}, completed{0};
std::mutex metadata_mutex;
std::string last_metadata{"{}"};
struct Target {
  tvm::runtime::PackedFunc function;
  tvm::runtime::TVMArgs parameters;
  Target(tvm::runtime::TVMArgs args) : parameters(nullptr, nullptr, 0) {
    ICHECK_GE(args.size(), 3) << "module, symbol and at least one tensor required";
    parameters = tvm::runtime::TVMArgs(args.values + 2, args.type_codes + 2, args.size() - 2);
    tvm::runtime::Module module = args[0];
    std::string symbol = args[1];
    function = module.GetFunction(symbol, true);
    ICHECK(function != nullptr) << "missing complete function " << symbol;
    for (int i = 2; i < args.size(); ++i)
      ICHECK(args.type_codes[i] == kTVMDLTensorHandle || args.type_codes[i] == kTVMNDArrayHandle)
          << "generic once helper accepts tensor parameters only";
  }
};
void SaveMetadata(tvm::runtime::TVMArgs args, const char* kind) {
  std::ostringstream out;
  out << "{\"schema\":\"c3_official_window_once_helper_v1\",\"started_calls\":" << started.load()
      << ",\"completed_calls\":" << completed.load() << ",\"parameter_count\":" << args.size()-2
      << ",\"last_kind\":\"" << kind << "\",\"parameters\":[";
  for (int i = 2; i < args.size(); ++i) {
    DLTensor* tensor = args[i];
    if (i > 2) out << ',';
    out << "{\"dtype_code\":" << unsigned(tensor->dtype.code) << ",\"dtype_bits\":" << unsigned(tensor->dtype.bits)
        << ",\"dtype_lanes\":" << tensor->dtype.lanes << ",\"shape\":[";
    for (int d = 0; d < tensor->ndim; ++d) { if (d) out << ','; out << tensor->shape[d]; }
    out << "],\"byte_offset\":" << tensor->byte_offset << '}';
  }
  out << "],\"adds_synchronize\":false,\"packed_calls_per_invocation\":1}";
  std::lock_guard<std::mutex> lock(metadata_mutex);
  last_metadata = out.str();
}
}  // namespace

TVM_REGISTER_GLOBAL("c3.official_window.reset_counts")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) {
      started.store(0); completed.store(0);
      std::lock_guard<std::mutex> lock(metadata_mutex); last_metadata = "{}";
    });
TVM_REGISTER_GLOBAL("c3.official_window.counts")
    .set_body_typed([]() { return static_cast<int64_t>(completed.load()); });
TVM_REGISTER_GLOBAL("c3.official_window.metadata")
    .set_body_typed([]() { std::lock_guard<std::mutex> lock(metadata_mutex); return last_metadata; });
TVM_REGISTER_GLOBAL("c3.official_window.fused_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      Target target(args);
      tvm::runtime::TVMRetValue ignored;
      ++started;
      const auto begin = std::chrono::steady_clock::now();
      target.function.CallPacked(target.parameters, &ignored);
      const auto end = std::chrono::steady_clock::now();
      ++completed;
      SaveMetadata(args, "timer");  // Metadata and counters are outside timing.
      *rv = std::chrono::duration<double, std::milli>(end - begin).count();
    });
TVM_REGISTER_GLOBAL("c3.official_window.fused_diagnostic_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      Target target(args);
      auto handle = reinterpret_cast<void* (*)()>(dlsym(RTLD_DEFAULT, "VTATLSCommandHandle"));
      auto debug = reinterpret_cast<void (*)(void*, int)>(dlsym(RTLD_DEFAULT, "VTASetDebugMode"));
      ICHECK(handle && debug);
      FILE* capture = std::tmpfile();
      ICHECK(capture);
      std::fflush(stdout);
      const int saved = dup(STDOUT_FILENO);
      ICHECK_GE(saved, 0);
      ICHECK_GE(dup2(fileno(capture), STDOUT_FILENO), 0);
      tvm::runtime::TVMRetValue ignored;
      ++started;
      try {
        debug(handle(), 6);  // Dumps only; FORCE_SERIAL remains disabled.
        target.function.CallPacked(target.parameters, &ignored);
        ++completed;
        debug(handle(), 0);
        std::fflush(stdout);
      } catch (...) {
        debug(handle(), 0); std::fflush(stdout);
        dup2(saved, STDOUT_FILENO); close(saved); std::fclose(capture);
        throw;
      }
      ICHECK_GE(dup2(saved, STDOUT_FILENO), 0);
      close(saved); std::rewind(capture);
      std::string text; char buffer[8192];
      while (size_t n = std::fread(buffer, 1, sizeof(buffer), capture)) text.append(buffer, n);
      std::fclose(capture);
      SaveMetadata(args, "diagnostic");
      *rv = text;
    });
