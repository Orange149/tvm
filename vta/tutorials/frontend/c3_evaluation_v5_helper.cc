// Board-side timing and synchronization controls for the frozen C3 v5 campaign.
// The helper resolves VTA symbols from the RPC process and never links libvta,
// which prevents it from accidentally owning a second command queue/runtime.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <iomanip>
#include <sstream>
#include <string>

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

namespace {

using CommandHandle = void*;
using GetHandle = CommandHandle (*)();
using Synchronize = void (*)(CommandHandle, unsigned int);

std::atomic<uint64_t> operator_calls{0};
std::atomic<uint64_t> graph_calls{0};
std::atomic<uint64_t> explicit_sync_calls{0};

template <typename T>
T Resolve(const char* name) {
  dlerror();
  auto function = reinterpret_cast<T>(dlsym(RTLD_DEFAULT, name));
  const char* error = dlerror();
  ICHECK(function != nullptr && error == nullptr) << "missing runtime symbol " << name;
  return function;
}

void Sync() {
  auto handle = Resolve<GetHandle>("VTATLSCommandHandle");
  auto synchronize = Resolve<Synchronize>("VTASynchronize");
  synchronize(handle(), 1U << 31);
  ++explicit_sync_calls;
}

void ClearProfiler() {
  const tvm::runtime::PackedFunc* clear =
      tvm::runtime::Registry::Get("vta.runtime.profiler_clear");
  ICHECK(clear != nullptr) << "missing vta.runtime.profiler_clear";
  (*clear)();
}

double Milliseconds(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string SymbolJSON(const char* name, void* pointer) {
  Dl_info info{};
  ICHECK(dladdr(pointer, &info) != 0) << "dladdr failed for " << name;
  std::ostringstream os;
  os << "\"" << name << "\":{";
  os << "\"address\":\"0x" << std::hex << reinterpret_cast<uintptr_t>(pointer) << "\"";
  os << ",\"dso\":\"" << (info.dli_fname == nullptr ? "" : info.dli_fname) << "\"}";
  return os.str();
}

tvm::runtime::PackedFunc ModuleFunction(tvm::runtime::Module module, const char* name) {
  auto function = module.GetFunction(name, true);
  ICHECK(function != nullptr) << "module lacks function " << name;
  return function;
}

}  // namespace

TVM_REGISTER_GLOBAL("c3.v5.runtime_symbols")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue* rv) {
      auto handle = Resolve<GetHandle>("VTATLSCommandHandle");
      auto synchronize = Resolve<Synchronize>("VTASynchronize");
      std::ostringstream os;
      os << "{" << SymbolJSON("VTATLSCommandHandle", reinterpret_cast<void*>(handle)) << ","
         << SymbolJSON("VTASynchronize", reinterpret_cast<void*>(synchronize)) << "}";
      *rv = os.str();
    });

TVM_REGISTER_GLOBAL("c3.v5.reset_counts")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) {
      operator_calls.store(0);
      graph_calls.store(0);
      explicit_sync_calls.store(0);
    });

TVM_REGISTER_GLOBAL("c3.v5.counts")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue* rv) {
      std::ostringstream os;
      os << "{\"operator_calls\":" << operator_calls.load()
         << ",\"graph_calls\":" << graph_calls.load()
         << ",\"explicit_sync_calls\":" << explicit_sync_calls.load() << "}";
      *rv = os.str();
    });

TVM_REGISTER_GLOBAL("c3.v5.empty_sync")
    .set_body([](tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) { Sync(); });

TVM_REGISTER_GLOBAL("c3.v5.operator_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      tvm::runtime::Module module = args[0];
      auto function = ModuleFunction(module, "main");
      const auto start = std::chrono::steady_clock::now();
      function(args[1], args[2], args[3]);
      const auto end = std::chrono::steady_clock::now();
      ++operator_calls;
      *rv = Milliseconds(start, end);
    });

TVM_REGISTER_GLOBAL("c3.v5.legacy_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      tvm::runtime::Module module = args[0];
      auto function = ModuleFunction(module, "main");
      // Establish a common state outside the timing/profile boundary.
      Sync();
      ClearProfiler();
      const auto start = std::chrono::steady_clock::now();
      function(args[1], args[2], args[3]);
      ++operator_calls;
      // Legacy completion is part of the measured controlled-isolation wrapper.
      Sync();
      const auto end = std::chrono::steady_clock::now();
      *rv = Milliseconds(start, end);
    });

TVM_REGISTER_GLOBAL("c3.v5.legacy_audit_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue*) {
      tvm::runtime::Module module = args[0];
      auto function = ModuleFunction(module, "main");
      Sync();
      ClearProfiler();
      function(args[1], args[2], args[3]);
      ++operator_calls;
      Sync();
    });


TVM_REGISTER_GLOBAL("c3.v5.graph_run_only_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      tvm::runtime::Module module = args[0];
      auto function = ModuleFunction(module, "run");
      const auto start = std::chrono::steady_clock::now();
      function();
      const auto end = std::chrono::steady_clock::now();
      ++graph_calls;
      *rv = Milliseconds(start, end);
    });

TVM_REGISTER_GLOBAL("c3.v5.graph_run_sync_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      tvm::runtime::Module module = args[0];
      auto function = ModuleFunction(module, "run");
      const auto start = std::chrono::steady_clock::now();
      function();
      ++graph_calls;
      Sync();
      const auto end = std::chrono::steady_clock::now();
      *rv = Milliseconds(start, end);
    });
