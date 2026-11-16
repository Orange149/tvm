// Independent graph helper: exactly one GraphExecutor.run(), no extra sync.
// A throwing call remains visible in live started/completed counts.
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {
std::atomic<uint64_t> started{0}, completed{0};
std::atomic_flag active = ATOMIC_FLAG_INIT;
const char* kind = "none";

struct Invocation {
  tvm::runtime::PackedFunc function;
  explicit Invocation(tvm::runtime::TVMArgs args, const char* name) {
    ICHECK_EQ(args.size(), 1) << "only the GraphExecutor module is accepted";
    tvm::runtime::Module module = args[0];
    function = module.GetFunction("run", false);
    ICHECK(function != nullptr) << "missing GraphExecutor.run";
    ICHECK(!active.test_and_set()) << "concurrent graph helper invocation";
    kind = name;
    ++started;
  }
  ~Invocation() { active.clear(); }
  void Run() { function(); }
  void Complete() { ++completed; }
};

// Capture and formatting belong only to diagnostics, never formal timing.
struct Capture {
  FILE* file{nullptr};
  int saved{-1};
  void* handle{nullptr};
  void (*debug)(void*, int){nullptr};
  Capture() {
    auto get_handle = reinterpret_cast<void* (*)()>(dlsym(RTLD_DEFAULT, "VTATLSCommandHandle"));
    debug = reinterpret_cast<void (*)(void*, int)>(dlsym(RTLD_DEFAULT, "VTASetDebugMode"));
    ICHECK(get_handle && debug);
    handle = get_handle();
    file = std::tmpfile();
    ICHECK(file);
    std::fflush(stdout);
    saved = dup(STDOUT_FILENO);
    if (saved < 0 || dup2(fileno(file), STDOUT_FILENO) < 0) {
      if (saved >= 0) close(saved);
      std::fclose(file);
      file = nullptr;
      LOG(FATAL) << "cannot capture graph diagnostic stream";
    }
    debug(handle, 6);  // Dumps, without FORCE_SERIAL.
  }
  ~Capture() {
    if (file) {
      debug(handle, 0);
      std::fflush(stdout);
      dup2(saved, STDOUT_FILENO);
      close(saved);
      std::fclose(file);
    }
  }
  std::string Text() {
    std::fflush(stdout);
    std::rewind(file);
    std::string text;
    char buffer[8192];
    while (size_t count = std::fread(buffer, 1, sizeof(buffer), file)) text.append(buffer, count);
    return text;
  }
};
}  // namespace

TVM_REGISTER_GLOBAL("c3.yolo416.graph_reset_counts")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue*) {
      ICHECK_EQ(args.size(), 0);
      ICHECK(!active.test_and_set());
      started = 0;
      completed = 0;
      kind = "none";
      active.clear();
    });

TVM_REGISTER_GLOBAL("c3.yolo416.graph_metadata")
    .set_body_typed([]() {
      std::ostringstream out;
      out << "{\"schema\":\"c3_yolo416_graph_once_v1\",\"started_calls\":" << started.load()
          << ",\"completed_calls\":" << completed.load() << ",\"last_kind\":\"" << kind
          << "\",\"packed_calls_per_invocation\":1,\"parameter_count\":0,\"adds_synchronize\":false}";
      return out.str();
    });

TVM_REGISTER_GLOBAL("c3.yolo416.graph_timed_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      Invocation call(args, "timer");
      const auto begin = std::chrono::steady_clock::now();
      call.Run();
      const auto end = std::chrono::steady_clock::now();
      call.Complete();
      *rv = std::chrono::duration<double, std::milli>(end - begin).count();
    });

TVM_REGISTER_GLOBAL("c3.yolo416.graph_diagnostic_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      Invocation call(args, "diagnostic");
      Capture capture;
      call.Run();
      call.Complete();
      *rv = capture.Text();
    });
