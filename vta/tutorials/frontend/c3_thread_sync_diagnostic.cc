// Experiment-only capture of existing runtime dumps. No driver changes.
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

TVM_REGISTER_GLOBAL("c3.thread_sync.diagnostic_once")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      auto handle = reinterpret_cast<void* (*)()>(dlsym(RTLD_DEFAULT, "VTATLSCommandHandle"));
      auto debug = reinterpret_cast<void (*)(void*, int)>(dlsym(RTLD_DEFAULT, "VTASetDebugMode"));
      ICHECK(handle && debug);
      tvm::runtime::Module module = args[0];
      auto function = module.GetFunction("main", true);
      ICHECK(function != nullptr);
      std::fflush(stdout);
      FILE* captured = std::tmpfile();
      ICHECK(captured != nullptr);
      int saved = dup(STDOUT_FILENO);
      ICHECK_GE(saved, 0);
      ICHECK_GE(dup2(fileno(captured), STDOUT_FILENO), 0);
      try {
        debug(handle(), (1 << 1) | (1 << 2));  // Dump only; never FORCE_SERIAL.
        function(args[1], args[2], args[3]);
        debug(handle(), 0);
        std::fflush(stdout);
      } catch (...) {
        debug(handle(), 0);
        std::fflush(stdout);
        dup2(saved, STDOUT_FILENO);
        close(saved);
        std::fclose(captured);
        throw;
      }
      ICHECK_GE(dup2(saved, STDOUT_FILENO), 0);
      close(saved);
      std::rewind(captured);
      std::string text;
      char buffer[8192];
      while (size_t n = std::fread(buffer, 1, sizeof(buffer), captured)) text.append(buffer, n);
      std::fclose(captured);
      *rv = text;
    });
