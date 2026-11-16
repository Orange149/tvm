// Licensed under the Apache License, Version 2.0.
// Forward the authenticated 32-bit-address VTA ABI without changing execution.
#include <vta/driver.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <mutex>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

static_assert(sizeof(vta_phy_addr_t)==4,"this observer requires the audited Ejjj VTA ABI");
namespace {
using Original=int(*)(VTADeviceHandle,vta_phy_addr_t,uint32_t,uint32_t);
struct Event {
  unsigned long long tid,wall_start,wall_end,cpu_start,cpu_end;
  unsigned instructions; int result,core_start,core_end;
};
struct Recorder {
  std::array<Event,4096> events{};
  size_t count=0;
  bool overflow=false;
  std::mutex lock;
  const char* path=std::getenv("VTA_DEVICE_PHASE_TRACE");
  Original original=nullptr;
  const char* provider=nullptr;
  unsigned long long provider_offset=0;
  Recorder() {
    original=reinterpret_cast<Original>(dlsym(RTLD_NEXT,"VTADeviceRun"));
    Dl_info info{};
    if (!original||!dladdr(reinterpret_cast<void*>(original),&info)||!info.dli_fname) {
      std::fputs("device observer could not resolve original VTADeviceRun\n",stderr);
      std::abort();
    }
    provider=info.dli_fname;
    provider_offset=reinterpret_cast<unsigned long long>(original)-
                    reinterpret_cast<unsigned long long>(info.dli_fbase);
  }
  void Add(const Event& event) {
    std::lock_guard<std::mutex> guard(lock);
    if (count==events.size()) {overflow=true;return;}
    events[count++]=event;
  }
  ~Recorder() {
    if (!path||!*path) return;
    // Deployment paths are constrained and cannot contain JSON delimiters.
    if (std::strpbrk(provider,"\"\\\n\r")) return;
    FILE* output=std::fopen(path,"w");
    if (!output) return;
    std::fprintf(output,"{\"kind\":\"status\",\"events\":%zu,\"overflow\":%s,"
        "\"provider\":\"%s\",\"provider_offset\":%llu}\n",count,
        overflow?"true":"false",provider,provider_offset);
    for(size_t i=0;i<count;++i) {
      const auto& e=events[i];
      std::fprintf(output,"{\"kind\":\"device\",\"tid\":%llu,\"wall_start_ns\":%llu,"
          "\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu,"
          "\"instructions\":%u,\"result\":%d,\"core_start\":%d,\"core_end\":%d}\n",
          e.tid,e.wall_start,e.wall_end,e.cpu_start,e.cpu_end,e.instructions,e.result,
          e.core_start,e.core_end);
    }
    std::fclose(output);
  }
};
Recorder& State() {static Recorder value;return value;}
unsigned long long Clock(clockid_t clock) {
  timespec value{};
  if (clock_gettime(clock,&value)) std::abort();
  return static_cast<unsigned long long>(value.tv_sec)*1000000000ULL+value.tv_nsec;
}
}
extern "C" int VTADeviceRun(VTADeviceHandle device,vta_phy_addr_t address,
                            uint32_t instructions,uint32_t wait_cycles) {
  auto& state=State();
  if (!state.path||!*state.path) return state.original(device,address,instructions,wait_cycles);
  Event event{};event.tid=syscall(SYS_gettid);event.instructions=instructions;
  event.core_start=sched_getcpu();event.wall_start=Clock(CLOCK_MONOTONIC);
  event.cpu_start=Clock(CLOCK_THREAD_CPUTIME_ID);
  int result=state.original(device,address,instructions,wait_cycles);
  event.cpu_end=Clock(CLOCK_THREAD_CPUTIME_ID);event.wall_end=Clock(CLOCK_MONOTONIC);
  event.core_end=sched_getcpu();event.result=result;state.Add(event);
  return result;
}
