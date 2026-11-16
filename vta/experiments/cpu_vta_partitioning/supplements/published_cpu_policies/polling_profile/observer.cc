// Derived from device_phase/observer.cc, Apache License 2.0.
// Read the existing driver counters once before/after an original call.
// No per-poll interception, additional sleeping, or driver changes.
#include <vta/driver.h>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <mutex>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

static_assert(sizeof(vta_phy_addr_t)==4,"audited Ejjj address ABI required");
static_assert(sizeof(VTADriverProfilerStats)==64,"audited stats ABI required");
static_assert(offsetof(VTADriverProfilerStats,poll_iters)==24,"poll counter offset differs");
static_assert(offsetof(VTADriverProfilerStats,run_total_us)==32,"driver time offset differs");
static_assert(offsetof(VTADriverProfilerStats,poll_wait_us)==56,"poll time offset differs");
namespace {
using U=unsigned long long;
using Run=int(*)(VTADeviceHandle,vta_phy_addr_t,uint32_t,uint32_t);
using Status=void(*)(VTADriverProfilerStats*);
struct Event {
  U tid,wall_start,wall_end,cpu_start,cpu_end;
  unsigned instructions;int result,core_start,core_end;
  VTADriverProfilerStats delta{};
  bool counters_monotonic=false;
};
U Clock(clockid_t clock) {
  timespec t{};if(clock_gettime(clock,&t))std::abort();
  return U(t.tv_sec)*1000000000ULL+t.tv_nsec;
}
struct Recorder {
  std::array<Event,4096> events{};size_t count=0;bool overflow=false;
  std::mutex lock;
  const char* path=std::getenv("VTA_DEVICE_PHASE_TRACE");
  Run original=nullptr;Status status=nullptr;
  const char* provider=nullptr;U provider_offset=0,status_offset=0;
  Recorder() {
    original=reinterpret_cast<Run>(dlsym(RTLD_NEXT,"VTADeviceRun"));
    status=reinterpret_cast<Status>(dlsym(RTLD_NEXT,"VTADriverProfilerStatus"));
    Dl_info a{},b{};
    if(!original||!status||!dladdr(reinterpret_cast<void*>(original),&a)||
       !dladdr(reinterpret_cast<void*>(status),&b)||!a.dli_fname||!b.dli_fname||
       a.dli_fbase!=b.dli_fbase) {
      std::fputs("poll observer original/status ABI provider unavailable\n",stderr);std::abort();
    }
    provider=a.dli_fname;
    provider_offset=reinterpret_cast<U>(original)-reinterpret_cast<U>(a.dli_fbase);
    status_offset=reinterpret_cast<U>(status)-reinterpret_cast<U>(b.dli_fbase);
  }
  void Add(const Event& e) {
    std::lock_guard<std::mutex> guard(lock);
    if(count==events.size()){overflow=true;return;}events[count++]=e;
  }
  ~Recorder() {
    if(!path||!*path)return;
    if(std::strpbrk(provider,"\"\\\n\r"))return;
    FILE* f=std::fopen(path,"w");if(!f)return;
    std::fprintf(f,"{\"kind\":\"status\",\"events\":%zu,\"overflow\":%s,\"provider\":\"%s\",\"provider_offset\":%llu,\"stats_offset\":%llu,\"stats_bytes\":64,\"counter_reads_per_device_call\":2,\"per_poll_probe\":false}\n",count,overflow?"true":"false",provider,provider_offset,status_offset);
    for(size_t i=0;i<count;++i) {
      const auto& e=events[i];const auto& d=e.delta;
      std::fprintf(f,"{\"kind\":\"device\",\"tid\":%llu,\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu,\"instructions\":%u,\"result\":%d,\"core_start\":%d,\"core_end\":%d,\"counters_monotonic\":%s,\"driver_run_calls\":%llu,\"driver_run_insns\":%llu,\"driver_timeout_calls\":%llu,\"driver_poll_iters\":%llu,\"driver_run_total_us\":%.9f,\"driver_submit_mmio_us\":%.9f,\"driver_post_start_sleep_us\":%.9f,\"driver_poll_wait_us\":%.9f}\n",
        e.tid,e.wall_start,e.wall_end,e.cpu_start,e.cpu_end,e.instructions,e.result,e.core_start,e.core_end,
        e.counters_monotonic?"true":"false",U(d.run_calls),U(d.run_insns),U(d.timeout_calls),U(d.poll_iters),
        d.run_total_us,d.submit_mmio_us,d.post_start_sleep_us,d.poll_wait_us);
    }
    std::fclose(f);
  }
};
Recorder& State(){static Recorder r;return r;}
}
extern "C" int VTADeviceRun(VTADeviceHandle device,vta_phy_addr_t address,
                            uint32_t instructions,uint32_t wait_cycles) {
  auto& state=State();
  if(!state.path||!*state.path)return state.original(device,address,instructions,wait_cycles);
  VTADriverProfilerStats before{},after{};
  state.status(&before);
  Event event{};event.tid=syscall(SYS_gettid);event.instructions=instructions;
  event.core_start=sched_getcpu();event.wall_start=Clock(CLOCK_MONOTONIC);
  event.cpu_start=Clock(CLOCK_THREAD_CPUTIME_ID);
  int result=state.original(device,address,instructions,wait_cycles);
  event.cpu_end=Clock(CLOCK_THREAD_CPUTIME_ID);event.wall_end=Clock(CLOCK_MONOTONIC);
  event.core_end=sched_getcpu();event.result=result;
  state.status(&after);
  event.counters_monotonic=(after.run_calls>=before.run_calls&&after.run_insns>=before.run_insns&&
      after.timeout_calls>=before.timeout_calls&&after.poll_iters>=before.poll_iters&&
      after.run_total_us>=before.run_total_us&&after.submit_mmio_us>=before.submit_mmio_us&&
      after.post_start_sleep_us>=before.post_start_sleep_us&&after.poll_wait_us>=before.poll_wait_us);
  event.delta.run_calls=after.run_calls-before.run_calls;
  event.delta.run_insns=after.run_insns-before.run_insns;
  event.delta.timeout_calls=after.timeout_calls-before.timeout_calls;
  event.delta.poll_iters=after.poll_iters-before.poll_iters;
  event.delta.run_total_us=after.run_total_us-before.run_total_us;
  event.delta.submit_mmio_us=after.submit_mmio_us-before.submit_mmio_us;
  event.delta.post_start_sleep_us=after.post_start_sleep_us-before.post_start_sleep_us;
  event.delta.poll_wait_us=after.poll_wait_us-before.poll_wait_us;
  state.Add(event);return result;
}
