// Derived from polling_profile/observer.cc, Apache License 2.0.
// Observe original MMIO calls once each. No extra register reads or sleeps.
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
namespace {
using U=unsigned long long;
using Run=int(*)(VTADeviceHandle,vta_phy_addr_t,uint32_t,uint32_t);
using Status=void(*)(VTADriverProfilerStats*);
using Read=uint32_t(*)(void*,uint32_t);
using Write=void(*)(void*,uint32_t,uint32_t);
struct Event {
  U tid=0,wall_start=0,wall_end=0,cpu_start=0,cpu_end=0;
  U launch_before=0,launch_after=0,last_false_before=0,last_false_after=0;
  U first_true_before=0,first_true_after=0,read_clock_envelope=0;
  U polls=0,false_reads=0,true_reads=0,start_writes=0,auto_writes=0;
  unsigned instructions=0;int result=0,core_start=0,core_end=0;
  void* compute_handle=nullptr;bool mmio_error=false,counters_monotonic=false;
  VTADriverProfilerStats delta{};
};
thread_local Event* active=nullptr;
U Clock(clockid_t c){timespec t{};if(clock_gettime(c,&t))std::abort();return U(t.tv_sec)*1000000000ULL+t.tv_nsec;}
struct Recorder {
  std::array<Event,4096> events{};size_t count=0;bool overflow=false;
  std::mutex lock;const char* path=std::getenv("VTA_DEVICE_PHASE_TRACE");
  Run original=nullptr;Status status=nullptr;Read read=nullptr;Write write=nullptr;
  const char* provider=nullptr;U provider_offset=0,status_offset=0,read_offset=0,write_offset=0;
  Recorder(){
    original=reinterpret_cast<Run>(dlsym(RTLD_NEXT,"VTADeviceRun"));
    status=reinterpret_cast<Status>(dlsym(RTLD_NEXT,"VTADriverProfilerStatus"));
    read=reinterpret_cast<Read>(dlsym(RTLD_NEXT,"VTAReadMappedReg"));
    write=reinterpret_cast<Write>(dlsym(RTLD_NEXT,"VTAWriteMappedReg"));
    Dl_info a{},b{},c{},d{};
    if(!original||!status||!read||!write||!dladdr(reinterpret_cast<void*>(original),&a)||
       !dladdr(reinterpret_cast<void*>(status),&b)||!dladdr(reinterpret_cast<void*>(read),&c)||
       !dladdr(reinterpret_cast<void*>(write),&d)||!a.dli_fname||a.dli_fbase!=b.dli_fbase||
       a.dli_fbase!=c.dli_fbase||a.dli_fbase!=d.dli_fbase){
      std::fputs("completion observer original MMIO/provider ABI unavailable\n",stderr);std::abort();}
    provider=a.dli_fname;U base=reinterpret_cast<U>(a.dli_fbase);
    provider_offset=reinterpret_cast<U>(original)-base;status_offset=reinterpret_cast<U>(status)-base;
    read_offset=reinterpret_cast<U>(read)-base;write_offset=reinterpret_cast<U>(write)-base;
  }
  void Add(const Event& e){std::lock_guard<std::mutex> g(lock);if(count==events.size()){overflow=true;return;}events[count++]=e;}
  ~Recorder(){
    if(!path||!*path||std::strpbrk(provider,"\"\\\n\r"))return;
    FILE* f=std::fopen(path,"w");if(!f)return;
    std::fprintf(f,"{\"kind\":\"status\",\"events\":%zu,\"overflow\":%s,\"provider\":\"%s\",\"provider_offset\":%llu,\"stats_offset\":%llu,\"read_offset\":%llu,\"write_offset\":%llu,\"stats_bytes\":64,\"per_poll_probe\":true,\"completion_brackets\":true,\"extra_MMIO_reads\":0}\n",count,overflow?"true":"false",provider,provider_offset,status_offset,read_offset,write_offset);
    for(size_t i=0;i<count;++i){const auto& e=events[i];const auto& d=e.delta;
      std::fprintf(f,"{\"kind\":\"device\",\"tid\":%llu,\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu,\"instructions\":%u,\"result\":%d,\"core_start\":%d,\"core_end\":%d,\"counters_monotonic\":%s,\"driver_run_calls\":%llu,\"driver_run_insns\":%llu,\"driver_timeout_calls\":%llu,\"driver_poll_iters\":%llu,\"driver_run_total_us\":%.9f,\"driver_submit_mmio_us\":%.9f,\"driver_post_start_sleep_us\":%.9f,\"driver_poll_wait_us\":%.9f,",e.tid,e.wall_start,e.wall_end,e.cpu_start,e.cpu_end,e.instructions,e.result,e.core_start,e.core_end,e.counters_monotonic?"true":"false",U(d.run_calls),U(d.run_insns),U(d.timeout_calls),U(d.poll_iters),d.run_total_us,d.submit_mmio_us,d.post_start_sleep_us,d.poll_wait_us);
      std::fprintf(f,"\"launch_before_ns\":%llu,\"launch_after_ns\":%llu,\"last_false_before_ns\":%llu,\"last_false_after_ns\":%llu,\"first_true_before_ns\":%llu,\"first_true_after_ns\":%llu,\"poll_read_envelope_ns\":%llu,\"observed_poll_reads\":%llu,\"false_reads\":%llu,\"true_reads\":%llu,\"start_writes\":%llu,\"auto_writes\":%llu,\"MMIO_error\":%s}\n",e.launch_before,e.launch_after,e.last_false_before,e.last_false_after,e.first_true_before,e.first_true_after,e.read_clock_envelope,e.polls,e.false_reads,e.true_reads,e.start_writes,e.auto_writes,e.mmio_error?"true":"false");
    }std::fclose(f);
  }
};
Recorder& State(){static Recorder r;return r;}
}
extern "C" void VTAWriteMappedReg(void* h,uint32_t offset,uint32_t value){
  auto& s=State();auto* e=active;
  if(!e||offset!=0||(value!=1&&value!=129)){s.write(h,offset,value);return;}
  U before=Clock(CLOCK_MONOTONIC);s.write(h,offset,value);U after=Clock(CLOCK_MONOTONIC);
  if(value==1){if(e->start_writes++)e->mmio_error=true;e->launch_before=before;}
  else {++e->auto_writes;e->launch_after=after;}
}
extern "C" uint32_t VTAReadMappedReg(void* h,uint32_t offset){
  auto& s=State();auto* e=active;
  // Audited deployed VTADeviceRun reads done only at compute offset 24.
  if(!e||offset!=24)return s.read(h,offset);
  U before=Clock(CLOCK_MONOTONIC);uint32_t value=s.read(h,offset);U after=Clock(CLOCK_MONOTONIC);
  if(e->compute_handle&&e->compute_handle!=h)e->mmio_error=true;e->compute_handle=h;
  ++e->polls;e->read_clock_envelope+=after-before;
  if(value==1){++e->true_reads;if(!e->first_true_before){e->first_true_before=before;e->first_true_after=after;}}
  else {if(e->true_reads)e->mmio_error=true;++e->false_reads;e->last_false_before=before;e->last_false_after=after;}
  return value;
}
extern "C" int VTADeviceRun(VTADeviceHandle h,vta_phy_addr_t a,uint32_t n,uint32_t w){
  auto& s=State();if(!s.path||!*s.path)return s.original(h,a,n,w);
  if(active){std::fputs("nested VTADeviceRun unsupported\n",stderr);std::abort();}
  VTADriverProfilerStats before{},after{};s.status(&before);
  Event e{};e.tid=syscall(SYS_gettid);e.instructions=n;e.core_start=sched_getcpu();
  e.wall_start=Clock(CLOCK_MONOTONIC);e.cpu_start=Clock(CLOCK_THREAD_CPUTIME_ID);active=&e;
  e.result=s.original(h,a,n,w);active=nullptr;
  e.cpu_end=Clock(CLOCK_THREAD_CPUTIME_ID);e.wall_end=Clock(CLOCK_MONOTONIC);e.core_end=sched_getcpu();s.status(&after);
  e.counters_monotonic=(after.run_calls>=before.run_calls&&after.run_insns>=before.run_insns&&after.timeout_calls>=before.timeout_calls&&after.poll_iters>=before.poll_iters&&after.run_total_us>=before.run_total_us&&after.submit_mmio_us>=before.submit_mmio_us&&after.post_start_sleep_us>=before.post_start_sleep_us&&after.poll_wait_us>=before.poll_wait_us);
  e.delta.run_calls=after.run_calls-before.run_calls;e.delta.run_insns=after.run_insns-before.run_insns;e.delta.timeout_calls=after.timeout_calls-before.timeout_calls;e.delta.poll_iters=after.poll_iters-before.poll_iters;e.delta.run_total_us=after.run_total_us-before.run_total_us;e.delta.submit_mmio_us=after.submit_mmio_us-before.submit_mmio_us;e.delta.post_start_sleep_us=after.post_start_sleep_us-before.post_start_sleep_us;e.delta.poll_wait_us=after.poll_wait_us-before.poll_wait_us;
  s.Add(e);return e.result;
}
