// A single-process, zero-DNN audit of the frozen runtime's event append path.
// Calls no buffer, command-queue, device, inference, or system-tuning API.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// This 152-byte by-value ABI is checked against the actual frozen caller's
// memcpy size and field offsets before building; not a portable VTA API.
struct Driver {
  uint64_t run_calls, run_insns, timeout_calls, poll_iters;
  double run_total_us, submit_mmio_us, post_start_sleep_us, poll_wait_us;
};
struct Event {
  uint64_t seq;
  double ts_us;
  int kind, scope;
  size_t bytes;
  double duration_us;
  uint64_t insns;
  size_t load_bytes, store_bytes;
  uint32_t memory_type, x_size, y_size, x_stride;
  bool padded;
  Driver driver;
};
static_assert(sizeof(Event)==152 && offsetof(Event,driver)==88, "frozen event ABI differs");

static uint64_t stamp(clockid_t id) {
  timespec t{};
  if(clock_gettime(id,&t))std::abort();
  return uint64_t(t.tv_sec)*1000000000+uint64_t(t.tv_nsec);
}
static int invalid(const char* reason) {
  std::fprintf(stderr,"event-cost probe invalid: %s\n",reason);
  return 2;
}
using Global=void*(*)();
using Push=void(*)(void*,Event);
using Status=std::string(*)(void*);
using Events=std::string(*)(void*,int);
static uint64_t empty_calls=0;
__attribute__((noinline)) static void empty_push(void*,Event e) {
  // Match one external by-value call; retain a visible result for correctness.
  empty_calls+=uint64_t(e.kind);
  asm volatile("":::"memory");
}
struct Result {uint64_t count,cpu,wall;};
static Result block(Push push,void* object,Event event,uint64_t n) {
  const auto a=stamp(CLOCK_MONOTONIC),c=stamp(CLOCK_THREAD_CPUTIME_ID);
  for(uint64_t i=0;i<n;++i)push(object,event);
  const auto d=stamp(CLOCK_THREAD_CPUTIME_ID),b=stamp(CLOCK_MONOTONIC);
  return {n,d-c,b-a};
}
int main(int argc,char** argv) {
  alarm(8);
  if(argc!=2)return invalid("one frozen libvta path required");
  cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(3,&mask);
  if(sched_setaffinity(0,sizeof(mask),&mask))return invalid("single-core affinity failed");
  CPU_ZERO(&mask);
  if(sched_getaffinity(0,sizeof(mask),&mask)||CPU_COUNT(&mask)!=1||!CPU_ISSET(3,&mask))
    return invalid("actual affinity differs");
  const std::string vta_path(argv[1]);
  const size_t slash=vta_path.find_last_of('/');
  if(slash==std::string::npos)return invalid("absolute frozen library path required");
  const std::string tvm_path=vta_path.substr(0,slash+1)+"libtvm_runtime.so";
  // Normal executors already provide TVM's global RTTI and registry symbols.
  // This standalone helper must load the same frozen provider first.
  if(!dlopen(tvm_path.c_str(),RTLD_NOW|RTLD_GLOBAL)){
    std::fprintf(stderr,"%s\n",dlerror());return invalid("TVM provider load failed");
  }
  void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!library){std::fprintf(stderr,"%s\n",dlerror());return invalid("load failed");}
  auto global=reinterpret_cast<Global>(dlsym(library,"_ZN3vta15RuntimeProfiler6GlobalEv"));
  auto push=reinterpret_cast<Push>(dlsym(library,"_ZN3vta15RuntimeProfiler15PushEventLockedENS0_5EventE"));
  auto status=reinterpret_cast<Status>(dlsym(library,"_ZN3vta15RuntimeProfiler6AsJSONB5cxx11Ev"));
  auto events=reinterpret_cast<Events>(dlsym(library,"_ZN3vta15RuntimeProfiler12EventsAsJSONB5cxx11Ei"));
  // EventsAsJSON may be fully inlined; require only the actually exported ABI.
  if(!global||!push||!status)return invalid("frozen symbols unavailable");
  void* object=global();
  const std::string before=status(object);
  if(before.find("\"event_count\":0")==std::string::npos)return invalid("fresh event history not empty");
  Event event{};event.kind=5;
  // Only one thread accesses this process-local profiler. The normal caller's
  // mutex and DMA-signature bookkeeping are explicitly outside this benchmark.
  block(push,object,event,5000);
  const Result baseline=block(empty_push,object,event,20000);
  const Result small=block(push,object,event,5000);
  const Result large=block(push,object,event,20000);
  const std::string after=status(object);
  if(empty_calls!=100000||after.find("\"event_count\":4096")==std::string::npos||
     after.find("\"load_buffer_2d_calls\":0")==std::string::npos||
     after.find("\"synchronize_calls\":0")==std::string::npos)
    return invalid("synthetic event or operation-count conservation failed");
  std::printf("{\"status\":\"VALIDATED\",\"DNN_inferences\":0,\"FPGA_access\":false,"
    "\"affinity\":[3],\"event_size\":%zu,\"warmup_events\":5000,\"total_events\":30000,"
    "\"single_thread_only\":true,\"mutex_and_signature_bookkeeping_measured\":false,"
    "\"synthetic_events_are_not_DMA_records\":true,\"blocks\":[",sizeof(Event));
  const Result rows[]={baseline,small,large};
  const char* names[]={"empty_by_value_call","actual_event_append_5000","actual_event_append_20000"};
  for(int i=0;i<3;++i)std::printf("%s{\"name\":\"%s\",\"calls\":%llu,\"CPU_ns\":%llu,\"wall_ns\":%llu}",
    i?",":"",names[i],(unsigned long long)rows[i].count,
    (unsigned long long)rows[i].cpu,(unsigned long long)rows[i].wall);
  std::printf("],\"retained_events\":4096,\"optional_event_dump_ABI_available\":%s,\"stats\":%s}\n",
    events?"true":"false",after.c_str());
  // Leave the library loaded until process exit; no FPGA/runtime shutdown API.
  return 0;
}
