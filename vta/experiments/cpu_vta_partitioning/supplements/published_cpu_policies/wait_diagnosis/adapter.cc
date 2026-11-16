// Public-C-ABI interposition for the frozen non-OpenMP TVM runtime.
// Original policy always calls the original launch/barrier/Yield functions.
// Blocking policy changes only synchronization waits, never the kernel body.
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

extern "C" {
struct TVMParallelGroupEnv { void* sync_handle; int32_t num_task; };
using Lambda = int (*)(int, TVMParallelGroupEnv*, void*);
}
static_assert(sizeof(TVMParallelGroupEnv)==16 && offsetof(TVMParallelGroupEnv,num_task)==8,
              "frozen runtime public ABI differs");
static_assert(sizeof(std::atomic<uint32_t>)==4,"futex word must be 32 bits");

namespace {
using Launch = int (*)(Lambda,void*,int);
using Barrier = int (*)(int,TVMParallelGroupEnv*);
using Yield = void (*)();
Launch original_launch;
Barrier original_barrier;
Yield original_yield;
pthread_once_t once=PTHREAD_ONCE_INIT;
bool initialized=false, blocking=false, probe=false;
const char *trace_path=nullptr,*status_path=nullptr;
#ifndef WAIT_TRACE_CAPACITY
#define WAIT_TRACE_CAPACITY 262144
#endif
constexpr uint64_t kCapacity=WAIT_TRACE_CAPACITY;
struct Record {
  uint64_t launch,tid,caller,callback,ws,we,cs,ce;
  uint64_t barrier_wall,barrier_cpu,barrier_calls,yields,completion_yields;
  uint64_t tail_ws,tail_cs,last_task_end;
  int kind,task,tasks,result,module_index;
};
struct ModuleInfo {uintptr_t function;uint64_t offset;char path[2048];};
ModuleInfo modules[4096];int module_count=0;
pthread_mutex_t module_mutex=PTHREAD_MUTEX_INITIALIZER;
Record* records=nullptr;
std::atomic<uint64_t> next_record{0},next_launch{1},launches{0},callbacks{0},barriers{0},
  yields{0},completion_yields{0},outside_yields{0},overflows{0},errors{0},
  futex_waits{0},futex_wakes{0},futex_interrupts{0},futex_races{0};

uint64_t clock_ns(clockid_t clock) {
  timespec t{};
  if(clock_gettime(clock,&t)!=0) { errors.fetch_add(1);return 0; }
  return uint64_t(t.tv_sec)*1000000000ull+uint64_t(t.tv_nsec);
}
uint64_t tid() { return static_cast<uint64_t>(syscall(SYS_gettid)); }
void quote(FILE* f,const char* s) {
  fputc('"',f);
  for(const unsigned char* p=reinterpret_cast<const unsigned char*>(s?s:"");*p;++p) {
    if(*p=='"'||*p=='\\') { fputc('\\',f);fputc(*p,f); }
    else if(*p<32)fprintf(f,"\\u%04x",*p);else fputc(*p,f);
  }
  fputc('"',f);
}
void add_record(const Record& r) {
  if(!probe)return;
  const auto i=next_record.fetch_add(1,std::memory_order_relaxed);
  if(i<kCapacity)records[i]=r;else overflows.fetch_add(1,std::memory_order_relaxed);
}
int module_for(Lambda function) {
  // Cache while the DSO is loaded, not at process exit after graph destruction.
  struct Entry {uintptr_t function;int module;};
  static thread_local Entry cached[64]{};static thread_local int used=0;
  const auto address=reinterpret_cast<uintptr_t>(function);
  for(int i=0;i<used;++i)if(cached[i].function==address)return cached[i].module;
  pthread_mutex_lock(&module_mutex);int index=-1;
  for(int i=0;i<module_count;++i)if(modules[i].function==address){index=i;break;}
  if(index<0&&module_count<4096) {
    Dl_info info{};
    if(dladdr(reinterpret_cast<void*>(function),&info)!=0) {
      index=module_count++;auto& m=modules[index];m.function=address;
      m.offset=address-reinterpret_cast<uintptr_t>(info.dli_fbase);
      if(strlen(info.dli_fname)>=sizeof(m.path)){errors.fetch_add(1);index=-1;}
      else strcpy(m.path,info.dli_fname);
    } else errors.fetch_add(1);
  }
  if(index<0)errors.fetch_add(1);
  pthread_mutex_unlock(&module_mutex);
  if(index>=0&&used<64)cached[used++]={address,index};return index;
}
void flush() {
  if(!initialized)return;
  if(probe) {
    FILE* f=trace_path?fopen(trace_path,"w"):nullptr;
    if(!f)errors.fetch_add(1);
    else {
      const uint64_t n=next_record.load()<kCapacity?next_record.load():kCapacity;
      for(uint64_t i=0;i<n;++i) {
        const auto& r=records[i];bool found=r.module_index>=0&&r.module_index<module_count;
        fprintf(f,"{\"kind\":%d,\"launch\":%llu,\"tid\":%llu,\"caller\":%llu,\"task\":%d,\"tasks\":%d,",
          r.kind,(unsigned long long)r.launch,(unsigned long long)r.tid,(unsigned long long)r.caller,r.task,r.tasks);
        fprintf(f,"\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu,",
          (unsigned long long)r.ws,(unsigned long long)r.we,(unsigned long long)r.cs,(unsigned long long)r.ce);
        fprintf(f,"\"barrier_wall_ns\":%llu,\"barrier_cpu_ns\":%llu,\"barrier_calls\":%llu,\"yield_calls\":%llu,\"completion_yield_calls\":%llu,",
          (unsigned long long)r.barrier_wall,(unsigned long long)r.barrier_cpu,(unsigned long long)r.barrier_calls,
          (unsigned long long)r.yields,(unsigned long long)r.completion_yields);
        fprintf(f,"\"caller_tail_wall_start_ns\":%llu,\"caller_tail_cpu_start_ns\":%llu,\"last_task_end_ns\":%llu,\"result\":%d,\"module\":",
          (unsigned long long)r.tail_ws,(unsigned long long)r.tail_cs,(unsigned long long)r.last_task_end,r.result);
        quote(f,found?modules[r.module_index].path:"");
        fprintf(f,",\"function_offset\":%llu}\n",(unsigned long long)(found?modules[r.module_index].offset:0));
      }
      if(fclose(f)!=0)errors.fetch_add(1);
    }
  }
  FILE* f=status_path?fopen(status_path,"w"):nullptr;
  if(f) {
    fprintf(f,"{\"schema\":1,\"policy\":\"%s\",\"probe\":%s,\"launches\":%llu,\"callbacks\":%llu,\"barriers\":%llu,\"yield_calls\":%llu,\"completion_yield_calls\":%llu,\"outside_yield_calls\":%llu,\"records\":%llu,\"overflow\":%llu,\"errors\":%llu,\"futex_waits\":%llu,\"futex_wakes\":%llu,\"futex_interrupts\":%llu,\"futex_races\":%llu,\"scope\":\"whole_native_session_not_scored_frames\"}\n",
      blocking?"blocking":"original",probe?"true":"false",
      (unsigned long long)launches.load(),(unsigned long long)callbacks.load(),(unsigned long long)barriers.load(),
      (unsigned long long)yields.load(),(unsigned long long)completion_yields.load(),(unsigned long long)outside_yields.load(),
      (unsigned long long)next_record.load(),(unsigned long long)overflows.load(),(unsigned long long)errors.load(),
      (unsigned long long)futex_waits.load(),(unsigned long long)futex_wakes.load(),
      (unsigned long long)futex_interrupts.load(),(unsigned long long)futex_races.load());
    fclose(f);
  }
}
void initialize() {
  original_launch=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"TVMBackendParallelLaunch"));
  original_barrier=reinterpret_cast<Barrier>(dlsym(RTLD_NEXT,"TVMBackendParallelBarrier"));
  original_yield=reinterpret_cast<Yield>(dlsym(RTLD_NEXT,"_ZN3tvm7runtime9threading5YieldEv"));
  const char* policy=getenv("VTA_WAIT_POLICY");const char* detail=getenv("VTA_WAIT_PROBE");
  if(!original_launch||!original_barrier||!original_yield||!policy||
      (strcmp(policy,"original")&&strcmp(policy,"blocking"))||!detail||
      (strcmp(detail,"0")&&strcmp(detail,"1"))) {
    fputs("wait adapter: unresolved frozen ABI or invalid explicit mode\n",stderr);_exit(80);
  }
  blocking=!strcmp(policy,"blocking");probe=!strcmp(detail,"1");
  trace_path=getenv("VTA_WAIT_TRACE");status_path=getenv("VTA_WAIT_STATUS");
  if(!status_path || (probe&&!trace_path))_exit(81);
  if(probe) {
    void* p=mmap(nullptr,kCapacity*sizeof(Record),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(p==MAP_FAILED)_exit(82);
    records=static_cast<Record*>(p);
  }
  initialized=true;atexit(flush);
}
struct Context {
  Lambda function;void* data;uint64_t id,caller;int module_index=-1;
  std::atomic<uint32_t> tasks{0},finished{0},aborted{0},generation[4]{};
  std::atomic<uint64_t> last_end{0};
  bool caller_done=false;uint64_t tail_wall=0,tail_cpu=0;
  uint64_t caller_completion_yields=0;
  Context(){for(auto& g:generation)g.store(0,std::memory_order_relaxed);}
};
struct Local {
  Context* ctx=nullptr;bool callback=false,barrier=false;
  uint64_t total_yields=0,barrier_yields=0,completion=0;
  uint64_t barrier_wall=0,barrier_cpu=0,barrier_calls=0;
};
thread_local Local local;
void wake(std::atomic<uint32_t>& word) {
  futex_wakes.fetch_add(1,std::memory_order_relaxed);
  if(syscall(SYS_futex,reinterpret_cast<uint32_t*>(&word),FUTEX_WAKE_PRIVATE,INT_MAX,nullptr,nullptr,0)<0)
    errors.fetch_add(1);
}
void wait(std::atomic<uint32_t>& word,uint32_t expected) {
  futex_waits.fetch_add(1,std::memory_order_relaxed);
#ifdef WAIT_ADAPTER_SPURIOUS_ONCE
  // Test-only injection of a wake without a changed predicate. Not shipped.
  static thread_local bool injected=false;
  if(!injected){injected=true;return;}
#endif
  const long res=syscall(SYS_futex,reinterpret_cast<uint32_t*>(&word),FUTEX_WAIT_PRIVATE,expected,nullptr,nullptr,0);
  if(res<0) {
    if(errno==EINTR)futex_interrupts.fetch_add(1);
    else if(errno==EAGAIN)futex_races.fetch_add(1);
    else errors.fetch_add(1);
  }
}
void abort_group(Context& c) {
  c.aborted.store(1,std::memory_order_release);
  wake(c.finished);for(auto& g:c.generation)wake(g);
}
int call_task(int task,TVMParallelGroupEnv* env,void* pointer) {
  auto& c=*static_cast<Context*>(pointer);Local previous=local;
  local=Local{};local.ctx=&c;local.callback=true;
  if(env->num_task<1||env->num_task>4||task<0||task>=env->num_task) {
    errors.fetch_add(1);abort_group(c);local=previous;return -1;
  }
  uint32_t expected=0;c.tasks.compare_exchange_strong(expected,env->num_task);
  if(c.tasks.load()!=uint32_t(env->num_task)) {errors.fetch_add(1);abort_group(c);local=previous;return -1;}
  callbacks.fetch_add(1,std::memory_order_relaxed);
  Record r{};r.kind=1;r.launch=c.id;r.tid=tid();r.caller=c.caller;r.task=task;r.tasks=env->num_task;
  r.module_index=c.module_index;
  r.callback=reinterpret_cast<uintptr_t>(c.function);
  if(probe) {r.ws=clock_ns(CLOCK_MONOTONIC);r.cs=clock_ns(CLOCK_THREAD_CPUTIME_ID);}
  try {r.result=c.function(task,env,c.data);}
  catch(...) {errors.fetch_add(1);r.result=-1;if(blocking)abort_group(c);}
  if(probe) {r.ce=clock_ns(CLOCK_THREAD_CPUTIME_ID);r.we=clock_ns(CLOCK_MONOTONIC);}
  r.barrier_wall=local.barrier_wall;r.barrier_cpu=local.barrier_cpu;
  r.barrier_calls=local.barrier_calls;r.yields=local.total_yields;
  yields.fetch_add(r.yields,std::memory_order_relaxed);
  if(probe) {
    auto end=c.last_end.load();while(end<r.we&&!c.last_end.compare_exchange_weak(end,r.we)){}
  }
  if(r.tid==c.caller) {c.caller_done=true;if(probe){c.tail_wall=r.we;c.tail_cpu=r.ce;}}
  if(r.result!=0&&blocking)abort_group(c);
  c.finished.fetch_add(1,std::memory_order_release);
  if(blocking)wake(c.finished);
  add_record(r);local=previous;return r.result;
}
}

extern "C" int TVMBackendParallelLaunch(Lambda f,void* data,int tasks) {
  pthread_once(&once,initialize);
  Context c;c.function=f;c.data=data;c.id=next_launch.fetch_add(1);c.caller=tid();
  if(probe)c.module_index=module_for(f);
  Local previous=local;local=Local{};local.ctx=&c;
  Record r{};r.kind=0;r.launch=c.id;r.tid=c.caller;r.caller=c.caller;r.task=-1;
  r.module_index=c.module_index;
  r.callback=reinterpret_cast<uintptr_t>(f);
  if(probe){r.ws=clock_ns(CLOCK_MONOTONIC);r.cs=clock_ns(CLOCK_THREAD_CPUTIME_ID);}
  launches.fetch_add(1,std::memory_order_relaxed);
  r.result=original_launch(call_task,&c,tasks);
  if(probe){r.ce=clock_ns(CLOCK_THREAD_CPUTIME_ID);r.we=clock_ns(CLOCK_MONOTONIC);}
  r.tasks=static_cast<int>(c.tasks.load());r.yields=local.total_yields;
  r.completion_yields=c.caller_completion_yields;r.tail_ws=c.tail_wall;r.tail_cs=c.tail_cpu;
  r.last_task_end=c.last_end.load();
  yields.fetch_add(r.yields,std::memory_order_relaxed);
  completion_yields.fetch_add(r.completion_yields,std::memory_order_relaxed);
  if(c.finished.load()!=c.tasks.load())errors.fetch_add(1);
  add_record(r);local=previous;return r.result;
}

extern "C" int TVMBackendParallelBarrier(int task,TVMParallelGroupEnv* env) {
  pthread_once(&once,initialize);barriers.fetch_add(1,std::memory_order_relaxed);
  if(!local.ctx||!local.callback) {
    if(blocking){errors.fetch_add(1);return -1;}
    return original_barrier(task,env);
  }
  auto& c=*local.ctx;Record r{};r.kind=2;r.launch=c.id;r.tid=tid();r.caller=c.caller;
  r.module_index=c.module_index;
  r.task=task;r.tasks=env->num_task;r.callback=reinterpret_cast<uintptr_t>(c.function);
  const uint64_t before=local.total_yields;
  if(probe){r.ws=clock_ns(CLOCK_MONOTONIC);r.cs=clock_ns(CLOCK_THREAD_CPUTIME_ID);}
  local.barrier=true;
  if(!blocking)r.result=original_barrier(task,env);
  else if(task<0||task>=env->num_task||env->num_task>4){errors.fetch_add(1);abort_group(c);r.result=-1;}
  else {
    auto& mine=c.generation[task];const uint32_t generation=mine.fetch_add(1,std::memory_order_release);
    wake(mine);
    for(int i=0;i<env->num_task;++i)if(i!=task) {
      auto& other=c.generation[i];
      while(!c.aborted.load(std::memory_order_acquire)) {
        const uint32_t value=other.load(std::memory_order_acquire);
        if(value>generation)break;
        wait(other,value);
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if(c.aborted.load())r.result=-1;
  }
  local.barrier=false;
  if(probe){r.ce=clock_ns(CLOCK_THREAD_CPUTIME_ID);r.we=clock_ns(CLOCK_MONOTONIC);}
  r.yields=local.total_yields-before;
  ++local.barrier_calls;local.barrier_wall+=r.we-r.ws;local.barrier_cpu+=r.ce-r.cs;
  add_record(r);return r.result;
}

extern "C" void WaitYield() asm("_ZN3tvm7runtime9threading5YieldEv");
extern "C" void WaitYield() {
  pthread_once(&once,initialize);++local.total_yields;
  auto* c=local.ctx;
  if(c&&!local.callback&&c->caller_done) {
    ++c->caller_completion_yields;
    if(blocking&&!c->aborted.load(std::memory_order_acquire)) {
      const auto done=c->finished.load(std::memory_order_acquire);
      if(done<c->tasks.load(std::memory_order_acquire)){wait(c->finished,done);return;}
    }
  } else if(!local.barrier)outside_yields.fetch_add(1,std::memory_order_relaxed);
  original_yield();
}
