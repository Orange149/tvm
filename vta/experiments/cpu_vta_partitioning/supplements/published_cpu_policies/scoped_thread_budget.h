#ifndef CPU_VTA_SCOPED_THREAD_BUDGET_H_
#define CPU_VTA_SCOPED_THREAD_BUDGET_H_
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/registry.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace scoped_thread_budget {
using NS=unsigned long long;
inline NS Clock(clockid_t id) {
  timespec t{};
  if(clock_gettime(id,&t))throw std::runtime_error("scoped CPU clock unavailable");
  return NS(t.tv_sec)*1000000000ULL+NS(t.tv_nsec);
}
inline bool Enabled() {
  const char* p=std::getenv("CPU_RUN_BUDGET_JSON");return p&&*p;
}
struct Slot {
  long tid=0;clockid_t clock=0;int error=0;std::vector<int> cpus;
};
struct Pool {std::string stage;std::vector<Slot> slots;NS registration_begin=0,registration_end=0;};
inline Pool& Current() {static thread_local Pool p;return p;}
inline int Register(int slot,TVMParallelGroupEnv* group,void* opaque) {
  auto* p=static_cast<Pool*>(opaque);
  if(slot<0||slot>=int(p->slots.size())||group->num_task!=int(p->slots.size()))return -1;
  auto& s=p->slots[slot];s.tid=long(syscall(SYS_gettid));
  s.error=pthread_getcpuclockid(pthread_self(),&s.clock);
  cpu_set_t mask;CPU_ZERO(&mask);
  if(!s.error)s.error=pthread_getaffinity_np(pthread_self(),sizeof(mask),&mask);
  if(!s.error)for(int core=0;core<CPU_SETSIZE;++core)if(CPU_ISSET(core,&mask))s.cpus.push_back(core);
  return s.error?-1:0;
}
struct Read {NS begin=0,end=0,cpu=0;};
inline Read ReadClock(const Slot& slot) {
  Read r;r.begin=Clock(CLOCK_MONOTONIC);r.cpu=Clock(slot.clock);r.end=Clock(CLOCK_MONOTONIC);return r;
}
struct Record {
  std::string stage;NS call=0,graph_begin=0,graph_end=0,process_begin=0,process_end=0;
  std::vector<Read> before,after;
};
class Recorder {
 public:
  Pool pool;std::vector<Record> records;
  void Flush() {
    if(!Enabled()||written_)return;
    const std::string path=std::getenv("CPU_RUN_BUDGET_JSON"),temporary=path+".tmp";
    FILE* f=std::fopen(temporary.c_str(),"w");if(!f)throw std::runtime_error("cannot open thread budget output");
    std::fprintf(f,"{\"schema\":\"scoped_thread_budget_v1\",\"registration_launches\":1,"
        "\"registration_callbacks\":%zu,\"registration_outside_stage_phases\":true,"
        "\"registration_begin_ns\":%llu,\"registration_end_ns\":%llu,"
        "\"stage\":\"%s\",\"slots\":[",pool.slots.size(),pool.registration_begin,pool.registration_end,pool.stage.c_str());
    for(size_t i=0;i<pool.slots.size();++i) {
      const auto& s=pool.slots[i];std::fprintf(f,"%s{\"slot\":%zu,\"tid\":%ld,\"clock_id\":%d,\"cpus\":[",i?",":"",i,s.tid,int(s.clock));
      for(size_t j=0;j<s.cpus.size();++j)std::fprintf(f,"%s%d",j?",":"",s.cpus[j]);
      std::fprintf(f,"]}");
    }
    std::fprintf(f,"],\"records\":[");
    for(size_t i=0;i<records.size();++i) {
      const auto& r=records[i];std::fprintf(f,"%s{\"stage\":\"%s\",\"call\":%llu,"
          "\"graph_begin_ns\":%llu,\"graph_end_ns\":%llu,\"process_CPU_begin_ns\":%llu,"
          "\"process_CPU_end_ns\":%llu,\"threads\":[",i?",":"",r.stage.c_str(),r.call,
          r.graph_begin,r.graph_end,r.process_begin,r.process_end);
      for(size_t j=0;j<r.before.size();++j) {
        const auto& a=r.before[j];const auto& b=r.after[j];
        std::fprintf(f,"%s{\"slot\":%zu,\"tid\":%ld,\"before_read_begin_ns\":%llu,"
            "\"before_read_end_ns\":%llu,\"CPU_begin_ns\":%llu,\"after_read_begin_ns\":%llu,"
            "\"after_read_end_ns\":%llu,\"CPU_end_ns\":%llu}",j?",":"",j,pool.slots[j].tid,
            a.begin,a.end,a.cpu,b.begin,b.end,b.cpu);
      }
      std::fprintf(f,"]}");
    }
    std::fprintf(f,"]}\n");
    if(std::fclose(f)||std::rename(temporary.c_str(),path.c_str()))throw std::runtime_error("cannot commit thread budget");
    written_=true;
  }
  ~Recorder(){try{Flush();}catch(const std::exception& e){std::fprintf(stderr,"thread budget flush failed: %s\n",e.what());}}
 private:bool written_=false;
};
inline Recorder& State(){static Recorder r;return r;}
inline void Ensure(const std::string& stage) {
  if(!Enabled())return;
  auto& p=Current();
  const auto* count=tvm::runtime::Registry::Get("runtime.NumThreads");
  const int n=count?(*count)().operator int():0;
  if(!p.slots.empty()) {
    if(p.stage!=stage||int(p.slots.size())!=n)throw std::runtime_error("scoped budget supports one unchanged CPU pool per process");
    return;
  }
  if(n<1||n>4||stage.find_first_of("\"\\\n\r")!=std::string::npos)throw std::runtime_error("invalid scoped budget pool");
  p.stage=stage;p.slots.resize(n);
  p.registration_begin=Clock(CLOCK_MONOTONIC);
  if(TVMBackendParallelLaunch(Register,&p,n))throw std::runtime_error("thread CPU clock registration failed");
  p.registration_end=Clock(CLOCK_MONOTONIC);
  std::set<long> tids;
  for(const auto& s:p.slots)if(s.error||s.cpus.empty()||!tids.insert(s.tid).second)throw std::runtime_error("thread clock/role registration incomplete");
  if(p.slots[0].tid!=long(syscall(SYS_gettid)))throw std::runtime_error("logical task zero is not caller");
  State().pool=p;State().records.reserve(64);
}
class Scope {
 public:
  Scope(const std::string& stage,NS call) {
    enabled_=Enabled();if(!enabled_)return;
    const auto& p=Current();if(p.stage!=stage||p.slots.empty())throw std::runtime_error("unregistered thread clocks");
    r_.stage=stage;r_.call=call;r_.before.reserve(p.slots.size());r_.after.reserve(p.slots.size());
    for(const auto& s:p.slots)r_.before.push_back(ReadClock(s));
    r_.graph_begin=Clock(CLOCK_MONOTONIC);r_.process_begin=Clock(CLOCK_PROCESS_CPUTIME_ID);
  }
  void End() {
    if(!enabled_)return;
    r_.process_end=Clock(CLOCK_PROCESS_CPUTIME_ID);r_.graph_end=Clock(CLOCK_MONOTONIC);
    for(const auto& s:Current().slots)r_.after.push_back(ReadClock(s));
    if(State().records.size()>=64)throw std::runtime_error("thread budget capacity exceeded");
    State().records.push_back(std::move(r_));enabled_=false;
  }
 private:bool enabled_=false;Record r_;
};
}  // namespace scoped_thread_budget
#endif
