// Counting-only prerequisite: no DNN, TVM, FPGA, sampling, or system changes.
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using U=unsigned long long;
U clock_ns(clockid_t c){timespec t{};if(clock_gettime(c,&t))throw std::runtime_error("clock_gettime");return U(t.tv_sec)*1000000000ULL+t.tv_nsec;}
void pin(int core){cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);if(sched_setaffinity(0,sizeof(s),&s))throw std::runtime_error("affinity");}
struct Reading {U value=0,enabled=0,running=0;};
struct Snapshot {std::vector<Reading> counts;U wall_start,wall_end,cpu_start,cpu_end;};
class Counters {
 std::vector<int> fds;
 public:
 explicit Counters(const std::vector<int>& cpus){
  try {
   auto open_counter=[&](int cpu){
    perf_event_attr a{};a.size=PERF_ATTR_SIZE_VER0;a.type=PERF_TYPE_SOFTWARE;a.config=PERF_COUNT_SW_TASK_CLOCK;
    a.disabled=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    const int fd=syscall(SYS_perf_event_open,&a,0,cpu,-1,PERF_FLAG_FD_CLOEXEC);
    if(fd<0)throw std::runtime_error("perf_event_open");fds.push_back(fd);
   };
   open_counter(-1);for(int cpu:cpus)open_counter(cpu);
   for(int fd:fds)if(ioctl(fd,PERF_EVENT_IOC_ENABLE,0))throw std::runtime_error("perf enable");
  }catch(...){const int saved=errno;for(int fd:fds)close(fd);fds.clear();errno=saved;throw;}
 }
 ~Counters(){for(int fd:fds)close(fd);}
 Snapshot read(){
  Snapshot s;s.wall_start=clock_ns(CLOCK_MONOTONIC);s.cpu_start=clock_ns(CLOCK_THREAD_CPUTIME_ID);
  for(int fd:fds){Reading r;if(::read(fd,&r,sizeof(r))!=sizeof(r))throw std::runtime_error("perf read");s.counts.push_back(r);}
  s.cpu_end=clock_ns(CLOCK_THREAD_CPUTIME_ID);s.wall_end=clock_ns(CLOCK_MONOTONIC);return s;
 }
};
struct Row {const char* name;int core;Snapshot before,after;U CPU,wall;};
void readings(const std::vector<Reading>& r){
 printf("[");for(size_t i=0;i<r.size();++i){if(i)printf(",");printf("{\"value\":%llu,\"enabled\":%llu,\"running\":%llu}",r[i].value,r[i].enabled,r[i].running);}printf("]");
}
int main(int argc,char** argv){
 if(argc==2&&!strcmp(argv[1],"--describe")){
  puts("{\"event\":\"PERF_COUNT_SW_TASK_CLOCK\",\"mode\":\"counting\",\"per_TID_per_CPU\":true,\"sampling\":false,\"DNN_inferences\":0,\"changes_system_configuration\":false}");return 0;
 }
 try {
  std::vector<int> cpus{0,1,2,3};
  if(argc==3&&!strcmp(argv[1],"--cpus")){
   cpus.clear();std::string text=argv[2];size_t start=0;
   while(start<text.size()) {size_t end=text.find(',',start);if(end==std::string::npos)end=text.size();std::string token=text.substr(start,end-start);
    size_t used=0;int value=std::stoi(token,&used);if(used!=token.size()||value<0||value>=CPU_SETSIZE)throw std::runtime_error("invalid CPU");
    for(int old:cpus)if(old==value)throw std::runtime_error("duplicate CPU");cpus.push_back(value);start=end+1;}
   if(cpus.size()<2||cpus.size()>4)throw std::runtime_error("require 2-4 CPU IDs");
  }else if(argc!=1)throw std::runtime_error("usage: --describe or --cpus A,B[,C,D]");
  pin(cpus[0]);Counters counters(cpus);std::vector<Row> rows;
  for(int phase=0;phase<3;++phase){
   const int core=cpus[phase==0?0:1];pin(core);Row r{};r.name=phase==0?"busy_first":phase==1?"busy_migrated":"sleep";r.core=core;
   r.before=counters.read();const U wall=clock_ns(CLOCK_MONOTONIC),cpu=clock_ns(CLOCK_THREAD_CPUTIME_ID);
   if(phase==2){timespec delay{0,100000000};while(nanosleep(&delay,&delay)&&errno==EINTR){}}
   else {volatile U value=1;while(clock_ns(CLOCK_THREAD_CPUTIME_ID)-cpu<150000000ULL)for(int i=0;i<1000;++i)value=value*6364136223846793005ULL+1;}
   r.CPU=clock_ns(CLOCK_THREAD_CPUTIME_ID)-cpu;r.wall=clock_ns(CLOCK_MONOTONIC)-wall;r.after=counters.read();rows.push_back(r);
  }
  printf("{\"status\":\"MEASURED\",\"TID\":%ld,\"cpus\":[",syscall(SYS_gettid));
  for(size_t i=0;i<cpus.size();++i){if(i)printf(",");printf("%d",cpus[i]);}
  printf("],\"DNN_inferences\":0,\"sampling\":false,\"rows\":[");
  for(size_t i=0;i<rows.size();++i){const auto& r=rows[i];if(i)printf(",");
   printf("{\"case\":\"%s\",\"requested_core\":%d,\"CPU_ns\":%llu,\"wall_ns\":%llu,\"read_envelope_ns\":%llu,\"before\":",r.name,r.core,r.CPU,r.wall,(r.before.wall_end-r.before.wall_start)+(r.after.wall_end-r.after.wall_start));
   readings(r.before.counts);printf(",\"after\":");readings(r.after.counts);printf("}");
  }printf("]}\n");return 0;
 }catch(const std::exception& e){const int err=errno;printf("{\"status\":\"UNAVAILABLE\",\"operation\":\"%s\",\"errno\":%d,\"DNN_inferences\":0,\"sampling\":false}\n",e.what(),err);return 2;}
}
