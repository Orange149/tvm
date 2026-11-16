// No DNN, TVM, FPGA, periodic sampling, kernel/sysctl changes, or reboot.
// The only helper threads are the finite controlled workloads being observed.
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using U=uint64_t;
U clock_ns(clockid_t id){timespec t{};if(clock_gettime(id,&t))throw std::runtime_error("clock_gettime failed");return U(t.tv_sec)*1000000000ULL+t.tv_nsec;}
void pin(int cpu){cpu_set_t m;CPU_ZERO(&m);CPU_SET(cpu,&m);if(sched_setaffinity(0,sizeof(m),&m))throw std::runtime_error("self-test affinity failed");}
std::string error_text(const char* action){return std::string(action)+": "+strerror(errno);}

class Ring {
 int fd_=-1;void* mapping_=MAP_FAILED;size_t map_bytes_=0,page_=0;
 public:
 U enable_start=0,enable_end=0,disable_start=0,disable_end=0,head=0,tail=0,data_size=0;bool overflow=false;
 std::vector<unsigned char> bytes;
 Ring(){
  perf_event_attr a{};a.size=PERF_ATTR_SIZE_VER5;a.type=PERF_TYPE_SOFTWARE;a.config=PERF_COUNT_SW_DUMMY;
  a.disabled=1;a.exclude_hv=1;a.context_switch=1;a.sample_id_all=1;
  a.sample_type=PERF_SAMPLE_TID|PERF_SAMPLE_TIME|PERF_SAMPLE_CPU;
  a.use_clockid=1;a.clockid=CLOCK_MONOTONIC; // No period or frequency sampling.
  fd_=syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC);
  if(fd_<0)throw std::runtime_error(error_text("perf_event_open context_switch"));
  page_=sysconf(_SC_PAGESIZE);map_bytes_=page_*17;
  mapping_=mmap(nullptr,map_bytes_,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
  if(mapping_==MAP_FAILED){const auto e=error_text("perf ring mmap");close(fd_);fd_=-1;throw std::runtime_error(e);}
 }
 ~Ring(){if(mapping_!=MAP_FAILED)munmap(mapping_,map_bytes_);if(fd_>=0)close(fd_);}
 void Start(){enable_start=clock_ns(CLOCK_MONOTONIC);if(ioctl(fd_,PERF_EVENT_IOC_ENABLE,0))throw std::runtime_error(error_text("perf enable"));enable_end=clock_ns(CLOCK_MONOTONIC);}
 void Stop(){
  disable_start=clock_ns(CLOCK_MONOTONIC);if(ioctl(fd_,PERF_EVENT_IOC_DISABLE,0))throw std::runtime_error(error_text("perf disable"));disable_end=clock_ns(CLOCK_MONOTONIC);
  auto* meta=static_cast<perf_event_mmap_page*>(mapping_);
  head=__atomic_load_n(&meta->data_head,__ATOMIC_ACQUIRE);tail=meta->data_tail;
  const U offset=meta->data_offset?meta->data_offset:page_;data_size=meta->data_size?meta->data_size:page_*16;
  if(offset<page_||offset+data_size>map_bytes_||!data_size||(data_size&(data_size-1)))throw std::runtime_error("invalid perf ring layout");
  if(head<tail)throw std::runtime_error("perf ring head moved backwards");
  overflow=head-tail>data_size;
  if(!overflow){auto* data=static_cast<unsigned char*>(mapping_)+offset;bytes.resize(head-tail);
   for(U i=0;i<head-tail;++i)bytes[i]=data[(tail+i)%data_size];}
  __atomic_store_n(&meta->data_tail,head,__ATOMIC_RELEASE);
 }
};

struct Segment {const char* kind;int requested_core,actual_start,actual_end;U start,end,cpu_start,cpu_end;};
struct Result {std::string name,error;int index=0,pid=0,tid=0,policy=0,nice=0;U enable_start=0,enable_end=0,disable_start=0,disable_end=0,head=0,tail=0,data_size=0;bool overflow=false;std::vector<Segment> segments;std::vector<unsigned char> bytes;};
struct Gate {std::mutex mutex;std::condition_variable cv;int ready=0;bool go=false,failed=false;};

void work(Result& r,const char* kind,int core,U amount){
 pin(core);Segment s{};s.kind=kind;s.requested_core=core;s.actual_start=sched_getcpu();
 s.start=clock_ns(CLOCK_MONOTONIC);s.cpu_start=clock_ns(CLOCK_THREAD_CPUTIME_ID);
 if(!strcmp(kind,"sleep")){timespec t{0,static_cast<long>(amount)};while(nanosleep(&t,&t)){if(errno!=EINTR)throw std::runtime_error("nanosleep failed");}}
 else {volatile U v=1;while(clock_ns(CLOCK_THREAD_CPUTIME_ID)-s.cpu_start<amount)for(int i=0;i<4096;++i)v=v*6364136223846793005ULL+1;}
 s.cpu_end=clock_ns(CLOCK_THREAD_CPUTIME_ID);s.end=clock_ns(CLOCK_MONOTONIC);s.actual_end=sched_getcpu();r.segments.push_back(s);
}

void run_case(const char* name,int count,std::vector<Result>& all){
 Gate gate;std::vector<Result> rows(count);std::vector<std::thread> workers;
 for(int i=0;i<count;++i)workers.emplace_back([&,i]{
  auto& r=rows[i];r.name=name;r.index=i;r.pid=getpid();r.tid=syscall(SYS_gettid);r.policy=sched_getscheduler(0);r.nice=getpriority(PRIO_PROCESS,0);
  try{
   pin(0);Ring ring;
   {std::unique_lock<std::mutex> lock(gate.mutex);++gate.ready;gate.cv.notify_all();gate.cv.wait(lock,[&]{return gate.go||gate.failed;});if(gate.failed)throw std::runtime_error("peer capability setup failed");}
   ring.Start();
   if(!strcmp(name,"sleep"))work(r,"sleep",0,150000000);
   else if(!strcmp(name,"migrate")){work(r,"busy",0,100000000);work(r,"busy",1,100000000);}
   else work(r,"busy",0,150000000);
   ring.Stop();r.enable_start=ring.enable_start;r.enable_end=ring.enable_end;r.disable_start=ring.disable_start;r.disable_end=ring.disable_end;
   r.head=ring.head;r.tail=ring.tail;r.data_size=ring.data_size;r.overflow=ring.overflow;r.bytes=std::move(ring.bytes);
  }catch(const std::exception& e){r.error=e.what();std::lock_guard<std::mutex> lock(gate.mutex);gate.failed=true;gate.cv.notify_all();}
 });
 {std::unique_lock<std::mutex> lock(gate.mutex);gate.cv.wait(lock,[&]{return gate.ready==count||gate.failed;});gate.go=true;gate.cv.notify_all();}
 for(auto& w:workers)w.join();for(auto& r:rows)all.push_back(std::move(r));
}

void quote(const std::string& s){putchar('"');for(unsigned char c:s){if(c=='"'||c=='\\')putchar('\\');if(c>=32)putchar(c);else printf("\\u%04x",c);}putchar('"');}
int main(int argc,char** argv){
 if(argc==2&&!strcmp(argv[1],"--describe")){
  puts("{\"interface\":\"PERF_RECORD_SWITCH\",\"clock\":\"CLOCK_MONOTONIC\",\"periodic_sampling\":false,\"sample_type\":[\"TID\",\"TIME\",\"CPU\"],\"DNN_inferences\":0,\"system_parameter_changes\":false,\"workload_threads\":5,\"cases\":[\"alone\",\"contended\",\"sleep\",\"migrate\"]}");return 0;
 }
 if(argc!=1)return 3;
 std::vector<Result> rows;
 for(const auto& c:std::vector<std::pair<const char*,int>>{{"alone",1},{"contended",2},{"sleep",1},{"migrate",1}}){
  run_case(c.first,c.second,rows);if(std::any_of(rows.begin(),rows.end(),[](const Result& r){return !r.error.empty();}))break;
 }
 const bool ok=rows.size()==5&&std::all_of(rows.begin(),rows.end(),[](const Result& r){return r.error.empty();});
 utsname kernel{};uname(&kernel);
 printf("{\"status\":\"%s\",\"DNN_inferences\":0,\"periodic_sampling\":false,\"byte_order\":\"little\",\"sample_id_layout\":\"TID,TIME,CPU\",\"kernel_release\":",ok?"MEASURED":"UNAVAILABLE");quote(kernel.release);
 printf(",\"streams\":[");
 for(size_t i=0;i<rows.size();++i){const auto& r=rows[i];if(i)putchar(',');printf("{\"case\":");quote(r.name);printf(",\"index\":%d,\"pid\":%d,\"tid\":%d,\"policy\":%d,\"nice\":%d,\"error\":",r.index,r.pid,r.tid,r.policy,r.nice);quote(r.error);
  printf(",\"enable_start_ns\":%llu,\"enable_end_ns\":%llu,\"disable_start_ns\":%llu,\"disable_end_ns\":%llu,\"head\":%llu,\"tail\":%llu,\"data_size\":%llu,\"overflow\":%s,\"segments\":[",(unsigned long long)r.enable_start,(unsigned long long)r.enable_end,(unsigned long long)r.disable_start,(unsigned long long)r.disable_end,(unsigned long long)r.head,(unsigned long long)r.tail,(unsigned long long)r.data_size,r.overflow?"true":"false");
  for(size_t j=0;j<r.segments.size();++j){const auto& s=r.segments[j];if(j)putchar(',');printf("{\"kind\":\"%s\",\"requested_core\":%d,\"actual_start_core\":%d,\"actual_end_core\":%d,\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu}",s.kind,s.requested_core,s.actual_start,s.actual_end,(unsigned long long)s.start,(unsigned long long)s.end,(unsigned long long)s.cpu_start,(unsigned long long)s.cpu_end);}
  printf("],\"records_hex\":\"");for(unsigned char b:r.bytes)printf("%02x",b);printf("\"}");
 }printf("]}\n");return ok?0:2;
}
