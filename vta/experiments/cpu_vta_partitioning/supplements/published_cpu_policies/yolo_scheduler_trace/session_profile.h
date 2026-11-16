// Optional task-scoped scheduler recording. No changes to affinity or waiting.
#ifndef VTA_YOLO_SCHEDULER_SESSION_H_
#define VTA_YOLO_SCHEDULER_SESSION_H_
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace vta_deploy {
class SessionProfile {
 using U=uint64_t;
 struct Count {U value=0,enabled=0,running=0;};
 struct Task {
   long tid=0;int event=-1,counter=-1;void* mapped=MAP_FAILED;size_t map_bytes=0,page=0;
   std::vector<int> allowed;U enable_start=0,enable_end=0,disable_start=0,disable_end=0;
   U head=0,tail=0,data_size=0;bool overflow=false,alive_at_stop=false;
   Count before,after;std::vector<unsigned char> bytes;
 };
 std::string path_,error_;bool started_=false,stopped_=false;
 std::vector<long> targets_;std::vector<Task> tasks_;
 U completion_start_=0,completion_end_=0,setup_start_=0,setup_end_=0,stop_start_=0,stop_end_=0;
 static U Clock(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))throw std::runtime_error("scheduler trace clock failed");return U(t.tv_sec)*1000000000ULL+t.tv_nsec;}
 static Count Read(int fd){Count c;if(::read(fd,&c,sizeof(c))!=sizeof(c))throw std::runtime_error("task-clock read failed");return c;}
 void Close(){for(auto& t:tasks_){if(t.mapped!=MAP_FAILED){munmap(t.mapped,t.map_bytes);t.mapped=MAP_FAILED;}if(t.event>=0){close(t.event);t.event=-1;}if(t.counter>=0){close(t.counter);t.counter=-1;}}}
 static void CountJSON(std::ostream& o,const Count& c){o<<"{\"value\":"<<c.value<<",\"enabled\":"<<c.enabled<<",\"running\":"<<c.running<<"}";}
 void Write(){
   std::ofstream o(path_);if(!o)throw std::runtime_error("scheduler profile output failed");
   o<<"{\"status\":\""<<(error_.empty()?"MEASURED":"UNAVAILABLE")<<"\",\"error\":\""<<error_<<"\",\"pid\":"<<getpid()
    <<",\"clock\":\"CLOCK_MONOTONIC\",\"byte_order\":\"little\",\"sample_id_layout\":\"TID,TIME,CPU\",\"periodic_sampling\":false,\"data_pages\":32,"
    <<"\"completion_start_ns\":"<<completion_start_<<",\"completion_end_ns\":"<<completion_end_
    <<",\"setup_start_ns\":"<<setup_start_<<",\"setup_end_ns\":"<<setup_end_<<",\"stop_start_ns\":"<<stop_start_<<",\"stop_end_ns\":"<<stop_end_<<",\"streams\":[";
   const char hex[]="0123456789abcdef";
   for(size_t i=0;i<tasks_.size();++i){if(i)o<<",";const auto& t=tasks_[i];
    o<<"{\"tid\":"<<t.tid<<",\"pid\":"<<getpid()<<",\"allowed\":[";
    for(size_t j=0;j<t.allowed.size();++j){if(j)o<<",";o<<t.allowed[j];}
    o<<"],\"enable_start_ns\":"<<t.enable_start<<",\"enable_end_ns\":"<<t.enable_end
     <<",\"disable_start_ns\":"<<t.disable_start<<",\"disable_end_ns\":"<<t.disable_end
     <<",\"head\":"<<t.head<<",\"tail\":"<<t.tail<<",\"data_size\":"<<t.data_size<<",\"overflow\":"<<(t.overflow?"true":"false")
     <<",\"alive_at_stop\":"<<(t.alive_at_stop?"true":"false")<<",\"counter_before\":";
    CountJSON(o,t.before);o<<",\"counter_after\":";CountJSON(o,t.after);o<<",\"records_hex\":\"";
    for(unsigned char b:t.bytes)o<<hex[b>>4]<<hex[b&15];o<<"\"}";
   }o<<"]}\n";if(!o)throw std::runtime_error("scheduler trace write failed");
 }
 public:
 explicit SessionProfile(std::string p):path_(std::move(p)){}
 ~SessionProfile(){Close();}
 bool enabled()const{return !path_.empty();}
 void StartAt(double ms,const std::vector<long>& tids)noexcept{
   if(enabled()){completion_start_=static_cast<U>(ms*1000000.0+0.5);targets_=tids;}Start();
 }
 void StopAt(double ms)noexcept{if(enabled())completion_end_=static_cast<U>(ms*1000000.0+0.5);Stop();}
 void Start()noexcept{
  if(!enabled()||started_)return;started_=true;
  try{
   setup_start_=Clock();std::sort(targets_.begin(),targets_.end());
   if(targets_.size()!=12||targets_[0]<=0||std::adjacent_find(targets_.begin(),targets_.end())!=targets_.end())throw std::runtime_error("scheduler target roles missing or duplicate");
   tasks_.reserve(targets_.size());
   for(long tid:targets_){tasks_.emplace_back();auto& t=tasks_.back();t.tid=tid;
    cpu_set_t mask;CPU_ZERO(&mask);if(sched_getaffinity(tid,sizeof(mask),&mask))throw std::runtime_error("scheduler target affinity unavailable");
    for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&mask))t.allowed.push_back(c);
    perf_event_attr a{};a.size=PERF_ATTR_SIZE_VER5;a.type=PERF_TYPE_SOFTWARE;a.config=PERF_COUNT_SW_DUMMY;a.disabled=1;a.exclude_hv=1;
    a.context_switch=1;a.sample_id_all=1;a.sample_type=PERF_SAMPLE_TID|PERF_SAMPLE_TIME|PERF_SAMPLE_CPU;a.use_clockid=1;a.clockid=CLOCK_MONOTONIC;
    t.event=syscall(SYS_perf_event_open,&a,tid,-1,-1,PERF_FLAG_FD_CLOEXEC);if(t.event<0)throw std::runtime_error("task-scoped switch open failed");
    t.page=sysconf(_SC_PAGESIZE);t.map_bytes=t.page*33;t.mapped=mmap(nullptr,t.map_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,t.event,0);
    if(t.mapped==MAP_FAILED)throw std::runtime_error("task-scoped switch mmap failed");
    perf_event_attr c{};c.size=PERF_ATTR_SIZE_VER0;c.type=PERF_TYPE_SOFTWARE;c.config=PERF_COUNT_SW_TASK_CLOCK;c.disabled=1;c.exclude_hv=1;
    c.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    t.counter=syscall(SYS_perf_event_open,&c,tid,-1,-1,PERF_FLAG_FD_CLOEXEC);if(t.counter<0)throw std::runtime_error("task-clock open failed");
   }
   for(auto& t:tasks_){t.enable_start=Clock();if(ioctl(t.counter,PERF_EVENT_IOC_ENABLE,0))throw std::runtime_error("task-clock enable failed");
    t.before=Read(t.counter);if(ioctl(t.event,PERF_EVENT_IOC_ENABLE,0))throw std::runtime_error("switch enable failed");t.enable_end=Clock();}
   setup_end_=Clock();
  }catch(const std::exception& e){error_=e.what();Close();}catch(...){error_="unknown scheduler setup failure";Close();}
 }
 void Stop()noexcept{
  if(!enabled()||stopped_)return;stopped_=true;
  try{
   stop_start_=Clock();if(!started_)throw std::runtime_error("scheduler trace not started");
   if(error_.empty()){
    for(auto& t:tasks_){t.disable_start=Clock();
     if(ioctl(t.event,PERF_EVENT_IOC_DISABLE,0))throw std::runtime_error("switch disable failed");
     t.after=Read(t.counter);if(ioctl(t.counter,PERF_EVENT_IOC_DISABLE,0))throw std::runtime_error("task-clock disable failed");t.disable_end=Clock();
     t.alive_at_stop=access(("/proc/self/task/"+std::to_string(t.tid)).c_str(),F_OK)==0;
    }
    stop_end_=Clock();
    for(auto& t:tasks_){auto* m=static_cast<perf_event_mmap_page*>(t.mapped);t.head=__atomic_load_n(&m->data_head,__ATOMIC_ACQUIRE);t.tail=m->data_tail;
     U offset=m->data_offset?m->data_offset:t.page;t.data_size=m->data_size?m->data_size:t.page*32;
     if(offset<t.page||!t.data_size||offset+t.data_size>t.map_bytes||(t.data_size&(t.data_size-1))||t.head<t.tail)throw std::runtime_error("invalid scheduler ring layout");
     t.overflow=t.head-t.tail>t.data_size;
     if(!t.overflow){auto* data=static_cast<unsigned char*>(t.mapped)+offset;t.bytes.resize(t.head-t.tail);for(U i=0;i<t.head-t.tail;++i)t.bytes[i]=data[(t.tail+i)%t.data_size];}
     __atomic_store_n(&m->data_tail,t.head,__ATOMIC_RELEASE);
    }
   }
   Close();Write();
  }catch(const std::exception& e){error_=e.what();Close();try{Write();}catch(...){}}
  catch(...){error_="unknown scheduler stop failure";Close();try{Write();}catch(...){}}
 }
};
} // namespace vta_deploy
#endif
