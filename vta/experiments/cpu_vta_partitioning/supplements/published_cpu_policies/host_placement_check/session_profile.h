// Optional, counting-only observation at the runner's existing completion window.
// No sampling threads, affinity changes, sleeps, or waiting-policy changes.
#ifndef VTA_HOST_PLACEMENT_SESSION_PROFILE_H_
#define VTA_HOST_PLACEMENT_SESSION_PROFILE_H_
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <linux/perf_event.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace vta_deploy {
class SessionProfile {
  using U=uint64_t;
  struct Reading {U value=0,enabled=0,running=0;};
  struct Snapshot {U start=0,end=0;std::array<Reading,5> counts{};};
  struct Task {pid_t tid=0;std::array<int,5> fds{{-1,-1,-1,-1,-1}};Snapshot before,after;std::vector<int> allowed;};
  std::string path_,error_;bool started_=false,stopped_=false;
  std::vector<Task> tasks_;U setup_start_=0,start_end_=0,stop_start_=0,stop_end_=0,completion_start_=0,completion_end_=0;
  static U Clock(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))throw std::runtime_error("monotonic clock");return U(t.tv_sec)*1000000000ULL+t.tv_nsec;}
  static std::vector<pid_t> Tids(){
    DIR* d=opendir("/proc/self/task");if(!d)throw std::runtime_error("task list unavailable");std::vector<pid_t> ids;
    while(dirent* e=readdir(d)){char* p=nullptr;long v=strtol(e->d_name,&p,10);if(p!=e->d_name&&!*p&&v>0)ids.push_back(v);}
    closedir(d);std::sort(ids.begin(),ids.end());return ids;
  }
  static Snapshot Read(const Task& t){
    Snapshot s;s.start=Clock();
    for(int i=0;i<5;++i)if(::read(t.fds[i],&s.counts[i],sizeof(Reading))!=sizeof(Reading))throw std::runtime_error("task-clock read failed");
    s.end=Clock();return s;
  }
  void Close(){for(auto& t:tasks_)for(int& fd:t.fds)if(fd>=0){close(fd);fd=-1;}}
  static void WriteSnapshot(std::ostream& o,const Snapshot& s){
    o<<"{\"read_start_ns\":"<<s.start<<",\"read_end_ns\":"<<s.end<<",\"counts\":[";
    for(int i=0;i<5;++i){if(i)o<<",";const auto& r=s.counts[i];o<<"{\"value\":"<<r.value<<",\"enabled\":"<<r.enabled<<",\"running\":"<<r.running<<"}";}
    o<<"]}";
  }
  void Write(){
    std::ofstream o(path_);if(!o)throw std::runtime_error("profile output unavailable");
    o<<"{\"status\":\""<<(error_.empty()?"MEASURED":"UNAVAILABLE")<<"\",\"error\":\""<<error_<<"\",\"sampling\":false,\"per_TID_per_CPU\":true,\"scaled\":false,\"cpus\":[0,1,2,3],"
      "\"setup_start_ns\":"<<setup_start_<<",\"start_reads_end_ns\":"<<start_end_<<",\"stop_reads_start_ns\":"<<stop_start_<<",\"stop_reads_end_ns\":"<<stop_end_
      <<",\"completion_start_ns\":"<<completion_start_<<",\"completion_end_ns\":"<<completion_end_<<",\"tasks\":[";
    for(size_t k=0;k<tasks_.size();++k){if(k)o<<",";const auto& t=tasks_[k];o<<"{\"tid\":"<<t.tid<<",\"allowed\":[";
      for(size_t i=0;i<t.allowed.size();++i){if(i)o<<",";o<<t.allowed[i];}
      o<<"],\"before\":";WriteSnapshot(o,t.before);o<<",\"after\":";WriteSnapshot(o,t.after);o<<"}";}
    o<<"]}\n";if(!o)throw std::runtime_error("profile output failed");
  }
 public:
  explicit SessionProfile(std::string path):path_(std::move(path)){}
  ~SessionProfile(){Close();}
  bool enabled()const{return !path_.empty();}
  void StartAt(double ms)noexcept{if(enabled())completion_start_=static_cast<U>(ms*1000000.0+0.5);Start();}
  void StopAt(double ms)noexcept{if(enabled())completion_end_=static_cast<U>(ms*1000000.0+0.5);Stop();}
  void Start()noexcept{
    if(!enabled()||started_)return;started_=true;
    try{
      setup_start_=Clock();const auto tids=Tids();if(tids.empty())throw std::runtime_error("empty task list");
      tasks_.reserve(tids.size());
      for(pid_t tid:tids){tasks_.emplace_back();auto& t=tasks_.back();t.tid=tid;
        cpu_set_t set;CPU_ZERO(&set);if(sched_getaffinity(tid,sizeof(set),&set))throw std::runtime_error("task affinity read failed");
        for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&set))t.allowed.push_back(c);
        for(int i=0;i<5;++i){perf_event_attr a{};a.size=PERF_ATTR_SIZE_VER0;a.type=PERF_TYPE_SOFTWARE;a.config=PERF_COUNT_SW_TASK_CLOCK;
          a.disabled=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
          t.fds[i]=syscall(SYS_perf_event_open,&a,tid,i-1,-1,PERF_FLAG_FD_CLOEXEC);
          if(t.fds[i]<0)throw std::runtime_error("task-clock open failed");}
      }
      for(auto& t:tasks_){for(int fd:t.fds)if(ioctl(fd,PERF_EVENT_IOC_ENABLE,0))throw std::runtime_error("task-clock enable failed");t.before=Read(t);}
      start_end_=Clock();
    }catch(const std::exception& e){error_=e.what();Close();}catch(...){error_="unknown start error";Close();}
  }
  void Stop()noexcept{
    if(!enabled()||stopped_)return;stopped_=true;
    try{
      stop_start_=Clock();if(!started_)throw std::runtime_error("counter window never started");
      if(error_.empty()){for(auto& t:tasks_)t.after=Read(t);stop_end_=Clock();}
      Close();Write();
    }catch(const std::exception& e){error_=e.what();Close();try{Write();}catch(...){}}
    catch(...){error_="unknown stop error";Close();try{Write();}catch(...){}}
  }
};
}  // namespace vta_deploy
#endif
