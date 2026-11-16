// Licensed under the Apache License, Version 2.0.
// Opt-in observation only: no scheduling, worker, queue or waiting changes.
#ifndef VTA_WORKLOAD_PHASE_PROBE_H_
#define VTA_WORKLOAD_PHASE_PROBE_H_
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace vta_workload_phase {
struct Event {
  std::string stage, operation;
  unsigned long long call=0, tid=0, wall_start=0, wall_end=0, cpu_start=0, cpu_end=0;
  int core_start=-1, core_end=-1;
};
inline unsigned long long Clock(clockid_t clock) {
  timespec value{};
  if (clock_gettime(clock,&value)) throw std::runtime_error("phase clock failed");
  return static_cast<unsigned long long>(value.tv_sec)*1000000000ULL+value.tv_nsec;
}
class Recorder {
 public:
  void Configure(const std::string& value) {
    if (configured_) throw std::runtime_error("phase recorder configured twice");
    configured_=true; path_=value;
    if (!value.empty()) events_.reserve(65536);
  }
  bool Enabled() const {return !path_.empty();}
  unsigned long long Next(const std::string& stage) {
    if (!Enabled()) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return calls_[stage]++;
  }
  void Add(Event value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size()>=65536) throw std::runtime_error("phase event capacity exceeded");
    events_.push_back(std::move(value));
  }
  void Flush() {
    if (!Enabled()||flushed_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string temporary=path_+".tmp";
    FILE* file=fopen(temporary.c_str(),"w");
    if (!file) throw std::runtime_error("cannot create phase trace");
    for (const auto& e:events_) {
      // Stage/operation strings originate from the strictly constrained CLI.
      fprintf(file,"{\"stage\":\"%s\",\"call\":%llu,\"operation\":\"%s\",\"tid\":%llu,"
          "\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu,"
          "\"core_start\":%d,\"core_end\":%d}\n",e.stage.c_str(),e.call,e.operation.c_str(),e.tid,
          e.wall_start,e.wall_end,e.cpu_start,e.cpu_end,e.core_start,e.core_end);
    }
    if (fclose(file)||rename(temporary.c_str(),path_.c_str()))
      throw std::runtime_error("cannot commit phase trace");
    flushed_=true;
  }
  ~Recorder() {
    try {Flush();} catch (const std::exception& e) {
      fprintf(stderr,"phase trace flush failed: %s\n",e.what());
    }
  }
 private:
  bool configured_=false,flushed_=false;
  std::string path_; std::mutex mutex_;
  std::map<std::string,unsigned long long> calls_;
  std::vector<Event> events_;
};
inline Recorder& State() {static Recorder value;return value;}
inline void Configure(const std::string& path) {State().Configure(path);}
inline unsigned long long Next(const std::string& stage) {return State().Next(stage);}
class Span {
 public:
  Span(const std::string& stage,unsigned long long call,const char* operation) {
    active_=State().Enabled(); if (!active_)return;
    if (stage.find_first_of("\"\\\n\r")!=std::string::npos)
      throw std::runtime_error("unsafe phase stage identity");
    event_.stage=stage;event_.operation=operation;event_.call=call;
    event_.tid=static_cast<unsigned long long>(syscall(SYS_gettid));
    event_.core_start=sched_getcpu();
    event_.wall_start=Clock(CLOCK_MONOTONIC);
    event_.cpu_start=Clock(CLOCK_THREAD_CPUTIME_ID);
  }
  void End() {
    if (!active_)return;
    event_.cpu_end=Clock(CLOCK_THREAD_CPUTIME_ID);
    event_.wall_end=Clock(CLOCK_MONOTONIC);event_.core_end=sched_getcpu();
    active_=false;State().Add(std::move(event_));
  }
 private:
  bool active_=false;Event event_;
};
}  // namespace vta_workload_phase
#endif
