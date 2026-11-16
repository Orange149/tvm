#ifndef CPU_COMMON_FRAME_FLOW_H_
#define CPU_COMMON_FRAME_FLOW_H_
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace cpu_frame_stream {
// Queue is the frozen runner's BoundedQueue<int>. No per-frame barrier.
template<template<typename> class Queue>
class Flow {
 public:
  Flow(size_t actors,size_t depth,int frames):frames_(frames) {
    for(size_t i=0;i<actors;++i) queues_.emplace_back(new Queue<int>(depth));
  }
  template<typename Clock,typename Published>
  void Produce(Clock clock,Published published) {
    for(int frame=0;frame<frames_&&!cancel_;++frame) {
      for(size_t actor=0;actor<queues_.size();++actor) {
        const auto before=clock();
        if(!queues_[actor]->Push(frame)) {Cancel();return;}
        published(actor,frame,before,clock());
      }
    }
    for(auto& q:queues_) q->Close();
  }
  template<typename Clock,typename Run>
  void Consume(size_t actor,Clock clock,Run run) {
    int frame;
    while(!cancel_) {
      const auto before=clock();
      if(!queues_.at(actor)->Pop(&frame)||cancel_) break;
      run(frame,before,clock());
    }
  }
  void Cancel() {
    cancel_=true;
    for(auto& q:queues_) q->Close();
  }
 private:
  std::vector<std::unique_ptr<Queue<int>>> queues_;
  int frames_;
  std::atomic<bool> cancel_{false};
};
}
#endif
