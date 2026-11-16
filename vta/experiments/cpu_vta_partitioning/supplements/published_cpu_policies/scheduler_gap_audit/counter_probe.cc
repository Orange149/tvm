#include "taskstats_client.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <sched.h>
#include <thread>
#include <time.h>

uint64_t ns(clockid_t c) {timespec t{};if(clock_gettime(c,&t))throw std::runtime_error("clock failed");return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
void pin(int c) {cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);if(sched_setaffinity(0,sizeof(s),&s))throw std::runtime_error("self-test affinity failed");}
int main() {
 try {
  pin(3);
  for(int mode=0;mode<3;++mode) {
    struct Result {uint32_t tid=0;uint64_t wall=0,cpu=0,delay=0,read_max=0;std::string error;};
    Result results[2];std::atomic<int> ready{0};std::atomic<bool> start{false};
    const int count=mode==1?2:1;std::vector<std::thread> threads;
    for(int i=0;i<count;++i)threads.emplace_back([&,i]{
      try {
        pin(0);scheduler_gap::Client client;auto& r=results[i];r.tid=syscall(SYS_gettid);
        for(int q=0;q<32;++q){const auto a=ns(CLOCK_MONOTONIC);client.read(r.tid);r.read_max=std::max(r.read_max,ns(CLOCK_MONOTONIC)-a);}
        ++ready;while(!start.load())std::this_thread::yield();
        const auto before=client.read(r.tid);const auto w=ns(CLOCK_MONOTONIC),c=ns(CLOCK_THREAD_CPUTIME_ID);
        if(mode==2)std::this_thread::sleep_for(std::chrono::seconds(1));
        else {volatile uint64_t value=1;while(ns(CLOCK_MONOTONIC)-w<1000000000ULL)for(int k=0;k<1000;++k)value=value*6364136223846793005ULL+1;}
        r.cpu=ns(CLOCK_THREAD_CPUTIME_ID)-c;r.wall=ns(CLOCK_MONOTONIC)-w;
        const auto after=client.read(r.tid);
        if(after.delay<before.delay||after.count<before.count)throw std::runtime_error("counter moved backwards");
        r.delay=after.delay-before.delay;
      }catch(const std::exception& e){results[i].error=e.what();++ready;}
    });
    while(ready.load()<count)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    start=true;for(auto& t:threads)t.join();
    for(int i=0;i<count;++i) {
      const auto& r=results[i];if(!r.error.empty())throw std::runtime_error(r.error);
      printf("{\"case\":\"%s\",\"tid\":%u,\"wall_ns\":%llu,\"cpu_ns\":%llu,\"queue_ns\":%llu,\"read_max_ns\":%llu}\n",
        mode==0?"alone":mode==1?"contended":"sleep",r.tid,(unsigned long long)r.wall,(unsigned long long)r.cpu,
        (unsigned long long)r.delay,(unsigned long long)r.read_max);
    }
  }
 }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}
 return 0;
}
