// Exercise the real runtime, including concurrent thread-local pools and errors.
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <pthread.h>
#include <unistd.h>
extern "C" {
struct TVMParallelGroupEnv {void* sync_handle;int32_t num_task;};
using Lambda=int(*)(int,TVMParallelGroupEnv*,void*);
int TVMBackendParallelLaunch(Lambda,void*,int);
int TVMBackendParallelBarrier(int,TVMParallelGroupEnv*);
}
struct State {std::atomic<int> values[4]{};int rounds;bool fail;};
void signal_handler(int){}
int task(int id,TVMParallelGroupEnv* env,void* ptr) {
  auto& s=*static_cast<State*>(ptr);
  if(s.fail) {
    // Blocking policy must release other workers when one callback fails.
    if(id==env->num_task-1){usleep(2000);return -1;}
    return TVMBackendParallelBarrier(id,env);
  }
  for(int epoch=1;epoch<=s.rounds;++epoch) {
    s.values[id].store(epoch,std::memory_order_relaxed);
    // Variable arrival order forces real waits and wake-before-wait races.
    if((epoch+id)%7==0)for(volatile int i=0;i<3000;++i){}
    if(TVMBackendParallelBarrier(id,env)!=0)return -1;
    for(int i=0;i<env->num_task;++i)if(s.values[i].load()!=epoch)return -1;
    if(TVMBackendParallelBarrier(id,env)!=0)return -1;
  }
  return 0;
}
int main(int argc,char** argv) {
  const bool fail=argc>1&&std::atoi(argv[1])!=0;
  struct sigaction sa{};sa.sa_handler=signal_handler;sigemptyset(&sa.sa_mask);sigaction(SIGUSR1,&sa,nullptr);
  const int groups=fail?1:3;std::atomic<int> errors{0},completed{0};std::atomic<bool> done{false};
  std::vector<std::thread> threads;
  for(int g=0;g<groups;++g)threads.emplace_back([&]{
    for(int n=1;n<=4;++n) {
      State s;s.rounds=32;s.fail=fail;
      const int result=TVMBackendParallelLaunch(task,&s,n);
      if((!fail&&result!=0)||(fail&&result==0&&n>1))++errors;
    }
    if(!fail){State s;s.rounds=10;s.fail=false;if(TVMBackendParallelLaunch(task,&s,0)!=0)++errors;}
    ++completed;
  });
  // Signals interrupt blocking syscalls; all predicates must be rechecked.
  std::thread interrupter([&]{while(!done.load()){for(auto& t:threads)pthread_kill(t.native_handle(),SIGUSR1);usleep(200);}});
  while(completed.load()!=groups)usleep(200);
  done=true;interrupter.join();for(auto& t:threads)t.join();
  printf("{\"passed\":%s,\"groups\":%d,\"error_case\":%s}\n",errors.load()?"false":"true",groups,fail?"true":"false");
  return errors.load()?2:0;
}
