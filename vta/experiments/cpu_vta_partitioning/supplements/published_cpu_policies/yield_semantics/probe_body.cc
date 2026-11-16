// Appended to the hash-checked task-scoped PERF_RECORD_SWITCH helper.
// This program never loads TVM, DNN modules or FPGA drivers.
#include <atomic>

struct Measurement {
  std::string name, error;
  int index=0, pid=0, tid=0, core=0, start_core=-1, end_core=-1, policy=0, nice=0;
  U start=0,end=0,cpu_start=0,cpu_end=0,yields=0;
  U enable_start=0,enable_end=0,disable_start=0,disable_end=0,head=0,tail=0,data_size=0;
  bool overflow=false;
  std::vector<unsigned char> bytes;
};

void run_pair(const char* name, bool yielding, bool separate,
              std::vector<Measurement>& output) {
  std::mutex mutex; std::condition_variable cv;
  bool go=false,failed=false; int ready=0;
  std::atomic<bool> done{false};
  std::vector<Measurement> rows(2); std::vector<std::thread> workers;
  for(int i=0;i<2;++i) workers.emplace_back([&,i] {
    auto& r=rows[i];r.name=name;r.index=i;r.pid=getpid();
    r.tid=syscall(SYS_gettid);r.core=(i==0&&separate)?1:0;
    r.policy=sched_getscheduler(0);r.nice=getpriority(PRIO_PROCESS,0);
    try {
      pin(r.core); Ring ring; ring.Start();
      {
        std::unique_lock<std::mutex> lock(mutex);++ready;cv.notify_all();
        cv.wait(lock,[&]{return go||failed;});
        if(failed)throw std::runtime_error("peer setup failed");
      }
      r.start_core=sched_getcpu();
      if(r.start_core!=r.core)throw std::runtime_error("unexpected initial CPU core");
      r.start=clock_ns(CLOCK_MONOTONIC);r.cpu_start=clock_ns(CLOCK_THREAD_CPUTIME_ID);
      volatile U value=1;
      if(i==0) {
        while(clock_ns(CLOCK_THREAD_CPUTIME_ID)-r.cpu_start<80000000) {
          for(int j=0;j<4096;++j)value=value*6364136223846793005ULL+1;
          if(clock_ns(CLOCK_MONOTONIC)-r.start>3000000000ULL)
            throw std::runtime_error("finite busy task exceeded wall bound");
          if(done.load(std::memory_order_acquire))throw std::runtime_error("peer cancelled");
        }
        done.store(true,std::memory_order_release);
      } else {
        while(!done.load(std::memory_order_acquire)) {
          if(yielding) {
            if(sched_yield())throw std::runtime_error("sched_yield failed");
            ++r.yields;
          } else for(int j=0;j<4096;++j)value=value*6364136223846793005ULL+1;
          if(clock_ns(CLOCK_MONOTONIC)-r.start>3000000000ULL)
            throw std::runtime_error("finite waiter exceeded wall bound");
        }
      }
      r.cpu_end=clock_ns(CLOCK_THREAD_CPUTIME_ID);r.end=clock_ns(CLOCK_MONOTONIC);
      r.end_core=sched_getcpu();
      if(r.end_core!=r.core)throw std::runtime_error("unexpected CPU core");
      ring.Stop();
      r.enable_start=ring.enable_start;r.enable_end=ring.enable_end;
      r.disable_start=ring.disable_start;r.disable_end=ring.disable_end;
      r.head=ring.head;r.tail=ring.tail;r.data_size=ring.data_size;
      r.overflow=ring.overflow;r.bytes=std::move(ring.bytes);
    } catch(const std::exception& e) {
      r.error=e.what();done.store(true,std::memory_order_release);
      std::lock_guard<std::mutex> lock(mutex);failed=true;cv.notify_all();
    }
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock,[&]{return ready==2||failed;});go=true;cv.notify_all();
  }
  for(auto& w:workers)w.join();
  for(auto& r:rows)output.push_back(std::move(r));
}

void quote_string(const std::string& s) {
  putchar('"');for(unsigned char c:s){if(c=='"'||c=='\\')putchar('\\');
    if(c>=32)putchar(c);else printf("\\u%04x",c);}putchar('"');
}

int main(int argc,char** argv) {
  if(argc==2&&!strcmp(argv[1],"--describe")) {
    puts("{\"DNN_inferences\":0,\"FPGA_access\":false,\"workload_threads\":6,\"native_sessions\":1,\"target_CPU_ns\":240000000,\"cases\":[\"busy_same\",\"yield_same\",\"yield_separate\"]}");return 0;
  }
  if(argc!=1)return 3;
  pin(2);std::vector<Measurement> rows;
  run_pair("busy_same",false,false,rows);
  if(std::all_of(rows.begin(),rows.end(),[](const Measurement& r){return r.error.empty();}))
    run_pair("yield_same",true,false,rows);
  if(std::all_of(rows.begin(),rows.end(),[](const Measurement& r){return r.error.empty();}))
    run_pair("yield_separate",true,true,rows);
  const bool ok=rows.size()==6&&std::all_of(rows.begin(),rows.end(),[](const Measurement& r){return r.error.empty();});
  printf("{\"status\":\"%s\",\"DNN_inferences\":0,\"periodic_sampling\":false,\"byte_order\":\"little\",\"sample_id_layout\":\"TID,TIME,CPU\",\"streams\":[",ok?"MEASURED":"UNAVAILABLE");
  for(size_t j=0;j<rows.size();++j) {
    const auto& r=rows[j];if(j)putchar(',');printf("{\"case\":");quote_string(r.name);
    printf(",\"index\":%d,\"pid\":%d,\"tid\":%d,\"policy\":%d,\"nice\":%d,\"error\":",r.index,r.pid,r.tid,r.policy,r.nice);quote_string(r.error);
    printf(",\"enable_start_ns\":%llu,\"enable_end_ns\":%llu,\"disable_start_ns\":%llu,\"disable_end_ns\":%llu,\"head\":%llu,\"tail\":%llu,\"data_size\":%llu,\"overflow\":%s,\"yield_calls\":%llu,\"segments\":[",(unsigned long long)r.enable_start,(unsigned long long)r.enable_end,(unsigned long long)r.disable_start,(unsigned long long)r.disable_end,(unsigned long long)r.head,(unsigned long long)r.tail,(unsigned long long)r.data_size,r.overflow?"true":"false",(unsigned long long)r.yields);
    printf("{\"kind\":\"%s\",\"requested_core\":%d,\"actual_start_core\":%d,\"actual_end_core\":%d,\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu}",r.index==0?"busy":"wait",r.core,r.start_core,r.end_core,(unsigned long long)r.start,(unsigned long long)r.end,(unsigned long long)r.cpu_start,(unsigned long long)r.cpu_end);
    printf("],\"records_hex\":\"");for(unsigned char b:r.bytes)printf("%02x",b);printf("\"}");
  }
  puts("]}");return ok?0:2;
}
