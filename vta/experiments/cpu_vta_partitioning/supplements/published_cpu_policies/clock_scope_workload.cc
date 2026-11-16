// Appended to the authenticated Ring implementation at local preparation.
// Zero DNN/TVM/VTA work. No runtime parameter, priority, or kernel changes.
struct Count3 {U value=0,enabled=0,running=0;};
struct TaskClock {
 int fd=-1;
 TaskClock(){
  perf_event_attr a{};a.size=PERF_ATTR_SIZE_VER0;a.type=PERF_TYPE_SOFTWARE;
  a.config=PERF_COUNT_SW_TASK_CLOCK;a.disabled=1;a.exclude_hv=1;
  a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
  fd=syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC);
  if(fd<0)throw std::runtime_error(error_text("task clock open"));
 }
 ~TaskClock(){if(fd>=0)close(fd);}
 Count3 read(){Count3 x{};if(::read(fd,&x,sizeof(x))!=sizeof(x))throw std::runtime_error("task clock read failed");return x;}
 void enable(){if(ioctl(fd,PERF_EVENT_IOC_ENABLE,0))throw std::runtime_error("task clock enable failed");}
 void disable(){if(ioctl(fd,PERF_EVENT_IOC_DISABLE,0))throw std::runtime_error("task clock disable failed");}
};
void sleep_ns(U n){timespec t{static_cast<time_t>(n/1000000000ULL),static_cast<long>(n%1000000000ULL)};
 while(nanosleep(&t,&t)){if(errno!=EINTR)throw std::runtime_error("sleep failed");}}
void busy_ns(U n){U until=clock_ns(CLOCK_MONOTONIC)+n;volatile U v=1;
 while(clock_ns(CLOCK_MONOTONIC)<until)for(int i=0;i<32;++i)v=v*6364136223846793005ULL+1;}
void count_json(const Count3& c){printf("{\"value\":%llu,\"enabled\":%llu,\"running\":%llu}",
 (unsigned long long)c.value,(unsigned long long)c.enabled,(unsigned long long)c.running);}
void scenario(const char* kind,bool trace){
 pin(3);Ring ring;TaskClock counter;counter.enable();if(trace)ring.Start();
 // These guard sleeps establish both sides of the work window in the ring.
 // They are excluded from all work-window clock comparisons.
 sleep_ns(1000000);
 U read0a=clock_ns(CLOCK_MONOTONIC);auto before=counter.read();U read0b=clock_ns(CLOCK_MONOTONIC);
 U ws=clock_ns(CLOCK_MONOTONIC),cs=clock_ns(CLOCK_THREAD_CPUTIME_ID);
 if(!strcmp(kind,"busy"))busy_ns(12000000);
 else for(int i=0;i<600;++i){busy_ns(5000);sleep_ns(1000);}
 U ce=clock_ns(CLOCK_THREAD_CPUTIME_ID),we=clock_ns(CLOCK_MONOTONIC);
 U read1a=clock_ns(CLOCK_MONOTONIC);auto after=counter.read();U read1b=clock_ns(CLOCK_MONOTONIC);
 sleep_ns(1000000);if(trace)ring.Stop();counter.disable();
 printf("{\"case\":\"%s\",\"trace\":%s,\"pid\":%d,\"tid\":%ld,\"allowed\":[3],\"iterations\":%d,",
  kind,trace?"true":"false",getpid(),syscall(SYS_gettid),!strcmp(kind,"busy")?1:600);
 printf("\"wall_start_ns\":%llu,\"wall_end_ns\":%llu,\"cpu_start_ns\":%llu,\"cpu_end_ns\":%llu,",
  (unsigned long long)ws,(unsigned long long)we,(unsigned long long)cs,(unsigned long long)ce);
 printf("\"read0_start_ns\":%llu,\"read0_end_ns\":%llu,\"read1_start_ns\":%llu,\"read1_end_ns\":%llu,",
  (unsigned long long)read0a,(unsigned long long)read0b,(unsigned long long)read1a,(unsigned long long)read1b);
 printf("\"enable_start_ns\":%llu,\"enable_end_ns\":%llu,\"disable_start_ns\":%llu,\"disable_end_ns\":%llu,",
  (unsigned long long)ring.enable_start,(unsigned long long)ring.enable_end,(unsigned long long)ring.disable_start,(unsigned long long)ring.disable_end);
 printf("\"head\":%llu,\"tail\":%llu,\"data_size\":%llu,\"overflow\":%s,\"counter_before\":",
  (unsigned long long)ring.head,(unsigned long long)ring.tail,(unsigned long long)ring.data_size,ring.overflow?"true":"false");
 count_json(before);printf(",\"counter_after\":");count_json(after);printf(",\"records_hex\":\"");
 for(unsigned char b:ring.bytes)printf("%02x",b);puts("\"}");
}
int main(int argc,char** argv){
 if(argc==2&&!strcmp(argv[1],"--describe")){
  puts("{\"DNN_inferences\":0,\"VTA_calls\":0,\"cases\":4,\"poll_iterations\":600,\"core\":3,\"busy_duration_ns\":12000000,\"poll_busy_ns\":5000,\"poll_sleep_request_ns\":1000,\"system_changes\":false}");return 0;}
 if(argc!=1)return 3;
 try{scenario("busy",false);scenario("busy",true);scenario("poll",false);scenario("poll",true);return 0;}
 catch(const std::exception& e){fprintf(stderr,"clock-scope self-test: %s\n",e.what());return 2;}
}
