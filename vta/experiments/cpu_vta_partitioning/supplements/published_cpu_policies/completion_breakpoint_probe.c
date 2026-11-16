#define _GNU_SOURCE
#include <asm/perf_regs.h>
#include <errno.h>
#include <inttypes.h>
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* The observed instruction follows a successful atomic decrement. No binary
 * rewriting, signal handler, periodic sampler, runtime library, or FPGA use. */
#ifndef __aarch64__
#error This interface self-test is AArch64 only
#endif
extern void completion_probe_signal(uint32_t*);
extern const char completion_probe_after_signal[];
__asm__(".text\n.p2align 2\n.global completion_probe_signal\n"
        ".type completion_probe_signal,%function\ncompletion_probe_signal:\n"
        "1: ldaxr w1,[x0]\nsub w1,w1,#1\nstlxr w2,w1,[x0]\ncbnz w2,1b\n"
        ".global completion_probe_after_signal\ncompletion_probe_after_signal:\nnop\nret\n"
        ".size completion_probe_signal,.-completion_probe_signal\n");

static uint64_t now_ns(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
  return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec;
}
static int unavailable(const char* operation) {
  printf("{\"status\":\"UNAVAILABLE\",\"operation\":\"%s\",\"errno\":%d,"
         "\"DNN_inferences\":0,\"FPGA_access\":false,\"sysctl_changed\":false}\n", operation, errno);
  return 0;
}
static int invalid(const char* reason) {
  printf("{\"status\":\"INVALID\",\"reason\":\"%s\",\"DNN_inferences\":0}\n", reason);
  return 2;
}
static void ring_read(const unsigned char* ring, uint64_t capacity, uint64_t at, void* target, size_t n) {
  size_t first = n < capacity-at%capacity ? n : capacity-at%capacity;
  memcpy(target, ring+at%capacity, first);
  memcpy((unsigned char*)target+first, ring, n-first);
}

int main(void) {
  alarm(3);
  struct perf_event_attr a; memset(&a, 0, sizeof(a));
  a.type=PERF_TYPE_BREAKPOINT; a.size=sizeof(a);
  a.bp_type=HW_BREAKPOINT_X; a.bp_len=HW_BREAKPOINT_LEN_4;
  a.bp_addr=(uintptr_t)completion_probe_after_signal;
  a.sample_period=1; a.disabled=1; a.pinned=1; a.exclude_kernel=1; a.exclude_hv=1;
  a.sample_type=PERF_SAMPLE_IP|PERF_SAMPLE_TID|PERF_SAMPLE_TIME|PERF_SAMPLE_CPU|PERF_SAMPLE_REGS_USER;
  a.sample_regs_user=(1ull<<PERF_REG_ARM64_X0)|(1ull<<PERF_REG_ARM64_X1)|(1ull<<PERF_REG_ARM64_X2);
  a.use_clockid=1; a.clockid=CLOCK_MONOTONIC; a.wakeup_events=1;
  const int fd=(int)syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC);
  if (fd<0) return unavailable("perf_event_open_execution_breakpoint");
  const long page=sysconf(_SC_PAGESIZE); const size_t length=(size_t)page*9;
  void* memory=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if (memory==MAP_FAILED) { const int code=unavailable("mmap_breakpoint_ring");close(fd);return code; }
  struct perf_event_mmap_page* meta=memory;
  if (meta->data_offset!=(uint64_t)page || meta->data_size!=(uint64_t)page*8) {
    munmap(memory,length);close(fd);return invalid("unexpected_ring_extent");
  }
  volatile uint32_t pending=0; uint32_t expected[8]; uint64_t begin[8],end[8];
  if (ioctl(fd,PERF_EVENT_IOC_RESET,0) || ioctl(fd,PERF_EVENT_IOC_ENABLE,0)) {
    const int code=unavailable("enable_breakpoint");munmap(memory,length);close(fd);return code;
  }
  for (int i=0;i<8;++i) {
    pending=(i%2)+1; expected[i]=pending-1; begin[i]=now_ns();
    completion_probe_signal((uint32_t*)&pending); end[i]=now_ns();
    if (pending!=expected[i]) { ioctl(fd,PERF_EVENT_IOC_DISABLE,0);return invalid("atomic_result_changed"); }
  }
  if (ioctl(fd,PERF_EVENT_IOC_DISABLE,0)) return invalid("disable_failed");
  uint64_t counted=0;
  if (read(fd,&counted,sizeof(counted))!=sizeof(counted)) return invalid("counter_read_failed");
  const uint64_t head=__atomic_load_n(&meta->data_head,__ATOMIC_ACQUIRE);
  const uint64_t tail=meta->data_tail, capacity=meta->data_size;
  if (head<tail || head-tail>=capacity) return invalid("ring_extent_or_loss");
  unsigned char* ring=(unsigned char*)memory+meta->data_offset;
  uint64_t cursor=tail, times[8]={0}, sampled[8]={0}; uint32_t cpus[8]={0}; int n=0;
  const uint32_t pid=(uint32_t)getpid(),tid=(uint32_t)syscall(SYS_gettid);
  while (cursor<head) {
    struct perf_event_header h; ring_read(ring,capacity,cursor,&h,sizeof(h));
    if (h.type!=PERF_RECORD_SAMPLE || h.size!=72 || head-cursor<h.size || n>=8)
      return invalid("lost_or_unexpected_sample_ABI");
    uint64_t words[8]; ring_read(ring,capacity,cursor+8,words,sizeof(words));
    /* ip; pid/tid; time; cpu/res; ABI; x0,x1,x2 */
    if (words[0]!=(uintptr_t)completion_probe_after_signal || (uint32_t)words[1]!=pid ||
        (uint32_t)(words[1]>>32)!=tid || words[4]!=PERF_SAMPLE_REGS_ABI_64 ||
        words[5]!=(uintptr_t)&pending || words[6]!=expected[n] || words[7]!=0 ||
        words[2]<begin[n] || words[2]>end[n] || (n && words[2]<times[n-1]) || words[3]>>32)
      return invalid("instruction_register_or_clock_contract");
    times[n]=words[2];sampled[n]=words[6];cpus[n]=(uint32_t)words[3];++n;cursor+=h.size;
  }
  if (counted!=8 || n!=8) return invalid("missing_or_duplicate_event");
  __atomic_store_n(&meta->data_tail,head,__ATOMIC_RELEASE);
  printf("{\"status\":\"VALIDATED\",\"events\":8,\"atomic_operations\":8,"
         "\"breakpoint_instruction\":\"after_successful_atomic_decrement\","
         "\"register_values_checked\":true,\"monotonic_clock_envelope_checked\":true,"
         "\"periodic_sampling\":false,\"binary_patched\":false,\"samples\":[");
  for(int i=0;i<8;++i)printf("%s{\"time_ns\":%" PRIu64 ",\"pending\":%" PRIu64 ",\"cpu\":%u}",
                            i?",":"",times[i],sampled[i],cpus[i]);
  printf("],\"DNN_inferences\":0,\"FPGA_access\":false,\"sysctl_changed\":false,"
         "\"does_not_validate_DNN_probe_perturbation\":true}\n");
  munmap(memory,length);close(fd);return 0;
}
