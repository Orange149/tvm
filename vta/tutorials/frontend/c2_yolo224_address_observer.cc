// Qualification-only transparent observer of the two actual CPU-pointer calls
// in the frozen last-reader compute function. Never used for performance data.
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

extern "C" void* VTABufferCPUPtr(void* command,void* buffer) {
  using Resolve=void*(*)(void*,void*);
  using Physical=uint64_t(*)(void*);
  static auto real=reinterpret_cast<Resolve>(dlsym(RTLD_NEXT,"VTABufferCPUPtr"));
  static auto physical=reinterpret_cast<Physical>(dlsym(RTLD_NEXT,"VTAMemGetPhyAddr"));
  if(!real || !physical)_exit(126);
  void* cpu=real(command,buffer);
  const char* path=getenv("C2_ADDRESS_LOG");
  const char* first=getenv("C2_ADDRESS_RETURN_PC0");
  const char* second=getenv("C2_ADDRESS_RETURN_PC1");
  if(!path || !first || !second)return cpu;
  Dl_info caller{};void* pc=__builtin_return_address(0);
  if(!dladdr(pc,&caller) || !caller.dli_fname || !strstr(caller.dli_fname,"/stages/stage1/graphlib.so"))return cpu;
  uintptr_t offset=reinterpret_cast<uintptr_t>(pc)-reinterpret_cast<uintptr_t>(caller.dli_fbase);
  int site=offset==strtoull(first,nullptr,0)?0:offset==strtoull(second,nullptr,0)?1:-1;
  if(site<0)return cpu;
  static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
  static uint64_t pair=0;
  pthread_mutex_lock(&lock);
  struct timespec stamp{};clock_gettime(CLOCK_MONOTONIC,&stamp);
  Dl_info implementation{};dladdr(reinterpret_cast<void*>(real),&implementation);
  FILE* stream=fopen(path,"a");
  if(!stream)_exit(125);
  fprintf(stream,"{\"pair\":%llu,\"site\":%d,\"tid\":%ld,\"return_pc\":\"0x%llx\","
      "\"handle\":\"0x%llx\",\"cpu\":\"0x%llx\",\"physical\":\"0x%llx\","
      "\"time_ns\":%llu,\"object\":\"%s\",\"real_resolver\":\"%s\"}\n",
      static_cast<unsigned long long>(pair),site,syscall(SYS_gettid),static_cast<unsigned long long>(offset),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buffer)),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(cpu)),
      static_cast<unsigned long long>(physical(cpu)),
      static_cast<unsigned long long>(stamp.tv_sec)*1000000000ULL+stamp.tv_nsec,
      caller.dli_fname,implementation.dli_fname?implementation.dli_fname:"");
  fclose(stream);if(site==1)++pair;
  pthread_mutex_unlock(&lock);
  return cpu;
}
