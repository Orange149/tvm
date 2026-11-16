// Bounded, loss-intolerant, write-forward perf ring retention.
#ifndef VTA_SCHEDULER_RING_DRAIN_H_
#define VTA_SCHEDULER_RING_DRAIN_H_
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace scheduler_ring {
using U=uint64_t;
struct Chunk {U tail,head,head_after_copy,head_after_publish,begin_ns,end_ns,offset;};
struct Retained {std::vector<unsigned char> bytes;std::vector<Chunk> chunks;U cursor=0;};
constexpr U kGuardBytes=64;
constexpr U kMaxRetainedBytes=4ULL*1024*1024; // <=48MiB for exactly 12 targets.
inline void CheckSpace(U head,U tail,U capacity) {
  if (!capacity || (capacity&(capacity-1)) || head<tail || head-tail>=capacity ||
      capacity-(head-tail)<=kGuardBytes)
    throw std::runtime_error("scheduler ring capacity exhausted or uncertain");
}
template<class Head,class Clock,class Publish>
void Drain(Retained& saved,const unsigned char* data,U capacity,U actual_tail,
           Head head,Clock clock,Publish publish) {
  U begin=clock(),h=head(),tail=saved.cursor;
  if(actual_tail!=tail)throw std::runtime_error("scheduler ring consumer cursor changed");
  CheckSpace(h,tail,capacity);
  U n=h-tail,offset=saved.bytes.size();
  if(offset+n>kMaxRetainedBytes)throw std::runtime_error("scheduler retained memory limit");
  saved.bytes.resize(offset+n);
  for(U i=0;i<n;++i)saved.bytes[offset+i]=data[(tail+i)&(capacity-1)];
  U after=head();
  if(after<h)throw std::runtime_error("scheduler ring producer moved backwards");
  CheckSpace(after,tail,capacity); // Also catch filling during the copy.
  // Required read-before-tail ordering on AArch64; kernel may reuse bytes only
  // after the entire prior head snapshot has been copied into private storage.
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  publish(h);
  U published=head();
  if(published<after)throw std::runtime_error("scheduler producer moved backwards after publish");
  // A preemption just before publication must not conceal a full ring. This
  // conservative check may reject a busy but lossless copy; it never credits
  // newly freed space as evidence that the old occupancy was safe.
  CheckSpace(published,tail,capacity);
  saved.cursor=h;
  saved.chunks.push_back({tail,h,after,published,begin,clock(),offset});
}
} // namespace scheduler_ring
#endif
