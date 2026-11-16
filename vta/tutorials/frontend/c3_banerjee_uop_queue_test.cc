// Test the actual experimental queue template with small SRAM and fake DMA.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define CHECK(x) assert(x)
#define CHECK_LE(a,b) assert((a)<=(b))
#define CHECK_GT(a,b) assert((a)>(b))
#define CHECK_EQ(a,b) assert((a)==(b))
#define C3_BANERJEE_VALUE_UOPS 1
#define C3_BANERJEE_GREEDY_UOPS 1
#define VTA_UOP_BUFF_DEPTH 16
#define VTA_OPCODE_LOAD 0
#define VTA_MEM_ID_UOP 0
struct VTAUop { unsigned dst_idx: 8; unsigned src_idx: 12; unsigned wgt_idx: 12; };
struct VTAMemInsn {
  unsigned opcode{}, memory_type{}, sram_base{}, dram_base{}, y_size{}, x_size{}, x_stride{};
  unsigned y_pad_0{}, y_pad_1{}, x_pad_0{}, x_pad_1{};
};
struct UopKernel {
  std::vector<VTAUop> ops;
  unsigned sram_begin_{}, sram_end_{};
  bool cached() const { return sram_begin_ != sram_end_; }
  size_t size() const { return ops.size(); }
  const VTAUop* data() const { return ops.data(); }
};
namespace queue_capacity {
unsigned FromEnv(const char*, unsigned x) { return x; }
bool DiagnosticEnabled() { return false; }
}
double NowMicros() { return 0; }
void VTAMemCopyFromHost(void* dst, const void* src, size_t size) { memcpy(dst,src,size); }
void VTAFlushCache(void*, unsigned, size_t) {}
template <class T> class BaseQueue {
 public:
  struct TransferTiming { double pack_us{}, copy_us{}, cache_us{}; } transfer_timing;
  unsigned fpga_buff_phy_{4096}, sram_begin_{}, sram_end_{};
  std::vector<T> dram_buffer_;
  char memory_[8192]{};
  void* fpga_buff_{memory_};
  bool coherent_{}, always_cache_{};
  bool pending() const { return sram_begin_ != sram_end_; }
  void InitSpace(unsigned, unsigned, bool, bool) {}
  void CheckCapacity(size_t size) const { assert(size <= sizeof(memory_)); }
  void Reset() { dram_buffer_.clear(); sram_begin_ = sram_end_ = 0; }
};
template <int N, bool C, bool A> class InsnQueue {
 public:
  std::vector<VTAMemInsn> instructions;
  size_t count() const { return instructions.size(); }
  VTAMemInsn* data() { return instructions.data(); }
  VTAMemInsn* CreateMemInsn(unsigned) {
    instructions.emplace_back(); return &instructions.back();
  }
};
#include "c3_banerjee_uop_queue.h"

UopKernel make(unsigned n, unsigned value) {
  UopKernel k;
  for (unsigned i=0;i<n;++i) k.ops.push_back({value+i, 1, 2});
  return k;
}
int main() {
  UopQueue<4096,false,false> q;
  InsnQueue<4096,false,false> insn;
  auto a=make(4,0), b=make(4,0), c=make(4,10);
  q.Push(&a, []{assert(false);}); q.FlushUopLoad(insn);
  auto bytes=q.SerializeCacheBytes();
  q.Push(&b, []{assert(false);});
  assert(!q.pending() && a.sram_begin_ == b.sram_begin_ && a.sram_end_ == b.sram_end_);
  assert(q.SerializedSize() == 4*sizeof(VTAUop) && q.SerializeCacheBytes() == bytes);
  q.Push(&c, []{assert(false);}); q.FlushUopLoad(insn);
  assert(insn.count()==1 && insn.data()[0].x_size==8); // greedy adjacent LOAD
  assert(q.SerializedSize()==8*sizeof(VTAUop));
  q.ReadBarrier();
  assert(memcmp(q.memory_,a.data(),4*sizeof(VTAUop))==0);
  assert(memcmp(q.memory_+4*sizeof(VTAUop),c.data(),4*sizeof(VTAUop))==0);
  auto d=make(12,30);
  q.Push(&d, []{assert(false);}); q.FlushUopLoad(insn); // SRAM wrap/eviction
  assert(!a.cached() && !b.cached() && !c.cached() && d.sram_begin_==0);
  assert(insn.count()==2 && insn.data()[1].dram_base==(4096+8*sizeof(VTAUop))/sizeof(VTAUop));
  assert(q.SerializedSize()==20*sizeof(VTAUop)); // old commands retain their DRAM data
  q.ReadBarrier();
  assert(memcmp(q.memory_+8*sizeof(VTAUop),d.data(),12*sizeof(VTAUop))==0);
  q.Reset();
  assert(!d.cached() && q.SerializedSize()==0);
  q.Push(&b, []{assert(false);}); q.FlushUopLoad(insn);
  assert(b.cached() && q.SerializedSize()==4*sizeof(VTAUop));
  // Small command buffer: synchronize BEFORE appending a new kernel.
  UopQueue<32,false,false> small;
  InsnQueue<32,false,false> small_insn;
  auto e=make(4,70), f=make(4,80), g=make(4,90);
  small.Push(&e, []{assert(false);}); small.FlushUopLoad(small_insn);
  small.Push(&f, []{assert(false);}); small.FlushUopLoad(small_insn);
  bool synced=false;
  small.Push(&g,[&]{synced=true; small.ReadBarrier(); small.Reset();});
  assert(synced && !e.cached() && !f.cached() && g.cached());
  small.FlushUopLoad(small_insn);
  assert(small.SerializedSize()==16);
}
