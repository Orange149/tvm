// Actual 8192-entry SRAM and actual 16-byte instruction ABI, fake host DMA only.
// This is a queue safety fixture, not a model/Tile/performance candidate.
#include <vta/hw_spec.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "c3_official_window_runtime_collection.h"
#define CHECK(x) assert(x)
#define CHECK_LE(a,b) assert((a)<=(b))
#define CHECK_GT(a,b) assert((a)>(b))
#define CHECK_EQ(a,b) assert((a)==(b))
#define C3_BANERJEE_VALUE_UOPS 1
#define C3_BANERJEE_GREEDY_UOPS 1
struct UopKernel {
  std::vector<VTAUop> ops;
  unsigned sram_begin_{},sram_end_{};
  bool cached() const { return sram_begin_ != sram_end_; }
  size_t size() const { return ops.size(); }
  const VTAUop* data() const { return ops.data(); }
};
namespace queue_capacity {
unsigned FromEnv(const char*,unsigned x) { return x; }
bool DiagnosticEnabled() { return false; }
}
double NowMicros() { return 0; }
void VTAMemCopyFromHost(void* dst,const void* src,size_t bytes) { memcpy(dst,src,bytes); }
void VTAFlushCache(void*,unsigned,size_t) {}
template <class T> class BaseQueue {
 public:
  struct TransferTiming { double pack_us{},copy_us{},cache_us{}; } transfer_timing;
  unsigned fpga_buff_phy_{4096},sram_begin_{},sram_end_{};
  std::vector<T> dram_buffer_;
  char memory_[1<<18]{};
  void* fpga_buff_{memory_};
  bool coherent_{},always_cache_{};
  bool pending() const { return sram_begin_ != sram_end_; }
  void InitSpace(unsigned,unsigned,bool,bool) {}
  void CheckCapacity(size_t bytes) const { assert(bytes <= sizeof(memory_)); }
  void Reset() { dram_buffer_.clear();sram_begin_=sram_end_=0; }
};
template <int N,bool C,bool A> class InsnQueue {
 public:
  std::vector<VTAInsn> instructions;
  size_t count() const { return instructions.size(); }
  VTAInsn* data() { return instructions.data(); }
  VTAMemInsn* CreateMemInsn(unsigned) {
    instructions.emplace_back();memset(&instructions.back(),0,sizeof(VTAInsn));
    return reinterpret_cast<VTAMemInsn*>(&instructions.back());
  }
};
#ifdef ORIGINAL_QUEUE
#include "c3_banerjee_uop_queue.h"
#else
#include "c3_official_window_runtime_uop_queue.h"
#endif
UopKernel make(unsigned n,unsigned id) {
  UopKernel k;
  for (unsigned i=0;i<n;++i) { VTAUop u{};u.dst_idx=i%2048;u.src_idx=id;u.wgt_idx=id;k.ops.push_back(u); }
  return k;
}
int main(int argc,char** argv) {
  static_assert(VTA_UOP_BUFF_DEPTH==8192,"fixture must use actual SRAM");
  static_assert(sizeof(VTAInsn)==16&&sizeof(VTAUop)==4,"actual instruction/uop units required");
  UopQueue<1<<25,false,false> q;
  InsnQueue<1<<25,false,false> instructions;
  if(argc>1&&std::strcmp(argv[1],"oversize")==0) {
    auto oversized=make(8193,1);q.Push(&oversized,[]{assert(false);});return 8;
  }
  if(argc>1&&std::strcmp(argv[1],"endpoint")==0) {
    auto exact=make(8192,1);q.Push(&exact,[]{assert(false);});q.FlushUopLoad(instructions);
    auto* g=reinterpret_cast<VTAGemInsn*>(instructions.CreateMemInsn(0));g->opcode=VTA_OPCODE_GEMM;
    g->uop_bgn=exact.sram_begin_;g->uop_end=exact.sram_end_;
    assert(g->uop_bgn==0&&g->uop_end==8192);
    printf("{\"single_kernel_8192\":\"representable\",\"begin\":0,\"exclusive_end\":8192}\n");return 0;
  }
  auto a=make(2048,1),b=make(2048,2),c=make(2048,3),d=make(2048,4),e=make(2048,5),f=make(2048,6);
  auto reference=[&](UopKernel& k) {
    q.Push(&k,[]{assert(false);});
    if(q.pending())q.FlushUopLoad(instructions);
    auto* g=reinterpret_cast<VTAGemInsn*>(instructions.CreateMemInsn(0));g->opcode=VTA_OPCODE_GEMM;
    g->uop_bgn=k.sram_begin_;g->uop_end=k.sram_end_;
  };
  for(auto* k:{&a,&b,&c,&d,&e})reference(*k);
  assert(b.cached());reference(b);
  const unsigned old_read=instructions.count()-1;
  reference(f);
  const auto packed=q.SerializeCacheBytes();
  assert(packed.size()==6*2048*4);
  const auto* source=reinterpret_cast<const VTAUop*>(packed.data());
  std::vector<VTAUop> live(8192);
  bool old_b_correct=false;
  for(unsigned i=0;i<instructions.count();++i) {
    auto* mem=reinterpret_cast<const VTAMemInsn*>(instructions.data())+i;
    if(mem->opcode==VTA_OPCODE_LOAD) {
      assert(mem->memory_type==VTA_MEM_ID_UOP&&mem->sram_base+mem->x_size<=8192);
      const unsigned offset=mem->dram_base-q.fpga_buff_phy_/4;
      assert(offset+mem->x_size<=packed.size()/4);
      std::copy(source+offset,source+offset+mem->x_size,live.begin()+mem->sram_base);
    } else if(i==old_read) {
      auto* g=reinterpret_cast<const VTAGemInsn*>(instructions.data())+i;
      assert(g->uop_bgn==2048&&g->uop_end==4096);
      old_b_correct=live[2048].src_idx==2&&live[4095].src_idx==2;
    }
  }
#ifdef ORIGINAL_QUEUE
  assert(!old_b_correct);
#else
  assert(old_b_correct);
#endif
  q.Reset();assert(!b.cached()&&!f.cached()&&q.SerializedSize()==0);
  unsigned clocks=0;
  c3_official_window::ClearLedger();
  c3_official_window::DetailClock([&](){++clocks;return 1.0;});
  assert(clocks==static_cast<unsigned>(c3_official_window::Detailed()));
  for(unsigned i=0;i<17;++i)c3_official_window::RecordSubmit("explicit_sync",8,16,i==3?1:0);
  auto& state=c3_official_window::State();
  assert(state.overflow&&state.attempted_submits==17&&state.recorded_submits==16);
  assert(!state.submits[3].completion&&state.submits[3].timeout==1);
  printf("{\"actual_sram_entries\":8192,\"serialized_elements\":12288,\"cached_old_read_correct\":%s,"
         "\"original_queue\":%s,\"successful_sync_resets_residency\":true,\"ledger_overflow_explicit_invalid\":true,"
         "\"collection_clock_gate\":true}\n",old_b_correct?"true":"false",
#ifdef ORIGINAL_QUEUE
         "true"
#else
         "false"
#endif
         );
}
