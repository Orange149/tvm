/* Experiment-only adaptation of pasqoc/incubator-tvm
 * 9bc19a8a90b6fef53ef05a4320032414e545f22f vta/runtime/runtime.cc.
 * Licensed under the Apache License, Version 2.0.
 *
 * New official-window experiment copy. Preserves value consolidation and
 * permits greedy LOAD extension only with a checked intervening-read guard.
 * Local additions retain capacity checks, contiguous host packing, transfer
 * diagnostics and replay serialization. Production runtime is not replaced.
 */
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue;

template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class UopQueue : public BaseQueue<UopKernel*> {
 public:
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, queue_capacity::FromEnv("VTA_UOP_BUFFER_BYTES", kMaxBytes),
                         kCoherent, kAlwaysCache);
  }
  size_t SerializedSize() const {
    size_t bytes = 0;
    for (const UopKernel* kernel : dram_buffer_) bytes += kernel->size() * kElemBytes;
    return bytes;
  }
  template <typename FAutoSync>
  void Push(UopKernel* kernel, FAutoSync fautosync) {
    if (kernel->cached()) return;
    size_t num_op = kernel->size();
    CHECK_LE(num_op, static_cast<size_t>(kMaxNumUop));
    CHECK_GT(num_op, 0U);
    if (C3_BANERJEE_VALUE_UOPS && !pending() && cache_idx_ > 0) {
      UopKernel* prev = cache_[cache_idx_ - 1];
      if (prev->cached() && num_op == prev->size()) {
        bool equal = true;
        for (size_t i = 0; i < num_op; ++i) {
          const VTAUop& a = prev->data()[i];
          const VTAUop& b = kernel->data()[i];
          if (a.dst_idx != b.dst_idx || a.src_idx != b.src_idx || a.wgt_idx != b.wgt_idx) {
            equal = false;
            break;
          }
        }
        if (equal) {
          kernel->sram_begin_ = prev->sram_begin_;
          kernel->sram_end_ = prev->sram_end_;
          cache_.insert(cache_.begin() + cache_idx_, kernel);
          ++cache_idx_;
          return;
        }
      }
    }
    // DRAM keeps kernels referenced by this whole batch, including evicted
    // SRAM entries. Offset computation and serialization use the same list.
    bool found = std::find(dram_buffer_.begin(), dram_buffer_.end(), kernel) != dram_buffer_.end();
    if (!found) {
      if (SerializedSize() + num_op * kElemBytes > static_cast<size_t>(kMaxBytes)) {
        fautosync();
        CHECK_LE(SerializedSize() + num_op * kElemBytes, static_cast<size_t>(kMaxBytes));
      }
      dram_buffer_.push_back(kernel);
    }
    uint32_t begin = 0;
    if (sram_end_ + num_op > kMaxNumUop) {
      cache_idx_ = 0;
      sram_begin_ = 0;
      sram_end_ = num_op;
    } else {
      begin = sram_end_;
      sram_end_ += num_op;
    }
    uint32_t evict_begin = cache_idx_;
    for (; cache_idx_ < cache_.size(); ++cache_idx_) {
      if (cache_[cache_idx_]->sram_begin_ >= sram_end_) break;
      cache_[cache_idx_]->sram_begin_ = 0;
      cache_[cache_idx_]->sram_end_ = 0;
    }
    kernel->sram_begin_ = begin;
    kernel->sram_end_ = sram_end_;
    cache_.insert(cache_.begin() + cache_idx_, kernel);
    cache_.erase(cache_.begin() + evict_begin, cache_.begin() + cache_idx_);
    cache_idx_ = evict_begin + 1;
  }
  void FlushUopLoad(InsnQueue<kMaxBytes, kCoherent, kAlwaysCache>& queue) {
    CHECK(pending());
    CHECK_GT(cache_idx_, 0U);
    UopKernel* cur = cache_[cache_idx_ - 1];
    CHECK_EQ(cur->size(), sram_end_ - sram_begin_);
    uint32_t offset = 0;
    bool found = false;
    for (UopKernel* kernel : dram_buffer_) {
      if (kernel == cur) { found = true; break; }
      offset += kernel->size() * kElemBytes;
    }
    CHECK(found);
    uint32_t dram_base = (fpga_buff_phy_ + offset) / kElemBytes;
    if (C3_BANERJEE_GREEDY_UOPS) {
      for (int64_t i = static_cast<int64_t>(queue.count()) - 1; i >= 0; --i) {
        VTAMemInsn* prior = reinterpret_cast<VTAMemInsn*>(queue.data()) + i;
        if (prior->opcode == VTA_OPCODE_LOAD && prior->memory_type == VTA_MEM_ID_UOP) {
          bool intervening_read_conflict = false;
          for (uint32_t j = static_cast<uint32_t>(i) + 1; j < queue.count(); ++j) {
            const VTAGemInsn* crossed = reinterpret_cast<const VTAGemInsn*>(queue.data()) + j;
            if (crossed->opcode == VTA_OPCODE_GEMM || crossed->opcode == VTA_OPCODE_ALU) {
              // Moving this new tail to an older LOAD must not overwrite any
              // UOP reference that would otherwise precede the new LOAD.
              if (crossed->uop_bgn < sram_end_ && sram_begin_ < crossed->uop_end) {
                intervening_read_conflict = true;
                break;
              }
            }
          }
          if (prior->sram_base + prior->x_size == sram_begin_ &&
              prior->dram_base + prior->x_size == dram_base && cache_idx_ > 1 &&
              cur->cached() && !intervening_read_conflict &&
              prior->sram_base + prior->x_size + (sram_end_ - sram_begin_) < kMaxNumUop) {
            prior->x_size += sram_end_ - sram_begin_;
            prior->x_stride += sram_end_ - sram_begin_;
            sram_begin_ = sram_end_;
            return;
          }
          break;
        }
      }
    }
    VTAMemInsn* insn = queue.CreateMemInsn(VTA_MEM_ID_UOP);
    insn->opcode = VTA_OPCODE_LOAD;
    insn->memory_type = VTA_MEM_ID_UOP;
    insn->sram_base = sram_begin_;
    insn->dram_base = dram_base;
    insn->y_size = 1;
    insn->x_size = sram_end_ - sram_begin_;
    insn->x_stride = sram_end_ - sram_begin_;
    insn->y_pad_0 = insn->y_pad_1 = insn->x_pad_0 = insn->x_pad_1 = 0;
    sram_begin_ = sram_end_;
  }
  void Reset() {
    for (UopKernel* kernel : cache_) kernel->sram_begin_ = kernel->sram_end_ = 0;
    cache_.clear();
    cache_idx_ = 0;
    BaseQueue<UopKernel*>::Reset();
  }
  std::vector<char> SerializeCacheBytes() const {
    std::vector<char> out(SerializedSize());
    size_t offset = 0;
    for (const UopKernel* kernel : dram_buffer_) {
      size_t size = kernel->size() * kElemBytes;
      memcpy(out.data() + offset, kernel->data(), size);
      offset += size;
    }
    return out;
  }
  void AutoReadBarrier() { ReadBarrier(); }
  void ReadBarrier() {
    CheckCapacity(SerializedSize());
    transfer_timing = {};
    const bool diagnostic = c3_official_window::Detailed() && queue_capacity::DiagnosticEnabled();
    double start = diagnostic ? NowMicros() : 0;
    auto bytes = SerializeCacheBytes();
    if (diagnostic) { transfer_timing.pack_us = NowMicros() - start; start = NowMicros(); }
    if (!bytes.empty()) VTAMemCopyFromHost(fpga_buff_, bytes.data(), bytes.size());
    if (diagnostic) { transfer_timing.copy_us = NowMicros() - start; start = NowMicros(); }
    if (!coherent_ && always_cache_ && !bytes.empty()) VTAFlushCache(fpga_buff_, fpga_buff_phy_, bytes.size());
    if (diagnostic) transfer_timing.cache_us = NowMicros() - start;
  }
  void LoadSerializedBytes(const std::vector<char>& bytes) {
    CheckCapacity(bytes.size());
    if (bytes.empty()) return;
    VTAMemCopyFromHost(fpga_buff_, bytes.data(), bytes.size());
    if (!coherent_ && always_cache_) VTAFlushCache(fpga_buff_, fpga_buff_phy_, bytes.size());
  }
 private:
  uint32_t cache_idx_{0};
  std::vector<UopKernel*> cache_;
  static constexpr int kElemBytes = sizeof(VTAUop);
  static constexpr int kMaxNumUop = VTA_UOP_BUFF_DEPTH;
};
