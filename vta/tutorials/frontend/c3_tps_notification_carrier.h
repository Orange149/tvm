// Local FSim-only carrier experiment. Included inside InsnQueue's public area.
// Call only AFTER queue generation/drain; greedy UOP packing remains frozen.
// Qualification requires an exact Python proposal/actual-encoding comparison,
// unchanged material reachability, UOP live-value proof and finite FIFO proof.
void C3TPSNotificationCarriers() {
  auto bits = [](VTAMemInsn* p, int a, int b, int c, int d) {
    return p->pop_prev_dep == a && p->pop_next_dep == b &&
           p->push_prev_dep == c && p->push_next_dep == d;
  };
  auto next_compute = [&](size_t start) {
    for (size_t j = start; j < dram_buffer_.size(); ++j) {
      auto* p = reinterpret_cast<VTAMemInsn*>(&dram_buffer_[j]);
      if (p->opcode == VTA_OPCODE_FINISH || GetPipelineStageAll(p) == kComputeStage)
        return j;
    }
    return dram_buffer_.size();
  };
  for (size_t i = 0; i < dram_buffer_.size();) {
    auto* p = reinterpret_cast<VTAMemInsn*>(&dram_buffer_[i]);
    bool erased = false;
    if (p->opcode == VTA_OPCODE_LOAD && p->memory_type == VTA_MEM_ID_UOP &&
        p->x_size == 0 && p->y_size == 0) {
      const size_t a = next_compute(i + 1);
      const size_t b = next_compute(a + 1);
      if (a < dram_buffer_.size()) {
        auto* first = reinterpret_cast<VTAMemInsn*>(&dram_buffer_[a]);
        if (bits(p, 1, 0, 0, 0) && first->opcode == VTA_OPCODE_LOAD &&
            first->memory_type == VTA_MEM_ID_UOP && first->x_size > 0 &&
            first->y_size > 0 && bits(first, 1, 0, 0, 0) && b < dram_buffer_.size()) {
          auto* second = reinterpret_cast<VTAGemInsn*>(&dram_buffer_[b]);
          if (second->opcode == VTA_OPCODE_GEMM && !second->reset_reg &&
              !second->pop_prev_dep && !second->pop_next_dep && !second->push_next_dep) {
            second->pop_prev_dep = 1;
            erased = true;
          }
        } else if (bits(p, 0, 0, 1, 0) && first->opcode == VTA_OPCODE_LOAD &&
                   first->memory_type == VTA_MEM_ID_ACC && first->x_size > 0 &&
                   first->y_size > 0 && bits(first, 0, 0, 0, 0)) {
          first->push_prev_dep = 1;
          erased = true;
        }
      }
    }
    if (erased) dram_buffer_.erase(dram_buffer_.begin() + i);
    else ++i;
  }
}
