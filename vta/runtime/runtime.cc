/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file runtime.cc
 * \brief Generic VTA runtime in C++11.
 *
 *  The runtime depends on specific instruction
 *  stream spec as specified in hw_spec.h
 */
#include "runtime.h"

#include <chrono>
#include <dmlc/logging.h>
#include <malloc.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <vta/driver.h>
#include <vta/hw_spec.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vta {

namespace {

constexpr size_t kMaxTrackedDirtyRanges = 32;
constexpr size_t kSmallDMATransferBytes = 4096;
constexpr size_t kRuntimeMemoryTypeBuckets = 7;
constexpr size_t kDMATopSignatureCount = 3;
constexpr size_t kMaxRuntimeEvents = 4096;

enum class CopyScopeKind : int {
  kNone = 0,
  kInput = 1,
  kOutput = 2,
  kDeviceCopy = 3,
};

using DirtyRange = std::pair<size_t, size_t>;

thread_local std::vector<CopyScopeKind> copy_scope_stack;

inline CopyScopeKind ParseCopyScope(const std::string& scope) {
  if (scope == "input") return CopyScopeKind::kInput;
  if (scope == "output") return CopyScopeKind::kOutput;
  if (scope == "device_copy") return CopyScopeKind::kDeviceCopy;
  return CopyScopeKind::kNone;
}

inline void PushCopyScope(CopyScopeKind scope) { copy_scope_stack.push_back(scope); }

inline void PopCopyScope() {
  if (!copy_scope_stack.empty()) {
    copy_scope_stack.pop_back();
  }
}

inline size_t MemoryTypeStatIndex(uint32_t memory_type) {
  switch (memory_type) {
    case VTA_MEM_ID_UOP:
      return 0;
    case VTA_MEM_ID_INP:
      return 1;
    case VTA_MEM_ID_WGT:
      return 2;
    case VTA_MEM_ID_ACC:
      return 3;
    case VTA_MEM_ID_OUT:
      return 4;
    case VTA_MEM_ID_ACC_8BIT:
      return 5;
    default:
      return 6;
  }
}

inline const char* MemoryTypeStatName(size_t index) {
  switch (index) {
    case 0:
      return "uop";
    case 1:
      return "inp";
    case 2:
      return "wgt";
    case 3:
      return "acc";
    case 4:
      return "out";
    case 5:
      return "acc8";
    default:
      return "unknown";
  }
}

inline uint32_t MemoryTypeElemBytes(uint32_t memory_type) {
  switch (memory_type) {
    case VTA_MEM_ID_UOP:
      return VTA_UOP_ELEM_BYTES;
    case VTA_MEM_ID_INP:
      return VTA_INP_ELEM_BYTES;
    case VTA_MEM_ID_WGT:
      return VTA_WGT_ELEM_BYTES;
    case VTA_MEM_ID_ACC:
      return VTA_ACC_ELEM_BYTES;
    case VTA_MEM_ID_OUT:
      return VTA_OUT_ELEM_BYTES;
    case VTA_MEM_ID_ACC_8BIT:
      return VTA_ACC_ELEM_BYTES / 4;
    default:
      return 1;
  }
}

inline uint64_t EncodeDMASignature(uint32_t memory_type, uint32_t x_size, uint32_t y_size,
                                   uint32_t x_stride) {
  return (static_cast<uint64_t>(memory_type & 0xffu) << 48) |
         (static_cast<uint64_t>(x_size & 0xffffu) << 32) |
         (static_cast<uint64_t>(y_size & 0xffffu) << 16) |
         static_cast<uint64_t>(x_stride & 0xffffu);
}

inline std::string DescribeDMASignature(uint64_t sig, uint64_t count) {
  uint32_t memory_type = static_cast<uint32_t>((sig >> 48) & 0xffu);
  uint32_t x_size = static_cast<uint32_t>((sig >> 32) & 0xffffu);
  uint32_t y_size = static_cast<uint32_t>((sig >> 16) & 0xffffu);
  uint32_t x_stride = static_cast<uint32_t>(sig & 0xffffu);
  std::ostringstream os;
  os << MemoryTypeStatName(MemoryTypeStatIndex(memory_type)) << ":x=" << x_size << ",y=" << y_size
     << ",stride=" << x_stride << ",count=" << count;
  return os.str();
}

inline std::string JSONQuote(const std::string& s) {
  std::ostringstream os;
  os << "\"";
  for (char ch : s) {
    switch (ch) {
      case '"':
        os << "\\\"";
        break;
      case '\\':
        os << "\\\\";
        break;
      default:
        os << ch;
        break;
    }
  }
  os << "\"";
  return os.str();
}

inline CopyScopeKind CurrentCopyScope() {
  return copy_scope_stack.empty() ? CopyScopeKind::kNone : copy_scope_stack.back();
}

inline bool LazyCoherenceEnabled() {
  static const bool enabled = []() {
    const char* v = getenv("VTA_LAZY_COHERENCE");
    return !(v && v[0] != '\0' && strcmp(v, "0") == 0);
  }();
  return enabled;
}

inline size_t DirtyRangeBytes(const DirtyRange& range) { return range.second - range.first; }

inline DirtyRange MakeDirtyRange(size_t begin, size_t size) { return {begin, begin + size}; }

inline void InsertDirtyRange(std::vector<DirtyRange>* ranges, DirtyRange range, size_t buffer_size) {
  if (range.first >= range.second) return;
  std::vector<DirtyRange> out;
  bool inserted = false;
  for (const DirtyRange& curr : *ranges) {
    if (curr.second < range.first) {
      out.push_back(curr);
    } else if (range.second < curr.first) {
      if (!inserted) {
        out.push_back(range);
        inserted = true;
      }
      out.push_back(curr);
    } else {
      range.first = std::min(range.first, curr.first);
      range.second = std::max(range.second, curr.second);
    }
  }
  if (!inserted) {
    out.push_back(range);
  }
  if (out.size() > kMaxTrackedDirtyRanges) {
    out.clear();
    out.push_back({0, buffer_size});
  }
  *ranges = std::move(out);
}

inline void RemoveDirtyRange(std::vector<DirtyRange>* ranges, DirtyRange range) {
  if (range.first >= range.second) return;
  std::vector<DirtyRange> out;
  for (const DirtyRange& curr : *ranges) {
    if (curr.second <= range.first || range.second <= curr.first) {
      out.push_back(curr);
      continue;
    }
    if (curr.first < range.first) {
      out.push_back({curr.first, range.first});
    }
    if (range.second < curr.second) {
      out.push_back({range.second, curr.second});
    }
  }
  *ranges = std::move(out);
}

inline size_t Compute2DSpanBytes(uint32_t elem_bytes, uint32_t elem_offset, uint32_t x_size,
                                 uint32_t y_size, uint32_t x_stride) {
  if (x_size == 0 || y_size == 0) return 0;
  uint64_t row_end = static_cast<uint64_t>(elem_offset) +
                     static_cast<uint64_t>(x_stride) * (static_cast<uint64_t>(y_size) - 1) +
                     static_cast<uint64_t>(x_size);
  return static_cast<size_t>(row_end * elem_bytes) - static_cast<size_t>(elem_offset) * elem_bytes;
}

inline VTADriverProfilerStats DiffDriverStats(const VTADriverProfilerStats& after,
                                              const VTADriverProfilerStats& before) {
  VTADriverProfilerStats out{};
  out.run_calls = after.run_calls - before.run_calls;
  out.run_insns = after.run_insns - before.run_insns;
  out.timeout_calls = after.timeout_calls - before.timeout_calls;
  out.poll_iters = after.poll_iters - before.poll_iters;
  out.run_total_us = after.run_total_us - before.run_total_us;
  out.submit_mmio_us = after.submit_mmio_us - before.submit_mmio_us;
  out.post_start_sleep_us = after.post_start_sleep_us - before.post_start_sleep_us;
  out.poll_wait_us = after.poll_wait_us - before.poll_wait_us;
  return out;
}

}  // namespace

inline CopyScopeKind CopyScopeFromString(const std::string& scope) { return ParseCopyScope(scope); }

inline void PushScopedCopy(CopyScopeKind scope) { PushCopyScope(scope); }

inline void PopScopedCopy() { PopCopyScope(); }

inline CopyScopeKind CurrentScopedCopy() { return CurrentCopyScope(); }

inline double NowMicros();

class RuntimeProfiler {
 public:
  enum class EventKind : int {
    kBufferCopy = 0,
    kMemCopyFromHost = 1,
    kMemCopyToHost = 2,
    kFlushCache = 3,
    kInvalidateCache = 4,
    kLoadBuffer2D = 5,
    kStoreBuffer2D = 6,
    kPushGEMMOp = 7,
    kPushALUOp = 8,
    kSynchronize = 9,
    kTemplateCapture = 10,
    kTemplateReplay = 11,
    kTemplateReplayFallback = 12,
  };

  struct Stats {
    uint64_t mem_copy_from_host_calls{0};
    uint64_t mem_copy_from_host_bytes{0};
    double mem_copy_from_host_us{0.0};

    uint64_t mem_copy_to_host_calls{0};
    uint64_t mem_copy_to_host_bytes{0};
    double mem_copy_to_host_us{0.0};

    uint64_t flush_cache_calls{0};
    uint64_t flush_cache_bytes{0};
    double flush_cache_us{0.0};

    uint64_t invalidate_cache_calls{0};
    uint64_t invalidate_cache_bytes{0};
    double invalidate_cache_us{0.0};

    uint64_t load_buffer_2d_calls{0};
    uint64_t load_buffer_2d_bytes{0};
    uint64_t load_buffer_2d_small_calls{0};
    uint64_t load_buffer_2d_strided_calls{0};
    uint64_t load_buffer_2d_padded_calls{0};
    uint64_t load_buffer_2d_xsize1_calls{0};
    uint64_t load_buffer_2d_ysize1_calls{0};
    double load_buffer_2d_enqueue_us{0.0};
    std::array<uint64_t, kRuntimeMemoryTypeBuckets> load_buffer_2d_mem_type_calls{};
    std::array<uint64_t, kRuntimeMemoryTypeBuckets> load_buffer_2d_mem_type_bytes{};
    uint64_t store_buffer_2d_calls{0};
    uint64_t store_buffer_2d_bytes{0};
    uint64_t store_buffer_2d_small_calls{0};
    uint64_t store_buffer_2d_strided_calls{0};
    uint64_t store_buffer_2d_xsize1_calls{0};
    uint64_t store_buffer_2d_ysize1_calls{0};
    double store_buffer_2d_enqueue_us{0.0};
    std::array<uint64_t, kRuntimeMemoryTypeBuckets> store_buffer_2d_mem_type_calls{};
    std::array<uint64_t, kRuntimeMemoryTypeBuckets> store_buffer_2d_mem_type_bytes{};
    uint64_t push_gemm_op_calls{0};
    double push_gemm_op_us{0.0};
    uint64_t push_alu_op_calls{0};
    double push_alu_op_us{0.0};
    uint64_t template_capture_calls{0};
    uint64_t template_replay_hits{0};
    uint64_t template_replay_fallbacks{0};
    double template_capture_us{0.0};
    double template_replay_us{0.0};
    uint64_t input_copy_calls{0};
    uint64_t input_copy_bytes{0};
    double input_copy_us{0.0};

    uint64_t output_copy_calls{0};
    uint64_t output_copy_bytes{0};
    double output_copy_us{0.0};

    uint64_t device_copy_calls{0};
    uint64_t device_copy_bytes{0};
    double device_copy_us{0.0};

    uint64_t synchronize_calls{0};
    uint64_t synchronize_insns{0};
    uint64_t synchronize_load_bytes{0};
    uint64_t synchronize_store_bytes{0};
    double device_run_wait_us{0.0};
    uint64_t driver_run_calls{0};
    uint64_t driver_run_insns{0};
    uint64_t driver_timeout_calls{0};
    uint64_t driver_poll_iters{0};
    double driver_run_total_us{0.0};
    double driver_submit_mmio_us{0.0};
    double driver_post_start_sleep_us{0.0};
    double driver_poll_wait_us{0.0};
  };

  struct Event {
    uint64_t seq{0};
    double ts_us{0.0};
    EventKind kind{EventKind::kBufferCopy};
    CopyScopeKind copy_scope{CopyScopeKind::kNone};
    size_t bytes{0};
    double duration_us{0.0};
    uint64_t insns{0};
    size_t load_bytes{0};
    size_t store_bytes{0};
    uint32_t memory_type{0};
    uint32_t x_size{0};
    uint32_t y_size{0};
    uint32_t x_stride{0};
    bool is_padded{false};
    VTADriverProfilerStats driver{};
  };

  static RuntimeProfiler& Global() {
    static RuntimeProfiler inst;
    return inst;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_ = Stats();
    load_signature_counts_.clear();
    store_signature_counts_.clear();
    events_.clear();
    next_event_seq_ = 1;
    VTADriverProfilerClear();
  }

  void AddMemCopyFromHost(size_t size, double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.mem_copy_from_host_calls += 1;
    stats_.mem_copy_from_host_bytes += size;
    stats_.mem_copy_from_host_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kMemCopyFromHost, size, us));
  }

  void AddMemCopyToHost(size_t size, double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.mem_copy_to_host_calls += 1;
    stats_.mem_copy_to_host_bytes += size;
    stats_.mem_copy_to_host_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kMemCopyToHost, size, us));
  }

  void AddFlushCache(size_t size, double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.flush_cache_calls += 1;
    stats_.flush_cache_bytes += size;
    stats_.flush_cache_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kFlushCache, size, us));
  }

  void AddInvalidateCache(size_t size, double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.invalidate_cache_calls += 1;
    stats_.invalidate_cache_bytes += size;
    stats_.invalidate_cache_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kInvalidateCache, size, us));
  }

  void AddLoadBuffer2D(uint32_t memory_type, uint32_t x_size, uint32_t y_size, uint32_t x_stride,
                       bool is_padded, double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    RecordLoadBuffer2D(&stats_, &load_signature_counts_, memory_type, x_size, y_size, x_stride,
                       is_padded);
    stats_.load_buffer_2d_enqueue_us += us;
    Event event;
    event.kind = EventKind::kLoadBuffer2D;
    event.duration_us = us;
    event.bytes =
        static_cast<size_t>(x_size) * y_size * static_cast<size_t>(MemoryTypeElemBytes(memory_type));
    event.memory_type = memory_type;
    event.x_size = x_size;
    event.y_size = y_size;
    event.x_stride = x_stride;
    event.is_padded = is_padded;
    PushEventLocked(event);
  }

  void AddStoreBuffer2D(uint32_t memory_type, uint32_t x_size, uint32_t y_size, uint32_t x_stride,
                        double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    RecordStoreBuffer2D(&stats_, &store_signature_counts_, memory_type, x_size, y_size, x_stride);
    stats_.store_buffer_2d_enqueue_us += us;
    Event event;
    event.kind = EventKind::kStoreBuffer2D;
    event.duration_us = us;
    event.bytes =
        static_cast<size_t>(x_size) * y_size * static_cast<size_t>(MemoryTypeElemBytes(memory_type));
    event.memory_type = memory_type;
    event.x_size = x_size;
    event.y_size = y_size;
    event.x_stride = x_stride;
    PushEventLocked(event);
  }

  void AddPushGEMMOp(double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.push_gemm_op_calls += 1;
    stats_.push_gemm_op_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kPushGEMMOp, 0, us));
  }

  void AddPushALUOp(double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.push_alu_op_calls += 1;
    stats_.push_alu_op_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kPushALUOp, 0, us));
  }

  void AddTemplateCapture(double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.template_capture_calls += 1;
    stats_.template_capture_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kTemplateCapture, 0, us));
  }

  void AddTemplateReplay(double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.template_replay_hits += 1;
    stats_.template_replay_us += us;
    PushEventLocked(MakeSizeEvent(EventKind::kTemplateReplay, 0, us));
  }

  void AddTemplateReplayFallback() {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.template_replay_fallbacks += 1;
    PushEventLocked(MakeSizeEvent(EventKind::kTemplateReplayFallback, 0, 0.0));
  }

  void AddScopedCopy(CopyScopeKind scope, size_t size, double us) {
    std::lock_guard<std::mutex> lock(mtx_);
    switch (scope) {
      case CopyScopeKind::kInput:
        stats_.input_copy_calls += 1;
        stats_.input_copy_bytes += size;
        stats_.input_copy_us += us;
        break;
      case CopyScopeKind::kOutput:
        stats_.output_copy_calls += 1;
        stats_.output_copy_bytes += size;
        stats_.output_copy_us += us;
        break;
      case CopyScopeKind::kDeviceCopy:
        stats_.device_copy_calls += 1;
        stats_.device_copy_bytes += size;
        stats_.device_copy_us += us;
        break;
      case CopyScopeKind::kNone:
        break;
    }
    Event event = MakeSizeEvent(EventKind::kBufferCopy, size, us);
    event.copy_scope = scope;
    PushEventLocked(event);
  }

  void AddSynchronize(uint64_t insns, size_t load_bytes, size_t store_bytes, double us,
                      const VTADriverProfilerStats& driver_delta) {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.synchronize_calls += 1;
    stats_.synchronize_insns += insns;
    stats_.synchronize_load_bytes += load_bytes;
    stats_.synchronize_store_bytes += store_bytes;
    stats_.device_run_wait_us += us;
    stats_.driver_run_calls += driver_delta.run_calls;
    stats_.driver_run_insns += driver_delta.run_insns;
    stats_.driver_timeout_calls += driver_delta.timeout_calls;
    stats_.driver_poll_iters += driver_delta.poll_iters;
    stats_.driver_run_total_us += driver_delta.run_total_us;
    stats_.driver_submit_mmio_us += driver_delta.submit_mmio_us;
    stats_.driver_post_start_sleep_us += driver_delta.post_start_sleep_us;
    stats_.driver_poll_wait_us += driver_delta.poll_wait_us;
    Event event;
    event.kind = EventKind::kSynchronize;
    event.duration_us = us;
    event.insns = insns;
    event.load_bytes = load_bytes;
    event.store_bytes = store_bytes;
    event.driver = driver_delta;
    PushEventLocked(event);
  }

  std::string AsJSON() {
    std::lock_guard<std::mutex> lock(mtx_);
    const Stats& s = stats_;
    std::ostringstream os;
    os << "{"
       << "\"mem_copy_from_host_calls\":" << s.mem_copy_from_host_calls << ","
       << "\"mem_copy_from_host_bytes\":" << s.mem_copy_from_host_bytes << ","
       << "\"mem_copy_from_host_us\":" << s.mem_copy_from_host_us << ","
       << "\"mem_copy_to_host_calls\":" << s.mem_copy_to_host_calls << ","
       << "\"mem_copy_to_host_bytes\":" << s.mem_copy_to_host_bytes << ","
       << "\"mem_copy_to_host_us\":" << s.mem_copy_to_host_us << ","
       << "\"flush_cache_calls\":" << s.flush_cache_calls << ","
       << "\"flush_cache_bytes\":" << s.flush_cache_bytes << ","
       << "\"flush_cache_us\":" << s.flush_cache_us << ","
       << "\"invalidate_cache_calls\":" << s.invalidate_cache_calls << ","
       << "\"invalidate_cache_bytes\":" << s.invalidate_cache_bytes << ","
       << "\"invalidate_cache_us\":" << s.invalidate_cache_us << ","
       << "\"load_buffer_2d_calls\":" << s.load_buffer_2d_calls << ","
       << "\"load_buffer_2d_bytes\":" << s.load_buffer_2d_bytes << ","
       << "\"load_buffer_2d_small_calls\":" << s.load_buffer_2d_small_calls << ","
       << "\"load_buffer_2d_strided_calls\":" << s.load_buffer_2d_strided_calls << ","
       << "\"load_buffer_2d_xsize1_calls\":" << s.load_buffer_2d_xsize1_calls << ","
       << "\"load_buffer_2d_ysize1_calls\":" << s.load_buffer_2d_ysize1_calls << ","
       << "\"load_buffer_2d_padded_calls\":" << s.load_buffer_2d_padded_calls << ","
       << "\"load_buffer_2d_enqueue_us\":" << s.load_buffer_2d_enqueue_us << ","
       << "\"store_buffer_2d_calls\":" << s.store_buffer_2d_calls << ","
       << "\"store_buffer_2d_bytes\":" << s.store_buffer_2d_bytes << ","
       << "\"store_buffer_2d_small_calls\":" << s.store_buffer_2d_small_calls << ","
       << "\"store_buffer_2d_strided_calls\":" << s.store_buffer_2d_strided_calls << ","
       << "\"store_buffer_2d_xsize1_calls\":" << s.store_buffer_2d_xsize1_calls << ","
       << "\"store_buffer_2d_ysize1_calls\":" << s.store_buffer_2d_ysize1_calls << ","
       << "\"store_buffer_2d_enqueue_us\":" << s.store_buffer_2d_enqueue_us << ","
       << "\"push_gemm_op_calls\":" << s.push_gemm_op_calls << ","
       << "\"push_gemm_op_us\":" << s.push_gemm_op_us << ","
       << "\"push_alu_op_calls\":" << s.push_alu_op_calls << ","
       << "\"push_alu_op_us\":" << s.push_alu_op_us << ","
       << "\"template_capture_calls\":" << s.template_capture_calls << ","
       << "\"template_replay_hits\":" << s.template_replay_hits << ","
       << "\"template_replay_fallbacks\":" << s.template_replay_fallbacks << ","
       << "\"template_capture_us\":" << s.template_capture_us << ","
       << "\"template_replay_us\":" << s.template_replay_us << ","
       << "\"input_copy_calls\":" << s.input_copy_calls << ","
       << "\"input_copy_bytes\":" << s.input_copy_bytes << ","
       << "\"input_copy_us\":" << s.input_copy_us << ","
       << "\"output_copy_calls\":" << s.output_copy_calls << ","
       << "\"output_copy_bytes\":" << s.output_copy_bytes << ","
       << "\"output_copy_us\":" << s.output_copy_us << ","
       << "\"device_copy_calls\":" << s.device_copy_calls << ","
       << "\"device_copy_bytes\":" << s.device_copy_bytes << ","
       << "\"device_copy_us\":" << s.device_copy_us << ","
       << "\"synchronize_calls\":" << s.synchronize_calls << ","
       << "\"synchronize_insns\":" << s.synchronize_insns << ","
       << "\"synchronize_load_bytes\":" << s.synchronize_load_bytes << ","
       << "\"synchronize_store_bytes\":" << s.synchronize_store_bytes << ","
       << "\"device_run_wait_us\":" << s.device_run_wait_us << ","
       << "\"driver_run_calls\":" << s.driver_run_calls << ","
       << "\"driver_run_insns\":" << s.driver_run_insns << ","
       << "\"driver_timeout_calls\":" << s.driver_timeout_calls << ","
       << "\"driver_poll_iters\":" << s.driver_poll_iters << ","
       << "\"driver_run_total_us\":" << s.driver_run_total_us << ","
       << "\"driver_submit_mmio_us\":" << s.driver_submit_mmio_us << ","
       << "\"driver_post_start_sleep_us\":" << s.driver_post_start_sleep_us << ","
       << "\"driver_poll_wait_us\":" << s.driver_poll_wait_us << ","
       << "\"event_count\":" << events_.size();
    for (size_t i = 0; i < kRuntimeMemoryTypeBuckets; ++i) {
      os << ",\"load_buffer_2d_" << MemoryTypeStatName(i) << "_calls\":"
         << s.load_buffer_2d_mem_type_calls[i];
      os << ",\"load_buffer_2d_" << MemoryTypeStatName(i) << "_bytes\":"
         << s.load_buffer_2d_mem_type_bytes[i];
      os << ",\"store_buffer_2d_" << MemoryTypeStatName(i) << "_calls\":"
         << s.store_buffer_2d_mem_type_calls[i];
      os << ",\"store_buffer_2d_" << MemoryTypeStatName(i) << "_bytes\":"
         << s.store_buffer_2d_mem_type_bytes[i];
    }
    os << ",\"load_buffer_2d_top_signatures\":";
    AppendTopSignaturesJSON(os, load_signature_counts_);
    os << ",\"store_buffer_2d_top_signatures\":";
    AppendTopSignaturesJSON(os, store_signature_counts_);
    os << "}";
    return os.str();
  }

  std::string EventsAsJSON(int max_events) {
    std::lock_guard<std::mutex> lock(mtx_);
    size_t keep = events_.size();
    if (max_events > 0) {
      keep = std::min(keep, static_cast<size_t>(max_events));
    }
    size_t begin = events_.size() - keep;
    std::ostringstream os;
    os << "[";
    for (size_t i = begin; i < events_.size(); ++i) {
      if (i != begin) os << ",";
      AppendEventJSON(os, events_[i]);
    }
    os << "]";
    return os.str();
  }

 private:
  static const char* EventKindName(EventKind kind) {
    switch (kind) {
      case EventKind::kBufferCopy:
        return "buffer_copy";
      case EventKind::kMemCopyFromHost:
        return "mem_copy_from_host";
      case EventKind::kMemCopyToHost:
        return "mem_copy_to_host";
      case EventKind::kFlushCache:
        return "flush_cache";
      case EventKind::kInvalidateCache:
        return "invalidate_cache";
      case EventKind::kLoadBuffer2D:
        return "load_buffer_2d";
      case EventKind::kStoreBuffer2D:
        return "store_buffer_2d";
      case EventKind::kPushGEMMOp:
        return "push_gemm_op";
      case EventKind::kPushALUOp:
        return "push_alu_op";
      case EventKind::kSynchronize:
        return "synchronize";
      case EventKind::kTemplateCapture:
        return "template_capture";
      case EventKind::kTemplateReplay:
        return "template_replay";
      case EventKind::kTemplateReplayFallback:
        return "template_replay_fallback";
    }
    return "unknown";
  }

  static const char* CopyScopeName(CopyScopeKind scope) {
    switch (scope) {
      case CopyScopeKind::kInput:
        return "input";
      case CopyScopeKind::kOutput:
        return "output";
      case CopyScopeKind::kDeviceCopy:
        return "device_copy";
      case CopyScopeKind::kNone:
        return "none";
    }
    return "none";
  }

  static Event MakeSizeEvent(EventKind kind, size_t bytes, double us) {
    Event event;
    event.kind = kind;
    event.bytes = bytes;
    event.duration_us = us;
    return event;
  }

  void PushEventLocked(Event event) {
    event.seq = next_event_seq_++;
    event.ts_us = NowMicros();
    if (events_.size() == kMaxRuntimeEvents) {
      events_.pop_front();
    }
    events_.push_back(std::move(event));
  }

  static void AppendDriverJSON(std::ostringstream& os, const VTADriverProfilerStats& driver) {
    os << "{"
       << "\"run_calls\":" << driver.run_calls << ","
       << "\"run_insns\":" << driver.run_insns << ","
       << "\"timeout_calls\":" << driver.timeout_calls << ","
       << "\"poll_iters\":" << driver.poll_iters << ","
       << "\"run_total_us\":" << driver.run_total_us << ","
       << "\"submit_mmio_us\":" << driver.submit_mmio_us << ","
       << "\"post_start_sleep_us\":" << driver.post_start_sleep_us << ","
       << "\"poll_wait_us\":" << driver.poll_wait_us << "}";
  }

  static void AppendEventJSON(std::ostringstream& os, const Event& event) {
    os << "{"
       << "\"seq\":" << event.seq << ","
       << "\"ts_us\":" << event.ts_us << ","
       << "\"kind\":" << JSONQuote(EventKindName(event.kind)) << ","
       << "\"scope\":" << JSONQuote(CopyScopeName(event.copy_scope)) << ","
       << "\"bytes\":" << event.bytes << ","
       << "\"duration_us\":" << event.duration_us << ","
       << "\"insns\":" << event.insns << ","
       << "\"load_bytes\":" << event.load_bytes << ","
       << "\"store_bytes\":" << event.store_bytes << ","
       << "\"memory_type\":" << event.memory_type << ","
       << "\"x_size\":" << event.x_size << ","
       << "\"y_size\":" << event.y_size << ","
       << "\"x_stride\":" << event.x_stride << ","
       << "\"is_padded\":" << (event.is_padded ? "true" : "false") << ","
       << "\"driver\":";
    AppendDriverJSON(os, event.driver);
    os << "}";
  }

  static void UpdateDMABuffer2D(std::array<uint64_t, kRuntimeMemoryTypeBuckets>* mem_type_calls,
                                std::array<uint64_t, kRuntimeMemoryTypeBuckets>* mem_type_bytes,
                                std::unordered_map<uint64_t, uint64_t>* signatures,
                                uint64_t* calls, uint64_t* bytes, uint64_t* small_calls,
                                uint64_t* strided_calls, uint64_t* padded_calls,
                                uint64_t* xsize1_calls, uint64_t* ysize1_calls,
                                uint32_t memory_type, uint32_t x_size, uint32_t y_size,
                                uint32_t x_stride, bool is_padded, int64_t sign) {
    const int64_t transfer_bytes =
        static_cast<int64_t>(static_cast<uint64_t>(x_size) * static_cast<uint64_t>(y_size) *
                             static_cast<uint64_t>(MemoryTypeElemBytes(memory_type)));
    *calls = static_cast<uint64_t>(static_cast<int64_t>(*calls) + sign);
    *bytes = static_cast<uint64_t>(static_cast<int64_t>(*bytes) + transfer_bytes * sign);
    *small_calls = static_cast<uint64_t>(
        static_cast<int64_t>(*small_calls) + (transfer_bytes < static_cast<int64_t>(kSmallDMATransferBytes) ? sign : 0));
    *strided_calls = static_cast<uint64_t>(
        static_cast<int64_t>(*strided_calls) + (x_stride != x_size ? sign : 0));
    if (padded_calls != nullptr) {
      *padded_calls = static_cast<uint64_t>(static_cast<int64_t>(*padded_calls) +
                                            (is_padded ? sign : 0));
    }
    *xsize1_calls = static_cast<uint64_t>(
        static_cast<int64_t>(*xsize1_calls) + (x_size == 1 ? sign : 0));
    *ysize1_calls = static_cast<uint64_t>(
        static_cast<int64_t>(*ysize1_calls) + (y_size == 1 ? sign : 0));
    size_t memory_index = MemoryTypeStatIndex(memory_type);
    (*mem_type_calls)[memory_index] = static_cast<uint64_t>(
        static_cast<int64_t>((*mem_type_calls)[memory_index]) + sign);
    (*mem_type_bytes)[memory_index] = static_cast<uint64_t>(
        static_cast<int64_t>((*mem_type_bytes)[memory_index]) + transfer_bytes * sign);
    uint64_t signature = EncodeDMASignature(memory_type, x_size, y_size, x_stride);
    auto it = signatures->find(signature);
    if (sign > 0) {
      if (it == signatures->end()) {
        signatures->emplace(signature, static_cast<uint64_t>(sign));
      } else {
        it->second += static_cast<uint64_t>(sign);
      }
    } else if (it != signatures->end()) {
      uint64_t delta = static_cast<uint64_t>(-sign);
      if (it->second <= delta) {
        signatures->erase(it);
      } else {
        it->second -= delta;
      }
    }
  }

  static void RecordLoadBuffer2D(Stats* stats, std::unordered_map<uint64_t, uint64_t>* signatures,
                                 uint32_t memory_type, uint32_t x_size, uint32_t y_size,
                                 uint32_t x_stride, bool is_padded) {
    UpdateDMABuffer2D(&stats->load_buffer_2d_mem_type_calls, &stats->load_buffer_2d_mem_type_bytes,
                      signatures, &stats->load_buffer_2d_calls, &stats->load_buffer_2d_bytes,
                      &stats->load_buffer_2d_small_calls, &stats->load_buffer_2d_strided_calls,
                      &stats->load_buffer_2d_padded_calls, &stats->load_buffer_2d_xsize1_calls,
                      &stats->load_buffer_2d_ysize1_calls, memory_type, x_size, y_size, x_stride,
                      is_padded, 1);
  }

  static void RecordStoreBuffer2D(Stats* stats, std::unordered_map<uint64_t, uint64_t>* signatures,
                                  uint32_t memory_type, uint32_t x_size, uint32_t y_size,
                                  uint32_t x_stride) {
    UpdateDMABuffer2D(&stats->store_buffer_2d_mem_type_calls,
                      &stats->store_buffer_2d_mem_type_bytes, signatures,
                      &stats->store_buffer_2d_calls, &stats->store_buffer_2d_bytes,
                      &stats->store_buffer_2d_small_calls, &stats->store_buffer_2d_strided_calls,
                      nullptr, &stats->store_buffer_2d_xsize1_calls,
                      &stats->store_buffer_2d_ysize1_calls, memory_type, x_size, y_size, x_stride,
                      false, 1);
  }

  static void AppendTopSignaturesJSON(std::ostringstream& os,
                                      const std::unordered_map<uint64_t, uint64_t>& signatures) {
    std::vector<std::pair<uint64_t, uint64_t>> items(signatures.begin(), signatures.end());
    std::sort(items.begin(), items.end(),
              [](const std::pair<uint64_t, uint64_t>& lhs,
                 const std::pair<uint64_t, uint64_t>& rhs) {
                if (lhs.second != rhs.second) return lhs.second > rhs.second;
                return lhs.first < rhs.first;
              });
    os << "[";
    for (size_t i = 0; i < std::min(items.size(), kDMATopSignatureCount); ++i) {
      if (i != 0) os << ",";
      os << JSONQuote(DescribeDMASignature(items[i].first, items[i].second));
    }
    os << "]";
  }

  std::mutex mtx_;
  Stats stats_;
  std::unordered_map<uint64_t, uint64_t> load_signature_counts_;
  std::unordered_map<uint64_t, uint64_t> store_signature_counts_;
  std::deque<Event> events_;
  uint64_t next_event_seq_{1};
};

inline double NowMicros() {
  using Clock = std::chrono::steady_clock;
  using Micros = std::chrono::duration<double, std::micro>;
  return std::chrono::duration_cast<Micros>(Clock::now().time_since_epoch()).count();
}

inline bool RuntimeTraceEnabled() {
  const char* v = getenv("VTA_RUNTIME_TRACE");
  return v && v[0] != '\0' && strcmp(v, "0") != 0;
}

inline void RuntimeTrace(const char* fmt, ...) {
  if (!RuntimeTraceEnabled()) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[vta_runtime][TRACE] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

// Avoid bad configurations.
static_assert(VTA_UOP_WIDTH == sizeof(VTAUop) * 8, "VTA_UOP_WIDTH do not match VTAUop size");

/*! \brief Enable coherent access of data buffers between VTA and CPU */
static const bool kBufferCoherent = VTA_COHERENT_ACCESSES;
/*! \brief Always cache buffers (otherwise, write back to DRAM from CPU) */
static const bool kAlwaysCache = true;

template <typename T, std::size_t N = ALLOC_ALIGNMENT>
class AlignmentAllocator : public std::allocator<T> {
 public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T* pointer;
  typedef const T* const_pointer;

  typedef T& reference;
  typedef const T& const_reference;

  inline AlignmentAllocator() throw() {}

  template <typename T2>
  inline AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw() {}

  inline ~AlignmentAllocator() throw() {}

  inline pointer address(reference r) { return &r; }

  inline const_pointer address(const_reference r) const { return &r; }

  inline pointer allocate(size_type n) { return (pointer)memalign(N, n * sizeof(value_type)); }

  inline void deallocate(pointer p, size_type) { free(p); }

  inline void construct(pointer p, const value_type& wert) { new (p) value_type(wert); }

  inline void destroy(pointer p) { p->~value_type(); }

  inline size_type max_size() const throw() { return size_type(-1) / sizeof(value_type); }

  template <typename T2>
  struct rebind {
    typedef AlignmentAllocator<T2, N> other;
  };

  bool operator!=(const AlignmentAllocator<T, N>& other) const { return !(*this == other); }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const AlignmentAllocator<T, N>& other) const { return true; }
};

class DeviceAllocStat {
 public:
  void AddAlloc(const void* ptr) {
    std::lock_guard<std::mutex> lock(mtx_);
    allocated_.insert(ptr);
  }

  bool CheckAlloc(const void* ptr) {
    std::lock_guard<std::mutex> lock(mtx_);
    return allocated_.count(ptr);
  }

  void DelAlloc(const void* ptr) {
    std::lock_guard<std::mutex> lock(mtx_);
    allocated_.erase(ptr);
  }

 private:
  std::set<const void*> allocated_;
  std::mutex mtx_;
};

// here we use a global variable to memorize the allocation stats
static std::shared_ptr<DeviceAllocStat> alloc_stat(new DeviceAllocStat());

/*!
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer {
  DataBuffer() { alloc_stat_ = alloc_stat; }

  /*! \return Virtual address of the data. */
  void* virt_addr() const { return data_; }
  /*! \return Physical address of the data. */
  vta_phy_addr_t phy_addr() const { return phy_addr_; }
  /*! \return Size of the buffer in bytes. */
  size_t size() const { return size_; }
  /*!
   * \brief Invalidate the cache of given location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void InvalidateCache(size_t offset, size_t size) {
    if (!kBufferCoherent && kAlwaysCache) {
      double t0 = NowMicros();
      VTAInvalidateCache(reinterpret_cast<char*>(data_) + offset, phy_addr_ + offset, size);
      RuntimeProfiler::Global().AddInvalidateCache(size, NowMicros() - t0);
    }
  }
  /*!
   * \brief Invalidate the cache of certain location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void FlushCache(size_t offset, size_t size) {
    if (!kBufferCoherent && kAlwaysCache) {
      double t0 = NowMicros();
      VTAFlushCache(reinterpret_cast<char*>(data_) + offset, phy_addr_ + offset, size);
      RuntimeProfiler::Global().AddFlushCache(size, NowMicros() - t0);
    }
  }

  void MarkHostWrite(size_t offset, size_t size) {
    if (size == 0) return;
    if (!LazyCoherenceEnabled()) {
      this->FlushCache(offset, size);
      return;
    }
    DirtyRange range = MakeDirtyRange(offset, size);
    RemoveDirtyRange(&fpga_dirty_ranges_, range);
    InsertDirtyRange(&cpu_dirty_ranges_, range, size_);
  }

  void MarkDeviceWrite(size_t offset, size_t size) {
    if (size == 0) return;
    if (!LazyCoherenceEnabled()) {
      this->InvalidateCache(offset, size);
      return;
    }
    DirtyRange range = MakeDirtyRange(offset, size);
    RemoveDirtyRange(&cpu_dirty_ranges_, range);
    InsertDirtyRange(&fpga_dirty_ranges_, range, size_);
  }

  void EnsureDeviceReadable(size_t offset, size_t size) {
    if (size == 0) return;
    if (!LazyCoherenceEnabled()) {
      this->FlushCache(offset, size);
      return;
    }
    DirtyRange range = MakeDirtyRange(offset, size);
    for (const DirtyRange& curr : cpu_dirty_ranges_) {
      size_t begin = std::max(curr.first, range.first);
      size_t end = std::min(curr.second, range.second);
      if (begin < end) {
        this->FlushCache(begin, end - begin);
      }
    }
    RemoveDirtyRange(&cpu_dirty_ranges_, range);
  }

  void EnsureHostReadable(size_t offset, size_t size) {
    if (size == 0) return;
    if (!LazyCoherenceEnabled()) {
      this->InvalidateCache(offset, size);
      return;
    }
    DirtyRange range = MakeDirtyRange(offset, size);
    for (const DirtyRange& curr : fpga_dirty_ranges_) {
      size_t begin = std::max(curr.first, range.first);
      size_t end = std::min(curr.second, range.second);
      if (begin < end) {
        this->InvalidateCache(begin, end - begin);
      }
    }
    RemoveDirtyRange(&fpga_dirty_ranges_, range);
  }
  /*!
   * \brief Performs a copy operation from host memory to buffer allocated with VTAMemAlloc.
   * \param dst The desination buffer in FPGA-accessible memory. Has to be allocated with
   * VTAMemAlloc(). \param src The source buffer in host memory. \param size Size of the region in
   * Bytes.
   */
  void MemCopyFromHost(void* dst, const void* src, size_t size) {
    double t0 = NowMicros();
    VTAMemCopyFromHost(dst, src, size);
    RuntimeProfiler::Global().AddMemCopyFromHost(size, NowMicros() - t0);
  }
  /*!
   * \brief Performs a copy operation from buffer allocated with VTAMemAlloc to host memory.
   * \param dst The desination buffer in host memory.
   * \param src The source buffer in FPGA-accessible memory. Has to be allocated with VTAMemAlloc().
   * \param size Size of the region in Bytes.
   */
  void MemCopyToHost(void* dst, const void* src, size_t size) {
    double t0 = NowMicros();
    VTAMemCopyToHost(dst, src, size);
    RuntimeProfiler::Global().AddMemCopyToHost(size, NowMicros() - t0);
  }
  /*!
   * \brief Allocate a buffer of a given size.
   * \param size The size of the buffer.
   */
  static DataBuffer* Alloc(size_t size) {
    RuntimeTrace("DataBuffer::Alloc begin size=%zu", size);
    void* data = VTAMemAlloc(size, kAlwaysCache);
    CHECK(data != nullptr);
    void* header = memalign(ALLOC_ALIGNMENT, sizeof(DataBuffer));
    CHECK(header != nullptr);
    DataBuffer* buffer = new (header) DataBuffer();
    buffer->data_ = data;
    buffer->phy_addr_ = VTAMemGetPhyAddr(data);
    buffer->size_ = size;
    RuntimeTrace("DataBuffer::Alloc done size=%zu data=%p phy=0x%llx", size, data,
                 static_cast<unsigned long long>(buffer->phy_addr_));

    alloc_stat->AddAlloc(buffer);
    return buffer;
  }
  /*!
   * \brief Free the data buffer.
   * \param buffer The buffer to be freed.
   */
  static void Free(DataBuffer* buffer) {
    alloc_stat->DelAlloc(buffer);
    VTAMemFree(buffer->data_);
    buffer->~DataBuffer();
    free(buffer);
  }
  /*!
   * \brief Create data buffer header from buffer ptr.
   * \param buffer The buffer pointer.
   * \return The corresponding data buffer header.
   */
  static DataBuffer* FromHandle(const void* buffer) {
    if (alloc_stat->CheckAlloc(buffer)) {
      return const_cast<DataBuffer*>(reinterpret_cast<const DataBuffer*>(buffer));
    } else {
      return nullptr;
    }
  }

 private:
  /*! \brief The internal data. */
  void* data_;
  /*! \brief The physical address of the buffer, excluding header. */
  vta_phy_addr_t phy_addr_;
  /*! \brief The size of the buffer in bytes. */
  size_t size_{0};
  /*! \brief Host-writable dirty ranges that are not yet visible to FPGA. */
  std::vector<DirtyRange> cpu_dirty_ranges_;
  /*! \brief Device-writable dirty ranges that are not yet visible to CPU. */
  std::vector<DirtyRange> fpga_dirty_ranges_;

  // a copy of global shared_ptr instance
  // to avoid the global instance is destructed before there are still some pending DataBuffers not
  // destructed
  std::shared_ptr<DeviceAllocStat> alloc_stat_;
};

/*!
 * \brief Micro op kernel.
 *  Contains functions to construct the kernel with prefix Push.
 */
class UopKernel {
 public:
  /*! \brief Loop information. */
  struct LoopEntry {
    uint32_t extent;
    uint32_t dst_factor;
    uint32_t src_factor;
    uint32_t wgt_factor;
  };
  /*!
   * \brief Construct UopKernel with signature.
   * \param signature The pointer to signature.
   * \param nbytes Number of bytes.
   */
  UopKernel(const char* signature, int nbytes) : signature_(signature, signature + nbytes) {}
  /*!
   * \brief Verify if the signature is correct.
   * \param signature Signature ptr.
   * \param nbytes Number of bytes.
   */
  bool MatchSignature(void* signature, int nbytes) const {
    if (static_cast<size_t>(nbytes) != signature_.size()) return false;
    return memcmp(signature, signature_.data(), nbytes) == 0;
  }
  /*! \return Whether the kernel is cached in SRAM. */
  bool cached() const { return sram_begin_ != sram_end_; }
  /*! \return The length of the micro op sequence. */
  size_t size() const { return seq_.size(); }
  /*! \return The micro-op data. */
  const VTAUop* data() const { return seq_.data(); }
  /*! \return The loop structure. */
  const std::vector<LoopEntry>& loop() const { return loop_; }
  /*!
   * \brief Declare loop start.
   * \param extent The loop extent.
   * \param dst_factor Loop factor of accum index.
   * \param src_factor Loop factor of input index
   * \param wgt_factor Loop factor of weight index.
   */
  void PushLoopBegin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                     uint32_t wgt_factor) {
    LoopEntry le;
    le.extent = extent;
    le.dst_factor = dst_factor;
    le.src_factor = src_factor;
    le.wgt_factor = wgt_factor;
    CHECK_EQ(seq_.size(), 0U);
    CHECK_LT(loop_.size(), 2U);
    loop_.push_back(le);
    ++loop_ptr_;
  }
  /*!
   * \brief Declare loop end.
   */
  void PushLoopEnd() { --loop_ptr_; }
  /*!
   * \brief Push micro op into kernel.
   * \param mode Set to GEMM mode if set to 0, ALU mode is set to 1.
   * \param reset_out Resets the accum to 0.
   * \param dst_index The accum memory index.
   * \param src_index The input memory (gemm) / accum memory (alu) index.
   * \param wgt_index The weight memory index.
   * \param opcode The ALU opcode.
   * \param use_imm Use immediate in ALU mode if set to true.
   * \param imm_val Immediate value in ALU mode.
   */
  void Push(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
            uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
    // The loop nest structure
    VerifyDep(dst_index);
    VTAUop op;
    op.dst_idx = dst_index;
    op.src_idx = src_index;
    op.wgt_idx = wgt_index;
    seq_.push_back(op);
    // Ensure that mode is consistent if set
    if (mode_ == 0xFFFFFFFF) {
      mode_ = mode;
    } else {
      CHECK(mode_ == mode);
    }
    // Set reset_out field if unset
    if (reset_out_ == 0xFFFFFFFF) {
      reset_out_ = reset_out;
    } else {
      CHECK(reset_out_ == reset_out);
    }
    // Check kernel op and imm/imm_val in ALU mode
    if (mode == 1) {
      if (opcode_ == 0xFFFFFFFF) {
        opcode_ = opcode;
        use_imm_ = use_imm;
        imm_val_ = imm_val;
      } else {
        CHECK(opcode_ == opcode);
        CHECK(use_imm_ == use_imm);
        CHECK(imm_val_ == imm_val);
      }
    }
  }
  /*! \brief Dump kernel micro ops to stdout. */
  void Dump() {
    uint32_t size = seq_.size();
    printf("There are %u uops\n", size);
    for (uint32_t i = 0; i < size; ++i) {
      printf("[%04u]\t acc=%u, inp=%u, wgt=%u\n", i, seq_[i].dst_idx, seq_[i].src_idx,
             seq_[i].wgt_idx);
    }
    printf("\n");
  }

 public:
  // The kernel's mode, opcode, immediate setting and value
  uint32_t mode_{0xFFFFFFFF};  // UOP type: 0xFFFFFFFF - unset, 0 - GEMM, 1 - ALU
  uint32_t opcode_{0xFFFFFFFF};
  uint32_t reset_out_{0xFFFFFFFF};
  bool use_imm_{false};
  int16_t imm_val_{0};

 private:
  // Verify that we don't write to the same acc_mem index two cycles in a row
  void VerifyDep(uint32_t dst_index) {
    size_t step = std::min(static_cast<size_t>(2U), seq_.size());
    for (size_t i = seq_.size() - step; i < seq_.size(); ++i) {
      CHECK(seq_[i].dst_idx != dst_index);
    }
  }
  // The uop buffer
  template <int, bool, bool>
  friend class UopQueue;
  friend class CommandQueue;
  // SRAM location if begin != end
  uint32_t sram_begin_{0};
  uint32_t sram_end_{0};
  // The signature used for verification
  std::vector<char> signature_;
  // Internal sequence
  std::vector<VTAUop> seq_;
  // The loop nest structure specific to ALU instructions
  std::vector<LoopEntry> loop_;
  // The loop pointer
  size_t loop_ptr_{0};
};

/*!
 * \brief Base class of all queues to send and recv serial data.
 */
template <class T>
class BaseQueue {
 public:
  virtual ~BaseQueue() {
    if (fpga_buff_ != nullptr) {
      VTAMemFree(fpga_buff_);
    }
  }
  /*! \return Content of DRAM buffer. */
  char* dram_buffer() const { return dram_buffer_; }
  /*! \return Physical address of DRAM. */
  vta_phy_addr_t dram_phy_addr() const {
    CHECK(fpga_buff_phy_);
    return fpga_buff_phy_;
  }
  /*! \return Whether there is pending information. */
  bool pending() const { return sram_begin_ != sram_end_; }
  /*! \brief Initialize the space of the buffer. */
  void InitSpace(uint32_t elem_bytes, uint32_t max_bytes, bool coherent, bool always_cache) {
    coherent_ = coherent;
    always_cache_ = always_cache;
    elem_bytes_ = elem_bytes;
    // Allocate buffer ahead of time
    fpga_buff_ = static_cast<char*>(VTAMemAlloc(max_bytes, coherent_ || always_cache_));
    CHECK(fpga_buff_ != nullptr);
    fpga_buff_phy_ = VTAMemGetPhyAddr(fpga_buff_);
    RuntimeTrace(
        "BaseQueue::InitSpace elem_bytes=%u max_bytes=%u coherent=%d always_cache=%d fpga=%p "
        "phy=0x%llx",
        elem_bytes, max_bytes, coherent ? 1 : 0, always_cache ? 1 : 0, fpga_buff_,
        static_cast<unsigned long long>(fpga_buff_phy_));
  }
  /*!
   * \brief Reset the pointer of the buffer.
   *  Set SRAM pointer to be the current end.
   */
  virtual void Reset() {
    dram_buffer_.clear();
    // reset to 0 as we always copy data to area starting from fpga_buff base
    // we do mem copy for every DeviceRun
    sram_end_ = 0;
    sram_begin_ = sram_end_;
  }

 protected:
  // Cache coherence access (shared memory only)
  bool coherent_{false};
  // Make the buffer cacheable
  bool always_cache_{false};
  // Element bytes
  uint32_t elem_bytes_{0};
  // Begin location of current SRAM read in FIFO mode
  uint32_t sram_begin_{0};
  // End location of current SRAM write in FIFO mode
  uint32_t sram_end_{0};
  // The buffer in DRAM
  std::vector<T, AlignmentAllocator<T, ALLOC_ALIGNMENT>> dram_buffer_;
  // FPGA accessible buffer
  void* fpga_buff_{NULL};
  // Physical address of the FPGA buffer
  vta_phy_addr_t fpga_buff_phy_{0};
};

/*!
 * \brief Micro op buffer that manages the micro op cache.
 */
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class UopQueue : public BaseQueue<VTAUop> {
 public:
  void InitSpace() { BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache); }
  // Push data to the queue
  template <typename FAutoSync>
  void Push(UopKernel* kernel, FAutoSync fautosync) {
    // if the micro-op is cached in VTA SRAM, skip
    if (kernel->cached()) return;
    // check if we've exceeded the size of the allocated FPGA readable buffer
    size_t num_op = kernel->size();
    if (dram_buffer_.size() + num_op > kMaxElems) {
      fautosync();
      CHECK(dram_buffer_.size() <= kMaxElems);
    }
    // Cannot have a micro-op kernel larger than SRAM buffer
    CHECK(num_op <= kMaxNumUop);
    uint32_t uop_begin = 0;
    if (sram_end_ + num_op > kMaxNumUop) {
      // Need to evict
      cache_idx_ = 0;
      sram_begin_ = 0;
      sram_end_ = num_op;
    } else {
      uop_begin = sram_end_;
      sram_end_ += num_op;
    }
    // Simple eviction policy
    uint32_t evict_begin = cache_idx_;
    for (; cache_idx_ < cache_.size(); ++cache_idx_) {
      if (cache_[cache_idx_]->sram_begin_ >= sram_end_) break;
      // Mark the kernel as "invalid"
      cache_[cache_idx_]->sram_begin_ = 0;
      cache_[cache_idx_]->sram_end_ = 0;
    }
    // Increase size of buffer
    kernel->sram_begin_ = uop_begin;
    kernel->sram_end_ = sram_end_;
    CHECK(kernel->cached());
    cache_.insert(cache_.begin() + cache_idx_, kernel);
    cache_.erase(cache_.begin() + evict_begin, cache_.begin() + cache_idx_);
    cache_idx_ = evict_begin + 1;
  }
  // Flush micro op load instruction
  void FlushUopLoad(VTAMemInsn* insn) {
    if (sram_begin_ != sram_end_) {
      // Derive offset in FPGA-readable buffer
      int32_t offset = 0;
      for (uint32_t i = 0; i < cache_idx_ - 1; ++i) {
        offset += cache_[i]->size() * kElemBytes;
      }
      insn->memory_type = VTA_MEM_ID_UOP;
      insn->sram_base = sram_begin_;
      // Update cache idx to physical address map
      insn->dram_base = (fpga_buff_phy_ + offset) / kElemBytes;
      insn->y_size = 1;
      insn->x_size = (sram_end_ - sram_begin_);
      insn->x_stride = (sram_end_ - sram_begin_);
      insn->y_pad_0 = 0;
      insn->y_pad_1 = 0;
      insn->x_pad_0 = 0;
      insn->x_pad_1 = 0;
      // Reset indices
      sram_begin_ = sram_end_;
    }
  }
  /*! \brief clear cache and reset base queue buffer.*/
  void Reset() {
    // unmark "cached" status
    // as we cannot assume it is still in SRAM across DeviceRun
    for (UopKernel* kernel : cache_) {
      kernel->sram_begin_ = 0;
      kernel->sram_end_ = 0;
    }

    cache_.clear();
    cache_idx_ = 0;
    BaseQueue<VTAUop>::Reset();
  }
  void AutoReadBarrier() { ReadBarrier(); }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier() {
    CHECK(fpga_buff_ != nullptr);
    CHECK(fpga_buff_phy_);
    // Iterate over caches; allocate buffer in FPGA-readable memory
    uint32_t buff_size = 0;
    for (uint32_t i = 0; i < cache_.size(); ++i) {
      buff_size += cache_[i]->size() * kElemBytes;
    }
    CHECK(buff_size <= kMaxBytes);

    // merge all the cache entries and do CopyFromHost once
    uint32_t total_size = 0;
    for (uint32_t i = 0; i < cache_.size(); ++i) {
      uint32_t ksize = cache_[i]->size() * kElemBytes;
      total_size += ksize;
    }

    char* lbuf = (char*)memalign(ALLOC_ALIGNMENT, total_size);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < cache_.size(); ++i) {
      uint32_t ksize = cache_[i]->size() * kElemBytes;
      memcpy(lbuf + offset, cache_[i]->data(), ksize);
      offset += ksize;
    }
    VTAMemCopyFromHost(static_cast<char*>(fpga_buff_), lbuf, total_size);
    free(lbuf);

    // Flush if we're using a shared memory system
    // and if interface is non-coherent
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_, fpga_buff_phy_, offset);
    }
  }

  std::vector<char> SerializeCacheBytes() const {
    uint32_t total_size = 0;
    for (const UopKernel* kernel : cache_) {
      total_size += kernel->size() * kElemBytes;
    }
    std::vector<char> out(total_size);
    uint32_t offset = 0;
    for (const UopKernel* kernel : cache_) {
      uint32_t ksize = kernel->size() * kElemBytes;
      memcpy(out.data() + offset, kernel->data(), ksize);
      offset += ksize;
    }
    return out;
  }

  void LoadSerializedBytes(const std::vector<char>& bytes) {
    if (bytes.empty()) return;
    CHECK_LE(bytes.size(), static_cast<size_t>(kMaxBytes));
    VTAMemCopyFromHost(fpga_buff_, bytes.data(), bytes.size());
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_, fpga_buff_phy_, bytes.size());
    }
  }

 private:
  // Cache pointer
  uint32_t cache_idx_{0};
  // Cached ring, sorted by sram_begin
  std::vector<UopKernel*> cache_;
  // Constants
  static constexpr int kElemBytes = sizeof(VTAUop);
  static constexpr int kMaxNumUop = VTA_UOP_BUFF_DEPTH;
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};

// Internal kernel structure
class UopKernelMap {
 public:
  // Simple hash map
  UopKernel** Get(void* signature, int nbytes) {
    uint32_t key = 0;
    CHECK(nbytes == 0 || nbytes == sizeof(int));
    if (nbytes == sizeof(int)) {
      memcpy(&key, signature, sizeof(int));
      key = key + 1;
    }
    CHECK_LT(key, 100);
    if (kmap_.size() <= key) {
      kmap_.resize(key + 1, nullptr);
    }
    return &(kmap_[key]);
  }

 private:
  std::vector<UopKernel*> kmap_;
};

enum PipelineStage : int { kNoneStage = 0, kLoadStage = 1, kComputeStage = 2, kStoreStage = 3 };

// Instruction Queue
template <int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue : public BaseQueue<VTAGenericInsn> {
 public:
  /*! \brief Initialize the space. */
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
    // Initialize the stage
    std::fill(pending_pop_prev_, pending_pop_prev_ + 4, 0);
    std::fill(pending_pop_next_, pending_pop_next_ + 4, 0);
  }
  /*! \return The data pointer. */
  VTAGenericInsn* data() { return dram_buffer_.data(); }
  /*! \return Number of instructions. */
  uint32_t count() { return dram_buffer_.size(); }
  // Insert dependency push of load
  void DepPop(int from, int to) {
    // NOTE: This instruction executes on queue[to]
    if (from < to) {
      if (pending_pop_prev_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_prev_[to] = 1;
    } else {
      if (pending_pop_next_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_next_[to] = 1;
    }
    // Impossible condition
    CHECK(from != kLoadStage || to != kStoreStage);
    CHECK(from != kStoreStage || to != kLoadStage);
  }
  // Insert dependency push of load
  void DepPush(int from, int to) {
    // NOTE: this instruction executes on queue[from]
    this->CommitPendingPop(from);
    if (!dram_buffer_.empty()) {
      VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(&dram_buffer_.back());
      if (GetPipelineStage(mptr) == from) {
        if (from < to && !mptr->push_next_dep) {
          // push(LD->C) or push(C->ST)
          mptr->push_next_dep = true;
          return;
        } else if (from > to && !mptr->push_prev_dep) {
          // push(C->LD) or push(ST->C)
          mptr->push_prev_dep = true;
          return;
        }
      }
    }
    if (from < to) {
      // Push next dep
      PushNoop(from, false, true, false, false);
    } else {
      // Push prev dep
      PushNoop(from, true, false, false, false);
    }
  }
  // Create a new instruction for a GEMM stage
  VTAGemInsn* CreateGemInsn() { return reinterpret_cast<VTAGemInsn*>(Create(kComputeStage)); }
  // Create a new instruction for a ALU stage
  VTAAluInsn* CreateAluInsn() { return reinterpret_cast<VTAAluInsn*>(Create(kComputeStage)); }
  // Create a new instruction for a memory stage
  VTAMemInsn* CreateMemInsn(int memory_type) {
    return reinterpret_cast<VTAMemInsn*>(Create(GetMemPipelineStage(memory_type)));
  }
  // create a new instruction for a store stage
  VTAMemInsn* CreateStoreInsn() { return reinterpret_cast<VTAMemInsn*>(Create(kStoreStage)); }
  // Rewrite instruction stream to force serial execution
  void RewriteForceSerial() {
    int insn_count = count();
    VTAMemInsn* mem_ptr = reinterpret_cast<VTAMemInsn*>(data());
    VTAMemInsn* mem_last_store_ptr = nullptr;
    VTAMemInsn* mem_last_ptr = nullptr;
    for (int i = 1; i < insn_count; ++i) {
      PipelineStage prev = GetPipelineStageAll(mem_ptr + i - 1);
      PipelineStage now = GetPipelineStageAll(mem_ptr + i);
      if (prev == kLoadStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else if (prev == kComputeStage && now == kLoadStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kStoreStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kComputeStage && now == kStoreStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = false;
      }
      if (now == kStoreStage) {
        mem_last_store_ptr = &mem_ptr[i];
      }
      mem_last_ptr = &mem_ptr[i];
    }
    // set dependency to make sure all core instruction get excuted
    // before last FINISH instruction
    if (mem_last_store_ptr && mem_last_ptr == mem_last_store_ptr) {
      mem_last_store_ptr->push_prev_dep = true;
      if (!pending_pop_next_[kComputeStage]) {
        DepPop(kStoreStage, kComputeStage);
      }
      CommitPendingPop(kComputeStage);
    } else {
      pending_pop_next_[kComputeStage] = 0;
    }
    DepPush(kComputeStage, kLoadStage);
    DepPop(kLoadStage, kComputeStage);
    if (!pending_pop_next_[kLoadStage]) {
      DepPop(kComputeStage, kLoadStage);
    }
    CommitPendingPop(kLoadStage);
    DepPush(kLoadStage, kComputeStage);
    CommitPendingPop(kComputeStage);
  }
  // Helper function: Get Opcode string
  const char* getOpcodeString(int opcode, bool use_imm) {
    // The string name
    if (opcode == VTA_ALU_OPCODE_MIN) {
      if (use_imm) {
        return "min imm";
      } else {
        return "min";
      }
    } else if (opcode == VTA_ALU_OPCODE_MAX) {
      if (use_imm) {
        return "max imm";
      } else {
        return "max";
      }
    } else if (opcode == VTA_ALU_OPCODE_ADD) {
      if (use_imm) {
        return "add imm";
      } else {
        return "add";
      }
    } else if (opcode == VTA_ALU_OPCODE_SHR) {
      return "shr";
    } else if (opcode == VTA_ALU_OPCODE_MUL) {
      return "mul";
    }

    return "unknown op";
  }
  // Dump instructions in the queue
  void DumpInsn() {
    // Keep tabs on dependence queues
    int l2g_queue = 0;
    int g2l_queue = 0;
    int s2g_queue = 0;
    int g2s_queue = 0;
    // Converter
    union VTAInsn c;
    // Iterate over all instructions
    int insn_count = count();
    const VTAGenericInsn* insn = data();
    printf("There are %u instructions\n", insn_count);
    for (int i = 0; i < insn_count; ++i) {
      // Fetch instruction and decode opcode
      c.generic = insn[i];
      printf("INSTRUCTION %u: ", i);
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.x_size == 0) {
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            printf("NOP-STORE-STAGE\n");
          } else if (GetMemPipelineStage(c.mem.memory_type) == kComputeStage) {
            printf("NOP-COMPUTE-STAGE\n");
          } else {
            printf("NOP-MEMORY-STAGE\n");
          }
          printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                 static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                 static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
          // Count status in queues
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            CHECK(c.mem.pop_next_dep == false);
            CHECK(c.mem.push_next_dep == false);
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
          } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                     (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
            CHECK(c.mem.pop_prev_dep == false);
            CHECK(c.mem.push_prev_dep == false);
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
          } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
          }
          printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
          printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
          continue;
        }
        // Print instruction field information
        if (c.mem.opcode == VTA_OPCODE_LOAD) {
          printf("LOAD ");
          if (c.mem.memory_type == VTA_MEM_ID_UOP) printf("UOP\n");
          if (c.mem.memory_type == VTA_MEM_ID_WGT) printf("WGT\n");
          if (c.mem.memory_type == VTA_MEM_ID_INP) printf("INP\n");
          if (c.mem.memory_type == VTA_MEM_ID_ACC) printf("ACC\n");
        }
        if (c.mem.opcode == VTA_OPCODE_STORE) {
          printf("STORE:\n");
        }
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\tDRAM: 0x%08x, SRAM:0x%04x\n", static_cast<int>(c.mem.dram_base),
               static_cast<int>(c.mem.sram_base));
        printf("\ty: size=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.y_size),
               static_cast<int>(c.mem.y_pad_0), static_cast<int>(c.mem.y_pad_1));
        printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.x_size),
               static_cast<int>(c.mem.x_stride), static_cast<int>(c.mem.x_pad_0),
               static_cast<int>(c.mem.x_pad_1));
      } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
        // Print instruction field information
        printf("GEMM\n");

        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
        printf("\trange (%d, %d)\n", static_cast<int>(c.gemm.uop_bgn),
               static_cast<int>(c.gemm.uop_end));
        printf("\touter loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_out), static_cast<int>(c.gemm.wgt_factor_out),
               static_cast<int>(c.gemm.src_factor_out), static_cast<int>(c.gemm.dst_factor_out));
        printf("\tinner loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_in), static_cast<int>(c.gemm.wgt_factor_in),
               static_cast<int>(c.gemm.src_factor_in), static_cast<int>(c.gemm.dst_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
        printf("\trange (%d, %d)\n", static_cast<int>(c.alu.uop_bgn),
               static_cast<int>(c.alu.uop_end));
        printf("\touter loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_out),
               static_cast<int>(c.alu.dst_factor_out), static_cast<int>(c.alu.src_factor_out));
        printf("\tinner loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_in),
               static_cast<int>(c.alu.dst_factor_in), static_cast<int>(c.alu.src_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
        printf("FINISH\n");
      }

      // Count status in queues
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.opcode == VTA_OPCODE_STORE) {
          CHECK(c.mem.pop_next_dep == false);
          CHECK(c.mem.push_next_dep == false);
          if (c.mem.pop_prev_dep) g2s_queue--;
          if (c.mem.push_prev_dep) s2g_queue++;
        } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                   (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
          CHECK(c.mem.pop_prev_dep == false);
          CHECK(c.mem.push_prev_dep == false);
          if (c.mem.pop_next_dep) g2l_queue--;
          if (c.mem.push_next_dep) l2g_queue++;
        } else {
          if (c.mem.pop_prev_dep) l2g_queue--;
          if (c.mem.push_prev_dep) g2l_queue++;
          if (c.mem.pop_next_dep) s2g_queue--;
          if (c.mem.push_next_dep) g2s_queue++;
        }
      } else if (c.mem.opcode == VTA_OPCODE_GEMM || c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        if (c.gemm.pop_prev_dep) l2g_queue--;
        if (c.gemm.push_prev_dep) g2l_queue++;
        if (c.gemm.pop_next_dep) s2g_queue--;
        if (c.gemm.push_next_dep) g2s_queue++;
      }
      printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
      printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
    }
  }
  // Commit all pending pop of corresponding stage
  void CommitPendingPop(int stage) {
    // Handle the LD<->compute queue
    // NOTE: pop executes on target(stage)
    CHECK(stage > 0 && stage < 4);
    if (pending_pop_prev_[stage] || pending_pop_next_[stage]) {
      PushNoop(stage, false, false, pending_pop_prev_[stage], pending_pop_next_[stage]);
      pending_pop_prev_[stage] = 0;
      pending_pop_next_[stage] = 0;
    }
  }
  void CommitPending() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      CommitPendingPop(i);
    }
  }
  bool HasPendingPop(int stage) const {
    CHECK(stage > 0 && stage < 4);
    return pending_pop_prev_[stage] || pending_pop_next_[stage];
  }
  bool PendingPop() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      if (pending_pop_prev_[i]) return true;
      if (pending_pop_next_[i]) return true;
    }
    return false;
  }
  void AutoReadBarrier() { ReadBarrier(); }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier() {
    CHECK(fpga_buff_ != nullptr);
    CHECK(fpga_buff_phy_);
    uint32_t buff_size = dram_buffer_.size() * elem_bytes_;
    CHECK(buff_size <= kMaxBytes);
    // Copy contents of DRAM buffer to FPGA buff
    VTAMemCopyFromHost(fpga_buff_, dram_buffer_.data(), buff_size);
    // Flush if we're using a shared memory system
    // and if interface is non-coherent
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_, fpga_buff_phy_, buff_size);
    }
  }

  std::vector<char> SerializeBytes() const {
    const char* begin = reinterpret_cast<const char*>(dram_buffer_.data());
    return std::vector<char>(begin, begin + dram_buffer_.size() * sizeof(VTAGenericInsn));
  }

  void LoadSerializedBytes(const std::vector<char>& bytes) {
    if (bytes.empty()) return;
    CHECK_EQ(bytes.size() % sizeof(VTAGenericInsn), 0U);
    CHECK_LE(bytes.size(), static_cast<size_t>(kMaxBytes));
    VTAMemCopyFromHost(fpga_buff_, bytes.data(), bytes.size());
    if (!coherent_ && always_cache_) {
      VTAFlushCache(fpga_buff_, fpga_buff_phy_, bytes.size());
    }
  }

 protected:
  /*! \return Add new instruction to the buffer. */
  VTAGenericInsn* NextInsn() {
    VTAGenericInsn insn = {};
    dram_buffer_.push_back(insn);
    return &dram_buffer_.back();
  }
  // Create a new instruction for a given stage
  VTAGenericInsn* Create(PipelineStage stage) {
    VTAGenericInsn* gptr = NextInsn();
    VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(gptr);
    mptr->pop_prev_dep = pending_pop_prev_[stage];
    mptr->pop_next_dep = pending_pop_next_[stage];
    mptr->push_prev_dep = false;
    mptr->push_next_dep = false;
    pending_pop_prev_[stage] = 0;
    pending_pop_next_[stage] = 0;
    return gptr;
  }
  // Get stage of the memory
  static PipelineStage GetMemPipelineStage(int memory_type) {
    if (memory_type == VTA_MEM_ID_ACC || memory_type == VTA_MEM_ID_ACC_8BIT) return kComputeStage;
    if (memory_type == VTA_MEM_ID_UOP) return kComputeStage;
    return kLoadStage;
  }
  // Get stage of the computation
  static PipelineStage GetPipelineStage(VTAMemInsn* insn) {
    if (insn->opcode == VTA_OPCODE_GEMM) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_ALU) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_LOAD) {
      if (insn->x_size == 0) return kNoneStage;
      if (insn->memory_type == VTA_MEM_ID_ACC || insn->memory_type == VTA_MEM_ID_ACC_8BIT)
        return kComputeStage;
      if (insn->memory_type == VTA_MEM_ID_UOP) return kComputeStage;
      return kLoadStage;
    }
    if (insn->opcode == VTA_OPCODE_STORE) {
      // FIXME: Right now memory_type is a 2-bit field which means that
      //        VTA_MEM_ID_OUT will appear as 0. For now we'll refrain from
      //        checking the memory_type to avoid an CHECK error...
      return kStoreStage;
    }
    LOG(FATAL) << "not reached";
  }

  // Get stage of memory and computation
  static PipelineStage GetPipelineStageAll(VTAMemInsn* insn) {
    PipelineStage stage = GetPipelineStage(insn);
    if (stage != kNoneStage) return stage;
    return GetMemPipelineStage(insn->memory_type);
  }

  // Push no-op
  void PushNoop(int stage, bool push_prev_dep, bool push_next_dep, bool pop_prev_dep,
                bool pop_next_dep) {
    VTAMemInsn* insn = reinterpret_cast<VTAMemInsn*>(NextInsn());
    insn->opcode = (stage == kStoreStage ? VTA_OPCODE_STORE : VTA_OPCODE_LOAD);
    insn->push_prev_dep = push_prev_dep;
    insn->push_next_dep = push_next_dep;
    insn->pop_prev_dep = pop_prev_dep;
    insn->pop_next_dep = pop_next_dep;
    insn->sram_base = 0;
    insn->dram_base = 0;
    insn->y_size = 0;
    insn->x_size = 0;
    insn->x_stride = 0;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    insn->memory_type = (stage == kLoadStage ? VTA_MEM_ID_INP : VTA_MEM_ID_UOP);
  }

 private:
  // Pending pop of each isntruction queue, qid=0 is not used
  int pending_pop_prev_[4];
  int pending_pop_next_[4];
  static constexpr int kElemBytes = sizeof(VTAGenericInsn);
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};

/*!
 * \brief The command queue object that handles the request.
 */
class CommandQueue {
 public:
  enum class ReplayMode : int {
    kDisabled = 0,
    kCapture = 1,
    kReplay = 2,
  };

  struct PendingDMA {
    uint32_t opcode{0};
    uint32_t memory_type{0};
    uint64_t dram_base{0};
    uint64_t sram_base{0};
    uint32_t x_size{0};
    uint32_t y_size{0};
    uint32_t x_stride{0};
    uint32_t y_pad_before{0};
    uint32_t y_pad_after{0};
    uint32_t x_pad_before{0};
    uint32_t x_pad_after{0};
  };

  struct CapturedTemplate {
    std::string label;
    std::vector<char> uop_bytes;
    std::vector<char> insn_bytes;
    uint64_t insn_count{0};
    size_t load_bytes{0};
    size_t store_bytes{0};
    std::vector<std::tuple<DataBuffer*, size_t, size_t>> store_ranges;
    uintptr_t first_load_handle{0};
    uintptr_t first_store_handle{0};

    bool valid() const { return !label.empty() && insn_count != 0 && !insn_bytes.empty(); }
  };

  CommandQueue() { this->InitSpace(); }
  void InitSpace() {
    RuntimeTrace("CommandQueue::InitSpace begin");
    uop_queue_.InitSpace();
    RuntimeTrace("CommandQueue::InitSpace uop_queue done");
    insn_queue_.InitSpace();
    RuntimeTrace("CommandQueue::InitSpace insn_queue done");
    device_ = VTADeviceAlloc();
    CHECK(device_ != nullptr);
    RuntimeTrace("CommandQueue::InitSpace device done handle=%p", device_);
  }

  ~CommandQueue() { VTADeviceFree(device_); }

  uint32_t GetElemBytes(uint32_t memory_id) {
    uint32_t elem_bytes = 0;
    switch (memory_id) {
      case VTA_MEM_ID_UOP:
        elem_bytes = VTA_UOP_ELEM_BYTES;
        break;
      case VTA_MEM_ID_INP:
        elem_bytes = VTA_INP_ELEM_BYTES;
        break;
      case VTA_MEM_ID_WGT:
        elem_bytes = VTA_WGT_ELEM_BYTES;
        break;
      case VTA_MEM_ID_ACC:
        elem_bytes = VTA_ACC_ELEM_BYTES;
        break;
      case VTA_MEM_ID_OUT:
        elem_bytes = VTA_OUT_ELEM_BYTES;
        break;
      case VTA_MEM_ID_ACC_8BIT:
        elem_bytes = VTA_ACC_ELEM_BYTES / 4;
        break;
      default:
        LOG(FATAL) << "Memory id not recognized:" << memory_id;
        break;
    }
    /*
     * elements size should not larger than VTA_PAGE_BYTES.
     *
     */
    CHECK_GE(VTA_PAGE_BYTES, elem_bytes);
    return elem_bytes;
  }

  void LoadBuffer2D(void* src_dram_addr, uint32_t src_elem_offset, uint32_t x_size, uint32_t y_size,
                    uint32_t x_stride, uint32_t x_pad_before, uint32_t y_pad_before,
                    uint32_t x_pad_after, uint32_t y_pad_after, uint32_t dst_sram_index,
                    uint32_t dst_memory_type) {
    if (mode_ == ReplayMode::kReplay) return;
    double t0 = NowMicros();
    const uint32_t elem_bytes = GetElemBytes(dst_memory_type);
    const size_t transfer_bytes = static_cast<size_t>(x_size) * y_size * elem_bytes;
    pending_load_bytes_ += transfer_bytes;
    DataBuffer* src = DataBuffer::FromHandle(src_dram_addr);
    CHECK(src != nullptr);
    if (mode_ == ReplayMode::kCapture && current_first_load_handle_ == 0) {
      current_first_load_handle_ = reinterpret_cast<uintptr_t>(src_dram_addr);
    }
    src->EnsureDeviceReadable(
        static_cast<size_t>(src_elem_offset) * elem_bytes,
        Compute2DSpanBytes(elem_bytes, src_elem_offset, x_size, y_size, x_stride));
    PendingDMA next = {};
    next.opcode = VTA_OPCODE_LOAD;
    next.memory_type = dst_memory_type;
    next.sram_base = dst_sram_index;
    next.dram_base = src->phy_addr() / GetElemBytes(dst_memory_type) + src_elem_offset;
    next.y_size = y_size;
    next.x_size = x_size;
    next.x_stride = x_stride;
    next.y_pad_before = y_pad_before;
    next.y_pad_after = y_pad_after;
    next.x_pad_before = x_pad_before;
    next.x_pad_after = x_pad_after;
    VTAMemInsn* insn = insn_queue_.CreateMemInsn(dst_memory_type);
    insn->opcode = VTA_OPCODE_LOAD;
    insn->memory_type = dst_memory_type;
    insn->sram_base = dst_sram_index;
    insn->dram_base = next.dram_base;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = y_pad_before;
    insn->y_pad_1 = y_pad_after;
    insn->x_pad_0 = x_pad_before;
    insn->x_pad_1 = x_pad_after;
    this->CheckInsnOverFlow();
    RuntimeProfiler::Global().AddLoadBuffer2D(
        dst_memory_type, x_size, y_size, x_stride,
        x_pad_before != 0 || y_pad_before != 0 || x_pad_after != 0 || y_pad_after != 0,
        NowMicros() - t0);
  }

  void StoreBuffer2D(uint32_t src_sram_index, uint32_t src_memory_type, void* dst_dram_addr,
                     uint32_t dst_elem_offset, uint32_t x_size, uint32_t y_size,
                     uint32_t x_stride) {
    if (mode_ == ReplayMode::kReplay) return;
    double t0 = NowMicros();
    const uint32_t elem_bytes = GetElemBytes(src_memory_type);
    const size_t transfer_bytes = static_cast<size_t>(x_size) * y_size * elem_bytes;
    pending_store_bytes_ += transfer_bytes;
    DataBuffer* dst = DataBuffer::FromHandle(dst_dram_addr);
    CHECK(dst != nullptr);
    if (mode_ == ReplayMode::kCapture && current_first_store_handle_ == 0) {
      current_first_store_handle_ = reinterpret_cast<uintptr_t>(dst_dram_addr);
    }
    size_t range_start = static_cast<size_t>(dst_elem_offset) * elem_bytes;
    size_t range_extent = Compute2DSpanBytes(elem_bytes, dst_elem_offset, x_size, y_size, x_stride);
    PendingDMA next = {};
    next.opcode = VTA_OPCODE_STORE;
    next.memory_type = src_memory_type;
    next.sram_base = src_sram_index;
    next.dram_base = dst->phy_addr() / GetElemBytes(src_memory_type) + dst_elem_offset;
    next.y_size = y_size;
    next.x_size = x_size;
    next.x_stride = x_stride;
    VTAMemInsn* insn = insn_queue_.CreateStoreInsn();
    insn->opcode = VTA_OPCODE_STORE;
    insn->memory_type = src_memory_type;
    insn->sram_base = src_sram_index;
    insn->dram_base = next.dram_base;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    pending_store_ranges_.push_back({dst, range_start, range_extent});
    this->CheckInsnOverFlow();
    RuntimeProfiler::Global().AddStoreBuffer2D(src_memory_type, x_size, y_size, x_stride,
                                               NowMicros() - t0);
  }

  void DepPush(int from_qid, int to_qid) {
    if (mode_ == ReplayMode::kReplay) return;
    insn_queue_.DepPush(from_qid, to_qid);
  }

  void DepPop(int from_qid, int to_qid) {
    if (mode_ == ReplayMode::kReplay) return;
    insn_queue_.DepPop(from_qid, to_qid);
  }

  void ReadBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (mode_ == ReplayMode::kReplay) return;
    if (!(debug_flag_ & VTA_DEBUG_SKIP_READ_BARRIER)) {
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->EnsureDeviceReadable(elem_bytes * start, elem_bytes * extent);
    }
  }

  void WriteBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (mode_ == ReplayMode::kReplay) return;
    if (!(debug_flag_ & VTA_DEBUG_SKIP_WRITE_BARRIER)) {
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      if (auto* data_buf = DataBuffer::FromHandle(buffer)) {
        pending_store_ranges_.push_back({data_buf, elem_bytes * start, elem_bytes * extent});
      }
    }
  }

  void Synchronize(uint32_t wait_cycles) {
    if (mode_ == ReplayMode::kReplay) {
      this->ReplayCapturedTemplate(wait_cycles);
      pending_store_ranges_.clear();
      pending_load_bytes_ = 0;
      pending_store_bytes_ = 0;
      uop_queue_.Reset();
      insn_queue_.Reset();
      this->ResetTemplateMode();
      return;
    }
    // Insert dependences to force serialization
    if (debug_flag_ & VTA_DEBUG_FORCE_SERIAL) {
      insn_queue_.RewriteForceSerial();
    } else {
      // This will issue finish after last store finishes
      insn_queue_.DepPush(kStoreStage, kComputeStage);
      insn_queue_.DepPush(kLoadStage, kComputeStage);
      insn_queue_.DepPop(kStoreStage, kComputeStage);
      insn_queue_.DepPop(kLoadStage, kComputeStage);
      insn_queue_.CommitPendingPop(kComputeStage);
    }
    // NOTE: FINISH cannot contain pop
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_FINISH;
    CHECK(!insn_queue_.PendingPop());
    // Check if there are no instruction to execute at all
    if (insn_queue_.count() == 0) {
      pending_store_ranges_.clear();
      pending_load_bytes_ = 0;
      pending_store_bytes_ = 0;
      return;
    }
    // Synchronization for the queues
    uop_queue_.AutoReadBarrier();
    insn_queue_.AutoReadBarrier();
    // Dump instructions if debug enabled
    if (debug_flag_ & VTA_DEBUG_DUMP_INSN) {
      insn_queue_.DumpInsn();
    }
    // Make sure that the last instruction is a finish instruction
    CHECK(reinterpret_cast<VTAMemInsn*>(insn_queue_.data())[insn_queue_.count() - 1].opcode ==
          VTA_OPCODE_FINISH);

    // Make sure that we don't exceed contiguous physical memory limits
    CHECK(insn_queue_.count() * sizeof(VTAGenericInsn) <= VTA_MAX_XFER);
    uint64_t insn_count = insn_queue_.count();
    VTADriverProfilerStats driver_before{};
    VTADriverProfilerStatus(&driver_before);
    double t0 = NowMicros();
    int timeout = VTADeviceRun(device_, insn_queue_.dram_phy_addr(), insn_count, wait_cycles);
    VTADriverProfilerStats driver_after{};
    VTADriverProfilerStatus(&driver_after);
    for (const auto& range : pending_store_ranges_) {
      if (std::get<0>(range) != nullptr) {
        std::get<0>(range)->MarkDeviceWrite(std::get<1>(range), std::get<2>(range));
      }
    }
    pending_store_ranges_.clear();
    RuntimeProfiler::Global().AddSynchronize(insn_count, pending_load_bytes_, pending_store_bytes_,
                                             NowMicros() - t0,
                                             DiffDriverStats(driver_after, driver_before));
    CHECK_EQ(timeout, 0);
    if (mode_ == ReplayMode::kCapture) {
      double tc0 = NowMicros();
      this->CaptureCurrentTemplate();
      RuntimeProfiler::Global().AddTemplateCapture(NowMicros() - tc0);
    }
    // Reset buffers
    pending_load_bytes_ = 0;
    pending_store_bytes_ = 0;
    uop_queue_.Reset();
    insn_queue_.Reset();
    this->ResetTemplateMode();
  }

  // Get record kernel
  UopKernel* record_kernel() const {
    CHECK(record_kernel_ != nullptr);
    return record_kernel_;
  }

  // Set debug flag
  void SetDebugFlag(int debug_flag) { debug_flag_ = debug_flag; }

  void BeginTemplateCapture(const std::string& label) {
    mode_ = ReplayMode::kCapture;
    active_template_label_ = label;
    current_first_load_handle_ = 0;
    current_first_store_handle_ = 0;
  }

  void BeginTemplateReplay(const std::string& label) {
    active_template_label_ = label;
    auto it = captured_templates_.find(label);
    if (it == captured_templates_.end() || !it->second.valid()) {
      mode_ = ReplayMode::kDisabled;
      RuntimeProfiler::Global().AddTemplateReplayFallback();
      return;
    }
    mode_ = ReplayMode::kReplay;
    current_first_load_handle_ = it->second.first_load_handle;
    current_first_store_handle_ = it->second.first_store_handle;
  }

  void ResetTemplateMode() {
    mode_ = ReplayMode::kDisabled;
    active_template_label_.clear();
    current_first_load_handle_ = 0;
    current_first_store_handle_ = 0;
  }

  void ClearCapturedTemplates() {
    this->ResetTemplateMode();
    captured_templates_.clear();
  }

  std::string TemplateReplayStatusJSON() const {
    std::ostringstream os;
    os << "{"
       << "\"mode\":" << JSONQuote(ReplayModeName(mode_)) << ","
       << "\"active_label\":" << JSONQuote(active_template_label_) << ","
       << "\"template_count\":" << captured_templates_.size();
    auto it = captured_templates_.find(active_template_label_);
    if (it != captured_templates_.end()) {
      os << ",\"active_template_valid\":" << (it->second.valid() ? "true" : "false");
    }
    os << "}";
    return os.str();
  }

  void PushGEMMOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
    if (mode_ == ReplayMode::kReplay) return;
    double t0 = NowMicros();
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushGEMMOp(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
    RuntimeProfiler::Global().AddPushGEMMOp(NowMicros() - t0);
  }

  void PushALUUop(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
    if (mode_ == ReplayMode::kReplay) return;
    double t0 = NowMicros();
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushALUUop(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
    RuntimeProfiler::Global().AddPushALUOp(NowMicros() - t0);
  }

  static std::shared_ptr<CommandQueue>& ThreadLocal() {
    static std::shared_ptr<CommandQueue> inst = std::make_shared<CommandQueue>();
    if (inst == nullptr) {
      inst = std::make_shared<CommandQueue>();
    }
    return inst;
  }

  static void Shutdown() { ThreadLocal().reset(); }

 private:
  // Push GEMM uop to the command buffer
  void PushGEMMOp(UopKernel* kernel) {
    uop_queue_.Push(kernel, [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);
    }
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_GEMM;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    const std::vector<UopKernel::LoopEntry>& loop = kernel->loop();
    if (loop.size() > 0) {
      insn->iter_out = loop[0].extent;
      insn->wgt_factor_out = loop[0].wgt_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->dst_factor_out = loop[0].dst_factor;
    } else {
      insn->iter_out = 1;
      insn->wgt_factor_out = 0;
      insn->src_factor_out = 0;
      insn->dst_factor_out = 0;
    }
    if (loop.size() > 1) {
      insn->iter_in = loop[1].extent;
      insn->wgt_factor_in = loop[1].wgt_factor;
      insn->src_factor_in = loop[1].src_factor;
      insn->dst_factor_in = loop[1].dst_factor;
    } else {
      insn->iter_in = 1;
      insn->wgt_factor_in = 0;
      insn->src_factor_in = 0;
      insn->dst_factor_in = 0;
    }
  }

  // Push ALU uop to the command buffer
  void PushALUUop(UopKernel* kernel) {
    uop_queue_.Push(kernel, [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);
    }
    VTAAluInsn* insn = insn_queue_.CreateAluInsn();
    insn->opcode = VTA_OPCODE_ALU;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    insn->alu_opcode = kernel->opcode_;
    insn->use_imm = kernel->use_imm_;
    insn->imm = kernel->imm_val_;
    const std::vector<UopKernel::LoopEntry>& loop = kernel->loop();
    if (loop.size() == 0) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = 1;
      insn->dst_factor_in = 0;
      insn->src_factor_in = 0;
    } else if (loop.size() == 1) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = loop[0].extent;
      insn->dst_factor_in = loop[0].dst_factor;
      insn->src_factor_in = loop[0].src_factor;
    } else {
      insn->iter_out = loop[0].extent;
      insn->dst_factor_out = loop[0].dst_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->iter_in = loop[1].extent;
      insn->dst_factor_in = loop[1].dst_factor;
      insn->src_factor_in = loop[1].src_factor;
    }
  }

  void CheckInsnOverFlow() {
    // At each API call, we can at most commit:
    // at most: 2 NOP-COMPUTE-STAGE -> 2 NOP-MEMORY-STAGE -> 1 NOP-COMPUTE-STAGE -> 1 FINISH
    if ((insn_queue_.count() + 6) * sizeof(VTAGenericInsn) > VTA_MAX_XFER) {
      this->AutoSync();
    }
  }
  // Auto sync when instruction overflow
  void AutoSync() { this->Synchronize(1 << 31); }

  void CaptureCurrentTemplate() {
    CapturedTemplate templ;
    templ.label = active_template_label_;
    templ.uop_bytes = uop_queue_.SerializeCacheBytes();
    templ.insn_bytes = insn_queue_.SerializeBytes();
    templ.insn_count = insn_queue_.count();
    templ.load_bytes = pending_load_bytes_;
    templ.store_bytes = pending_store_bytes_;
    templ.store_ranges = pending_store_ranges_;
    templ.first_load_handle = current_first_load_handle_;
    templ.first_store_handle = current_first_store_handle_;
    captured_templates_[active_template_label_] = std::move(templ);
  }

  void ReplayCapturedTemplate(uint32_t wait_cycles) {
    auto it = captured_templates_.find(active_template_label_);
    CHECK(it != captured_templates_.end());
    const CapturedTemplate& templ = it->second;
    CHECK(templ.valid());
    VTADriverProfilerStats driver_before{};
    VTADriverProfilerStatus(&driver_before);
    double t0 = NowMicros();
    uop_queue_.LoadSerializedBytes(templ.uop_bytes);
    insn_queue_.LoadSerializedBytes(templ.insn_bytes);
    int timeout = VTADeviceRun(device_, insn_queue_.dram_phy_addr(), templ.insn_count, wait_cycles);
    VTADriverProfilerStats driver_after{};
    VTADriverProfilerStatus(&driver_after);
    for (const auto& range : templ.store_ranges) {
      if (std::get<0>(range) != nullptr) {
        std::get<0>(range)->MarkDeviceWrite(std::get<1>(range), std::get<2>(range));
      }
    }
    RuntimeProfiler::Global().AddSynchronize(
        templ.insn_count, templ.load_bytes, templ.store_bytes, NowMicros() - t0,
        DiffDriverStats(driver_after, driver_before));
    RuntimeProfiler::Global().AddTemplateReplay(NowMicros() - t0);
    CHECK_EQ(timeout, 0);
  }

  static const char* ReplayModeName(ReplayMode mode) {
    switch (mode) {
      case ReplayMode::kDisabled:
        return "disabled";
      case ReplayMode::kCapture:
        return "capture";
      case ReplayMode::kReplay:
        return "replay";
    }
    return "disabled";
  }

  // Internal debug flag
  int debug_flag_{0};
  // The kernel we are currently recording
  UopKernel* record_kernel_{nullptr};
  // Micro op queue
  UopQueue<VTA_MAX_XFER, kBufferCoherent, kAlwaysCache> uop_queue_;
  // instruction queue
  InsnQueue<VTA_MAX_XFER, kBufferCoherent, kAlwaysCache> insn_queue_;
  // Device handle
  VTADeviceHandle device_{nullptr};
  // DRAM ranges that become CPU-stale after the next synchronize completes.
  std::vector<std::tuple<DataBuffer*, size_t, size_t>> pending_store_ranges_;
  // Aggregate DMA bytes submitted since the last synchronize.
  size_t pending_load_bytes_{0};
  size_t pending_store_bytes_{0};
  ReplayMode mode_{ReplayMode::kDisabled};
  std::string active_template_label_;
  std::unordered_map<std::string, CapturedTemplate> captured_templates_;
  uintptr_t current_first_load_handle_{0};
  uintptr_t current_first_store_handle_{0};
};

}  // namespace vta

void* VTABufferAlloc(size_t size) { return vta::DataBuffer::Alloc(size); }

void VTABufferFree(void* buffer) { vta::DataBuffer::Free(vta::DataBuffer::FromHandle(buffer)); }

void VTABufferCopy(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                   int kind_mask) {
  vta::RuntimeTrace("VTABufferCopy kind_mask=%d from=%p+%zu to=%p+%zu size=%zu", kind_mask, from,
                    from_offset, to, to_offset, size);
  double t0 = vta::NowMicros();
  vta::DataBuffer* from_buffer = nullptr;
  vta::DataBuffer* to_buffer = nullptr;

  if (kind_mask & 2) {
    from_buffer = vta::DataBuffer::FromHandle(from);
    from = from_buffer->virt_addr();
  }
  if (kind_mask & 1) {
    to_buffer = vta::DataBuffer::FromHandle(to);
    to = to_buffer->virt_addr();
  }

  if (from_buffer && to_buffer) {
    from_buffer->EnsureHostReadable(from_offset, size);
    to_buffer->MemCopyFromHost(static_cast<char*>(to) + to_offset,
                               static_cast<const char*>(from) + from_offset, size);
    to_buffer->MarkHostWrite(to_offset, size);
  } else if (from_buffer) {
    // This is an FPGA to host mem transfer
    from_buffer->EnsureHostReadable(from_offset, size);
    from_buffer->MemCopyToHost(static_cast<char*>(to) + to_offset,
                               static_cast<const char*>(from) + from_offset, size);
  } else if (to_buffer) {
    // This is a host to FPGA mem transfer
    to_buffer->MemCopyFromHost(static_cast<char*>(to) + to_offset,
                               static_cast<const char*>(from) + from_offset, size);
    to_buffer->MarkHostWrite(to_offset, size);
  }
  vta::RuntimeProfiler::Global().AddScopedCopy(vta::CurrentScopedCopy(), size,
                                               vta::NowMicros() - t0);
}

VTACommandHandle VTATLSCommandHandle() { return vta::CommandQueue::ThreadLocal().get(); }

void VTARuntimeShutdown() { vta::CommandQueue::Shutdown(); }

void VTASetDebugMode(VTACommandHandle cmd, int debug_flag) {
  static_cast<vta::CommandQueue*>(cmd)->SetDebugFlag(debug_flag);
}

void* VTABufferCPUPtr(VTACommandHandle cmd, void* buffer) {
  auto data_buf = vta::DataBuffer::FromHandle(buffer);
  if (data_buf) {
    return data_buf->virt_addr();
  } else {  // it is a raw ptr allocated by CPU
    return buffer;
  }
}

void VTAWriteBarrier(VTACommandHandle cmd, void* buffer, uint32_t elem_bits, uint32_t start,
                     uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->WriteBarrier(buffer, elem_bits, start, extent);
}

void VTAReadBarrier(VTACommandHandle cmd, void* buffer, uint32_t elem_bits, uint32_t start,
                    uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->ReadBarrier(buffer, elem_bits, start, extent);
}

void VTALoadBuffer2D(VTACommandHandle cmd, void* src_dram_addr, uint32_t src_elem_offset,
                     uint32_t x_size, uint32_t y_size, uint32_t x_stride, uint32_t x_pad_before,
                     uint32_t y_pad_before, uint32_t x_pad_after, uint32_t y_pad_after,
                     uint32_t dst_sram_index, uint32_t dst_memory_type) {
  static_cast<vta::CommandQueue*>(cmd)->LoadBuffer2D(
      src_dram_addr, src_elem_offset, x_size, y_size, x_stride, x_pad_before, y_pad_before,
      x_pad_after, y_pad_after, dst_sram_index, dst_memory_type);
}

void VTAStoreBuffer2D(VTACommandHandle cmd, uint32_t src_sram_index, uint32_t src_memory_type,
                      void* dst_dram_addr, uint32_t dst_elem_offset, uint32_t x_size,
                      uint32_t y_size, uint32_t x_stride) {
  static_cast<vta::CommandQueue*>(cmd)->StoreBuffer2D(
      src_sram_index, src_memory_type, dst_dram_addr, dst_elem_offset, x_size, y_size, x_stride);
}

void VTAUopPush(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
                uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
  vta::CommandQueue::ThreadLocal()->record_kernel()->Push(mode, reset_out, dst_index, src_index,
                                                          wgt_index, opcode, use_imm, imm_val);
}

void VTAUopLoopBegin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                     uint32_t wgt_factor) {
  vta::CommandQueue::ThreadLocal()->record_kernel()->PushLoopBegin(extent, dst_factor, src_factor,
                                                                   wgt_factor);
}

void VTAUopLoopEnd() { vta::CommandQueue::ThreadLocal()->record_kernel()->PushLoopEnd(); }

int VTAPushGEMMOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
  vta::CommandQueue::ThreadLocal()->PushGEMMOp(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTAPushALUOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes) {
  vta::CommandQueue::ThreadLocal()->PushALUUop(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTADepPush(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->DepPush(from_qid, to_qid);
  return 0;
}

int VTADepPop(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->DepPop(from_qid, to_qid);
  return 0;
}

void VTASynchronize(VTACommandHandle cmd, uint32_t wait_cycles) {
  static_cast<vta::CommandQueue*>(cmd)->Synchronize(wait_cycles);
}

TVM_REGISTER_GLOBAL("vta.runtime.profiler_clear").set_body_typed([]() {
  vta::RuntimeProfiler::Global().Clear();
});

TVM_REGISTER_GLOBAL("vta.runtime.profiler_status").set_body_typed([]() {
  return vta::RuntimeProfiler::Global().AsJSON();
});

TVM_REGISTER_GLOBAL("vta.runtime.profiler_events")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
  int max_events = 0;
  if (args.size() >= 1) {
    max_events = args[0];
  }
  *rv = vta::RuntimeProfiler::Global().EventsAsJSON(max_events);
});

TVM_REGISTER_GLOBAL("vta.runtime.replay_begin_capture").set_body_typed([](std::string label) {
  vta::CommandQueue::ThreadLocal()->BeginTemplateCapture(label);
});

TVM_REGISTER_GLOBAL("vta.runtime.replay_begin_replay").set_body_typed([](std::string label) {
  vta::CommandQueue::ThreadLocal()->BeginTemplateReplay(label);
});

TVM_REGISTER_GLOBAL("vta.runtime.replay_reset").set_body_typed([]() {
  vta::CommandQueue::ThreadLocal()->ClearCapturedTemplates();
});

TVM_REGISTER_GLOBAL("vta.runtime.replay_status").set_body_typed([]() {
  return vta::CommandQueue::ThreadLocal()->TemplateReplayStatusJSON();
});

TVM_REGISTER_GLOBAL("vta.runtime.push_copy_scope").set_body_typed([](std::string scope) {
  vta::PushScopedCopy(vta::CopyScopeFromString(scope));
});

TVM_REGISTER_GLOBAL("vta.runtime.pop_copy_scope").set_body_typed([]() {
  vta::PopScopedCopy();
});

TVM_REGISTER_GLOBAL("vta.runtime.ndarray_shared_cpu_view")
    .set_body_typed([](tvm::runtime::NDArray arr) {
      using namespace tvm::runtime;
      tvm::Device dev = arr->device;
      if (dev.device_type >= kRPCSessMask) {
        dev.device_type = static_cast<DLDeviceType>(dev.device_type % kRPCSessMask);
      }
      ICHECK_EQ(dev.device_type, kDLExtDev)
          << "shared CPU view only supports ext_dev NDArrays";
      auto* orig =
          const_cast<NDArray::Container*>(static_cast<const NDArray::Container*>(arr.get()));
      auto* data_buf = vta::DataBuffer::FromHandle(arr->data);
      ICHECK(data_buf != nullptr) << "NDArray is not backed by a VTA shared buffer";
      auto* view = new NDArray::Container(data_buf->virt_addr(), arr.Shape(), arr->dtype,
                                          tvm::Device{kDLCPU, 0});
      orig->IncRef();
      view->manager_ctx = orig;
      view->SetDeleter([](Object* ptr_obj) {
        auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
        if (ptr->manager_ctx != nullptr) {
          static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
        }
        delete ptr;
      });
      return NDArray(GetObjectPtr<Object>(view));
    });

TVM_REGISTER_GLOBAL("vta.runtime.ndarray_mark_host_write")
    .set_body_typed([](tvm::runtime::NDArray arr) {
      auto* data_buf = vta::DataBuffer::FromHandle(arr->data);
      ICHECK(data_buf != nullptr) << "expected ext_dev NDArray backed by VTA shared buffer";
      data_buf->MarkHostWrite(0, data_buf->size());
    });

TVM_REGISTER_GLOBAL("vta.runtime.ndarray_sync_host_read")
    .set_body_typed([](tvm::runtime::NDArray arr) {
      auto* data_buf = vta::DataBuffer::FromHandle(arr->data);
      ICHECK(data_buf != nullptr) << "expected ext_dev NDArray backed by VTA shared buffer";
      data_buf->EnsureHostReadable(0, data_buf->size());
    });
