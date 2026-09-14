/* Queue backing capacity policy; independent of hardware/AutoSync limits. */
#ifndef VTA_RUNTIME_QUEUE_CAPACITY_H_
#define VTA_RUNTIME_QUEUE_CAPACITY_H_
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace vta {
namespace queue_capacity {
inline uint64_t ByteHash(const void* data, size_t bytes) {
  auto* p = static_cast<const unsigned char*>(data);
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ULL;
  return hash;
}
inline size_t Parse(const char* text, size_t fallback, size_t limit, size_t alignment) {
  if (text == nullptr) return fallback;
  if (*text == '\0') throw std::invalid_argument("empty queue capacity");
  size_t value = 0;
  for (const char* p = text; *p; ++p) {
    if (*p < '0' || *p > '9' || static_cast<size_t>(*p - '0') > limit ||
        value > (limit - (*p - '0')) / 10) {
      throw std::invalid_argument("queue capacity must be decimal bytes within VTA_MAX_XFER");
    }
    value = value * 10 + (*p - '0');
  }
  if (value == 0 || value > limit || alignment == 0 || value % alignment != 0) {
    throw std::invalid_argument("queue capacity must be positive and aligned");
  }
  return value;
}
inline size_t FromEnv(const char* name, size_t limit) {
  try {
    return Parse(std::getenv(name), limit, limit, 256);
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }
}
inline size_t SubmitThresholdFromEnv(size_t limit, size_t instruction_bytes,
                                     bool* explicitly_configured) {
  const char* value = std::getenv("VTA_INSN_SUBMIT_THRESHOLD_BYTES");
  *explicitly_configured = value != nullptr;
  size_t threshold = Parse(value, limit, limit, instruction_bytes);
  // Keep room for the triggering instruction and the conservative six-insn tail.
  if (threshold < 7 * instruction_bytes) {
    throw std::invalid_argument("VTA_INSN_SUBMIT_THRESHOLD_BYTES: threshold is too small");
  }
  return threshold;
}
inline bool ShouldSubmitForThreshold(size_t required_bytes, size_t hard_limit,
                                     bool threshold_configured, size_t threshold_bytes) {
  return required_bytes > hard_limit ||
         (threshold_configured && required_bytes > threshold_bytes);
}
inline void RequireFits(size_t bytes, size_t capacity) {
  if (bytes > capacity) throw std::runtime_error("VTA queue backing capacity exceeded before submission");
}
inline bool DiagnosticEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VTA_QUEUE_DIAGNOSTICS");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}
inline bool BoundaryAuditEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VTA_QUEUE_BOUNDARY_AUDIT");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}
// A necessary closure check only: does not prove schedulability or SRAM lifetime.
struct DependencyBalance {
  long load_compute{0}, compute_load{0}, compute_store{0}, store_compute{0};
  bool valid{true};
  void Add(int stage, bool pop_prev, bool pop_next, bool push_prev, bool push_next) {
    if (stage == 1) {
      valid = valid && !pop_prev && !push_prev;
      compute_load -= pop_next;
      load_compute += push_next;
    } else if (stage == 2) {
      load_compute -= pop_prev;
      store_compute -= pop_next;
      compute_load += push_prev;
      compute_store += push_next;
    } else if (stage == 3) {
      valid = valid && !pop_next && !push_next;
      compute_store -= pop_prev;
      store_compute += push_prev;
    } else {
      valid = false;
    }
  }
  bool Closed() const {
    return valid && load_compute == 0 && compute_load == 0 &&
           compute_store == 0 && store_compute == 0;
  }
};
}  // namespace queue_capacity
}  // namespace vta
#endif
