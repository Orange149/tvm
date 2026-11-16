// Experiment-only collection switch and bounded per-complete-call submit ledger.
// No execution, queue allocation, cache, barrier or AutoSync policy lives here.
#ifndef C3_OFFICIAL_WINDOW_RUNTIME_COLLECTION_H_
#define C3_OFFICIAL_WINDOW_RUNTIME_COLLECTION_H_

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

namespace c3_official_window {
struct Submit {
  uint64_t ordinal{}, insns{}, serialized_uop_bytes{};
  const char* reason{"unset"};
  int timeout{};
  bool completion{};
};
struct Collection {
  bool detailed{true}, overflow{false};
  uint64_t attempted_submits{}, recorded_submits{}, detailed_clock_reads{};
  std::array<Submit, 16> submits{};
  Collection() {
    const char* value = std::getenv("C3_OFFICIAL_WINDOW_COLLECTION");
    if (!value || std::strcmp(value, "detailed") == 0) detailed = true;
    else if (std::strcmp(value, "minimal") == 0) detailed = false;
    else std::abort();
  }
};
inline Collection& State() { static Collection state; return state; }
inline bool Detailed() { return State().detailed; }
template <typename Clock> inline double DetailClock(Clock clock) {
  if (!Detailed()) return 0.0;
  ++State().detailed_clock_reads;
  return clock();
}
inline void ClearLedger() {
  auto& s = State();
  s.overflow = false;
  s.attempted_submits = s.recorded_submits = s.detailed_clock_reads = 0;
  s.submits = {};
}
inline void RecordSubmit(const char* reason, uint64_t insns, uint64_t serialized_uop_bytes,
                         int timeout) {
  auto& s = State();
  const uint64_t ordinal = ++s.attempted_submits;
  if (s.recorded_submits == s.submits.size()) {
    // Explicitly INVALID, never silently truncate a PASS certificate. Collection
    // cannot stop or change execution; the outside-call protocol rejects this.
    s.overflow = true;
    return;
  }
  s.submits[s.recorded_submits++] = {ordinal, insns, serialized_uop_bytes, reason, timeout,
                                    timeout == 0};
}
inline std::string LedgerJSON() {
  const auto& s = State();
  std::ostringstream out;
  out << "{\"schema\":\"c3_official_window_submit_ledger_v1\",\"mode\":\""
      << (s.detailed ? "detailed" : "minimal") << "\",\"overflow\":"
      << (s.overflow ? "true" : "false") << ",\"attempted_submits\":" << s.attempted_submits
      << ",\"recorded_submits\":" << s.recorded_submits << ",\"submits\":[";
  for (uint64_t i = 0; i < s.recorded_submits; ++i) {
    const auto& row = s.submits[i];
    if (i) out << ',';
    out << "{\"ordinal\":" << row.ordinal << ",\"reason\":\"" << row.reason
        << "\",\"insns\":" << row.insns << ",\"insn_bytes\":" << row.insns * 16
        << ",\"serialized_uop_bytes\":" << row.serialized_uop_bytes
        << ",\"completion\":" << (row.completion ? "true" : "false")
        << ",\"timeout\":" << row.timeout << '}';
  }
  out << "]}";
  return out.str();
}
}  // namespace c3_official_window
#endif
