// Experimental HLS output-valid handshake. Offset is an explicitly recorded
// hypothesis (DONE_RD+4), NOT a generated-map attestation for this bitstream.
#ifndef C3_FRESH_COMPLETION_H_
#define C3_FRESH_COMPLETION_H_
#include "c3_completion_probe.h"

inline bool C3AcceptFreshCompletion(uint32_t valid, uint32_t done) {
  return valid == 1 && done == 1;
}

struct C3FreshCompletion : C3CompletionProbe {
  uint32_t before_valid = 0, after_clear_valid = 0, first_valid = 0, last_valid = 0;
  uint64_t valid_events = 0;
  explicit C3FreshCompletion(bool trace) : C3CompletionProbe(trace) {}
  void ObserveFresh(uint32_t poll, uint32_t valid, uint32_t done) {
    if (!samples) first_valid = valid;
    last_valid = valid;
    valid_events += valid == 1;
    Observe(poll, done);
  }
  void ReportFresh(uint64_t address, uint32_t insns, bool timeout, double run_us) const {
    if (!enabled) return;
    fprintf(stderr,
      "[C3_COMPLETION] {\"insn_phy_addr\":%llu,\"insn_count\":%u,"
      "\"before_done\":%u,\"first_done\":%u,\"return_done\":%u,"
      "\"samples\":%llu,\"changes\":%llu,\"saw_zero\":%s,"
      "\"before_valid\":%u,\"after_clear_valid\":%u,\"first_valid\":%u,"
      "\"return_valid\":%u,\"valid_events\":%llu,\"valid_offset\":28,"
      "\"timeout\":%s,\"driver_run_total_us\":%.3f}\n",
      static_cast<unsigned long long>(address), insns, before_done, first_done, last_done,
      static_cast<unsigned long long>(samples), static_cast<unsigned long long>(changes),
      saw_zero ? "true" : "false", before_valid, after_clear_valid, first_valid, last_valid,
      static_cast<unsigned long long>(valid_events), timeout ? "true" : "false", run_us);
  }
};
#endif
