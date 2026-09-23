// Isolated experiment instrumentation; does not change the completion decision.
#ifndef C3_COMPLETION_PROBE_H_
#define C3_COMPLETION_PROBE_H_
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct C3CompletionProbe {
  bool enabled = false;
  uint32_t before_done = 0, first_done = 0, last_done = 0;
  uint64_t samples = 0, changes = 0;
  bool saw_zero = false, saw_high_then_zero = false;
  uint32_t first_zero_poll = 0;
  explicit C3CompletionProbe(bool active) : enabled(active) {}
  void Observe(uint32_t poll, uint32_t value) {
    if (!enabled) return;
    if (!samples) first_done = value;
    if (samples && last_done != value) ++changes;
    if (value == 0 && !saw_zero) {
      first_zero_poll = poll;
      saw_high_then_zero = samples && first_done == 1;
    }
    saw_zero |= value == 0;
    last_done = value;
    ++samples;
  }
  void Report(uint64_t address, uint32_t insns, bool timeout, double run_us) const {
    if (!enabled) return;
    fprintf(stderr,
      "[C3_COMPLETION] {\"insn_phy_addr\":%llu,\"insn_count\":%u,"
      "\"before_done\":%u,\"first_done\":%u,\"return_done\":%u,"
      "\"samples\":%llu,\"changes\":%llu,\"saw_zero\":%s,"
      "\"first_zero_poll\":%u,\"saw_high_then_zero\":%s,"
      "\"timeout\":%s,\"driver_run_total_us\":%.3f}\n",
      static_cast<unsigned long long>(address), insns, before_done, first_done, last_done,
      static_cast<unsigned long long>(samples), static_cast<unsigned long long>(changes),
      saw_zero ? "true" : "false", first_zero_poll, saw_high_then_zero ? "true" : "false",
      timeout ? "true" : "false", run_us);
  }
};
#endif
