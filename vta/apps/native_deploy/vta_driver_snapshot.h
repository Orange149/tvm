#ifndef VTA_DEPLOY_DRIVER_SNAPSHOT_H_
#define VTA_DEPLOY_DRIVER_SNAPSHOT_H_
#include <vta/driver.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

namespace vta_deploy {
static_assert(sizeof(VTADriverProfilerStats) == 64, "Frozen driver profiler ABI changed");
inline double SnapshotWallMillis() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double SnapshotThreadCpuMillis() {
  timespec t{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t))
    throw std::runtime_error("Thread CPU clock unavailable");
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
struct ThreadSchedCounters {
  uint64_t runtime_ns{0}, runqueue_ns{0}, dispatches{0};
};
inline ThreadSchedCounters ReadThreadSchedCounters(long tid) {
  std::ifstream f("/proc/self/task/" + std::to_string(tid) + "/schedstat");
  ThreadSchedCounters s;
  if (!(f >> s.runtime_ns >> s.runqueue_ns >> s.dispatches))
    throw std::runtime_error("Thread schedstat unavailable/malformed");
  return s;
}
inline uint64_t SnapshotCounterDelta(uint64_t a, uint64_t b) {
  if (b < a) throw std::runtime_error("Snapshot counter decreased");
  return b-a;
}
inline double SnapshotDurationDelta(double a, double b) {
  if (!std::isfinite(a) || !std::isfinite(b) || b < a)
    throw std::runtime_error("Snapshot duration decreased/nonfinite");
  return b-a;
}
struct DriverRunSnapshot {
  bool valid{false};
  long tid{0};
  VTADriverProfilerStats before{}, after{}, delta{};
  ThreadSchedCounters sched_before{}, sched_after{};
  double bracket_start_ms{0}, bracket_end_ms{0}, cpu_before_ms{0}, cpu_after_ms{0};
  double probe_before_ms{0}, probe_after_ms{0}, thread_cpu_ms{0}, runqueue_ms{0};
  uint64_t dispatches{0};
  void Begin() {
    bracket_start_ms=SnapshotWallMillis();
    cpu_before_ms=SnapshotThreadCpuMillis();
    tid=syscall(SYS_gettid);
    VTADriverProfilerStatus(&before);
    sched_before=ReadThreadSchedCounters(tid);
    probe_before_ms=SnapshotWallMillis()-bracket_start_ms;
  }
  void End(double run_end_ms) {
    VTADriverProfilerStatus(&after);
    sched_after=ReadThreadSchedCounters(tid);
    cpu_after_ms=SnapshotThreadCpuMillis();
    bracket_end_ms=SnapshotWallMillis();
    probe_after_ms=bracket_end_ms-run_end_ms;
    thread_cpu_ms=SnapshotDurationDelta(cpu_before_ms,cpu_after_ms);
    runqueue_ms=SnapshotCounterDelta(sched_before.runqueue_ns,sched_after.runqueue_ns)/1e6;
    dispatches=SnapshotCounterDelta(sched_before.dispatches,sched_after.dispatches);
    delta.run_calls=SnapshotCounterDelta(before.run_calls,after.run_calls);
    delta.run_insns=SnapshotCounterDelta(before.run_insns,after.run_insns);
    delta.timeout_calls=SnapshotCounterDelta(before.timeout_calls,after.timeout_calls);
    delta.poll_iters=SnapshotCounterDelta(before.poll_iters,after.poll_iters);
    delta.run_total_us=SnapshotDurationDelta(before.run_total_us,after.run_total_us);
    delta.submit_mmio_us=SnapshotDurationDelta(before.submit_mmio_us,after.submit_mmio_us);
    delta.post_start_sleep_us=SnapshotDurationDelta(before.post_start_sleep_us,after.post_start_sleep_us);
    delta.poll_wait_us=SnapshotDurationDelta(before.poll_wait_us,after.poll_wait_us);
    valid=true;
  }
  void WriteJSON(std::ostream& os) const {
    if (!valid) { os << "null"; return; }
    os << "{\"tid\":" << tid << ",\"bracket_start_monotonic_ms\":" << bracket_start_ms
       << ",\"bracket_end_monotonic_ms\":" << bracket_end_ms
       << ",\"probe_before_ms\":" << probe_before_ms << ",\"probe_after_ms\":" << probe_after_ms
       << ",\"thread_cpu_ms\":" << thread_cpu_ms << ",\"sched_runqueue_ms\":" << runqueue_ms
       << ",\"sched_dispatches\":" << dispatches
       << ",\"driver_run_calls\":" << delta.run_calls << ",\"driver_run_insns\":" << delta.run_insns
       << ",\"driver_timeout_calls\":" << delta.timeout_calls << ",\"driver_poll_iters\":" << delta.poll_iters
       << ",\"driver_total_ms\":" << delta.run_total_us/1000
       << ",\"driver_mmio_ms\":" << delta.submit_mmio_us/1000
       << ",\"driver_post_start_sleep_ms\":" << delta.post_start_sleep_us/1000
       << ",\"driver_poll_wait_ms\":" << delta.poll_wait_us/1000 << "}";
  }
};
}  // namespace vta_deploy
#endif
