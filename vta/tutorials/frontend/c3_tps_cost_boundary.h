// Experiment-only boundary observation. No execution/cache/token policy changes.
#ifndef C3_TPS_COST_BOUNDARY_H_
#define C3_TPS_COST_BOUNDARY_H_
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
namespace c3_tps_cost {
struct State {
  bool enabled;
  uint64_t begins{}, ends{}, clock_reads{};
  int64_t driver_begin_ns{}, driver_end_ns{};
  int timeout{};
  State() {
    const char* mode = std::getenv("C3_TPS_COST_MODE");
    if (mode && std::strcmp(mode, "on") == 0) enabled = true;
    else if (mode && std::strcmp(mode, "off") == 0) enabled = false;
    else std::abort();  // No implicit measurement mode.
  }
};
inline State& Get() { static State state; return state; }
inline int64_t Clock() {
  ++Get().clock_reads;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void Clear() {
  auto& s = Get();
  s.begins = s.ends = s.clock_reads = 0;
  s.driver_begin_ns = s.driver_end_ns = 0;
  s.timeout = 0;
}
inline void BeginDriver() {
  auto& s = Get();
  ++s.begins;
  if (s.enabled) s.driver_begin_ns = Clock();
}
inline void EndDriver(int timeout) {
  auto& s = Get();
  if (s.enabled) s.driver_end_ns = Clock();
  ++s.ends;
  s.timeout = timeout;
}
inline std::string JSON() {
  const auto& s = Get();
  std::ostringstream out;
  out << "{\"schema\":\"c3_tps_cost_boundary_v1\",\"mode\":\""
      << (s.enabled ? "on" : "off") << "\",\"begins\":" << s.begins
      << ",\"ends\":" << s.ends << ",\"clock_reads\":" << s.clock_reads
      << ",\"driver_begin_ns\":" << s.driver_begin_ns
      << ",\"driver_end_ns\":" << s.driver_end_ns << ",\"timeout\":" << s.timeout << '}';
  return out.str();
}
}  // namespace c3_tps_cost
#endif
