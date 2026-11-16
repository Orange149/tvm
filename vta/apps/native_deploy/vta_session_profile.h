/*
 * Session-scoped, opt-in Linux scheduler and PMU diagnostics.
 * This header deliberately has no dependency on TVM or VTA runtime state.
 */
#ifndef TVM_VTA_APPS_NATIVE_DEPLOY_VTA_SESSION_PROFILE_H_
#define TVM_VTA_APPS_NATIVE_DEPLOY_VTA_SESSION_PROFILE_H_

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace vta_deploy {

class SessionProfile {
 public:
  explicit SessionProfile(std::string output_path) : output_path_(std::move(output_path)) {}
  ~SessionProfile() { CloseCounters(); }

  bool enabled() const { return !output_path_.empty(); }

  void Start() noexcept {
    if (!enabled() || started_) return;
    started_ = true;
#if defined(__linux__)
    try {
      system_start_ = CaptureSystem();
      tids_start_ = TaskIds();
      if (tids_start_.empty()) {
        fatal_reason_ = "no_process_tasks_found";
        return;
      }
      for (pid_t tid : tids_start_) {
        TaskSample sample;
        if (ReadTask(tid, &sample)) start_[tid] = std::move(sample);
      }
      OpenEvents();
      for (auto& event : events_) event.Start();
    } catch (const std::exception& err) {
      fatal_reason_ = std::string("start_exception:") + err.what();
    } catch (...) {
      fatal_reason_ = "start_unknown_exception";
    }
#else
    fatal_reason_ = "unsupported_platform";
#endif
  }

  void Stop() noexcept {
    if (!enabled() || stopped_) return;
    stopped_ = true;
#if defined(__linux__)
    try {
      for (auto& event : events_) event.Stop();
      tids_end_ = TaskIds();
      for (pid_t tid : tids_start_) {
        TaskSample sample;
        if (ReadTask(tid, &sample)) end_[tid] = std::move(sample);
      }
      system_end_ = CaptureSystem();
      WriteJson();
    } catch (const std::exception& err) {
      fatal_reason_ = std::string("stop_exception:") + err.what();
      WriteFallbackJson();
    } catch (...) {
      fatal_reason_ = "stop_unknown_exception";
      WriteFallbackJson();
    }
#else
    WriteFallbackJson();
#endif
  }

 private:
  struct TaskSample {
    std::string comm;
    std::vector<unsigned int> cpus;
    uint64_t runtime_ns{0};
    uint64_t runqueue_wait_ns{0};
    uint64_t timeslices{0};
    uint64_t voluntary_switches{0};
    uint64_t nonvoluntary_switches{0};
  };

  struct PerfRead {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
  };

  struct SystemMetric {
    std::string path;
    bool available{false};
    int64_t value{0};
  };

  struct SystemSample {
    std::vector<SystemMetric> cpu_frequency_khz;
    std::vector<SystemMetric> temperature_millic;
  };

  struct EventCounter {
    std::string name;
    uint64_t config{0};
    bool available{false};
    std::string reason{"not_opened"};
    std::vector<int> fds;
    uint64_t raw_value{0};
    double scaled_value{0.0};
    uint64_t time_enabled{0};
    uint64_t time_running{0};

    void Close() {
#if defined(__linux__)
      for (int fd : fds) close(fd);
#endif
      fds.clear();
    }

    void Start() {
#if defined(__linux__)
      if (!available) return;
      for (int fd : fds) {
        if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) != 0 ||
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) != 0) {
          available = false;
          reason = std::string("perf_ioctl_start:") + std::strerror(errno);
          return;
        }
      }
#endif
    }

    void Stop() {
#if defined(__linux__)
      if (!available) return;
      for (int fd : fds) {
        if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) != 0) {
          available = false;
          reason = std::string("perf_ioctl_stop:") + std::strerror(errno);
          return;
        }
        PerfRead value{};
        if (read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value))) {
          available = false;
          reason = std::string("perf_read:") + std::strerror(errno);
          return;
        }
        raw_value += value.value;
        time_enabled += value.time_enabled;
        time_running += value.time_running;
        if (value.time_running != 0) {
          scaled_value += static_cast<double>(value.value) *
                          static_cast<double>(value.time_enabled) /
                          static_cast<double>(value.time_running);
        }
      }
      if (available) reason.clear();
#endif
    }
  };

#if defined(__linux__)
  static SystemMetric ReadMetric(const std::string& path) {
    SystemMetric metric;
    metric.path = path;
    std::ifstream input(path);
    if (input >> metric.value) metric.available = true;
    return metric;
  }

  static SystemSample CaptureSystem() {
    SystemSample sample;
    const long cpu_count = sysconf(_SC_NPROCESSORS_CONF);
    for (long cpu = 0; cpu < std::max<long>(cpu_count, 0); ++cpu) {
      sample.cpu_frequency_khz.push_back(ReadMetric(
          "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
          "/cpufreq/scaling_cur_freq"));
    }
    for (int zone = 0; zone < 64; ++zone) {
      const std::string path =
          "/sys/class/thermal/thermal_zone" + std::to_string(zone) + "/temp";
      std::ifstream probe(path);
      if (probe.good()) sample.temperature_millic.push_back(ReadMetric(path));
    }
    return sample;
  }
#endif

#if defined(__linux__)
  static std::vector<pid_t> TaskIds() {
    std::vector<pid_t> tids;
    DIR* dir = opendir("/proc/self/task");
    if (dir == nullptr) return tids;
    while (dirent* entry = readdir(dir)) {
      if (entry->d_name[0] == '.') continue;
      char* end = nullptr;
      long value = std::strtol(entry->d_name, &end, 10);
      if (end != entry->d_name && *end == '\0' && value > 0) {
        tids.push_back(static_cast<pid_t>(value));
      }
    }
    closedir(dir);
    std::sort(tids.begin(), tids.end());
    return tids;
  }

  static bool ReadTask(pid_t tid, TaskSample* out) {
    const std::string base = "/proc/self/task/" + std::to_string(tid) + "/";
    std::ifstream sched(base + "schedstat");
    if (!(sched >> out->runtime_ns >> out->runqueue_wait_ns >> out->timeslices)) return false;
    std::ifstream comm(base + "comm");
    std::getline(comm, out->comm);
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(tid, sizeof(set), &set) == 0) {
      for (unsigned int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) out->cpus.push_back(cpu);
      }
    }
    std::ifstream status(base + "status");
    std::string line;
    while (std::getline(status, line)) {
      const char* voluntary = "voluntary_ctxt_switches:";
      const char* involuntary = "nonvoluntary_ctxt_switches:";
      if (line.rfind(voluntary, 0) == 0) {
        out->voluntary_switches = std::strtoull(line.c_str() + std::strlen(voluntary), nullptr, 10);
      } else if (line.rfind(involuntary, 0) == 0) {
        out->nonvoluntary_switches =
            std::strtoull(line.c_str() + std::strlen(involuntary), nullptr, 10);
      }
    }
    return true;
  }

  void OpenEvents() {
    const std::pair<uint64_t, const char*> definitions[] = {
        {PERF_COUNT_HW_CPU_CYCLES, "cycles"},
        {PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
        {PERF_COUNT_HW_CACHE_REFERENCES, "cache_references"},
        {PERF_COUNT_HW_CACHE_MISSES, "cache_misses"},
    };
    for (const auto& definition : definitions) {
      EventCounter event;
      event.name = definition.second;
      event.config = definition.first;
      event.available = true;
      for (pid_t tid : tids_start_) {
        perf_event_attr attr{};
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(attr);
        attr.config = event.config;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, tid, -1, -1, 0));
        if (fd < 0) {
          event.available = false;
          event.reason = std::string("perf_event_open:") + std::strerror(errno);
          break;
        }
        event.fds.push_back(fd);
      }
      if (!event.available) event.Close();
      events_.push_back(std::move(event));
    }
  }
#endif

  static std::string Escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
      if (c == '"' || c == '\\') out << '\\' << c;
      else if (c == '\n') out << "\\n";
      else if (c < 0x20) out << "?";
      else out << c;
    }
    return out.str();
  }

  static uint64_t Delta(uint64_t before, uint64_t after) {
    return after >= before ? after - before : 0;
  }

  static void WriteSystem(std::ostream& out, const SystemSample& sample) {
    auto write_metrics = [&](const std::vector<SystemMetric>& metrics) {
      out << "[";
      for (size_t i = 0; i < metrics.size(); ++i) {
        if (i != 0) out << ",";
        out << "{\"path\":\"" << Escape(metrics[i].path) << "\",\"available\":"
            << (metrics[i].available ? "true" : "false") << ",\"value\":";
        if (metrics[i].available) out << metrics[i].value;
        else out << "null";
        out << "}";
      }
      out << "]";
    };
    out << "{\"cpu_frequency_khz\":";
    write_metrics(sample.cpu_frequency_khz);
    out << ",\"temperature_millic\":";
    write_metrics(sample.temperature_millic);
    out << "}";
  }

  void WriteJson() {
    std::ofstream out(output_path_, std::ios::out);
    if (!out) {
      fatal_reason_ = "unable_to_write_profile";
      return;
    }
    uint64_t runtime = 0, wait = 0, slices = 0, voluntary = 0, nonvoluntary = 0;
    out << std::fixed << std::setprecision(3)
        << "{\"schema\":\"vta_session_profile_v1\",\"scope\":\"whole_process_existing_threads\""
        << ",\"started\":" << (started_ ? "true" : "false")
        << ",\"fatal_reason\":";
    if (fatal_reason_.empty()) out << "null";
    else out << "\"" << Escape(fatal_reason_) << "\"";
    out << ",\"threads\":[";
    bool first = true;
    for (pid_t tid : tids_start_) {
      auto a = start_.find(tid), b = end_.find(tid);
      if (a == start_.end() || b == end_.end()) continue;
      uint64_t r = Delta(a->second.runtime_ns, b->second.runtime_ns);
      uint64_t w = Delta(a->second.runqueue_wait_ns, b->second.runqueue_wait_ns);
      uint64_t s = Delta(a->second.timeslices, b->second.timeslices);
      uint64_t v = Delta(a->second.voluntary_switches, b->second.voluntary_switches);
      uint64_t n = Delta(a->second.nonvoluntary_switches, b->second.nonvoluntary_switches);
      runtime += r; wait += w; slices += s; voluntary += v; nonvoluntary += n;
      if (!first) out << ',';
      first = false;
      out << "{\"tid\":" << tid << ",\"comm\":\"" << Escape(a->second.comm)
          << "\",\"cpus\":[";
      for (size_t i = 0; i < a->second.cpus.size(); ++i) {
        if (i) out << ',';
        out << a->second.cpus[i];
      }
      out << "],\"runtime_ns\":" << r << ",\"runqueue_wait_ns\":" << w
          << ",\"timeslices\":" << s << ",\"voluntary_switches\":" << v
          << ",\"nonvoluntary_switches\":" << n << '}';
    }
    out << "],\"aggregate\":{\"runtime_ns\":" << runtime
        << ",\"runqueue_wait_ns\":" << wait << ",\"timeslices\":" << slices
        << ",\"voluntary_switches\":" << voluntary
        << ",\"nonvoluntary_switches\":" << nonvoluntary << "},\"pmu\":{";
    for (size_t i = 0; i < events_.size(); ++i) {
      if (i) out << ',';
      const auto& event = events_[i];
      out << "\"" << event.name << "\":{\"available\":"
          << (event.available ? "true" : "false") << ",\"reason\":";
      if (event.reason.empty()) out << "null";
      else out << "\"" << Escape(event.reason) << "\"";
      out << ",\"raw\":" << event.raw_value << ",\"scaled\":" << event.scaled_value
          << ",\"time_enabled\":" << event.time_enabled
          << ",\"time_running\":" << event.time_running << '}';
    }
    out << "},\"system_start\":";
    WriteSystem(out, system_start_);
    out << ",\"system_end\":";
    WriteSystem(out, system_end_);
    out << ",\"start_tid_count\":" << tids_start_.size()
        << ",\"end_tid_count\":" << tids_end_.size() << "}\n";
  }

  void WriteFallbackJson() noexcept {
    try {
      std::ofstream out(output_path_, std::ios::out);
      if (out) out << "{\"schema\":\"vta_session_profile_v1\",\"fatal_reason\":\""
                   << Escape(fatal_reason_) << "\"}\n";
    } catch (...) {
    }
  }

  void CloseCounters() {
    for (auto& event : events_) event.Close();
  }

  std::string output_path_;
  bool started_{false};
  bool stopped_{false};
  std::string fatal_reason_;
  std::vector<pid_t> tids_start_;
  std::vector<pid_t> tids_end_;
  std::map<pid_t, TaskSample> start_;
  std::map<pid_t, TaskSample> end_;
  std::vector<EventCounter> events_;
  SystemSample system_start_;
  SystemSample system_end_;
};

}  // namespace vta_deploy

#endif  // TVM_VTA_APPS_NATIVE_DEPLOY_VTA_SESSION_PROFILE_H_
