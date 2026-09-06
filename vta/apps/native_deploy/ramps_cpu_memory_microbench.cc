#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

double WallMs() {
  using Clock = std::chrono::steady_clock;
  static const auto origin = Clock::now();
  return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
}

double ProcessCpuMs() {
  timespec value{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
  }
  return static_cast<double>(value.tv_sec) * 1000.0 +
         static_cast<double>(value.tv_nsec) / 1.0e6;
}

struct Args {
  std::string operation;
  std::string case_id;
  std::string output;
  std::string pressure_class{"custom"};
  std::string ready_file;
  std::string start_file;
  std::string barrier_token;
  int threads{1};
  int streams{1};
  int warmup{5};
  int runs{20};
  int inner_repeats{8};
  int start_timeout_s{120};
  size_t bytes{8 * 1024 * 1024};
  bool profile_pmu{false};
  bool barrier_each_run{false};
};

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (char ch : value) {
    if (ch == '\\' || ch == '"') output << '\\';
    if (ch == '\n') {
      output << "\\n";
    } else {
      output << ch;
    }
  }
  return output.str();
}

std::string Trim(const std::string& value) {
  const size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

void SignalReadyAndWait(const Args& args, int run_index = -1) {
  if (args.ready_file.empty()) return;
  const std::string suffix = run_index < 0 ? "" : "." + std::to_string(run_index);
  const std::string ready_file = args.ready_file + suffix;
  const std::string start_file = args.start_file + suffix;
  const std::filesystem::path ready(ready_file);
  if (!ready.parent_path().empty()) {
    std::filesystem::create_directories(ready.parent_path());
  }
  {
    std::ofstream output(ready, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to write ready barrier " + ready_file);
    output << args.barrier_token;
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(args.start_timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::is_regular_file(start_file)) {
      std::ifstream input(start_file);
      std::ostringstream contents;
      contents << input.rdbuf();
      if (Trim(contents.str()) == args.barrier_token) return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  throw std::runtime_error("timed out waiting for start barrier " + start_file);
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("missing value for " + key);
      return argv[i];
    };
    if (key == "--operation") args.operation = value();
    else if (key == "--case-id") args.case_id = value();
    else if (key == "--output") args.output = value();
    else if (key == "--pressure-class") args.pressure_class = value();
    else if (key == "--ready-file") args.ready_file = value();
    else if (key == "--start-file") args.start_file = value();
    else if (key == "--barrier-token") args.barrier_token = value();
    else if (key == "--threads") args.threads = std::stoi(value());
    else if (key == "--streams") args.streams = std::stoi(value());
    else if (key == "--warmup") args.warmup = std::stoi(value());
    else if (key == "--runs") args.runs = std::stoi(value());
    else if (key == "--inner-repeats") args.inner_repeats = std::stoi(value());
    else if (key == "--start-timeout-s") args.start_timeout_s = std::stoi(value());
    else if (key == "--bytes") args.bytes = static_cast<size_t>(std::stoull(value()));
    else if (key == "--profile-pmu") args.profile_pmu = true;
    else if (key == "--barrier-each-run") args.barrier_each_run = true;
    else throw std::runtime_error("unknown argument " + key);
  }
  if (args.operation != "read" && args.operation != "write" && args.operation != "copy") {
    throw std::runtime_error("--operation must be read, write, or copy");
  }
  if (args.pressure_class != "cache_resident" && args.pressure_class != "transition" &&
      args.pressure_class != "streaming" && args.pressure_class != "custom") {
    throw std::runtime_error(
        "--pressure-class must be cache_resident, transition, streaming, or custom");
  }
  const bool any_barrier = !args.ready_file.empty() || !args.start_file.empty() ||
                           !args.barrier_token.empty();
  const bool complete_barrier = !args.ready_file.empty() && !args.start_file.empty() &&
                                !args.barrier_token.empty();
  if (any_barrier && !complete_barrier) {
    throw std::runtime_error(
        "--ready-file, --start-file and --barrier-token must be specified together");
  }
  if (args.barrier_each_run && !complete_barrier) {
    throw std::runtime_error("--barrier-each-run requires a complete barrier configuration");
  }
  if (args.case_id.empty() || args.output.empty() || args.threads < 1 || args.streams < 1 ||
      args.warmup < 0 || args.runs < 1 || args.inner_repeats < 1 || args.start_timeout_s < 1 ||
      args.bytes < 4096) {
    throw std::runtime_error("invalid CPU memory microbenchmark arguments");
  }
  return args;
}

struct Buffers {
  std::vector<std::vector<uint64_t>> source;
  std::vector<std::vector<uint64_t>> destination;
};

struct PmuSnapshot {
  bool available{false};
  std::string unavailable_reason{"disabled_by_cli"};
  uint64_t cycles{0};
  uint64_t instructions{0};
  uint64_t cache_references{0};
  uint64_t cache_misses{0};
};

class ProcessPmuCounters {
 public:
  explicit ProcessPmuCounters(bool requested) {
    if (!requested) return;
#if defined(__linux__)
    const std::pair<uint64_t, const char*> events[] = {
        {PERF_COUNT_HW_CPU_CYCLES, "cycles"},
        {PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
        {PERF_COUNT_HW_CACHE_REFERENCES, "cache_references"},
        {PERF_COUNT_HW_CACHE_MISSES, "cache_misses"},
    };
    DIR* directory = opendir("/proc/self/task");
    if (directory == nullptr) {
      unavailable_reason_ = "unable_to_list_process_tasks";
      return;
    }
    std::vector<pid_t> tids;
    while (dirent* entry = readdir(directory)) {
      if (entry->d_name[0] == '.') continue;
      char* end = nullptr;
      const long value = std::strtol(entry->d_name, &end, 10);
      if (end != entry->d_name && *end == '\0' && value > 0) {
        tids.push_back(static_cast<pid_t>(value));
      }
    }
    closedir(directory);
    for (size_t event_index = 0; event_index < 4; ++event_index) {
      for (pid_t tid : tids) {
        perf_event_attr attr{};
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(attr);
        attr.config = events[event_index].first;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        const int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, tid, -1, -1, 0));
        if (fd < 0) {
          unavailable_reason_ = std::string("perf_event_open_") + events[event_index].second +
                                ":" + std::strerror(errno);
          CloseAll();
          return;
        }
        counters_.push_back({fd, event_index});
      }
    }
    available_ = true;
    unavailable_reason_.clear();
#else
    unavailable_reason_ = "perf_event_open_unsupported_platform";
#endif
  }

  ~ProcessPmuCounters() { CloseAll(); }

  void Start() {
#if defined(__linux__)
    if (!available_) return;
    for (const Counter& counter : counters_) {
      if (ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0) != 0 ||
          ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0) != 0) {
        MarkUnavailable(std::string("perf_ioctl_start:") + std::strerror(errno));
        return;
      }
    }
#endif
  }

  PmuSnapshot Stop() {
    PmuSnapshot result;
    result.available = available_;
    result.unavailable_reason = unavailable_reason_;
#if defined(__linux__)
    if (!available_) return result;
    uint64_t totals[4] = {0, 0, 0, 0};
    for (const Counter& counter : counters_) {
      uint64_t value = 0;
      if (ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0) != 0 ||
          read(counter.fd, &value, sizeof(value)) != sizeof(value)) {
        MarkUnavailable(std::string("perf_stop_or_read:") + std::strerror(errno));
        result.available = false;
        result.unavailable_reason = unavailable_reason_;
        return result;
      }
      totals[counter.event_index] += value;
    }
    result.cycles = totals[0];
    result.instructions = totals[1];
    result.cache_references = totals[2];
    result.cache_misses = totals[3];
#endif
    return result;
  }

 private:
  struct Counter {
    int fd;
    size_t event_index;
  };

  void MarkUnavailable(const std::string& reason) {
    available_ = false;
    unavailable_reason_ = reason;
    CloseAll();
  }

  void CloseAll() {
#if defined(__linux__)
    for (const Counter& counter : counters_) close(counter.fd);
#endif
    counters_.clear();
  }

  bool available_{false};
  std::string unavailable_reason_{"disabled_by_cli"};
  std::vector<Counter> counters_;
};

std::string ConfigureAffinity(int threads) {
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads != 0 && static_cast<unsigned int>(threads) > hardware_threads) {
    throw std::runtime_error("thread count exceeds available CPU cores");
  }
  cpu_set_t requested;
  CPU_ZERO(&requested);
  for (int cpu = 0; cpu < threads; ++cpu) CPU_SET(cpu, &requested);
  if (sched_setaffinity(0, sizeof(requested), &requested) != 0) {
    throw std::runtime_error("sched_setaffinity failed");
  }
  cpu_set_t effective;
  CPU_ZERO(&effective);
  if (sched_getaffinity(0, sizeof(effective), &effective) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  std::string result = "[";
  int observed = 0;
  for (unsigned int cpu = 0; cpu < hardware_threads; ++cpu) {
    if (!CPU_ISSET(cpu, &effective)) continue;
    if (observed++) result += ",";
    result += std::to_string(cpu);
  }
  result += "]";
  if (observed != threads) throw std::runtime_error("effective CPU affinity count mismatch");
  return result;
}

class MemoryWorkers {
 public:
  MemoryWorkers(const Args& args, Buffers* buffers)
      : args_(args), buffers_(buffers), partial_(static_cast<size_t>(args.threads), 0) {
    for (int thread_id = 0; thread_id < args_.threads; ++thread_id) {
      workers_.emplace_back(&MemoryWorkers::WorkerLoop, this, thread_id);
    }
  }

  ~MemoryWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    work_ready_.notify_all();
    for (std::thread& worker : workers_) worker.join();
  }

  uint64_t RunOnce() {
    std::unique_lock<std::mutex> lock(mutex_);
    completed_ = 0;
    ++generation_;
    work_ready_.notify_all();
    work_done_.wait(lock, [&]() { return completed_ == args_.threads; });
    uint64_t checksum = 0;
    for (uint64_t value : partial_) checksum ^= value;
    return checksum;
  }

 private:
  void WorkerLoop(int thread_id) {
    uint64_t observed_generation = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_ready_.wait(lock, [&]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) return;
      observed_generation = generation_;
      lock.unlock();
      const uint64_t local = ExecutePartition(thread_id);
      lock.lock();
      partial_[static_cast<size_t>(thread_id)] = local;
      ++completed_;
      if (completed_ == args_.threads) work_done_.notify_one();
    }
  }

  uint64_t ExecutePartition(int thread_id) {
    const size_t words = args_.bytes / sizeof(uint64_t);
    const size_t begin = words * static_cast<size_t>(thread_id) /
                         static_cast<size_t>(args_.threads);
    const size_t end = words * static_cast<size_t>(thread_id + 1) /
                       static_cast<size_t>(args_.threads);
    uint64_t local = 0;
    for (int repeat = 0; repeat < args_.inner_repeats; ++repeat) {
      for (int stream = 0; stream < args_.streams; ++stream) {
        uint64_t* dst = buffers_->destination[static_cast<size_t>(stream)].data();
        const uint64_t* src = buffers_->source[static_cast<size_t>(stream)].data();
        if (args_.operation == "read") {
          for (size_t index = begin; index < end; ++index) local += src[index];
        } else if (args_.operation == "write") {
          const uint64_t value = static_cast<uint64_t>(repeat + stream + thread_id + 1);
          std::fill(dst + begin, dst + end, value);
          local += dst[begin];
        } else {
          std::memcpy(dst + begin, src + begin, (end - begin) * sizeof(uint64_t));
          local += dst[begin];
        }
      }
    }
    return local;
  }

  const Args& args_;
  Buffers* buffers_;
  std::vector<uint64_t> partial_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  uint64_t generation_{0};
  int completed_{0};
  bool stopping_{false};
};

uint64_t ReferenceChecksum(const Args& args, const Buffers& buffers) {
  const size_t words = args.bytes / sizeof(uint64_t);
  uint64_t checksum = 0;
  for (int thread_id = 0; thread_id < args.threads; ++thread_id) {
    const size_t begin = words * static_cast<size_t>(thread_id) /
                         static_cast<size_t>(args.threads);
    const size_t end = words * static_cast<size_t>(thread_id + 1) /
                       static_cast<size_t>(args.threads);
    uint64_t local = 0;
    for (int repeat = 0; repeat < args.inner_repeats; ++repeat) {
      for (int stream = 0; stream < args.streams; ++stream) {
        const uint64_t* src = buffers.source[static_cast<size_t>(stream)].data();
        if (args.operation == "read") {
          for (size_t index = begin; index < end; ++index) local += src[index];
        } else if (args.operation == "write") {
          local += static_cast<uint64_t>(repeat + stream + thread_id + 1);
        } else {
          local += src[begin];
        }
      }
    }
    checksum ^= local;
  }
  return checksum;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const std::string cpu_affinity = ConfigureAffinity(args.threads);
    const size_t words = args.bytes / sizeof(uint64_t);
    Buffers buffers;
    buffers.source.resize(static_cast<size_t>(args.streams));
    buffers.destination.resize(static_cast<size_t>(args.streams));
    for (int stream = 0; stream < args.streams; ++stream) {
      buffers.source[static_cast<size_t>(stream)].resize(words);
      buffers.destination[static_cast<size_t>(stream)].resize(words);
      for (size_t index = 0; index < words; ++index) {
        buffers.source[static_cast<size_t>(stream)][index] = index + static_cast<size_t>(stream) + 1;
      }
    }
    MemoryWorkers workers(args, &buffers);
    for (int index = 0; index < args.warmup; ++index) workers.RunOnce();
    const uint64_t expected_checksum = ReferenceChecksum(args, buffers);
    if (!args.barrier_each_run) SignalReadyAndWait(args);
    ProcessPmuCounters process_pmu(args.profile_pmu);
    std::ofstream output(args.output, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to open output " + args.output);
    const uint64_t bytes_per_repeat = static_cast<uint64_t>(args.bytes) * args.streams *
                                      (args.operation == "copy" ? 2 : 1);
    const uint64_t active_working_set_bytes =
        static_cast<uint64_t>(args.bytes) * args.streams *
        (args.operation == "copy" ? 2 : 1);
    for (int index = 0; index < args.runs; ++index) {
      if (args.barrier_each_run) SignalReadyAndWait(args, index);
      process_pmu.Start();
      const double cpu_start = ProcessCpuMs();
      const double wall_start = WallMs();
      const uint64_t checksum = workers.RunOnce();
      const double wall_ms = WallMs() - wall_start;
      const double process_cpu_ms = ProcessCpuMs() - cpu_start;
      const PmuSnapshot pmu = process_pmu.Stop();
      const uint64_t traffic_bytes = bytes_per_repeat * args.inner_repeats;
      const double bandwidth = static_cast<double>(traffic_bytes) / std::max(wall_ms, 1.0e-12) / 1.0e6;
      output << std::fixed << std::setprecision(9)
             << "{\"case_id\":\"" << args.case_id << "\",\"operation\":\""
             << args.operation << "\",\"run_index\":" << index
             << ",\"pressure_class\":\"" << args.pressure_class << "\""
             << ",\"barrier_synchronized\":"
             << (args.ready_file.empty() ? "false" : "true")
             << ",\"threads\":" << args.threads << ",\"streams\":" << args.streams
             << ",\"cpu_affinity\":" << cpu_affinity
             << ",\"operand_bytes_per_stream\":" << args.bytes
             << ",\"working_set_bytes\":" << active_working_set_bytes
             << ",\"traffic_bytes\":" << traffic_bytes << ",\"wall_ms\":" << wall_ms
             << ",\"process_cpu_ms\":" << process_cpu_ms
             << ",\"average_cpu_cores\":" << process_cpu_ms / std::max(wall_ms, 1.0e-12)
             << ",\"worker_lifecycle\":\"persistent_process_threads\""
             << ",\"pmu_available\":" << (pmu.available ? "true" : "false")
             << ",\"pmu_unavailable_reason\":";
      if (pmu.available) {
        output << "null,\"pmu_cycles\":" << pmu.cycles
               << ",\"pmu_instructions\":" << pmu.instructions
               << ",\"pmu_cache_references\":" << pmu.cache_references
               << ",\"pmu_cache_misses\":" << pmu.cache_misses;
      } else {
        output << "\"" << JsonEscape(pmu.unavailable_reason) << "\""
               << ",\"pmu_cycles\":null,\"pmu_instructions\":null"
               << ",\"pmu_cache_references\":null,\"pmu_cache_misses\":null";
      }
      output
             << ",\"bandwidth_GBps\":" << bandwidth << ",\"checksum\":" << checksum
             << ",\"expected_checksum\":" << expected_checksum
             << ",\"correct\":" << (checksum == expected_checksum ? "true" : "false")
             << "}\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[RAMPS CPU MEMORY ERROR] " << error.what() << std::endl;
    return 2;
  }
}
