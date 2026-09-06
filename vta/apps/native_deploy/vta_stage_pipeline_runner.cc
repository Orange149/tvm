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

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <vta/driver.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <errno.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

using tvm::runtime::Module;
using tvm::runtime::NDArray;
using tvm::runtime::PackedFunc;
using tvm::runtime::Registry;

struct StageArgs {
  std::string name;
  std::string device;
  std::string graph;
  std::string lib;
  std::string params;
  std::vector<std::string> input_names{"data0"};
  std::vector<std::string> input_sources;
  std::vector<unsigned int> cpu_affinity;
  int runtime_num_threads = -1;
};

struct Args {
  std::vector<StageArgs> stages;
  std::vector<unsigned int> process_cpu_affinity;
  std::string input;
  std::vector<std::string> input_files;
  std::string input_list;
  std::string output_jsonl = "native_result.jsonl";
  std::string output_mode = "classification";
  std::string output_dump_dir;
  std::string profile_dir;
  int runs = 1;
  int warmup_runs = 0;
  int queue_depth = 2;
  int runtime_num_threads = 4;
  int events_limit = 200;
  int checkpoint_every = 0;
  int start_timeout_s = 120;
  std::string ready_file;
  std::string start_file;
  std::string barrier_token;
  bool profile_pmu = false;
  bool barrier_each_run = false;
  bool serial = false;
  bool host_empty = false;
  int host_empty_stage_count = 1;
  int host_empty_inner_repeats = 1000;
  int p8a_edge = -1;
  int p8_managed_slots = 0;
  int p8_slot_timeout_s = 120;
};

double NowMillis() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(now).count();
}

double ProcessCpuMillis() {
  struct timespec value;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_PROCESS_CPUTIME_ID) failed");
  }
  return static_cast<double>(value.tv_sec) * 1000.0 +
         static_cast<double>(value.tv_nsec) / 1.0e6;
}

std::string ReadFile(const std::string& path, bool binary = false) {
  std::ios::openmode mode = std::ios::in;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ifstream in(path, mode);
  if (!in) {
    throw std::runtime_error("Unable to open " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void WriteFile(const std::string& path, const std::string& data) {
  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    throw std::runtime_error("Unable to write " + path);
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out << "\n";
}

void EnsureDir(const std::string& path) {
  if (!path.empty()) {
    std::filesystem::create_directories(path);
  }
}

std::string TrimLine(const std::string& line);

void WriteBarrierFile(const std::string& path, const std::string& token) {
  const std::filesystem::path file(path);
  if (!file.parent_path().empty()) {
    EnsureDir(file.parent_path().string());
  }
  WriteFile(path, token);
}

void SignalReadyAndWait(const Args& args, int run_index = -1) {
  if (args.ready_file.empty()) return;
  const std::string suffix = run_index < 0 ? "" : "." + std::to_string(run_index);
  const std::string ready_file = args.ready_file + suffix;
  const std::string start_file = args.start_file + suffix;
  WriteBarrierFile(ready_file, args.barrier_token);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(args.start_timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    if (std::filesystem::is_regular_file(start_file)) {
      std::ifstream in(start_file);
      std::ostringstream contents;
      contents << in.rdbuf();
      if (TrimLine(contents.str()) == args.barrier_token) return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  throw std::runtime_error("timed out waiting for start barrier " + start_file);
}

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
    if (!requested) {
      unavailable_reason_ = "disabled_by_cli";
      return;
    }
#if defined(__linux__)
    const std::vector<pid_t> tids = CurrentTaskIds();
    if (tids.empty()) {
      unavailable_reason_ = "no_process_tasks_found";
      return;
    }
    const std::pair<uint64_t, const char*> events[] = {
        {PERF_COUNT_HW_CPU_CYCLES, "cycles"},
        {PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
        {PERF_COUNT_HW_CACHE_REFERENCES, "cache_references"},
        {PERF_COUNT_HW_CACHE_MISSES, "cache_misses"},
    };
    for (size_t event_index = 0; event_index < 4; ++event_index) {
      for (pid_t tid : tids) {
        perf_event_attr attr{};
        attr.type = PERF_TYPE_HARDWARE;
        attr.size = sizeof(attr);
        attr.config = events[event_index].first;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        const int fd = static_cast<int>(
            syscall(__NR_perf_event_open, &attr, tid, -1, -1, 0));
        if (fd < 0) {
          unavailable_reason_ = std::string("perf_event_open_") + events[event_index].second +
                                ":" + std::strerror(errno);
          CloseAll();
          return;
        }
        counters_.push_back(Counter{fd, event_index});
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
    std::string failure_reason;
    for (const Counter& counter : counters_) {
      if (ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0) != 0 ||
          ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0) != 0) {
        failure_reason = std::string("perf_ioctl_start:") + std::strerror(errno);
        break;
      }
    }
    if (!failure_reason.empty()) MarkUnavailable(failure_reason);
#endif
  }

  PmuSnapshot Stop() {
    PmuSnapshot result;
    result.available = available_;
    result.unavailable_reason = unavailable_reason_;
#if defined(__linux__)
    if (!available_) return result;
    uint64_t totals[4] = {0, 0, 0, 0};
    std::string failure_reason;
    for (const Counter& counter : counters_) {
      if (ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0) != 0) {
        failure_reason = std::string("perf_ioctl_stop:") + std::strerror(errno);
        break;
      }
      uint64_t value = 0;
      if (read(counter.fd, &value, sizeof(value)) != sizeof(value)) {
        failure_reason = std::string("perf_read:") + std::strerror(errno);
        break;
      }
      totals[counter.event_index] += value;
    }
    if (!failure_reason.empty()) {
      MarkUnavailable(failure_reason);
      result.available = false;
      result.unavailable_reason = unavailable_reason_;
      return result;
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

#if defined(__linux__)
  static std::vector<pid_t> CurrentTaskIds() {
    std::vector<pid_t> tids;
    DIR* dir = opendir("/proc/self/task");
    if (dir == nullptr) return tids;
    while (dirent* entry = readdir(dir)) {
      if (entry->d_name[0] == '.') continue;
      char* end = nullptr;
      const long value = std::strtol(entry->d_name, &end, 10);
      if (end != entry->d_name && *end == '\0' && value > 0) {
        tids.push_back(static_cast<pid_t>(value));
      }
    }
    closedir(dir);
    std::sort(tids.begin(), tids.end());
    return tids;
  }
#endif

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
  std::string unavailable_reason_{"perf_event_open_not_initialized"};
  std::vector<Counter> counters_;
};

void LoadParamsBlob(const PackedFunc& load_params, const std::string& params_data) {
  TVMByteArray params;
  params.data = params_data.data();
  params.size = params_data.size();
  load_params(params);
}

std::string TrimLine(const std::string& line) {
  const size_t begin = line.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const size_t end = line.find_last_not_of(" \t\r\n");
  return line.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitCSV(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = TrimLine(item);
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  if (items.empty()) {
    throw std::runtime_error("empty comma-separated argument");
  }
  return items;
}

std::vector<unsigned int> ParseCPUList(const std::string& value) {
  std::vector<unsigned int> cpus;
  for (const std::string& item : SplitCSV(value)) {
    if (item.empty() || !std::all_of(item.begin(), item.end(), [](unsigned char ch) {
          return std::isdigit(ch);
        })) {
      throw std::runtime_error("invalid CPU id " + item);
    }
    size_t consumed = 0;
    const unsigned long parsed = std::stoul(item, &consumed);
    if (consumed != item.size()) {
      throw std::runtime_error("invalid CPU id " + item);
    }
    cpus.push_back(static_cast<unsigned int>(parsed));
  }
  std::sort(cpus.begin(), cpus.end());
  if (std::adjacent_find(cpus.begin(), cpus.end()) != cpus.end()) {
    throw std::runtime_error("CPU affinity contains duplicate core ids");
  }
  return cpus;
}

std::string CPUListString(const std::vector<unsigned int>& cpus) {
  std::ostringstream os;
  for (size_t i = 0; i < cpus.size(); ++i) {
    if (i != 0) os << ',';
    os << cpus[i];
  }
  return os.str();
}

std::string CurrentThreadAffinity() {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) {
    return "unavailable";
  }
  std::vector<unsigned int> cpus;
  for (unsigned int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &cpuset)) cpus.push_back(cpu);
  }
  return CPUListString(cpus);
#else
  return "unsupported";
#endif
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream os;
  for (char c : value) {
    switch (c) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << c;
        break;
    }
  }
  return os.str();
}

void Usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --stageN-graph PATH --stageN-lib PATH --stageN-params PATH\n"
      << "  --stageN-input-names data0[,data1]\n"
      << "  --stageN-input-sources input:0|stageM:K[,stageM:K]\n"
      << "  --stageN-device cpu|vta\n"
      << "  --stageN-name NAME\n"
      << "  --input PATH | --input-files PATH0[,PATH1] | --input-list PATH\n"
      << "  --runs N\n"
      << "  --warmup-runs N\n"
      << "  --queue-depth N\n"
      << "  --runtime-num-threads N\n"
      << "  --process-cpu-affinity CPU0[,CPU1]\n"
      << "  --stageN-runtime-num-threads N\n"
      << "  --stageN-cpu-affinity CPU0[,CPU1]\n"
      << "  --serial\n"
      << "  --output-mode classification|raw|raw_all_stages\n"
      << "  --output-jsonl PATH\n"
      << "  --output-dump-dir DIR\n"
      << "  --vta-runtime-profile-dir DIR\n"
      << "  --vta-runtime-profile-events-limit N\n"
      << "  --vta-runtime-profile-checkpoint-every N\n"
      << "  --ready-file PATH --start-file PATH --barrier-token TOKEN\n"
      << "  --barrier-each-run\n"
      << "  --start-timeout-s N\n"
      << "  --profile-pmu\n"
      << "  --host-empty\n"
      << "  --host-empty-stage-count N\n"
      << "  --host-empty-inner-repeats N\n"
      << "  --p8-edge N (legacy alias: --p8a-edge)\n"
      << "  --p8-managed-slots N (1 for serial B1, 2 for pipeline B2)\n"
      << "  --p8-slot-timeout-s N\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  auto ensure_stage = [&](size_t index) -> StageArgs& {
    if (args.stages.size() <= index) {
      args.stages.resize(index + 1);
    }
    return args.stages[index];
  };
  auto parse_stage_key = [](const std::string& key, size_t* index,
                            std::string* suffix) -> bool {
    const std::string prefix = "--stage";
    if (key.rfind(prefix, 0) != 0) {
      return false;
    }
    size_t pos = prefix.size();
    if (pos >= key.size() || !std::isdigit(static_cast<unsigned char>(key[pos]))) {
      return false;
    }
    size_t value = 0;
    while (pos < key.size() && std::isdigit(static_cast<unsigned char>(key[pos]))) {
      value = value * 10 + static_cast<size_t>(key[pos] - '0');
      ++pos;
    }
    if (pos >= key.size() || key[pos] != '-') {
      return false;
    }
    *index = value;
    *suffix = key.substr(pos + 1);
    return true;
  };
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string& opt) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + opt);
      }
      return argv[++i];
    };

    size_t stage_index = 0;
    std::string stage_suffix;
    if (parse_stage_key(key, &stage_index, &stage_suffix)) {
      StageArgs& stage = ensure_stage(stage_index);
      if (stage_suffix == "graph") {
        stage.graph = need_value(key);
      } else if (stage_suffix == "lib") {
        stage.lib = need_value(key);
      } else if (stage_suffix == "params") {
        stage.params = need_value(key);
      } else if (stage_suffix == "input-names") {
        stage.input_names = SplitCSV(need_value(key));
      } else if (stage_suffix == "input-sources") {
        stage.input_sources = SplitCSV(need_value(key));
      } else if (stage_suffix == "device") {
        stage.device = need_value(key);
      } else if (stage_suffix == "name") {
        stage.name = need_value(key);
      } else if (stage_suffix == "runtime-num-threads") {
        stage.runtime_num_threads = std::stoi(need_value(key));
      } else if (stage_suffix == "cpu-affinity") {
        stage.cpu_affinity = ParseCPUList(need_value(key));
      } else {
        throw std::runtime_error("Unknown stage option " + key);
      }
    } else if (key == "--input") {
      args.input = need_value(key);
    } else if (key == "--input-files") {
      args.input_files = SplitCSV(need_value(key));
    } else if (key == "--input-list") {
      args.input_list = need_value(key);
    } else if (key == "--runs") {
      args.runs = std::stoi(need_value(key));
    } else if (key == "--warmup-runs") {
      args.warmup_runs = std::stoi(need_value(key));
    } else if (key == "--queue-depth") {
      args.queue_depth = std::stoi(need_value(key));
    } else if (key == "--runtime-num-threads") {
      args.runtime_num_threads = std::stoi(need_value(key));
    } else if (key == "--process-cpu-affinity") {
      args.process_cpu_affinity = ParseCPUList(need_value(key));
    } else if (key == "--output-jsonl") {
      args.output_jsonl = need_value(key);
    } else if (key == "--output-mode") {
      args.output_mode = need_value(key);
    } else if (key == "--output-dump-dir") {
      args.output_dump_dir = need_value(key);
    } else if (key == "--serial") {
      args.serial = true;
    } else if (key == "--vta-runtime-profile-dir") {
      args.profile_dir = need_value(key);
    } else if (key == "--vta-runtime-profile-events-limit") {
      args.events_limit = std::stoi(need_value(key));
    } else if (key == "--vta-runtime-profile-checkpoint-every") {
      args.checkpoint_every = std::stoi(need_value(key));
    } else if (key == "--ready-file") {
      args.ready_file = need_value(key);
    } else if (key == "--start-file") {
      args.start_file = need_value(key);
    } else if (key == "--barrier-token") {
      args.barrier_token = need_value(key);
    } else if (key == "--start-timeout-s") {
      args.start_timeout_s = std::stoi(need_value(key));
    } else if (key == "--profile-pmu") {
      args.profile_pmu = true;
    } else if (key == "--barrier-each-run") {
      args.barrier_each_run = true;
    } else if (key == "--host-empty") {
      args.host_empty = true;
    } else if (key == "--host-empty-stage-count") {
      args.host_empty_stage_count = std::stoi(need_value(key));
    } else if (key == "--host-empty-inner-repeats") {
      args.host_empty_inner_repeats = std::stoi(need_value(key));
    } else if (key == "--p8-edge" || key == "--p8a-edge") {
      args.p8a_edge = std::stoi(need_value(key));
    } else if (key == "--p8-managed-slots") {
      args.p8_managed_slots = std::stoi(need_value(key));
    } else if (key == "--p8-slot-timeout-s") {
      args.p8_slot_timeout_s = std::stoi(need_value(key));
    } else if (key == "--help" || key == "-h") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option " + key);
    }
  }

  if (args.stages.empty() && !args.host_empty) {
    throw std::runtime_error("at least one stage is required");
  }
  if (args.host_empty && !args.stages.empty()) {
    throw std::runtime_error("--host-empty cannot be combined with graph stages");
  }
  for (size_t i = 0; i < args.stages.size(); ++i) {
    StageArgs& stage = args.stages[i];
    if (stage.name.empty()) {
      stage.name = "stage" + std::to_string(i);
    }
    if (stage.device.empty()) {
      stage.device = (i == 1 ? "vta" : "cpu");
    }
    if (stage.device != "cpu" && stage.device != "vta") {
      throw std::runtime_error("--stage" + std::to_string(i) + "-device must be cpu or vta");
    }
    if (stage.graph.empty() || stage.lib.empty() || stage.params.empty()) {
      throw std::runtime_error("all stage graph/lib/params arguments are required");
    }
    if (!stage.input_sources.empty() && stage.input_sources.size() != stage.input_names.size()) {
      throw std::runtime_error("--stage" + std::to_string(i) +
                               "-input-sources must match --stage-input-names count");
    }
    if (stage.runtime_num_threads < -1) {
      throw std::runtime_error("--stage*-runtime-num-threads must be >= -1");
    }
  }
  if (args.stages.size() > 1) {
    for (size_t i = 0; i < args.stages.size(); ++i) {
      if (args.stages[i].graph.empty() || args.stages[i].lib.empty() ||
          args.stages[i].params.empty()) {
        throw std::runtime_error("stage indices must be contiguous from stage0");
      }
    }
  }
  if (args.stages.empty() && !args.host_empty) {
    throw std::runtime_error("all stage graph/lib/params arguments are required");
  }
  const int input_mode_count = static_cast<int>(!args.input.empty()) +
                               static_cast<int>(!args.input_files.empty()) +
                               static_cast<int>(!args.input_list.empty());
  if (input_mode_count > 1) {
    throw std::runtime_error("--input, --input-files and --input-list are mutually exclusive");
  }
  if (input_mode_count == 0 && !args.host_empty) {
    throw std::runtime_error("--input, --input-files or --input-list is required");
  }
  if (input_mode_count != 0 && args.host_empty) {
    throw std::runtime_error("--host-empty does not accept tensor inputs");
  }
  if (args.runs <= 0) {
    throw std::runtime_error("--runs must be positive");
  }
  if (args.warmup_runs < 0) {
    throw std::runtime_error("--warmup-runs must be non-negative");
  }
  if (args.host_empty_stage_count <= 0) {
    throw std::runtime_error("--host-empty-stage-count must be positive");
  }
  if (args.host_empty_inner_repeats <= 0) {
    throw std::runtime_error("--host-empty-inner-repeats must be positive");
  }
  if (args.queue_depth <= 0) {
    throw std::runtime_error("--queue-depth must be positive");
  }
  if (args.runtime_num_threads < 0) {
    throw std::runtime_error("--runtime-num-threads must be non-negative");
  }
  if (args.output_mode != "classification" && args.output_mode != "raw" &&
      args.output_mode != "raw_all_stages") {
    throw std::runtime_error("--output-mode must be classification, raw, or raw_all_stages");
  }
  if (args.events_limit < 0) {
    throw std::runtime_error("--vta-runtime-profile-events-limit must be non-negative");
  }
  if (args.checkpoint_every < 0) {
    throw std::runtime_error("--vta-runtime-profile-checkpoint-every must be non-negative");
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
  if (args.start_timeout_s <= 0) {
    throw std::runtime_error("--start-timeout-s must be positive");
  }
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (!args.process_cpu_affinity.empty() && hardware_threads != 0 &&
      args.process_cpu_affinity.back() >= hardware_threads) {
    throw std::runtime_error("process affinity references a core outside the system");
  }
  if (!args.host_empty && (any_barrier || args.profile_pmu) &&
      (!args.serial || args.stages.size() != 1)) {
    throw std::runtime_error(
        "barrier/PMU instrumentation requires --serial with exactly one stage");
  }
  if (args.p8a_edge >= 0) {
    if (!args.serial) {
      throw std::runtime_error("--p8-edge requires --serial");
    }
    if (static_cast<size_t>(args.p8a_edge + 1) >= args.stages.size()) {
      throw std::runtime_error("--p8-edge must name an adjacent stage boundary");
    }
    const std::string& producer = args.stages[static_cast<size_t>(args.p8a_edge)].device;
    const std::string& consumer =
        args.stages[static_cast<size_t>(args.p8a_edge + 1)].device;
    if (!((producer == "cpu" && consumer == "vta") ||
          (producer == "vta" && consumer == "cpu"))) {
      throw std::runtime_error("P8 requires an adjacent heterogeneous CPU/VTA edge");
    }
    if (args.barrier_each_run || any_barrier || args.profile_pmu) {
      throw std::runtime_error("P8 paired audit does not accept barrier or PMU instrumentation");
    }
  }
  if (args.p8_managed_slots < 0 || args.p8_managed_slots > 2) {
    throw std::runtime_error("--p8-managed-slots must be 0, 1, or 2");
  }
  if (args.p8_slot_timeout_s <= 0) {
    throw std::runtime_error("--p8-slot-timeout-s must be positive");
  }
  if (args.p8_managed_slots > 0) {
    if (args.p8a_edge >= 0) {
      throw std::runtime_error("--p8-managed-slots and --p8-edge are mutually exclusive");
    }
    if (args.stages.size() < 2) {
      throw std::runtime_error("P8 managed slots require at least two stages");
    }
    if (args.output_mode == "raw_all_stages") {
      throw std::runtime_error(
          "P8 managed slots do not retain intermediate tensors for raw_all_stages");
    }
    if (args.serial && args.p8_managed_slots != 1) {
      throw std::runtime_error("serial P8 managed mode requires exactly one slot per edge");
    }
    if (!args.serial && args.p8_managed_slots < 2) {
      throw std::runtime_error("pipeline P8 managed mode requires at least two slots per edge");
    }
    if (args.barrier_each_run || any_barrier || args.profile_pmu) {
      throw std::runtime_error(
          "P8 managed mode does not accept barrier or PMU instrumentation");
    }
    for (size_t i = 0; i + 1 < args.stages.size(); ++i) {
      const std::string& producer = args.stages[i].device;
      const std::string& consumer = args.stages[i + 1].device;
      if (!((producer == "cpu" && consumer == "vta") ||
            (producer == "vta" && consumer == "cpu"))) {
        throw std::runtime_error(
            "P8 managed mode requires every adjacent edge to be heterogeneous");
      }
      const std::vector<std::string>& sources = args.stages[i + 1].input_sources;
      const std::string expected = "stage" + std::to_string(i) + ":0";
      if (!sources.empty() && (sources.size() != 1 || sources[0] != expected)) {
        throw std::runtime_error(
            "P8 managed mode currently requires a linear single-tensor stage chain");
      }
    }
  }
  return args;
}

void ResolveStageThreadDefaults(Args* args) {
  for (size_t i = 0; i < args->stages.size(); ++i) {
    StageArgs& stage = args->stages[i];
    if (stage.runtime_num_threads >= 0) {
      continue;
    }
    stage.runtime_num_threads = stage.device == "cpu" ? args->runtime_num_threads : 1;
  }
}

void ValidateStageAffinities(const Args& args) {
  for (size_t i = 0; i < args.stages.size(); ++i) {
    const StageArgs& stage = args.stages[i];
    if (stage.device == "vta" && !stage.cpu_affinity.empty()) {
      throw std::runtime_error("--stage" + std::to_string(i) +
                               "-cpu-affinity is only valid for CPU stages");
    }
    if (stage.device != "cpu") continue;
    if (!stage.cpu_affinity.empty() &&
        stage.cpu_affinity.size() != static_cast<size_t>(stage.runtime_num_threads)) {
      throw std::runtime_error("--stage" + std::to_string(i) +
                               "-cpu-affinity count must equal runtime-num-threads");
    }
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (!stage.cpu_affinity.empty() && hardware_threads != 0 &&
        stage.cpu_affinity.back() >= hardware_threads) {
      throw std::runtime_error("CPU stage affinity references a core outside the system");
    }
  }
}

void ConfigureProcessAffinity(const std::vector<unsigned int>& cpus) {
  if (cpus.empty()) return;
#if defined(__linux__)
  cpu_set_t requested;
  CPU_ZERO(&requested);
  for (unsigned int cpu : cpus) CPU_SET(cpu, &requested);
  DIR* directory = opendir("/proc/self/task");
  if (directory == nullptr) {
    throw std::runtime_error("unable to list process threads for affinity");
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
  if (tids.empty()) {
    throw std::runtime_error("process affinity found no threads");
  }
  for (pid_t tid : tids) {
    if (sched_setaffinity(tid, sizeof(requested), &requested) != 0) {
      throw std::runtime_error("sched_setaffinity failed for process thread");
    }
  }
  if (CurrentThreadAffinity() != CPUListString(cpus)) {
    throw std::runtime_error("effective process affinity does not match requested cores");
  }
#else
  throw std::runtime_error("process affinity is unsupported on this platform");
#endif
}

size_t TensorNBytes(const DLTensor* tensor) {
  size_t elems = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    elems *= static_cast<size_t>(tensor->shape[i]);
  }
  const size_t bits = static_cast<size_t>(tensor->dtype.bits) *
                      static_cast<size_t>(tensor->dtype.lanes);
  return (elems * bits + 7) / 8;
}

size_t TensorNElements(const DLTensor* tensor) {
  size_t elems = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    elems *= static_cast<size_t>(tensor->shape[i]);
  }
  return elems;
}

const uint8_t* TensorDataBytes(const DLTensor* tensor) {
  return static_cast<const uint8_t*>(tensor->data) + tensor->byte_offset;
}

std::string DTypeString(const DLDataType& dtype) {
  std::string code = "unknown";
  if (dtype.code == kDLFloat) {
    code = "float";
  } else if (dtype.code == kDLInt) {
    code = "int";
  } else if (dtype.code == kDLUInt) {
    code = "uint";
  } else if (dtype.code == kDLBfloat) {
    code = "bfloat";
  }
  std::ostringstream os;
  os << code << static_cast<int>(dtype.bits);
  if (dtype.lanes != 1) {
    os << "x" << static_cast<int>(dtype.lanes);
  }
  return os.str();
}

uint64_t FNV1a64(const uint8_t* data, size_t nbytes) {
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < nbytes; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

NDArray MakeCPUInputLike(DLTensor* input_tensor, const std::string& input_bytes) {
  const size_t expected = TensorNBytes(input_tensor);
  if (input_bytes.size() != expected) {
    std::ostringstream os;
    os << "Input byte size mismatch: got " << input_bytes.size() << ", expected " << expected;
    throw std::runtime_error(os.str());
  }
  std::vector<int64_t> shape(input_tensor->shape, input_tensor->shape + input_tensor->ndim);
  NDArray input =
      NDArray::Empty(tvm::runtime::ShapeTuple(shape), input_tensor->dtype, tvm::Device{kDLCPU, 0});
  std::memcpy(input->data, input_bytes.data(), input_bytes.size());
  return input;
}

struct InputRecord {
  int input_index{0};
  std::string input_file;
  std::vector<NDArray> data;
};

struct StageExecutor {
  std::string name;
  std::string device;
  std::vector<std::string> input_names;
  Module graph;
  PackedFunc set_input;
  PackedFunc set_input_zero_copy;
  PackedFunc set_output_zero_copy;
  PackedFunc run;
  PackedFunc get_output;
  PackedFunc get_input;
  PackedFunc get_num_outputs;

  int NumOutputs() const { return get_num_outputs().operator int(); }
};

StageExecutor LoadStage(const std::string& name, const std::string& device,
                        const StageArgs& stage_args) {
  const std::string graph_json = ReadFile(stage_args.graph);
  const std::string params_data = ReadFile(stage_args.params, true);
  Module lib = Module::LoadFromFile(stage_args.lib);
  const PackedFunc* create = Registry::Get("tvm.graph_executor.create");
  if (create == nullptr) {
    throw std::runtime_error("Missing registry function tvm.graph_executor.create");
  }

  Module graph;
  if (device == "vta") {
    graph = (*create)(graph_json, lib, static_cast<int>(kDLExtDev), 0,
                      static_cast<int>(kDLCPU), 0);
  } else {
    graph = (*create)(graph_json, lib, static_cast<int>(kDLCPU), 0);
  }

  PackedFunc load_params = graph.GetFunction("load_params");
  LoadParamsBlob(load_params, params_data);

  StageExecutor stage;
  stage.name = name;
  stage.device = device;
  stage.input_names = stage_args.input_names;
  stage.graph = graph;
  stage.set_input = graph.GetFunction("set_input");
  stage.set_input_zero_copy = graph.GetFunction("set_input_zero_copy", false);
  stage.set_output_zero_copy = graph.GetFunction("set_output_zero_copy", false);
  stage.run = graph.GetFunction("run");
  stage.get_output = graph.GetFunction("get_output");
  stage.get_input = graph.GetFunction("get_input");
  stage.get_num_outputs = graph.GetFunction("get_num_outputs");

  std::cout << "[LOAD] " << name << " device=" << device
            << " inputs=" << stage.input_names.size()
            << " outputs=" << stage.NumOutputs() << "\n";
  return stage;
}

std::vector<InputRecord> LoadInputs(const Args& args, StageExecutor* first_stage) {
  auto make_input = [&](size_t slot, const std::string& path) {
    if (slot >= first_stage->input_names.size()) {
      throw std::runtime_error("input slot exceeds first-stage input arity");
    }
    DLTensor* tensor = first_stage->get_input(first_stage->input_names[slot]);
    if (tensor == nullptr) {
      throw std::runtime_error("Unable to resolve first-stage input tensor slot " +
                               std::to_string(slot));
    }
    return MakeCPUInputLike(tensor, ReadFile(path, true));
  };
  std::vector<InputRecord> inputs;
  if (!args.input_files.empty()) {
    if (args.input_files.size() != first_stage->input_names.size()) {
      throw std::runtime_error("--input-files count must match first-stage input arity");
    }
    std::vector<NDArray> arrays;
    arrays.reserve(args.input_files.size());
    for (size_t slot = 0; slot < args.input_files.size(); ++slot) {
      arrays.push_back(make_input(slot, args.input_files[slot]));
    }
    std::ostringstream names;
    for (size_t slot = 0; slot < args.input_files.size(); ++slot) {
      if (slot != 0) names << ",";
      names << args.input_files[slot];
    }
    inputs.push_back(InputRecord{0, names.str(), std::move(arrays)});
    return inputs;
  }
  if (args.input_list.empty()) {
    if (first_stage->input_names.size() != 1) {
      throw std::runtime_error("multi-input first stage requires --input-files");
    }
    inputs.push_back(InputRecord{0, args.input, {make_input(0, args.input)}});
    return inputs;
  }
  if (first_stage->input_names.size() != 1) {
    throw std::runtime_error("--input-list only supports a single-input first stage");
  }

  std::ifstream list(args.input_list);
  if (!list) {
    throw std::runtime_error("Unable to open " + args.input_list);
  }
  std::filesystem::path base = std::filesystem::path(args.input_list).parent_path();
  if (base.empty()) {
    base = ".";
  }

  std::string line;
  int input_index = 0;
  while (std::getline(list, line)) {
    std::string rel = TrimLine(line);
    if (rel.empty()) continue;
    std::filesystem::path input_path(rel);
    std::filesystem::path resolved = input_path.is_absolute() ? input_path : base / input_path;
    inputs.push_back(InputRecord{input_index, rel, {make_input(0, resolved.string())}});
    ++input_index;
  }
  if (inputs.empty()) {
    throw std::runtime_error("No inputs listed in " + args.input_list);
  }
  return inputs;
}

int Top1Float32(const NDArray& output_cpu) {
  const DLTensor* tensor = output_cpu.operator->();
  if (tensor->dtype.code != kDLFloat || tensor->dtype.bits != 32 || tensor->dtype.lanes != 1) {
    throw std::runtime_error("Top1 calculation only supports float32 output");
  }
  const size_t nbytes = TensorNBytes(tensor);
  const size_t count = nbytes / sizeof(float);
  const float* data = static_cast<const float*>(tensor->data);
  return static_cast<int>(std::max_element(data, data + count) - data);
}

std::string ShapeJSON(const DLTensor* tensor) {
  std::ostringstream os;
  os << "[";
  for (int i = 0; i < tensor->ndim; ++i) {
    if (i) os << ",";
    os << tensor->shape[i];
  }
  os << "]";
  return os.str();
}

std::string UInt64Hex(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setfill('0') << std::setw(16) << value;
  return os.str();
}

std::string TensorSummaryJSON(const NDArray& array, int index) {
  const DLTensor* tensor = array.operator->();
  const size_t nbytes = TensorNBytes(tensor);
  const size_t count = TensorNElements(tensor);
  const uint8_t* bytes = TensorDataBytes(tensor);

  double sum = 0.0;
  double min_value = 0.0;
  double max_value = 0.0;
  bool numeric = false;

  auto update_stats = [&](double value, size_t idx) {
    if (idx == 0) {
      min_value = value;
      max_value = value;
    } else {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    sum += value;
  };

  if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32) {
    const float* data = reinterpret_cast<const float*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(static_cast<double>(data[i]), i);
    }
    numeric = true;
  } else if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLFloat &&
             tensor->dtype.bits == 64) {
    const double* data = reinterpret_cast<const double*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(data[i], i);
    }
    numeric = true;
  } else if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLInt &&
             tensor->dtype.bits == 32) {
    const int32_t* data = reinterpret_cast<const int32_t*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(static_cast<double>(data[i]), i);
    }
    numeric = true;
  } else if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLInt &&
             tensor->dtype.bits == 64) {
    const int64_t* data = reinterpret_cast<const int64_t*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(static_cast<double>(data[i]), i);
    }
    numeric = true;
  }

  std::ostringstream os;
  os << std::fixed << std::setprecision(9);
  os << "{\"index\":" << index
     << ",\"shape\":" << ShapeJSON(tensor)
     << ",\"dtype\":\"" << JsonEscape(DTypeString(tensor->dtype)) << "\""
     << ",\"size\":" << count
     << ",\"nbytes\":" << nbytes
     << ",\"fnv1a64\":\"" << UInt64Hex(FNV1a64(bytes, nbytes)) << "\"";
  if (numeric) {
    os << ",\"min\":" << min_value
       << ",\"max\":" << max_value
       << ",\"mean\":" << (count ? sum / static_cast<double>(count) : 0.0)
       << ",\"sum\":" << sum;
  }
  os << "}";
  return os.str();
}

std::string RawOutputsJSON(const std::vector<NDArray>& outputs) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i) os << ",";
    os << TensorSummaryJSON(outputs[i], static_cast<int>(i));
  }
  os << "]";
  return os.str();
}

std::string DumpOutputsJSON(const std::string& output_dump_dir,
                            const std::vector<NDArray>& outputs,
                            int completion_index,
                            const std::string& tag) {
  if (output_dump_dir.empty()) {
    return "[]";
  }
  const std::filesystem::path run_dir =
      std::filesystem::path(output_dump_dir) /
      ("run_" + std::to_string(completion_index));
  EnsureDir(run_dir.string());
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < outputs.size(); ++i) {
    const DLTensor* tensor = outputs[i].operator->();
    const size_t nbytes = TensorNBytes(tensor);
    const uint8_t* bytes = TensorDataBytes(tensor);
    const std::string file_name = tag + "_output_" + std::to_string(i) + ".bin";
    const std::filesystem::path file_path = run_dir / file_name;
    std::ofstream out(file_path, std::ios::out | std::ios::binary);
    if (!out) {
      throw std::runtime_error("failed to open output dump " + file_path.string());
    }
    out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(nbytes));
    if (!out) {
      throw std::runtime_error("failed to write output dump " + file_path.string());
    }
    if (i) os << ",";
    os << "{\"index\":" << i
       << ",\"path\":\"" << JsonEscape(file_path.string()) << "\""
       << ",\"shape\":" << ShapeJSON(tensor)
       << ",\"dtype\":\"" << JsonEscape(DTypeString(tensor->dtype)) << "\""
       << ",\"nbytes\":" << nbytes
       << "}";
  }
  os << "]";
  return os.str();
}

struct StageTiming {
  double set_start_ms{0.0};
  double set_end_ms{0.0};
  double run_start_ms{0.0};
  double get_start_ms{0.0};
  double end_ms{0.0};
  double process_cpu_start_ms{0.0};
  double process_cpu_after_set_ms{0.0};
  double process_cpu_after_run_ms{0.0};
  double process_cpu_end_ms{0.0};
  bool process_cpu_time_valid{false};
  double vta_mutex_request_ms{0.0};
  double vta_mutex_acquire_ms{0.0};
  double vta_mutex_release_ms{0.0};
  double vta_run_call_start_ms{0.0};
  double vta_run_call_end_ms{0.0};
  bool vta_mutex_timing_valid{false};

  double SetMs() const { return set_end_ms - set_start_ms; }
  double RunMs() const { return get_start_ms - run_start_ms; }
  double GetMs() const { return end_ms - get_start_ms; }
  double ServiceMs() const { return end_ms - set_start_ms; }
  double ProcessCpuMs() const { return process_cpu_end_ms - process_cpu_start_ms; }
  double SetProcessCpuMs() const {
    return process_cpu_after_set_ms - process_cpu_start_ms;
  }
  double RunProcessCpuMs() const {
    return process_cpu_after_run_ms - process_cpu_after_set_ms;
  }
  double GetProcessCpuMs() const {
    return process_cpu_end_ms - process_cpu_after_run_ms;
  }
  double VtaMutexWaitMs() const {
    return vta_mutex_acquire_ms - vta_mutex_request_ms;
  }
  double VtaMutexHeldMs() const {
    return vta_mutex_release_ms - vta_mutex_acquire_ms;
  }
  double VtaRunCallMs() const {
    return vta_run_call_end_ms - vta_run_call_start_ms;
  }
  double ScheduledServiceMs() const {
    const double start_ms =
        vta_mutex_timing_valid ? std::min(set_start_ms, vta_mutex_request_ms) : set_start_ms;
    const double finish_ms =
        vta_mutex_timing_valid ? std::max(end_ms, vta_mutex_release_ms) : end_ms;
    return finish_ms - start_ms;
  }
};

struct P8SlotToken {
  size_t edge_index{0};
  size_t slot_id{0};
  uint64_t generation{0};
  int frame_id{-1};
};

struct P8BoundaryObservation {
  bool valid{false};
  size_t edge_index{0};
  size_t slot_id{0};
  uint64_t generation{0};
  size_t boundary_bytes{0};
  uint64_t physical_address{0};
  std::vector<size_t> tensor_bytes;
  std::vector<uint64_t> physical_addresses;
  std::string direction;
  double producer_wait_ms{0.0};
  double consumer_wait_ms{0.0};
};

struct Frame {
  int frame_id{0};
  int input_index{0};
  std::string input_file;
  double enqueue_ms{0.0};
  std::vector<NDArray> stage_inputs;
  std::vector<std::vector<NDArray>> stage_outputs;
  std::vector<StageTiming> stage_timings;
  std::vector<P8SlotToken> p8_edge_tokens;
  std::vector<P8BoundaryObservation> p8_boundaries;
  PmuSnapshot process_pmu;
  double done_ms{0.0};
  int top1{-1};
};

std::pair<int, int> ParseInputSource(const std::string& source) {
  const size_t colon = source.find(':');
  if (colon == std::string::npos) {
    throw std::runtime_error("input source must be input:0 or stageN:K, got " + source);
  }
  const std::string owner = source.substr(0, colon);
  const int index = std::stoi(source.substr(colon + 1));
  if (index < 0) {
    throw std::runtime_error("negative input source index in " + source);
  }
  if (owner == "input") {
    return {-1, index};
  }
  const std::string prefix = "stage";
  if (owner.rfind(prefix, 0) != 0 || owner.size() == prefix.size()) {
    throw std::runtime_error("input source must be input:0 or stageN:K, got " + source);
  }
  return {std::stoi(owner.substr(prefix.size())), index};
}

const NDArray& ResolveInputSource(const Frame& frame, const std::string& source) {
  const auto parsed = ParseInputSource(source);
  const int stage_index = parsed.first;
  const int output_index = parsed.second;
  if (stage_index < 0) {
    if (static_cast<size_t>(output_index) >= frame.stage_inputs.size()) {
      throw std::runtime_error("input source out of range: " + source);
    }
    return frame.stage_inputs[static_cast<size_t>(output_index)];
  }
  if (static_cast<size_t>(stage_index) >= frame.stage_outputs.size()) {
    throw std::runtime_error("stage source out of range: " + source);
  }
  const std::vector<NDArray>& outputs = frame.stage_outputs[static_cast<size_t>(stage_index)];
  if (static_cast<size_t>(output_index) >= outputs.size()) {
    throw std::runtime_error("stage output source out of range: " + source);
  }
  return outputs[static_cast<size_t>(output_index)];
}

std::vector<NDArray> ResolveStageInputs(const Args& args, const Frame& frame, size_t stage_index) {
  const StageArgs& stage = args.stages[stage_index];
  if (stage.input_sources.empty()) {
    if (stage_index == 0) {
      return frame.stage_inputs;
    }
    return frame.stage_outputs[stage_index - 1];
  }
  std::vector<NDArray> inputs;
  inputs.reserve(stage.input_sources.size());
  for (const std::string& source : stage.input_sources) {
    inputs.push_back(ResolveInputSource(frame, source));
  }
  return inputs;
}

std::vector<NDArray> RunStage(StageExecutor* stage, const std::vector<NDArray>& inputs,
                              StageTiming* timing, bool measure_process_cpu_time) {
  if (inputs.size() != stage->input_names.size()) {
    std::ostringstream os;
    os << stage->name << " expected " << stage->input_names.size() << " input(s), got "
       << inputs.size();
    throw std::runtime_error(os.str());
  }
  if (measure_process_cpu_time) {
    timing->process_cpu_start_ms = ProcessCpuMillis();
    timing->process_cpu_time_valid = true;
  }
  timing->set_start_ms = NowMillis();
  for (size_t i = 0; i < inputs.size(); ++i) {
    stage->set_input(stage->input_names[i], inputs[i]);
  }
  timing->set_end_ms = NowMillis();
  if (measure_process_cpu_time) {
    timing->process_cpu_after_set_ms = ProcessCpuMillis();
  }
  timing->run_start_ms = timing->set_end_ms;
  stage->run();
  if (measure_process_cpu_time) {
    timing->process_cpu_after_run_ms = ProcessCpuMillis();
  }
  timing->get_start_ms = NowMillis();

  std::vector<NDArray> outputs;
  const int num_outputs = stage->NumOutputs();
  outputs.reserve(static_cast<size_t>(num_outputs));
  for (int i = 0; i < num_outputs; ++i) {
    NDArray output = stage->get_output(i).operator NDArray();
    outputs.push_back(output.CopyTo(tvm::Device{kDLCPU, 0}));
  }
  timing->end_ms = NowMillis();
  if (measure_process_cpu_time) {
    timing->process_cpu_end_ms = ProcessCpuMillis();
  }
  return outputs;
}

bool SameDType(const DLDataType& left, const DLDataType& right) {
  return left.code == right.code && left.bits == right.bits && left.lanes == right.lanes;
}

bool SameShape(const DLTensor* left, const DLTensor* right) {
  if (left->ndim != right->ndim) return false;
  for (int i = 0; i < left->ndim; ++i) {
    if (left->shape[i] != right->shape[i]) return false;
  }
  return true;
}

bool EnvironmentFlagEnabled(const char* value, bool default_value) {
  if (value == nullptr) return default_value;
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return normalized != "0" && normalized != "false" && normalized != "off";
}

std::vector<uint64_t> OutputHashes(const std::vector<NDArray>& outputs) {
  std::vector<uint64_t> hashes;
  hashes.reserve(outputs.size());
  for (const NDArray& output : outputs) {
    const DLTensor* tensor = output.operator->();
    hashes.push_back(FNV1a64(TensorDataBytes(tensor), TensorNBytes(tensor)));
  }
  return hashes;
}

std::string HashesJSON(const std::vector<uint64_t>& hashes) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < hashes.size(); ++i) {
    if (i != 0) os << ",";
    os << "\"" << UInt64Hex(hashes[i]) << "\"";
  }
  os << "]";
  return os.str();
}

struct P8SharedEdge {
  size_t producer_stage{0};
  size_t consumer_stage{0};
  size_t nbytes{0};
  std::string direction;
  std::shared_ptr<void> allocation_owner;
  NDArray ext_owner;
  NDArray cpu_owner;
  NDArray producer_owner;
  NDArray consumer_owner;
  DLDeviceType producer_device_type{kDLCPU};
  DLDeviceType consumer_device_type{kDLExtDev};
  uint64_t physical_address{0};
};

P8SharedEdge AllocateP8HeterogeneousEdgeTensor(std::vector<StageExecutor>* stages,
                                               size_t producer_stage,
                                               size_t tensor_index) {
  const size_t consumer_stage = producer_stage + 1;
  StageExecutor& producer = stages->at(producer_stage);
  StageExecutor& consumer = stages->at(consumer_stage);
  if (!producer.set_output_zero_copy.defined() || !consumer.set_input_zero_copy.defined()) {
    throw std::runtime_error("GraphExecutor zero-copy functions are unavailable");
  }
  if (tensor_index >= static_cast<size_t>(producer.NumOutputs()) ||
      tensor_index >= consumer.input_names.size()) {
    throw std::runtime_error("P8 boundary tensor index is out of range");
  }

  NDArray producer_output = producer.get_output(static_cast<int>(tensor_index)).operator NDArray();
  const DLTensor* producer_tensor = producer_output.operator->();
  NDArray consumer_input = consumer.get_input(consumer.input_names[tensor_index]).operator NDArray();
  const DLTensor* consumer_tensor = consumer_input.operator->();
  if (producer_tensor == nullptr || consumer_tensor == nullptr) {
    throw std::runtime_error("P8 failed to resolve the boundary tensors");
  }
  const bool cpu_to_vta = producer_tensor->device.device_type == kDLCPU &&
                          consumer_tensor->device.device_type == kDLExtDev;
  const bool vta_to_cpu = producer_tensor->device.device_type == kDLExtDev &&
                          consumer_tensor->device.device_type == kDLCPU;
  if (!cpu_to_vta && !vta_to_cpu) {
    throw std::runtime_error("P8 boundary does not have complementary CPU/VTA views");
  }
  if (!SameDType(producer_tensor->dtype, consumer_tensor->dtype) ||
      !SameShape(producer_tensor, consumer_tensor) ||
      TensorNBytes(producer_tensor) != TensorNBytes(consumer_tensor)) {
    throw std::runtime_error("P8 boundary tensor contracts do not match");
  }
  if (producer_tensor->strides != nullptr || consumer_tensor->strides != nullptr) {
    throw std::runtime_error("P8 only supports contiguous boundary tensors");
  }
  if (producer_tensor->byte_offset != 0 || consumer_tensor->byte_offset != 0) {
    throw std::runtime_error("P8 requires zero byte offsets at the stage boundary");
  }

  const size_t nbytes = TensorNBytes(producer_tensor);
  void* slot_data = VTAMemAlloc(nbytes, VTA_CACHED);
  if (slot_data == nullptr) {
    throw std::runtime_error("P8 failed to allocate a cached u-dma-buf slot");
  }
  std::shared_ptr<void> allocation_owner(slot_data, [](void* data) { VTAMemFree(data); });
  const DLTensor* ext_template = cpu_to_vta ? consumer_tensor : producer_tensor;
  DLTensor ext_view = *ext_template;
  ext_view.data = slot_data;
  ext_view.byte_offset = 0;
  ext_view.strides = nullptr;
  NDArray ext_owner = NDArray::FromExternalDLTensor(ext_view);
  DLTensor cpu_view = ext_view;
  cpu_view.device = tvm::Device{kDLCPU, 0};
  NDArray cpu_owner = NDArray::FromExternalDLTensor(cpu_view);
  const uintptr_t virtual_address =
      reinterpret_cast<uintptr_t>(static_cast<char*>(cpu_view.data) + cpu_view.byte_offset);
  if (virtual_address % 256U != 0U) {
    throw std::runtime_error("P8 shared boundary pointer is not 256-byte aligned");
  }

  NDArray producer_owner = cpu_to_vta ? cpu_owner : ext_owner;
  NDArray consumer_owner = cpu_to_vta ? ext_owner : cpu_owner;

  P8SharedEdge edge;
  edge.producer_stage = producer_stage;
  edge.consumer_stage = consumer_stage;
  edge.nbytes = nbytes;
  edge.direction = cpu_to_vta ? "cpu_to_vta" : "vta_to_cpu";
  edge.allocation_owner = std::move(allocation_owner);
  edge.ext_owner = std::move(ext_owner);
  edge.cpu_owner = std::move(cpu_owner);
  edge.producer_owner = std::move(producer_owner);
  edge.consumer_owner = std::move(consumer_owner);
  edge.producer_device_type = producer_tensor->device.device_type;
  edge.consumer_device_type = consumer_tensor->device.device_type;
  edge.physical_address = static_cast<uint64_t>(VTAMemGetPhyAddr(edge.cpu_owner->data));
  if (edge.physical_address == 0) {
    throw std::runtime_error("P8 shared boundary has no VTA physical address");
  }
  if (edge.physical_address % 256U != 0U) {
    throw std::runtime_error("P8 shared boundary physical address is not 256-byte aligned");
  }
  return edge;
}

P8SharedEdge AllocateP8HeterogeneousEdgeSlot(std::vector<StageExecutor>* stages,
                                             size_t producer_stage) {
  StageExecutor& producer = stages->at(producer_stage);
  StageExecutor& consumer = stages->at(producer_stage + 1);
  if (producer.NumOutputs() != 1 || consumer.input_names.size() != 1) {
    throw std::runtime_error("P8A requires a single-output to single-input adjacent edge");
  }
  return AllocateP8HeterogeneousEdgeTensor(stages, producer_stage, 0);
}

std::vector<P8SharedEdge> AllocateP8HeterogeneousEdgeBundle(
    std::vector<StageExecutor>* stages, size_t producer_stage) {
  StageExecutor& producer = stages->at(producer_stage);
  StageExecutor& consumer = stages->at(producer_stage + 1);
  const size_t output_count = static_cast<size_t>(producer.NumOutputs());
  if (output_count == 0 || output_count != consumer.input_names.size()) {
    throw std::runtime_error(
        "P8 managed edge requires matching non-empty producer output and consumer input arity");
  }
  std::vector<P8SharedEdge> tensors;
  tensors.reserve(output_count);
  for (size_t tensor_index = 0; tensor_index < output_count; ++tensor_index) {
    tensors.push_back(
        AllocateP8HeterogeneousEdgeTensor(stages, producer_stage, tensor_index));
  }
  const std::string direction = tensors.front().direction;
  if (!std::all_of(tensors.begin(), tensors.end(), [&](const P8SharedEdge& tensor) {
        return tensor.direction == direction;
      })) {
    throw std::runtime_error("P8 managed edge tensors do not share one direction");
  }
  return tensors;
}

P8SharedEdge BindP8HeterogeneousEdge(std::vector<StageExecutor>* stages,
                                     size_t producer_stage) {
  P8SharedEdge edge = AllocateP8HeterogeneousEdgeSlot(stages, producer_stage);
  StageExecutor& producer = stages->at(edge.producer_stage);
  StageExecutor& consumer = stages->at(edge.consumer_stage);
  producer.set_output_zero_copy(0, edge.producer_owner);
  consumer.set_input_zero_copy(consumer.input_names[0], edge.consumer_owner);
  return edge;
}

class BoundarySlotManager {
 public:
  BoundarySlotManager(std::vector<StageExecutor>* stages, size_t edge_index,
                      size_t slot_count, int timeout_s)
      : edge_index_(edge_index), timeout_(std::chrono::seconds(timeout_s)) {
    slots_.reserve(slot_count);
    for (size_t slot_id = 0; slot_id < slot_count; ++slot_id) {
      Slot slot;
      slot.tensors = AllocateP8HeterogeneousEdgeBundle(stages, edge_index);
      slots_.push_back(std::move(slot));
    }
  }

  P8SlotToken AcquireForProducer(int frame_id, double* wait_ms) {
    const double start_ms = NowMillis();
    std::unique_lock<std::mutex> lock(mutex_);
    const auto ready = [&]() { return aborted_ || FindFreeSlotLocked() < slots_.size(); };
    if (!condition_.wait_for(lock, timeout_, ready)) {
      throw std::runtime_error("P8 producer timed out waiting for a FREE slot on edge " +
                               std::to_string(edge_index_));
    }
    ThrowIfAbortedLocked();
    const size_t slot_id = FindFreeSlotLocked();
    Slot& slot = slots_.at(slot_id);
    slot.state = SlotState::kProducerWriting;
    slot.frame_id = frame_id;
    ++slot.generation;
    next_slot_ = (slot_id + 1) % slots_.size();
    *wait_ms = NowMillis() - start_ms;
    return P8SlotToken{edge_index_, slot_id, slot.generation, frame_id};
  }

  const NDArray& ProducerView(const P8SlotToken& token, size_t tensor_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = ValidateTokenLocked(token, SlotState::kProducerWriting);
    return slot.tensors.at(tensor_index).producer_owner;
  }

  const NDArray& ConsumerView(const P8SlotToken& token, size_t tensor_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (token.edge_index != edge_index_ || token.slot_id >= slots_.size()) {
      throw std::runtime_error("P8 consumer view token references the wrong edge or slot");
    }
    Slot& slot = slots_[token.slot_id];
    if (slot.generation != token.generation || slot.frame_id != token.frame_id ||
        (slot.state != SlotState::kReady &&
         slot.state != SlotState::kConsumerUsing)) {
      throw std::runtime_error("P8 consumer view token is stale or not readable");
    }
    return slot.tensors.at(tensor_index).consumer_owner;
  }

  void Publish(const P8SlotToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = ValidateTokenLocked(token, SlotState::kProducerWriting);
    slot.state = SlotState::kReady;
    condition_.notify_all();
  }

  void AcquireForConsumer(const P8SlotToken& token, double* wait_ms) {
    const double start_ms = NowMillis();
    std::unique_lock<std::mutex> lock(mutex_);
    const auto ready = [&]() {
      if (aborted_) return true;
      const Slot& slot = slots_.at(token.slot_id);
      return slot.state == SlotState::kReady && slot.generation == token.generation &&
             slot.frame_id == token.frame_id;
    };
    if (!condition_.wait_for(lock, timeout_, ready)) {
      throw std::runtime_error("P8 consumer timed out waiting for a READY slot on edge " +
                               std::to_string(edge_index_));
    }
    ThrowIfAbortedLocked();
    Slot& slot = ValidateTokenLocked(token, SlotState::kReady);
    slot.state = SlotState::kConsumerUsing;
    *wait_ms = NowMillis() - start_ms;
  }

  void Release(const P8SlotToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = ValidateTokenLocked(token, SlotState::kConsumerUsing);
    slot.state = SlotState::kFree;
    slot.frame_id = -1;
    condition_.notify_all();
  }

  void Abort(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    abort_reason_ = reason;
    condition_.notify_all();
  }

  void VerifyAllFree() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t slot_id = 0; slot_id < slots_.size(); ++slot_id) {
      if (slots_[slot_id].state != SlotState::kFree) {
        throw std::runtime_error("P8 edge " + std::to_string(edge_index_) + " slot " +
                                 std::to_string(slot_id) + " was not released");
      }
    }
  }

  size_t slot_count() const { return slots_.size(); }
  size_t tensor_count() const { return slots_.at(0).tensors.size(); }
  size_t nbytes() const {
    size_t total = 0;
    for (const P8SharedEdge& tensor : slots_.at(0).tensors) total += tensor.nbytes;
    return total;
  }
  std::vector<size_t> tensor_bytes() const {
    std::vector<size_t> result;
    for (const P8SharedEdge& tensor : slots_.at(0).tensors) result.push_back(tensor.nbytes);
    return result;
  }
  const std::string& direction() const { return slots_.at(0).tensors.at(0).direction; }
  uint64_t physical_address(size_t slot_id) const {
    return slots_.at(slot_id).tensors.at(0).physical_address;
  }
  std::vector<uint64_t> physical_addresses(size_t slot_id) const {
    std::vector<uint64_t> result;
    for (const P8SharedEdge& tensor : slots_.at(slot_id).tensors) {
      result.push_back(tensor.physical_address);
    }
    return result;
  }
  std::vector<std::pair<uint64_t, uint64_t>> physical_ranges(size_t slot_id) const {
    std::vector<std::pair<uint64_t, uint64_t>> result;
    for (const P8SharedEdge& tensor : slots_.at(slot_id).tensors) {
      result.push_back(
          {tensor.physical_address, tensor.physical_address + tensor.nbytes});
    }
    return result;
  }

 private:
  enum class SlotState { kFree, kProducerWriting, kReady, kConsumerUsing };

  struct Slot {
    std::vector<P8SharedEdge> tensors;
    SlotState state{SlotState::kFree};
    uint64_t generation{0};
    int frame_id{-1};
  };

  size_t FindFreeSlotLocked() const {
    for (size_t offset = 0; offset < slots_.size(); ++offset) {
      const size_t slot_id = (next_slot_ + offset) % slots_.size();
      if (slots_[slot_id].state == SlotState::kFree) return slot_id;
    }
    return slots_.size();
  }

  Slot& ValidateTokenLocked(const P8SlotToken& token, SlotState expected) {
    if (token.edge_index != edge_index_ || token.slot_id >= slots_.size()) {
      throw std::runtime_error("P8 slot token references the wrong edge or slot");
    }
    Slot& slot = slots_[token.slot_id];
    if (slot.generation != token.generation || slot.frame_id != token.frame_id) {
      throw std::runtime_error("P8 stale slot token generation/owner mismatch");
    }
    if (slot.state != expected) {
      throw std::runtime_error("P8 slot state transition mismatch");
    }
    return slot;
  }

  void ThrowIfAbortedLocked() const {
    if (aborted_) {
      throw std::runtime_error("P8 slot manager aborted: " + abort_reason_);
    }
  }

  size_t edge_index_{0};
  std::chrono::seconds timeout_;
  std::vector<Slot> slots_;
  size_t next_slot_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
  bool aborted_{false};
  std::string abort_reason_;
};

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

  bool Push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&]() { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    queue_.push_back(std::move(value));
    not_empty_.notify_one();
    return true;
  }

  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    *value = std::move(queue_.front());
    queue_.pop_front();
    not_full_.notify_one();
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  size_t capacity_;
  std::deque<T> queue_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool closed_{false};
};

uint64_t EmptyStageTransform(uint64_t value, size_t stage_index) {
  return value ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(stage_index));
}

int RunHostEmpty(const Args& args) {
  std::vector<std::unique_ptr<BoundedQueue<uint64_t>>> queues;
  queues.reserve(static_cast<size_t>(args.host_empty_stage_count) + 1);
  for (int i = 0; i <= args.host_empty_stage_count; ++i) {
    queues.emplace_back(std::make_unique<BoundedQueue<uint64_t>>(args.queue_depth));
  }

  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(args.host_empty_stage_count));
  for (int stage_index = 0; stage_index < args.host_empty_stage_count; ++stage_index) {
    workers.emplace_back([&, stage_index]() {
      uint64_t value = 0;
      while (queues[static_cast<size_t>(stage_index)]->Pop(&value)) {
        value = EmptyStageTransform(value, static_cast<size_t>(stage_index));
        if (!queues[static_cast<size_t>(stage_index) + 1]->Push(value)) break;
      }
      queues[static_cast<size_t>(stage_index) + 1]->Close();
    });
  }

  auto execute_batch = [&]() {
    uint64_t checksum = 0;
    bool correct = true;
    for (int i = 0; i < args.host_empty_inner_repeats; ++i) {
      const uint64_t input = static_cast<uint64_t>(i + 1);
      if (!queues[0]->Push(input)) throw std::runtime_error("host-empty input queue closed");
      uint64_t observed = 0;
      if (!queues.back()->Pop(&observed)) {
        throw std::runtime_error("host-empty output queue closed");
      }
      uint64_t expected = input;
      for (int stage_index = 0; stage_index < args.host_empty_stage_count; ++stage_index) {
        expected = EmptyStageTransform(expected, static_cast<size_t>(stage_index));
      }
      correct = correct && observed == expected;
      checksum ^= observed;
    }
    return std::make_pair(checksum, correct);
  };

  for (int i = 0; i < args.warmup_runs; ++i) (void)execute_batch();
  SignalReadyAndWait(args);
  ProcessPmuCounters process_pmu(args.profile_pmu);
  std::ofstream result(args.output_jsonl, std::ios::out);
  if (!result) throw std::runtime_error("Unable to write " + args.output_jsonl);

  for (int sample = 0; sample < args.runs; ++sample) {
    const double cpu_start_ms = ProcessCpuMillis();
    process_pmu.Start();
    const double start_ms = NowMillis();
    const auto batch = execute_batch();
    const double end_ms = NowMillis();
    PmuSnapshot pmu = process_pmu.Stop();
    const double cpu_end_ms = ProcessCpuMillis();
    const double divisor = static_cast<double>(args.host_empty_inner_repeats);
    result << std::fixed << std::setprecision(9)
           << "{\"mode\":\"host_empty\""
           << ",\"sample_index\":" << sample
           << ",\"host_empty_stage_count\":" << args.host_empty_stage_count
           << ",\"inner_repeats\":" << args.host_empty_inner_repeats
           << ",\"wall_ms\":" << (end_ms - start_ms) / divisor
           << ",\"process_cpu_ms\":" << (cpu_end_ms - cpu_start_ms) / divisor
           << ",\"checksum\":" << batch.first
           << ",\"correct\":" << (batch.second ? "true" : "false")
           << ",\"reference_correctness_passed\":"
           << (batch.second ? "true" : "false")
           << ",\"pmu_available\":" << (pmu.available ? "true" : "false")
           << ",\"pmu_unavailable_reason\":";
    if (pmu.available) {
      result << "null";
    } else {
      result << "\"" << JsonEscape(pmu.unavailable_reason) << "\"";
    }
    auto write_counter = [&](const char* name, uint64_t value) {
      result << ",\"" << name << "\":";
      if (pmu.available) {
        result << static_cast<double>(value) / divisor;
      } else {
        result << "null";
      }
    };
    write_counter("pmu_cycles", pmu.cycles);
    write_counter("pmu_instructions", pmu.instructions);
    write_counter("pmu_cache_references", pmu.cache_references);
    write_counter("pmu_cache_misses", pmu.cache_misses);
    result << "}\n";
    result.flush();
  }

  queues[0]->Close();
  for (std::thread& worker : workers) worker.join();
  return 0;
}

std::string ProfileCallString(const PackedFunc* func) {
  if (func == nullptr) {
    return "{}";
  }
  return (*func)().operator std::string();
}

std::string ProfileEventsString(const PackedFunc* func, int limit) {
  if (func == nullptr) {
    return "[]";
  }
  return (*func)(limit).operator std::string();
}

std::string MetaJSON(const std::string& snapshot, int run_index, int events_limit) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"snapshot\": \"" << JsonEscape(snapshot) << "\",\n";
  os << "  \"run_index\": " << run_index << ",\n";
  os << "  \"events_limit\": " << events_limit << "\n";
  os << "}";
  return os.str();
}

void DumpProfileSnapshot(const std::string& profile_dir, const std::string& name, int run_index,
                         int events_limit, const PackedFunc* status_func,
                         const PackedFunc* events_func) {
  if (profile_dir.empty() || status_func == nullptr) {
    return;
  }
  EnsureDir(profile_dir);
  WriteFile(profile_dir + "/" + name + "_status.json", ProfileCallString(status_func));
  if (events_func != nullptr) {
    WriteFile(profile_dir + "/" + name + "_events.json",
              ProfileEventsString(events_func, events_limit));
  }
  WriteFile(profile_dir + "/" + name + "_meta.json", MetaJSON(name, run_index, events_limit));
}

std::string ResultJSON(const Args& args, const Frame& frame, int completion_index,
                       const std::string& mode, double benchmark_start_ms) {
  auto rel = [&](double value) { return value - benchmark_start_ms; };
  std::ostringstream os;
  os << std::fixed << std::setprecision(6);
  os << "{\"mode\":\"" << JsonEscape(mode) << "\""
     << ",\"completion_index\":" << completion_index
     << ",\"frame_id\":" << frame.frame_id
     << ",\"input_index\":" << frame.input_index
     << ",\"input_file\":\"" << JsonEscape(frame.input_file) << "\""
     << ",\"output_mode\":\"" << JsonEscape(args.output_mode) << "\""
     << ",\"process_cpu_affinity_requested\":\""
     << JsonEscape(CPUListString(args.process_cpu_affinity)) << "\""
     << ",\"process_cpu_affinity_effective\":\""
     << JsonEscape(CurrentThreadAffinity()) << "\""
     << ",\"process_cpu_affinity_scope\":\"all_existing_threads_future_threads_inherit\"";
  if (args.output_mode == "classification") {
    os << ",\"top1\":" << frame.top1;
  } else {
    os << ",\"top1\":" << frame.top1
       << ",\"raw_outputs\":" << RawOutputsJSON(frame.stage_outputs.back());
    if (!args.output_dump_dir.empty()) {
      os << ",\"raw_output_files\":"
         << DumpOutputsJSON(args.output_dump_dir, frame.stage_outputs.back(),
                            completion_index, "final");
    }
    if (args.output_mode == "raw_all_stages") {
      os << ",\"stage_raw_outputs\":[";
      for (size_t i = 0; i < frame.stage_outputs.size(); ++i) {
        if (i != 0) os << ",";
        os << "{\"stage_index\":" << i
           << ",\"outputs\":" << RawOutputsJSON(frame.stage_outputs[i]);
        if (!args.output_dump_dir.empty()) {
          os << ",\"output_files\":"
             << DumpOutputsJSON(args.output_dump_dir, frame.stage_outputs[i], completion_index,
                                "stage" + std::to_string(i));
        }
        os << "}";
      }
      os << "]";
    }
  }
  os
     << ",\"total_latency_ms\":" << (frame.done_ms - frame.enqueue_ms)
     << ",\"completion_ms\":" << rel(frame.done_ms)
     << ",\"stage_count\":" << frame.stage_timings.size()
     << ",\"stage_cpu_time_scope\":\""
     << (mode == "serial" ? "exclusive_process_cpu_time" : "disabled_concurrent_pipeline")
     << "\""
     << ",\"process_pmu_scope\":\""
     << (frame.process_pmu.available ? "single_stage_process_all_existing_threads"
                                     : "unavailable")
     << "\""
     << ",\"pmu_available\":" << (frame.process_pmu.available ? "true" : "false")
     << ",\"pmu_unavailable_reason\":";
  if (frame.process_pmu.available) {
    os << "null";
  } else {
    os << "\"" << JsonEscape(frame.process_pmu.unavailable_reason) << "\"";
  }
  os << ",\"pmu_cycles\":";
  if (frame.process_pmu.available) {
    os << frame.process_pmu.cycles
       << ",\"pmu_instructions\":" << frame.process_pmu.instructions
       << ",\"pmu_cache_references\":" << frame.process_pmu.cache_references
       << ",\"pmu_cache_misses\":" << frame.process_pmu.cache_misses;
  } else {
    os << "null,\"pmu_instructions\":null,\"pmu_cache_references\":null"
       << ",\"pmu_cache_misses\":null";
  }
  for (size_t i = 0; i < frame.stage_timings.size(); ++i) {
    const StageTiming& timing = frame.stage_timings[i];
    os << ",\"stage" << i << "_ms\":" << timing.ServiceMs()
       << ",\"stage" << i << "_scheduled_service_ms\":" << timing.ScheduledServiceMs()
       << ",\"stage" << i << "_set_ms\":" << timing.SetMs()
       << ",\"stage" << i << "_run_ms\":" << timing.RunMs()
       << ",\"stage" << i << "_get_ms\":" << timing.GetMs()
       << ",\"stage" << i << "_start_ms\":" << rel(timing.set_start_ms)
       << ",\"stage" << i << "_set_end_ms\":" << rel(timing.set_end_ms)
       << ",\"stage" << i << "_end_ms\":" << rel(timing.end_ms)
       << ",\"stage" << i << "_process_cpu_ms\":";
    if (timing.process_cpu_time_valid) {
      os << timing.ProcessCpuMs()
         << ",\"stage" << i << "_set_process_cpu_ms\":" << timing.SetProcessCpuMs()
         << ",\"stage" << i << "_run_process_cpu_ms\":" << timing.RunProcessCpuMs()
         << ",\"stage" << i << "_get_process_cpu_ms\":" << timing.GetProcessCpuMs();
    } else {
      os << "null"
         << ",\"stage" << i << "_set_process_cpu_ms\":null"
         << ",\"stage" << i << "_run_process_cpu_ms\":null"
         << ",\"stage" << i << "_get_process_cpu_ms\":null";
    }
    os << ",\"stage" << i << "_vta_mutex_wait_ms\":";
    if (timing.vta_mutex_timing_valid) {
      os << timing.VtaMutexWaitMs()
         << ",\"stage" << i << "_vta_mutex_held_ms\":" << timing.VtaMutexHeldMs()
         << ",\"stage" << i << "_vta_run_call_ms\":" << timing.VtaRunCallMs()
         << ",\"stage" << i << "_vta_mutex_request_ms\":"
         << rel(timing.vta_mutex_request_ms)
         << ",\"stage" << i << "_vta_mutex_acquire_ms\":"
         << rel(timing.vta_mutex_acquire_ms)
         << ",\"stage" << i << "_vta_mutex_release_ms\":"
         << rel(timing.vta_mutex_release_ms);
    } else {
      os << "null"
         << ",\"stage" << i << "_vta_mutex_held_ms\":null"
         << ",\"stage" << i << "_vta_run_call_ms\":null"
         << ",\"stage" << i << "_vta_mutex_request_ms\":null"
         << ",\"stage" << i << "_vta_mutex_acquire_ms\":null"
         << ",\"stage" << i << "_vta_mutex_release_ms\":null";
    }
  }
  if (!frame.p8_boundaries.empty()) {
    double total_producer_wait_ms = 0.0;
    double total_consumer_wait_ms = 0.0;
    os << ",\"p8_managed_slot_count\":" << args.p8_managed_slots
       << ",\"p8_framework_materialization_bytes\":0"
       << ",\"p8_boundaries\":[";
    for (size_t i = 0; i < frame.p8_boundaries.size(); ++i) {
      const P8BoundaryObservation& boundary = frame.p8_boundaries[i];
      if (i != 0) os << ",";
      if (!boundary.valid) {
        os << "null";
        continue;
      }
      total_producer_wait_ms += boundary.producer_wait_ms;
      total_consumer_wait_ms += boundary.consumer_wait_ms;
      os << "{\"edge_index\":" << boundary.edge_index
         << ",\"direction\":\"" << JsonEscape(boundary.direction) << "\""
         << ",\"slot_id\":" << boundary.slot_id
         << ",\"generation\":" << boundary.generation
         << ",\"boundary_bytes\":" << boundary.boundary_bytes
         << ",\"tensor_count\":" << boundary.tensor_bytes.size()
         << ",\"tensor_bytes\":[";
      for (size_t tensor_index = 0; tensor_index < boundary.tensor_bytes.size();
           ++tensor_index) {
        if (tensor_index != 0) os << ",";
        os << boundary.tensor_bytes[tensor_index];
      }
      os << "]"
         << ",\"physical_addresses\":[";
      for (size_t tensor_index = 0; tensor_index < boundary.physical_addresses.size();
           ++tensor_index) {
        if (tensor_index != 0) os << ",";
        os << "\"0x" << std::hex << boundary.physical_addresses[tensor_index]
           << std::dec << "\"";
      }
      os << "]"
         << ",\"physical_address\":\"0x" << std::hex
         << boundary.physical_address << std::dec << "\""
         << ",\"producer_wait_ms\":" << boundary.producer_wait_ms
         << ",\"consumer_wait_ms\":" << boundary.consumer_wait_ms << "}";
    }
    os << "]"
       << ",\"p8_producer_slot_wait_ms\":" << total_producer_wait_ms
       << ",\"p8_consumer_slot_wait_ms\":" << total_consumer_wait_ms
       << ",\"p8_total_slot_wait_ms\":"
       << (total_producer_wait_ms + total_consumer_wait_ms);
  }
  os << "}";
  return os.str();
}

void ConfigureThreadPool(int runtime_num_threads, const std::vector<unsigned int>& cpu_affinity = {},
                         const std::string& label = "") {
  if (runtime_num_threads <= 0) {
    return;
  }
  const PackedFunc* config = Registry::Get("runtime.config_threadpool");
  if (config != nullptr) {
    if (cpu_affinity.empty()) {
      (*config)(0, runtime_num_threads);
    } else {
      tvm::runtime::Array<tvm::runtime::String> cpu_array;
      for (unsigned int cpu : cpu_affinity) {
        cpu_array.push_back(tvm::runtime::String(std::to_string(cpu)));
      }
      (*config)(-3, runtime_num_threads, cpu_array);
    }
  }
  const PackedFunc* num_threads = Registry::Get("runtime.NumThreads");
  if (num_threads != nullptr) {
    std::cout << "[THREADPOOL] " << (label.empty() ? "thread" : label)
              << " runtime.NumThreads=" << (*num_threads)().operator int()
              << " requested_cpus="
              << (cpu_affinity.empty() ? "default" : CPUListString(cpu_affinity))
              << " effective_cpus=" << CurrentThreadAffinity() << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);
    if (args.host_empty) {
      ConfigureProcessAffinity(args.process_cpu_affinity);
      return RunHostEmpty(args);
    }
    ResolveStageThreadDefaults(&args);
    ValidateStageAffinities(args);
    int max_runtime_threads = args.runtime_num_threads;
    for (const StageArgs& stage : args.stages) {
      max_runtime_threads = std::max(max_runtime_threads, stage.runtime_num_threads);
    }
    if (max_runtime_threads > 0) {
      const std::string max_threads_env = std::to_string(max_runtime_threads);
      setenv("TVM_NUM_THREADS", max_threads_env.c_str(), 1);
    }
    ConfigureThreadPool(args.runtime_num_threads, {}, "main");
    ConfigureProcessAffinity(args.process_cpu_affinity);
    std::cout << "[THREADPOOL] resolved stage threads:";
    for (size_t i = 0; i < args.stages.size(); ++i) {
      std::cout << " stage" << i << "=" << args.stages[i].runtime_num_threads;
    }
    std::cout << "\n";

    std::vector<StageExecutor> stages;
    stages.reserve(args.stages.size());
    for (const StageArgs& stage_arg : args.stages) {
      stages.push_back(LoadStage(stage_arg.name, stage_arg.device, stage_arg));
    }

    std::vector<InputRecord> inputs = LoadInputs(args, &stages[0]);
    std::vector<std::unique_ptr<BoundarySlotManager>> p8_slot_managers;
    if (args.p8_managed_slots > 0) {
      p8_slot_managers.reserve(stages.size() - 1);
      for (size_t edge_index = 0; edge_index + 1 < stages.size(); ++edge_index) {
        p8_slot_managers.emplace_back(std::make_unique<BoundarySlotManager>(
            &stages, edge_index, static_cast<size_t>(args.p8_managed_slots),
            args.p8_slot_timeout_s));
      }
      std::vector<std::pair<uint64_t, uint64_t>> ranges;
      for (const auto& manager : p8_slot_managers) {
        for (size_t slot_id = 0; slot_id < manager->slot_count(); ++slot_id) {
          for (const auto& range : manager->physical_ranges(slot_id)) {
            for (const auto& existing : ranges) {
              if (range.first < existing.second && existing.first < range.second) {
                throw std::runtime_error("P8 managed slot physical ranges overlap");
              }
            }
            ranges.push_back(range);
          }
        }
      }
    }
    std::mutex vta_run_mutex;

    auto prepare_managed_frame = [&](Frame* frame) {
      const size_t edge_count = stages.size() - 1;
      frame->p8_edge_tokens.resize(edge_count);
      frame->p8_boundaries.resize(edge_count);
    };

    auto run_managed_stage = [&](Frame* frame, size_t stage_index) {
      StageExecutor& stage = stages.at(stage_index);
      StageTiming& timing = frame->stage_timings.at(stage_index);
      const bool has_incoming = stage_index > 0;
      const bool has_outgoing = stage_index + 1 < stages.size();

      if (has_incoming) {
        const size_t edge_index = stage_index - 1;
        P8BoundaryObservation& observation = frame->p8_boundaries.at(edge_index);
        p8_slot_managers.at(edge_index)->AcquireForConsumer(
            frame->p8_edge_tokens.at(edge_index), &observation.consumer_wait_ms);
      }

      if (has_outgoing) {
        const size_t edge_index = stage_index;
        P8BoundaryObservation& observation = frame->p8_boundaries.at(edge_index);
        double wait_ms = 0.0;
        P8SlotToken token =
            p8_slot_managers.at(edge_index)->AcquireForProducer(frame->frame_id, &wait_ms);
        frame->p8_edge_tokens.at(edge_index) = token;
        observation.valid = true;
        observation.edge_index = edge_index;
        observation.slot_id = token.slot_id;
        observation.generation = token.generation;
        observation.boundary_bytes = p8_slot_managers.at(edge_index)->nbytes();
        observation.physical_address =
            p8_slot_managers.at(edge_index)->physical_address(token.slot_id);
        observation.tensor_bytes = p8_slot_managers.at(edge_index)->tensor_bytes();
        observation.physical_addresses =
            p8_slot_managers.at(edge_index)->physical_addresses(token.slot_id);
        observation.direction = p8_slot_managers.at(edge_index)->direction();
        observation.producer_wait_ms = wait_ms;
      }

      timing.set_start_ms = NowMillis();
      if (has_incoming) {
        const size_t edge_index = stage_index - 1;
        if (stage.input_names.size() != p8_slot_managers.at(edge_index)->tensor_count()) {
          throw std::runtime_error("P8 managed consumer input arity changed at runtime");
        }
        for (size_t tensor_index = 0; tensor_index < stage.input_names.size();
             ++tensor_index) {
          stage.set_input_zero_copy(
              stage.input_names[tensor_index],
              p8_slot_managers.at(edge_index)->ConsumerView(
                  frame->p8_edge_tokens.at(edge_index), tensor_index));
        }
      } else {
        const std::vector<NDArray> first_inputs = ResolveStageInputs(args, *frame, stage_index);
        if (first_inputs.size() != stage.input_names.size()) {
          throw std::runtime_error("P8 managed first-stage input arity mismatch");
        }
        for (size_t input_index = 0; input_index < first_inputs.size(); ++input_index) {
          stage.set_input(stage.input_names[input_index], first_inputs[input_index]);
        }
      }
      if (has_outgoing) {
        const P8SlotToken& token = frame->p8_edge_tokens.at(stage_index);
        const size_t tensor_count = p8_slot_managers.at(stage_index)->tensor_count();
        if (static_cast<size_t>(stage.NumOutputs()) != tensor_count) {
          throw std::runtime_error("P8 managed producer output arity changed at runtime");
        }
        for (size_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
          stage.set_output_zero_copy(
              static_cast<int>(tensor_index),
              p8_slot_managers.at(stage_index)->ProducerView(token, tensor_index));
        }
      }
      timing.set_end_ms = NowMillis();
      if (stage.device == "vta") {
        timing.vta_mutex_timing_valid = true;
        timing.vta_mutex_request_ms = NowMillis();
        std::unique_lock<std::mutex> lock(vta_run_mutex);
        timing.vta_mutex_acquire_ms = NowMillis();
        timing.run_start_ms = NowMillis();
        timing.vta_run_call_start_ms = timing.run_start_ms;
        stage.run();
        timing.vta_run_call_end_ms = NowMillis();
        timing.get_start_ms = timing.vta_run_call_end_ms;
        timing.vta_mutex_release_ms = NowMillis();
        lock.unlock();
      } else {
        timing.run_start_ms = timing.set_end_ms;
        stage.run();
        timing.get_start_ms = NowMillis();
      }

      if (has_outgoing) {
        const P8SlotToken& token = frame->p8_edge_tokens.at(stage_index);
        p8_slot_managers.at(stage_index)->Publish(token);
        // Managed queues carry only the slot token.  The manager owns the shared
        // views, so no borrowed intermediate NDArray escapes into the frame queue.
        frame->stage_outputs.at(stage_index).clear();
      } else {
        const int num_outputs = stage.NumOutputs();
        frame->stage_outputs.at(stage_index).reserve(static_cast<size_t>(num_outputs));
        for (int output_index = 0; output_index < num_outputs; ++output_index) {
          NDArray output = stage.get_output(output_index).operator NDArray();
          frame->stage_outputs.at(stage_index).push_back(
              output.CopyTo(tvm::Device{kDLCPU, 0}));
        }
      }
      if (has_incoming) {
        p8_slot_managers.at(stage_index - 1)->Release(
            frame->p8_edge_tokens.at(stage_index - 1));
      }
      timing.end_ms = NowMillis();
    };

    const PackedFunc* profiler_clear = Registry::Get("vta.runtime.profiler_clear");
    const PackedFunc* profiler_status = Registry::Get("vta.runtime.profiler_status");
    const PackedFunc* profiler_events = Registry::Get("vta.runtime.profiler_events");
    if (!args.profile_dir.empty()) {
      EnsureDir(args.profile_dir);
      if (profiler_clear != nullptr) {
        (*profiler_clear)();
      }
    }

    auto run_serial_frame_on = [&](std::vector<StageExecutor>* executors, int frame_id,
                                   bool measure_process_cpu_time) {
      const InputRecord& input = inputs[static_cast<size_t>(frame_id) % inputs.size()];
      Frame frame;
      frame.frame_id = frame_id;
      frame.input_index = input.input_index;
      frame.input_file = input.input_file;
      frame.enqueue_ms = NowMillis();
      frame.stage_inputs = input.data;
      frame.stage_outputs.resize(executors->size());
      frame.stage_timings.resize(executors->size());
      for (size_t stage_index = 0; stage_index < executors->size(); ++stage_index) {
        ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                            args.stages[stage_index].cpu_affinity,
                            "serial.stage" + std::to_string(stage_index));
        ConfigureProcessAffinity(args.process_cpu_affinity);
        const std::vector<NDArray> current_inputs = ResolveStageInputs(args, frame, stage_index);
        frame.stage_outputs[stage_index] =
            RunStage(&executors->at(stage_index), current_inputs,
                     &frame.stage_timings[stage_index],
                     measure_process_cpu_time);
      }
      frame.done_ms = NowMillis();
      frame.top1 = Top1Float32(frame.stage_outputs.back()[0]);
      return frame;
    };
    auto run_serial_frame = [&](int frame_id, bool measure_process_cpu_time) {
      return run_serial_frame_on(&stages, frame_id, measure_process_cpu_time);
    };
    auto run_managed_serial_frame = [&](int frame_id) {
      const InputRecord& input = inputs[static_cast<size_t>(frame_id) % inputs.size()];
      Frame frame;
      frame.frame_id = frame_id;
      frame.input_index = input.input_index;
      frame.input_file = input.input_file;
      frame.enqueue_ms = NowMillis();
      frame.stage_inputs = input.data;
      frame.stage_outputs.resize(stages.size());
      frame.stage_timings.resize(stages.size());
      prepare_managed_frame(&frame);
      for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
        ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                            args.stages[stage_index].cpu_affinity,
                            "p8_managed_serial.stage" + std::to_string(stage_index));
        ConfigureProcessAffinity(args.process_cpu_affinity);
        run_managed_stage(&frame, stage_index);
      }
      frame.done_ms = NowMillis();
      frame.top1 = Top1Float32(frame.stage_outputs.back()[0]);
      return frame;
    };

    if (args.p8a_edge >= 0) {
      if (inputs.size() < 2 || args.runs < 2) {
        throw std::runtime_error("P8A requires at least two alternating inputs and two runs");
      }
      struct BaselineObservation {
        int input_index{0};
        uint64_t input_hash{0};
        std::vector<uint64_t> output_hashes;
        double total_latency_ms{0.0};
        StageTiming producer_timing;
        StageTiming consumer_timing;
      };

      std::vector<StageExecutor> baseline_stages;
      baseline_stages.reserve(args.stages.size());
      for (const StageArgs& stage_arg : args.stages) {
        baseline_stages.push_back(LoadStage(stage_arg.name, stage_arg.device, stage_arg));
      }

      auto observe_baseline = [&](Frame frame) {
        const size_t producer = static_cast<size_t>(args.p8a_edge);
        const size_t consumer = producer + 1;
        return BaselineObservation{
            frame.input_index,
            FNV1a64(TensorDataBytes(frame.stage_inputs[0].operator->()),
                    TensorNBytes(frame.stage_inputs[0].operator->())),
            OutputHashes(frame.stage_outputs.back()),
            frame.done_ms - frame.enqueue_ms,
            frame.stage_timings[producer],
            frame.stage_timings[consumer],
        };
      };

      P8SharedEdge shared =
          BindP8HeterogeneousEdge(&stages, static_cast<size_t>(args.p8a_edge));
      auto run_zero_copy_frame = [&](int frame_id) {
        const InputRecord& input = inputs[static_cast<size_t>(frame_id) % inputs.size()];
        Frame frame;
        frame.frame_id = frame_id;
        frame.input_index = input.input_index;
        frame.input_file = input.input_file;
        frame.enqueue_ms = NowMillis();
        frame.stage_inputs = input.data;
        frame.stage_outputs.resize(stages.size());
        frame.stage_timings.resize(stages.size());
        for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
          ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                              args.stages[stage_index].cpu_affinity,
                              "p8a.stage" + std::to_string(stage_index));
          ConfigureProcessAffinity(args.process_cpu_affinity);
          const std::vector<NDArray> current_inputs =
              ResolveStageInputs(args, frame, stage_index);
          if (stage_index != shared.producer_stage && stage_index != shared.consumer_stage) {
            frame.stage_outputs[stage_index] =
                RunStage(&stages[stage_index], current_inputs,
                         &frame.stage_timings[stage_index], false);
            continue;
          }

          StageExecutor& stage = stages[stage_index];
          StageTiming& timing = frame.stage_timings[stage_index];
          timing.set_start_ms = NowMillis();
          if (stage_index == shared.producer_stage) {
            for (size_t input_index = 0; input_index < current_inputs.size(); ++input_index) {
              stage.set_input(stage.input_names[input_index], current_inputs[input_index]);
            }
          } else if (current_inputs.size() != 1 ||
                     current_inputs[0]->data != shared.consumer_owner->data) {
            throw std::runtime_error("P8 consumer did not resolve the bound shared slot");
          }
          timing.set_end_ms = NowMillis();
          timing.run_start_ms = timing.set_end_ms;
          stage.run();
          timing.get_start_ms = NowMillis();
          if (stage_index == shared.producer_stage) {
            frame.stage_outputs[stage_index] = {shared.consumer_owner};
          } else {
            const int num_outputs = stage.NumOutputs();
            frame.stage_outputs[stage_index].reserve(static_cast<size_t>(num_outputs));
            for (int output_index = 0; output_index < num_outputs; ++output_index) {
              NDArray output = stage.get_output(output_index).operator NDArray();
              frame.stage_outputs[stage_index].push_back(
                  output.CopyTo(tvm::Device{kDLCPU, 0}));
            }
          }
          timing.end_ms = NowMillis();
        }
        frame.done_ms = NowMillis();
        frame.top1 = Top1Float32(frame.stage_outputs.back()[0]);
        return frame;
      };

      for (int i = 0; i < args.warmup_runs; ++i) {
        if (i % 2 == 0) {
          (void)run_serial_frame_on(&baseline_stages, i, false);
          (void)run_zero_copy_frame(i);
        } else {
          (void)run_zero_copy_frame(i);
          (void)run_serial_frame_on(&baseline_stages, i, false);
        }
      }
      std::ofstream p8a_result(args.output_jsonl, std::ios::out);
      if (!p8a_result) throw std::runtime_error("Unable to write " + args.output_jsonl);
      const char* safe_copy_env = std::getenv("AXU5EVB_DRIVER_SAFE_COPY");
      const bool safe_copy_effective = EnvironmentFlagEnabled(safe_copy_env, true);
      bool all_correct = true;
      std::vector<BaselineObservation> first_references;
      for (int i = 0; i < args.runs; ++i) {
        Frame baseline_frame;
        Frame frame;
        const bool ordinary_first = i % 2 == 0;
        if (ordinary_first) {
          baseline_frame = run_serial_frame_on(&baseline_stages, i, false);
          frame = run_zero_copy_frame(i);
        } else {
          frame = run_zero_copy_frame(i);
          baseline_frame = run_serial_frame_on(&baseline_stages, i, false);
        }
        const BaselineObservation reference = observe_baseline(std::move(baseline_frame));
        if (first_references.size() < 2) first_references.push_back(reference);
        const std::vector<uint64_t> observed_hashes = OutputHashes(frame.stage_outputs.back());
        const bool correct = frame.input_index == reference.input_index &&
                             observed_hashes == reference.output_hashes;
        all_correct = all_correct && correct;
        const StageTiming& producer =
            frame.stage_timings[static_cast<size_t>(args.p8a_edge)];
        const StageTiming& consumer =
            frame.stage_timings[static_cast<size_t>(args.p8a_edge + 1)];
        const double baseline_edge_copy_ms =
            reference.producer_timing.GetMs() + reference.consumer_timing.SetMs();
        p8a_result << std::fixed << std::setprecision(9)
                   << "{\"mode\":\"p8_heterogeneous_single_slot\""
                   << ",\"direction\":\"" << shared.direction << "\""
                   << ",\"frame_id\":" << i
                   << ",\"pair_order\":\""
                   << (ordinary_first ? "ordinary_then_zero_copy"
                                      : "zero_copy_then_ordinary")
                   << "\""
                   << ",\"input_index\":" << frame.input_index
                   << ",\"input_hash\":\"" << UInt64Hex(reference.input_hash) << "\""
                   << ",\"reference_output_hashes\":"
                   << HashesJSON(reference.output_hashes)
                   << ",\"zero_copy_output_hashes\":" << HashesJSON(observed_hashes)
                   << ",\"reference_scope\":\"ordinary_copy_same_compiled_graph\""
                   << ",\"reference_correctness_passed\":"
                   << (correct ? "true" : "false")
                   << ",\"producer_stage\":" << shared.producer_stage
                   << ",\"consumer_stage\":" << shared.consumer_stage
                   << ",\"boundary_shape\":" << ShapeJSON(shared.cpu_owner.operator->())
                   << ",\"boundary_dtype\":\""
                   << JsonEscape(DTypeString(shared.cpu_owner->dtype)) << "\""
                   << ",\"boundary_bytes\":" << shared.nbytes
                   << ",\"slot_virtual_address\":\"0x" << std::hex
                   << reinterpret_cast<uintptr_t>(shared.cpu_owner->data) << std::dec << "\""
                   << ",\"slot_physical_address\":\"0x" << std::hex
                   << shared.physical_address << std::dec << "\""
                   << ",\"slot_alignment_bytes\":256"
                   << ",\"slot_allocation_owner\":\"direct_vta_driver_cached\""
                   << ",\"same_virtual_address\":true"
                   << ",\"producer_view_device_type\":"
                   << static_cast<int>(shared.producer_device_type)
                   << ",\"consumer_view_device_type\":"
                   << static_cast<int>(shared.consumer_device_type)
                   << ",\"baseline_materialization_bytes\":" << (2 * shared.nbytes)
                   << ",\"zero_copy_materialization_bytes\":0"
                   << ",\"baseline_edge_copy_ms\":" << baseline_edge_copy_ms
                   << ",\"baseline_total_latency_ms\":" << reference.total_latency_ms
                   << ",\"zero_copy_total_latency_ms\":"
                   << (frame.done_ms - frame.enqueue_ms)
                   << ",\"baseline_producer_run_ms\":"
                   << reference.producer_timing.RunMs()
                   << ",\"zero_copy_producer_run_ms\":" << producer.RunMs()
                   << ",\"baseline_consumer_run_ms\":"
                   << reference.consumer_timing.RunMs()
                   << ",\"zero_copy_consumer_run_ms\":" << consumer.RunMs()
                   << ",\"safe_copy_env\":";
        if (safe_copy_env == nullptr) {
          p8a_result << "null";
        } else {
          p8a_result << "\"" << JsonEscape(safe_copy_env) << "\"";
        }
        p8a_result << ",\"safe_copy_effective\":"
                   << (safe_copy_effective ? "true" : "false") << "}\n";
        p8a_result.flush();
      }
      if (first_references.size() < 2 ||
          first_references[0].input_index == first_references[1].input_index ||
          first_references[0].input_hash == first_references[1].input_hash ||
          first_references[0].output_hashes == first_references[1].output_hashes) {
        throw std::runtime_error(
            "P8A alternating inputs must have distinct inputs and reference outputs");
      }
      if (!all_correct) {
        throw std::runtime_error("P8A zero-copy output differs from the ordinary-copy reference");
      }
      return 0;
    }

    if (args.warmup_runs > 0) {
      for (int i = 0; i < args.warmup_runs; ++i) {
        if (args.p8_managed_slots > 0) {
          (void)run_managed_serial_frame(i);
        } else {
          (void)run_serial_frame(i, false);
        }
      }
      if (profiler_clear != nullptr) {
        (*profiler_clear)();
      }
    }

    if (!args.barrier_each_run) SignalReadyAndWait(args);
    ProcessPmuCounters process_pmu(args.profile_pmu);

    std::ofstream result(args.output_jsonl, std::ios::out);
    if (!result) {
      throw std::runtime_error("Unable to write " + args.output_jsonl);
    }

    const double benchmark_start_ms = NowMillis();
    int completed = 0;

    auto write_completed = [&](const Frame& frame, const std::string& mode) {
      result << ResultJSON(args, frame, completed, mode, benchmark_start_ms) << "\n";
      result.flush();
      ++completed;
      const bool allow_intermediate_profile = args.serial;
      if (allow_intermediate_profile && !args.profile_dir.empty() && completed == 1) {
        DumpProfileSnapshot(args.profile_dir, "single_run", frame.frame_id, args.events_limit,
                            profiler_status, profiler_events);
      }
      if (allow_intermediate_profile && !args.profile_dir.empty() && args.checkpoint_every > 0 &&
          completed % args.checkpoint_every == 0) {
        std::ostringstream name;
        name << "checkpoints/run_" << std::setw(4) << std::setfill('0') << completed;
        const std::filesystem::path full = std::filesystem::path(args.profile_dir) / name.str();
        EnsureDir(full.parent_path().string());
        DumpProfileSnapshot(args.profile_dir, name.str(), frame.frame_id, args.events_limit,
                            profiler_status, profiler_events);
      }
    };

    if (args.serial) {
      for (int i = 0; i < args.runs; ++i) {
        if (args.barrier_each_run) SignalReadyAndWait(args, i);
        if (args.p8_managed_slots > 0) {
          Frame frame = run_managed_serial_frame(i);
          write_completed(frame, "p8_managed_serial");
        } else {
          process_pmu.Start();
          Frame frame = run_serial_frame(i, true);
          frame.process_pmu = process_pmu.Stop();
          write_completed(frame, "serial");
        }
      }
    } else {
      std::vector<std::unique_ptr<BoundedQueue<std::shared_ptr<Frame>>>> queues;
      queues.reserve(stages.size() + 1);
      for (size_t i = 0; i <= stages.size(); ++i) {
        queues.emplace_back(
            std::make_unique<BoundedQueue<std::shared_ptr<Frame>>>(args.queue_depth));
      }
      std::mutex error_mutex;
      std::exception_ptr first_error = nullptr;

      auto record_error = [&](std::exception_ptr err) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = err;
          for (auto& manager : p8_slot_managers) {
            manager->Abort("pipeline worker failure");
          }
          for (auto& queue : queues) {
            queue->Close();
          }
        }
      };

      std::vector<std::thread> stage_threads;
      stage_threads.reserve(stages.size());
      for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
        stage_threads.emplace_back([&, stage_index]() {
          try {
            ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                                args.stages[stage_index].cpu_affinity,
                                "pipeline.stage" + std::to_string(stage_index));
            ConfigureProcessAffinity(args.process_cpu_affinity);
            std::shared_ptr<Frame> frame;
            while (queues[stage_index]->Pop(&frame)) {
              if (args.p8_managed_slots > 0) {
                run_managed_stage(frame.get(), stage_index);
              } else {
                const std::vector<NDArray> stage_inputs =
                    ResolveStageInputs(args, *frame, stage_index);
                if (stages[stage_index].device == "vta") {
                  StageTiming& timing = frame->stage_timings[stage_index];
                  timing.vta_mutex_timing_valid = true;
                  timing.vta_mutex_request_ms = NowMillis();
                  std::unique_lock<std::mutex> lock(vta_run_mutex);
                  timing.vta_mutex_acquire_ms = NowMillis();
                  frame->stage_outputs[stage_index] = RunStage(
                      &stages[stage_index], stage_inputs, &timing, false);
                  timing.vta_run_call_start_ms = timing.run_start_ms;
                  timing.vta_run_call_end_ms = timing.get_start_ms;
                  timing.vta_mutex_release_ms = NowMillis();
                  lock.unlock();
                } else {
                  frame->stage_outputs[stage_index] = RunStage(
                      &stages[stage_index], stage_inputs,
                      &frame->stage_timings[stage_index], false);
                }
              }
              if (stage_index + 1 == stages.size()) {
                frame->done_ms = NowMillis();
                frame->top1 = Top1Float32(frame->stage_outputs[stage_index][0]);
              }
              if (!queues[stage_index + 1]->Push(frame)) break;
            }
            queues[stage_index + 1]->Close();
          } catch (...) {
            record_error(std::current_exception());
          }
        });
      }

      std::thread producer_thread([&]() {
        try {
          for (int i = 0; i < args.runs; ++i) {
            const InputRecord& input = inputs[static_cast<size_t>(i) % inputs.size()];
            auto frame = std::make_shared<Frame>();
            frame->frame_id = i;
            frame->input_index = input.input_index;
            frame->input_file = input.input_file;
            frame->enqueue_ms = NowMillis();
            frame->stage_inputs = input.data;
            frame->stage_outputs.resize(stages.size());
            frame->stage_timings.resize(stages.size());
            if (args.p8_managed_slots > 0) prepare_managed_frame(frame.get());
            if (!queues[0]->Push(frame)) {
              break;
            }
          }
          queues[0]->Close();
        } catch (...) {
          queues[0]->Close();
          record_error(std::current_exception());
        }
      });

      std::shared_ptr<Frame> frame;
      while (queues.back()->Pop(&frame)) {
        write_completed(*frame, "pipeline");
      }

      producer_thread.join();
      for (std::thread& stage_thread : stage_threads) {
        stage_thread.join();
      }

      if (first_error != nullptr) {
        std::rethrow_exception(first_error);
      }
      if (completed != args.runs) {
        std::ostringstream os;
        os << "Pipeline completed " << completed << " frame(s), expected " << args.runs;
        throw std::runtime_error(os.str());
      }
    }

    for (auto& manager : p8_slot_managers) {
      manager->VerifyAllFree();
    }

    DumpProfileSnapshot(args.profile_dir, "benchmark_totals", args.runs - 1, args.events_limit,
                        profiler_status, profiler_events);
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "vta_stage_pipeline_runner error: " << err.what() << "\n";
    return 1;
  }
}
