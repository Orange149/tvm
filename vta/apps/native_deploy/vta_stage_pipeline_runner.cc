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
#include "vta_shared_buffer_plan.h"
#include "vta_managed_lifecycle_control.h"
#include "vta_command_capacity_certificate.h"
#include "vta_cpu_stage_gate.h"
#include "vta_cpu_resource_groups.h"
#include "vta_session_profile.h"

#include <algorithm>
#include <dlfcn.h>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
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
  bool cpu_input_zero_copy = false;
};

struct Args {
  std::vector<StageArgs> stages;
  std::vector<unsigned int> process_cpu_affinity;
  std::string input;
  std::vector<std::string> input_files;
  std::string input_list;
  bool input_list_multi{false};
  std::string output_jsonl = "native_result.jsonl";
  std::string output_mode = "classification";
  std::string output_dump_dir;
  std::string profile_dir;
  std::string allocation_snapshot;
  std::string thread_affinity_report;
  std::string session_profile_json;
  int session_profile_first_completion{-1};
  int session_profile_completion_count{0};
  std::string block_profile_json;
  std::string component_thread_transition;
  std::string shared_buffer_plan;
  std::string managed_lifecycle_plan;
  std::string command_capacity_certificate;
  std::string bitstream_artifact;
  int runs = 1;
  int warmup_runs = 0;
  int queue_depth = 2;
  int runtime_num_threads = 4;
  int events_limit = 200;
  int checkpoint_every = 0;
  int start_timeout_s = 120;
  int process_timeout_s = 0;
  std::string ready_file;
  std::string start_file;
  std::string barrier_token;
  bool profile_pmu = false;
  bool barrier_each_run = false;
  bool serial = false;
  bool queue_timing = false;
  bool defer_result_write = false;
  bool c1_memory_audit = false;
  int64_t shared_slot_budget_bytes = -1;
  std::string cpu_stage_gate = "off";
  bool cpu_input_zero_copy = false;
  bool host_empty = false;
  int host_empty_stage_count = 1;
  int host_empty_inner_repeats = 1000;
  int p8a_edge = -1;
  int p8_managed_slots = 0;
  int p8_slot_timeout_s = 120;
  int block_profile_frames = 0;
  int transition_wait_us = 0;
};

double NowMillis() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(now).count();
}

// Qualification-only markers delimit the actual worker's submissions while the
// global VTA mutex is held. No marker, debug flag, or queue query is enabled in
// performance sessions. A main-thread queue cannot certify a worker's TLS queue.
void C1MemoryAudit(const Args& args, int frame, size_t stage, bool begin) {
  if (!args.c1_memory_audit) return;
  using TLSHandle = void* (*)();
  using DebugMode = void (*)(void*, int);
  auto handle = reinterpret_cast<TLSHandle>(dlsym(RTLD_DEFAULT, "VTATLSCommandHandle"));
  auto debug = reinterpret_cast<DebugMode>(dlsym(RTLD_DEFAULT, "VTASetDebugMode"));
  const PackedFunc* status = Registry::Get("vta.runtime.queue_capacity_status");
  if (handle == nullptr || debug == nullptr || status == nullptr)
    throw std::runtime_error("C1 memory audit requires worker queue/debug status ABI");
  void* queue = handle();
  if (begin) debug(queue, 2);  // VTA_DEBUG_DUMP_INSN, on this worker only.
  std::cout.flush();
  std::fflush(stdout);
  std::string queue_status = begin ? "null" : (*status)().operator std::string();
  long tid = 0;
#if defined(__linux__)
  tid = static_cast<long>(syscall(SYS_gettid));
#endif
  std::fprintf(stderr, "[C1_MEMORY_STAGE] {\"event\":\"%s\",\"frame\":%d,"
               "\"stage\":%zu,\"tid\":%ld,\"queue_id\":\"%p\",\"time_ms\":%.6f,"
               "\"queue_status\":%s}\n", begin ? "begin" : "end", frame, stage,
               tid, queue, NowMillis(), queue_status.c_str());
  std::fflush(stderr);
  if (!begin) debug(queue, 0);
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
      << "  --input-list-multi (each input-list line contains comma-separated frontier files)\n"
      << "  --runs N\n"
      << "  --warmup-runs N\n"
      << "  --allocation-snapshot PATH (diagnostic driver required)\n"
      << "  --process-timeout-s N (Linux process deadline; 0 disables)\n"
      << "  --shared-buffer-plan PATH (fixed K2 input-pool reuse)\n"
      << "  --command-capacity-certificate PATH (exact-program P-CAP preflight)\n"
      << "  --bitstream-artifact PATH (required by command-capacity certificate)\n"
      << "  --queue-depth N\n"
      << "  --queue-timing (pipeline-only opt-in queue timestamps)\n"
      << "  --cpu-stage-gate off|observe|exclusive|resource_group (CPU set/run/get admission)\n"
      << "  --defer-result-write (buffer JSON strings until execution finishes)\n"
      << "  --c1-memory-audit (qualification-only worker/queue/submission markers)\n"
      << "  --shared-slot-budget-bytes N (reject before execution; does not resize udmabuf)\n"
      << "  --thread-affinity-report PATH (Linux stage/pool thread masks)\n"
      << "  --session-profile-json PATH (opt-in PMU/schedstat diagnostics)\n"
      << "  --session-profile-window FIRST,COUNT (pipeline completion window; default whole session)\n"
      << "  --block-profile-json PATH --block-profile-frames N (serial diagnostic blocks)\n"
      << "  --component-thread-transition fixed|4-4|1-4|3-1-4|3-1-wait-4\n"
      << "  --transition-wait-us N (only with 3-1-wait-4)\n"
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
  std::cerr << "  --cpu-input-zero-copy (CPU-only serial; computed outputs remain snapshots)\n";
  std::cerr << "  --managed-lifecycle-plan PATH (functional-test trace and FIFO gates)\n";
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
    } else if (key == "--input-list-multi") {
      args.input_list_multi = true;
    } else if (key == "--runs") {
      args.runs = std::stoi(need_value(key));
    } else if (key == "--process-timeout-s") {
      args.process_timeout_s = std::stoi(need_value(key));
      if (args.process_timeout_s < 0) throw std::runtime_error("process timeout must be non-negative");
    } else if (key == "--shared-buffer-plan") {
      args.shared_buffer_plan = need_value(key);
    } else if (key == "--command-capacity-certificate") {
      args.command_capacity_certificate = need_value(key);
    } else if (key == "--bitstream-artifact") {
      args.bitstream_artifact = need_value(key);
    } else if (key == "--allocation-snapshot") {
      args.allocation_snapshot = need_value(key);
    } else if (key == "--warmup-runs") {
      args.warmup_runs = std::stoi(need_value(key));
    } else if (key == "--queue-depth") {
      args.queue_depth = std::stoi(need_value(key));
    } else if (key == "--queue-timing") {
      args.queue_timing = true;
    } else if (key == "--defer-result-write") {
      args.defer_result_write = true;
    } else if (key == "--cpu-stage-gate") {
      args.cpu_stage_gate = need_value(key);
    } else if (key == "--thread-affinity-report") {
      args.thread_affinity_report = need_value(key);
    } else if (key == "--session-profile-json") {
      args.session_profile_json = need_value(key);
    } else if (key == "--session-profile-window") {
      auto values = SplitCSV(need_value(key));
      if (values.size() != 2) throw std::runtime_error("profile window requires FIRST,COUNT");
      args.session_profile_first_completion = std::stoi(values[0]);
      args.session_profile_completion_count = std::stoi(values[1]);
    } else if (key == "--block-profile-json") {
      args.block_profile_json = need_value(key);
    } else if (key == "--block-profile-frames") {
      args.block_profile_frames = std::stoi(need_value(key));
    } else if (key == "--component-thread-transition") {
      args.component_thread_transition = need_value(key);
    } else if (key == "--transition-wait-us") {
      args.transition_wait_us = std::stoi(need_value(key));
    } else if (key == "--runtime-num-threads") {
      args.runtime_num_threads = std::stoi(need_value(key));
    } else if (key == "--process-cpu-affinity") {
      args.process_cpu_affinity = ParseCPUList(need_value(key));
    } else if (key == "--c1-memory-audit") {
      args.c1_memory_audit = true;
    } else if (key == "--shared-slot-budget-bytes") {
      args.shared_slot_budget_bytes = std::stoll(need_value(key));
      if (args.shared_slot_budget_bytes < 0)
        throw std::runtime_error("shared slot budget must be non-negative");
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
    } else if (key == "--cpu-input-zero-copy") {
      args.cpu_input_zero_copy = true;
    } else if (key == "--managed-lifecycle-plan") {
      args.managed_lifecycle_plan = need_value(key);
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
    stage.cpu_input_zero_copy = args.cpu_input_zero_copy;
  }
  if (args.cpu_input_zero_copy &&
      (!args.serial || args.p8_managed_slots > 0 || args.p8a_edge >= 0 || args.host_empty ||
       std::any_of(args.stages.begin(), args.stages.end(),
                   [](const StageArgs& stage) { return stage.device != "cpu"; }))) {
    throw std::runtime_error("--cpu-input-zero-copy requires CPU-only ordinary serial execution");
  }
  if (args.queue_timing && (args.serial || args.host_empty || args.p8a_edge >= 0)) {
    throw std::runtime_error("--queue-timing requires graph pipeline execution");
  }
  if (args.cpu_stage_gate != "off" && args.cpu_stage_gate != "observe" &&
      args.cpu_stage_gate != "exclusive" && args.cpu_stage_gate != "resource_group") {
    throw std::runtime_error("--cpu-stage-gate must be off, observe, exclusive or resource_group");
  }
  if (args.cpu_stage_gate != "off" &&
      (args.serial || args.host_empty || args.p8a_edge >= 0 || args.p8_managed_slots > 0)) {
    throw std::runtime_error("--cpu-stage-gate requires ordinary graph pipeline execution");
  }
  if (args.defer_result_write && (args.host_empty || args.p8a_edge >= 0)) {
    throw std::runtime_error("--defer-result-write requires ordinary graph execution");
  }
  if (args.session_profile_first_completion != -1 &&
      (args.serial || args.session_profile_json.empty() ||
       args.session_profile_first_completion < 0 || args.session_profile_completion_count < 2 ||
       args.session_profile_first_completion + args.session_profile_completion_count > args.runs)) {
    throw std::runtime_error("session profile window requires pipeline mode and a valid completion range");
  }
  if (!args.thread_affinity_report.empty() && (args.host_empty || args.p8a_edge >= 0)) {
    throw std::runtime_error("--thread-affinity-report requires ordinary graph execution");
  }
  if (!args.session_profile_json.empty() &&
      (args.host_empty || args.p8a_edge >= 0 || args.p8_managed_slots > 0)) {
    throw std::runtime_error("--session-profile-json requires ordinary graph execution");
  }
  if (!args.block_profile_json.empty() &&
      (!args.serial || args.host_empty || args.p8a_edge >= 0 || args.p8_managed_slots > 0 ||
       args.block_profile_frames <= 0)) {
    throw std::runtime_error(
        "--block-profile-json requires ordinary serial execution and positive block frames");
  }
  if (args.block_profile_json.empty() && args.block_profile_frames != 0) {
    throw std::runtime_error("--block-profile-frames requires --block-profile-json");
  }
  if (!args.block_profile_json.empty() && !args.session_profile_json.empty()) {
    throw std::runtime_error("block and whole-session profiles are mutually exclusive");
  }
  const std::vector<std::string> transition_modes{
      "", "fixed", "4-4", "1-4", "3-1-4", "3-1-wait-4"};
  if (std::find(transition_modes.begin(), transition_modes.end(),
                args.component_thread_transition) == transition_modes.end()) {
    throw std::runtime_error("unknown --component-thread-transition mode");
  }
  if (!args.component_thread_transition.empty() &&
      (!args.serial || args.stages.size() != 1 || args.stages[0].device != "cpu" ||
       args.stages[0].runtime_num_threads != 4)) {
    throw std::runtime_error(
        "component thread transitions require a one-stage CPU serial t4 session");
  }
  if (args.transition_wait_us < 0 ||
      (args.transition_wait_us != 0 &&
       args.component_thread_transition != "3-1-wait-4")) {
    throw std::runtime_error(
        "--transition-wait-us is non-negative and only valid with 3-1-wait-4");
  }
  if (!args.managed_lifecycle_plan.empty() &&
      (args.serial || args.p8_managed_slots != 2 || args.p8a_edge >= 0 || args.host_empty ||
       args.warmup_runs != 0)) {
    throw std::runtime_error("managed lifecycle plan requires the ordinary two-slot managed pipeline");
  }
  if (!args.command_capacity_certificate.empty() &&
      (args.host_empty || args.bitstream_artifact.empty())) {
    throw std::runtime_error(
        "command capacity certificate requires graph stages and --bitstream-artifact");
  }
  if (args.command_capacity_certificate.empty() && !args.bitstream_artifact.empty()) {
    throw std::runtime_error(
        "--bitstream-artifact is only accepted with --command-capacity-certificate");
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
    if (args.output_mode == "raw_all_stages" && !args.c1_memory_audit) {
      throw std::runtime_error(
          "P8 managed slots do not retain intermediate tensors for raw_all_stages");
    }
    if (args.serial && args.p8_managed_slots != 1 && args.shared_buffer_plan.empty()) {
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
    // An explicit VTA mask configures its host thread/pool. Its logical VTA
    // thread count is independent of the number of eligible host CPU cores.
    if (stage.device == "cpu" && !stage.cpu_affinity.empty() &&
        stage.cpu_affinity.size() != static_cast<size_t>(stage.runtime_num_threads)) {
      throw std::runtime_error("--stage" + std::to_string(i) +
                               "-cpu-affinity count must equal runtime-num-threads");
    }
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (!stage.cpu_affinity.empty() && hardware_threads != 0 &&
        stage.cpu_affinity.back() >= hardware_threads) {
      throw std::runtime_error("stage affinity references a core outside the system");
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

struct ThreadAffinityRecord {
  long caller_tid{0};
  int active_runtime_threads{0};
  std::vector<long> thread_ids;
  std::vector<std::vector<unsigned int>> effective_affinities;
};

std::vector<long> ProcessThreadIds() {
#if defined(__linux__)
  DIR* directory = opendir("/proc/self/task");
  if (directory == nullptr) throw std::runtime_error("unable to list affinity-report threads");
  std::vector<long> tids;
  while (dirent* entry = readdir(directory)) {
    char* end = nullptr;
    const long tid = std::strtol(entry->d_name, &end, 10);
    if (end != entry->d_name && *end == '\0' && tid > 0) tids.push_back(tid);
  }
  closedir(directory);
  std::sort(tids.begin(), tids.end());
  return tids;
#else
  throw std::runtime_error("thread affinity reports require Linux");
#endif
}

void CaptureThreadAffinities(ThreadAffinityRecord* record) {
#if defined(__linux__)
  record->effective_affinities.clear();
  for (long tid : record->thread_ids) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(static_cast<pid_t>(tid), sizeof(cpuset), &cpuset) != 0) {
      throw std::runtime_error("unable to read affinity for thread " + std::to_string(tid));
    }
    std::vector<unsigned int> cpus;
    for (unsigned int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &cpuset)) cpus.push_back(cpu);
    }
    record->effective_affinities.push_back(std::move(cpus));
  }
#else
  throw std::runtime_error("thread affinity reports require Linux");
#endif
}

void WriteThreadAffinityReport(const Args& args, const std::vector<ThreadAffinityRecord>& records) {
  std::ostringstream os;
  auto cpus_json = [&](const std::vector<unsigned int>& cpus) {
    os << "[";
    for (size_t i = 0; i < cpus.size(); ++i) {
      if (i != 0) os << ",";
      os << cpus[i];
    }
    os << "]";
  };
  os << "{\"mode\":\"" << (args.serial ? "serial" : "pipeline")
     << "\",\"pool_ownership\":\""
     << (args.serial ? "shared_main_thread_pool" : "distinct_stage_thread_pools")
     << "\",\"stages\":[";
  for (size_t stage = 0; stage < records.size(); ++stage) {
    if (stage != 0) os << ",";
    const auto& record = records[stage];
    os << "{\"stage_index\":" << stage << ",\"caller_tid\":" << record.caller_tid
       << ",\"active_runtime_threads\":" << record.active_runtime_threads
       << ",\"requested_cpus\":";
    cpus_json(args.stages[stage].cpu_affinity);
    os << ",\"thread_ids\":[";
    for (size_t i = 0; i < record.thread_ids.size(); ++i) {
      if (i != 0) os << ",";
      os << record.thread_ids[i];
    }
    os << "],\"effective_affinities\":[";
    for (size_t i = 0; i < record.thread_ids.size(); ++i) {
      if (i != 0) os << ",";
      os << "{\"tid\":" << record.thread_ids[i] << ",\"cpus\":";
      cpus_json(record.effective_affinities.at(i));
      os << "}";
    }
    os << "]}";
  }
  os << "]}\n";
  WriteFile(args.thread_affinity_report, os.str());
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

NDArray MakeCPUInputLike(const DLTensor* input_tensor, const std::string& input_bytes) {
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
  bool has_linked_parameters{false};
  std::string name;
  std::string device;
  std::vector<std::string> input_names;
  // -1 denotes a computed output. Other values retain an actual incoming view.
  // Derive this from graph heads, never from an unverified eligibility flag.
  std::vector<int> output_live_input_indices;
  bool cpu_input_zero_copy{false};
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
  stage.has_linked_parameters = lib.GetFunction("__tvm_lookup_linked_param", true).defined();
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
  stage.cpu_input_zero_copy = stage_args.cpu_input_zero_copy;
  if (stage.cpu_input_zero_copy && !stage.set_input_zero_copy.defined()) {
    throw std::runtime_error("CPU input zero-copy function is unavailable");
  }
  using namespace shared_buffer;
  const JSON compiled_graph = Parse(graph_json);
  const auto& heads = As<Array>(Field(compiled_graph, "heads"));
  const auto& nodes = As<Array>(Field(compiled_graph, "nodes"));
  Require(heads.size() == static_cast<size_t>(stage.NumOutputs()), "graph output arity mismatch");
  stage.output_live_input_indices.assign(heads.size(), -1);
  for (size_t output_index = 0; output_index < heads.size(); ++output_index) {
    const auto& head = As<Array>(heads[output_index]);
    Require(head.size() == 3, "invalid graph head");
    const auto& node = nodes.at(Number(head.at(0)));
    if (Text(Field(node, "op")) != "null") continue;
    const auto name = Text(Field(node, "name"));
    const auto input = std::find(stage.input_names.begin(), stage.input_names.end(), name);
    if (input == stage.input_names.end()) continue; // A parameter/constant head is not a live input.
    Require(Number(head.at(1)) == 0 && As<Array>(Field(node, "inputs")).empty(),
            "live-input head is not a plain argument");
    const auto& args = As<Array>(Field(compiled_graph, "arg_nodes"));
    Require(std::any_of(args.begin(), args.end(), [&](const JSON& arg) {
      return Number(arg) == Number(head.at(0));
    }), "live-input head is not in arg_nodes");
    stage.output_live_input_indices[output_index] = static_cast<int>(input - stage.input_names.begin());
  }

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
  if (first_stage->input_names.size() != 1 && !args.input_list_multi) {
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
    auto paths = args.input_list_multi ? SplitCSV(rel) : std::vector<std::string>{rel};
    if (paths.size() != first_stage->input_names.size()) {
      throw std::runtime_error("input-list frontier arity mismatch at record " +
                               std::to_string(input_index));
    }
    std::vector<NDArray> arrays;
    for (size_t slot = 0; slot < paths.size(); ++slot) {
      std::filesystem::path input_path(paths[slot]);
      auto resolved = input_path.is_absolute() ? input_path : base / input_path;
      arrays.push_back(make_input(slot, resolved.string()));
    }
    inputs.push_back(InputRecord{input_index, rel, std::move(arrays)});
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

int ResultTop1(const Args& args, const NDArray& output_cpu) {
  const DLDataType dtype = output_cpu->dtype;
  if (args.output_mode != "classification" &&
      (dtype.code != kDLFloat || dtype.bits != 32 || dtype.lanes != 1)) {
    // Raw component profiling can end at an integer quantized frontier.
    return -1;
  }
  return Top1Float32(output_cpu);
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
  uint64_t c1_input_copy_bytes{0};
  uint64_t c1_output_snapshot_bytes{0};
  // Passive G0 trace points. These do not change queue/slot admission or execution.
  double queue_pop_ms{0.0};
  double slots_acquired_ms{0.0};
  bool queue_pop_timing_valid{false};
  bool slots_acquired_timing_valid{false};
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
  double cpu_gate_request_ms{0.0};
  double cpu_gate_acquire_ms{0.0};
  double cpu_gate_release_ms{0.0};
  bool cpu_gate_timing_valid{false};

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
  size_t live_input_copy_bytes{0};
};

// One record per frame/queue. A consumer can dequeue before Push finishes;
// publishing the post-unlock return timestamp therefore needs its own barrier.
// This synchronization is allocated and used only with --queue-timing.
class QueueTiming {
 public:
  struct Snapshot {
    double push_request_ms{0.0};
    double enqueue_ms{0.0};
    double dequeue_ms{0.0};
    double push_return_ms{0.0};
  };

  void PushRequested(double time_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.push_request_ms = time_ms;
  }
  void Enqueued(double time_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.enqueue_ms = time_ms;
  }
  void Dequeued(double time_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_.dequeue_ms = time_ms;
    dequeued_ = true;
  }
  void PushReturned(double time_ms) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot_.push_return_ms = time_ms;
      push_returned_ = true;
    }
    ready_.notify_all();
  }
  Snapshot CompletedSnapshot() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [&] { return push_returned_; });
    if (!dequeued_) throw std::runtime_error("queue timing serialized before dequeue");
    return snapshot_;
  }

 private:
  Snapshot snapshot_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool push_returned_{false};
  bool dequeued_{false};
};

struct Frame {
  int frame_id{0};
  int input_index{0};
  std::string input_file;
  double enqueue_ms{0.0};
  std::vector<NDArray> stage_inputs;
  std::vector<std::vector<NDArray>> stage_outputs;
  std::vector<StageTiming> stage_timings;
  std::vector<std::shared_ptr<QueueTiming>> queue_timings;
  std::vector<P8SlotToken> p8_edge_tokens;
  std::vector<P8BoundaryObservation> p8_boundaries;
  PmuSnapshot process_pmu;
  double done_ms{0.0};
  int top1{-1};

  std::shared_ptr<QueueTiming> QueueTimingFor(size_t edge_index) const {
    return queue_timings.empty() ? nullptr : queue_timings.at(edge_index);
  }
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

struct StageRunHooks {
  std::function<void()> before_run;
  std::function<void(double)> started;
  std::function<void(double)> finished;
  std::function<void()> after_consumer_acquire, after_producer_acquire;
  std::function<void()> after_run, before_publish, before_release;
};

NDArray ResolveStageOutput(StageExecutor* stage, const std::vector<NDArray>& inputs,
                           int output_index) {
  const int live_input = stage->output_live_input_indices.at(static_cast<size_t>(output_index));
  if (live_input >= 0) return inputs.at(static_cast<size_t>(live_input));
  return stage->get_output(output_index).operator NDArray();
}

// Opt-in experiment instrumentation: reread actual executor/slot storage before
// publication or release. CPU snapshots alone cannot certify DMA completion.
void CheckStableStageOutputs(StageExecutor* stage,
                             const std::function<NDArray(int)>& resolve) {
  const char* directory = std::getenv("THESIS_OUTPUT_STABILITY_DIR");
  if (directory == nullptr || directory[0] == '\0') return;
  static size_t call = 0;  // Enabled only in serial correctness sessions.
  const size_t identity = call++;
  std::filesystem::create_directories(directory);
  std::vector<NDArray> first;
  for (int i = 0; i < stage->NumOutputs(); ++i)
    first.push_back(resolve(i).CopyTo(tvm::Device{kDLCPU, 0}));
  const double immediate_ms = NowMillis();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  for (int i = 0; i < stage->NumOutputs(); ++i) {
    NDArray second = resolve(i).CopyTo(tvm::Device{kDLCPU, 0});
    const double delayed_ms = NowMillis();
    const size_t bytes = TensorNBytes(first[i].operator->());
    const bool stable = bytes == TensorNBytes(second.operator->()) &&
        std::memcmp(TensorDataBytes(first[i].operator->()),
                    TensorDataBytes(second.operator->()), bytes) == 0;
    const std::string stem = stage->name + "_call" + std::to_string(identity) +
        "_out" + std::to_string(i);
    for (const auto& item : std::vector<std::pair<std::string, NDArray>>{
             {"immediate", first[i]}, {"delayed", second}}) {
      std::ofstream output(std::string(directory) + "/" + stem + "_" + item.first + ".bin",
                           std::ios::binary);
      output.write(reinterpret_cast<const char*>(TensorDataBytes(item.second.operator->())), bytes);
      if (!output) throw std::runtime_error("Cannot preserve stability output");
    }
    std::cout << "[THESIS_STABILITY] {\"stage\":\"" << JsonEscape(stage->name)
              << "\",\"call\":" << identity << ",\"output_index\":" << i
              << ",\"shape\":" << ShapeJSON(first[i].operator->())
              << ",\"dtype\":\"" << DTypeString(first[i]->dtype)
              << "\",\"stem\":\"" << stem << "\",\"immediate_ms\":"
              << std::fixed << std::setprecision(6) << immediate_ms
              << ",\"delayed_ms\":" << delayed_ms
              << ",\"stable\":" << (stable ? "true" : "false") << "}\n";
    if (!stable || delayed_ms - immediate_ms < 10.0)
      throw std::runtime_error("Returned output failed 10 ms stability: " + stem);
  }
}

std::vector<NDArray> RunStage(StageExecutor* stage, const std::vector<NDArray>& inputs,
                              StageTiming* timing, bool measure_process_cpu_time,
                              const StageRunHooks* hooks = nullptr, bool count_c1_copy_bytes = false) {
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
    if (stage->cpu_input_zero_copy) {
      stage->set_input_zero_copy(stage->input_names[i], inputs[i]);
    } else {
      stage->set_input(stage->input_names[i], inputs[i]);
      if (count_c1_copy_bytes) timing->c1_input_copy_bytes += TensorNBytes(inputs[i].operator->());
    }
  }
  timing->set_end_ms = NowMillis();
  if (measure_process_cpu_time) {
    timing->process_cpu_after_set_ms = ProcessCpuMillis();
  }
  timing->run_start_ms = timing->set_end_ms;
  if (hooks != nullptr) {
    if (hooks->before_run) hooks->before_run();
    timing->run_start_ms = NowMillis();
    if (hooks->started) hooks->started(timing->run_start_ms);
  }
  stage->run();
  if (measure_process_cpu_time) {
    timing->process_cpu_after_run_ms = ProcessCpuMillis();
  }
  timing->get_start_ms = NowMillis();
  if (hooks != nullptr && hooks->finished) hooks->finished(timing->get_start_ms);
  CheckStableStageOutputs(stage, [&](int i) { return ResolveStageOutput(stage, inputs, i); });

  std::vector<NDArray> outputs;
  const int num_outputs = stage->NumOutputs();
  outputs.reserve(static_cast<size_t>(num_outputs));
  for (int i = 0; i < num_outputs; ++i) {
    const int live_input = stage->output_live_input_indices.at(static_cast<size_t>(i));
    if (live_input >= 0) {
      // SetInputZeroCopy updates operator pointers, not the Var-backed data_entry
      // returned by GetOutput. Retain the real incoming buffer and its owner.
      outputs.push_back(inputs.at(static_cast<size_t>(live_input)));
    } else {
      NDArray output = ResolveStageOutput(stage, inputs, i);
      // Snapshot computed outputs before a later frame reuses this executor pool.
      outputs.push_back(output.CopyTo(tvm::Device{kDLCPU, 0}));
      if (count_c1_copy_bytes) timing->c1_output_snapshot_bytes += TensorNBytes(output.operator->());
    }
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
  // Retains the original view and its storage pool. Never VTAMemFree a borrowed pointer.
  NDArray borrowed_pool_owner;
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
                                               size_t tensor_index, bool borrow_input = false) {
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
  if (borrow_input && (!cpu_to_vta || consumer.has_linked_parameters)) {
    throw std::runtime_error("shared-buffer: unsupported anchor direction or linked parameters");
  }
  NDArray borrowed_cpu_view;
  if (borrow_input) {
    const PackedFunc* view = Registry::Get("vta.runtime.ndarray_shared_cpu_view");
    shared_buffer::Require(view != nullptr, "shared CPU view helper unavailable");
    // ext_dev data is a DataBuffer handle, not a CPU virtual address.
    borrowed_cpu_view = (*view)(consumer_input).operator NDArray();
  }
  void* slot_data = borrow_input ? borrowed_cpu_view->data : VTAMemAlloc(nbytes, VTA_CACHED);
  if (slot_data == nullptr) {
    throw std::runtime_error("P8 failed to allocate a cached u-dma-buf slot");
  }
  std::shared_ptr<void> allocation_owner;
  if (!borrow_input) allocation_owner = std::shared_ptr<void>(slot_data, [](void* data) { VTAMemFree(data); });
  const DLTensor* ext_template = cpu_to_vta ? consumer_tensor : producer_tensor;
  DLTensor ext_view = *ext_template;
  ext_view.data = slot_data;
  ext_view.byte_offset = 0;
  ext_view.strides = nullptr;
  NDArray ext_owner = NDArray::FromExternalDLTensor(ext_view);
  DLTensor cpu_view = ext_view;
  cpu_view.device = tvm::Device{kDLCPU, 0};
  NDArray cpu_owner = borrow_input ? borrowed_cpu_view : NDArray::FromExternalDLTensor(cpu_view);
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
  if (borrow_input) edge.borrowed_pool_owner = consumer_input;
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
  if (producer.output_live_input_indices.at(0) < 0) {
    producer.set_output_zero_copy(0, edge.producer_owner);
  }
  consumer.set_input_zero_copy(consumer.input_names[0], edge.consumer_owner);
  return edge;
}

std::string LoadedObjectPath(const char* symbol_name) {
  void* symbol = dlsym(RTLD_DEFAULT, symbol_name);
  Dl_info info{};
  if (symbol == nullptr || dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
    throw std::runtime_error(std::string("cannot identify loaded object for ") + symbol_name);
  }
  return info.dli_fname;
}

std::string ExecutablePath() {
#if defined(__linux__)
  std::vector<char> buffer(4096);
  const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0 || static_cast<size_t>(length) >= buffer.size() - 1) {
    throw std::runtime_error("cannot identify /proc/self/exe");
  }
  buffer[static_cast<size_t>(length)] = '\0';
  return buffer.data();
#else
  throw std::runtime_error("command capacity certificate requires /proc/self/exe");
#endif
}

command_capacity::VerifiedCertificate ValidateCommandCapacityCertificate(const Args& args) {
  if (args.command_capacity_certificate.empty()) {
    return command_capacity::VerifiedCertificate{};
  }
  command_capacity::ProgramIdentity program;
  for (const StageArgs& stage : args.stages) {
    program.stages.push_back(command_capacity::StageIdentity{
        stage.name, stage.device, stage.graph, stage.lib, stage.params, stage.input_names});
  }
  program.runner = ExecutablePath();
  program.runtime = LoadedObjectPath("TVMFuncGetGlobal");
  program.driver = LoadedObjectPath("VTAMemAlloc");
  program.bitstream = args.bitstream_artifact;
  program.shared_buffer_plan = args.shared_buffer_plan;
  program.vta_max_xfer_bytes = VTA_MAX_XFER;
  auto verified = command_capacity::Validate(args.command_capacity_certificate, program);
  std::cout << "[COMMAND-CAPACITY] certificate=" << verified.certificate_sha256
            << " program=" << verified.program_id
            << " evidence_scope=" << verified.evidence_scope
            << " insn=" << verified.insn_capacity_bytes
            << " uop=" << verified.uop_capacity_bytes << " verified before graph load\n";
  return verified;
}

shared_buffer::JSON ValidateSharedBufferPlan(const Args& args) {
  using namespace shared_buffer;
  if (args.shared_buffer_plan.empty()) return JSON();
  Require(args.p8_managed_slots == 2 && args.p8a_edge < 0 && !args.host_empty,
          "plan requires managed K2, no P8A/host-empty");
  JSON plan = Parse(Read(args.shared_buffer_plan));
  Require(Text(Field(plan, "schema")) == "shared-buffer-plan-v1", "unsupported plan schema");
  const auto& stages = As<Array>(Field(plan, "stages"));
  Require(stages.size() == args.stages.size(), "stage count mismatch");
  Require(As<Array>(Field(plan, "edges")).size() + 1 == stages.size(), "edge count mismatch");
  for (size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    const auto& actual = args.stages[i];
    Require(Text(Field(stage, "device")) == actual.device, "device mismatch");
    const auto& names = As<Array>(Field(stage, "input_names"));
    Require(names.size() == actual.input_names.size(), "input arity mismatch");
    for (size_t n = 0; n < names.size(); ++n)
      Require(Text(names[n]) == actual.input_names[n], "input name mismatch");
    CheckHash(actual.graph, Field(stage, "graph_sha256"));
    CheckHash(actual.lib, Field(stage, "lib_sha256"));
    CheckHash(actual.params, Field(stage, "params_sha256"));
  }
  for (const auto& runtime : std::vector<std::pair<const char*, const char*>>{
           {"TVMFuncGetGlobal", "runtime_sha256"}, {"VTAMemAlloc", "driver_sha256"}}) {
    Dl_info info{};
    void* symbol = dlsym(RTLD_DEFAULT, runtime.first);
    Require(symbol != nullptr && dladdr(symbol, &info) != 0 && info.dli_fname != nullptr,
            "cannot identify loaded runtime library");
    CheckHash(info.dli_fname, Field(plan, runtime.second));
  }
  std::cout << "[SHARED-BUFFER] plan/artifact/runtime hashes verified\n";
  return plan;
}

// Prepare all borrowed views before allocating ANY extra storage for this edge.
// Any rejection discards the entire bundle, retaining the original external-K2 path.
std::vector<P8SharedEdge> PrepareSharedBufferAnchor(
    std::vector<StageExecutor>* stages, const Args& args, const shared_buffer::JSON& plan,
    size_t edge_index, std::vector<std::pair<uint64_t, uint64_t>>* claimed) {
  using namespace shared_buffer;
  if (args.shared_buffer_plan.empty()) return {};
  const auto& edge = As<Array>(Field(plan, "edges")).at(edge_index);
  Require(Number(Field(edge, "producer_stage")) == edge_index &&
          Number(Field(edge, "consumer_stage")) == edge_index + 1, "edge identity mismatch");
  auto source = Text(Field(edge, "source"));
  Require(source == "external" || source == "pool-anchor", "unknown source");
  if (source == "external") {
    std::cout << "[SHARED-BUFFER] edge=" << edge_index << " source=external reason=static_rule\n";
    return {};
  }
  try {
    const auto& consumer_arg = args.stages.at(edge_index + 1);
    auto& consumer = stages->at(edge_index + 1);
    Require(args.stages.at(edge_index).device == "cpu" && consumer_arg.device == "vta",
            "direction outside input-anchor v1");
    Require(!consumer.has_linked_parameters, "linked parameter module");
    const auto& tensors = As<Array>(Field(edge, "tensors"));
    Require(tensors.size() == consumer.input_names.size() && !tensors.empty() &&
            tensors.size() == static_cast<size_t>(stages->at(edge_index).NumOutputs()), "bundle arity mismatch");
    JSON graph = Parse(Read(consumer_arg.graph));
    auto parameters = ParameterNames(Read(consumer_arg.params));
    using Snapshot = int (*)(uint64_t*, size_t);
    auto snapshot = reinterpret_cast<Snapshot>(dlsym(RTLD_DEFAULT, "VTAMemAllocationSnapshotV1"));
    uint64_t pool[8]{};
    Require(snapshot != nullptr && snapshot(pool, 8) == 0, "pool snapshot ABI unavailable");
    std::vector<P8SharedEdge> anchors;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      Require(Number(Field(tensor, "tensor_index")) == i &&
              Text(Field(tensor, "input_name")) == consumer.input_names[i], "tensor identity mismatch");
      CheckExclusiveInput(graph, consumer.input_names[i], Number(Field(tensor, "entry")),
                          Number(Field(tensor, "storage_id")), parameters);
      NDArray input = consumer.get_input(consumer.input_names[i]).operator NDArray();
      Require(input->byte_offset == 0 && input->strides == nullptr &&
              input->device.device_type == kDLExtDev && input->device.device_id == 0,
              "unsupported input view");
      size_t bytes = TensorNBytes(input.operator->());
      Require(bytes == Number(Field(tensor, "contract_bytes")), "anchor size mismatch");
      const PackedFunc* make_view = Registry::Get("vta.runtime.ndarray_shared_cpu_view");
      Require(make_view != nullptr, "shared CPU view helper unavailable");
      NDArray cpu_view = (*make_view)(input).operator NDArray();
      uint64_t physical = CheckPhysicalRange(reinterpret_cast<uintptr_t>(cpu_view->data), bytes, pool);
      for (const auto& prior : *claimed)
        Require(!Overlaps(physical, bytes, prior.first, prior.second), "cross-edge range conflict");
      for (const auto& prior : ranges)
        Require(!Overlaps(physical, bytes, prior.first, prior.second), "bundle range conflict");
      auto anchor = AllocateP8HeterogeneousEdgeTensor(stages, edge_index, i, true);
      Require(anchor.physical_address == physical, "physical mapping mismatch");
      anchors.push_back(std::move(anchor));
      ranges.emplace_back(physical, bytes);
    }
    claimed->insert(claimed->end(), ranges.begin(), ranges.end());
    for (size_t i = 0; i < anchors.size(); ++i) {
      std::cout << "[SHARED-BUFFER] edge=" << edge_index << " tensor=" << i
                << " source=pool-anchor slot=0 bytes=" << anchors[i].nbytes
                << " physical=" << anchors[i].physical_address << "\n";
    }
    return anchors;
  } catch (const std::exception& error) {
    std::cout << "[SHARED-BUFFER] edge=" << edge_index << " source=external fallback="
              << error.what() << "\n";
    return {};
  }
}

class BoundarySlotManager {
 public:
  BoundarySlotManager(std::vector<StageExecutor>* stages, size_t edge_index,
                      size_t slot_count, int timeout_s, std::vector<P8SharedEdge> anchor = {},
                      std::function<void(const char*, const P8SlotToken&)> observer = {})
      : edge_index_(edge_index), timeout_(std::chrono::seconds(timeout_s)), observer_(std::move(observer)) {
    slots_.reserve(slot_count);
    for (size_t slot_id = 0; slot_id < slot_count; ++slot_id) {
      Slot slot;
      slot.tensors = slot_id == 0 && !anchor.empty() ? std::move(anchor) :
          AllocateP8HeterogeneousEdgeBundle(stages, edge_index);
      slots_.push_back(std::move(slot));
      EmitLocked("slot_initialized", P8SlotToken{edge_index_, slot_id, 0, -1});
    }
  }

  P8SlotToken AcquireForProducer(int frame_id, double* wait_ms) {
    const double start_ms = NowMillis();
    std::unique_lock<std::mutex> lock(mutex_);
    const auto ready = [&]() { return aborted_ || FindFreeSlotLocked() < slots_.size(); };
    if (!ready()) EmitLocked("producer_wait_free", P8SlotToken{edge_index_, slots_.size(), 0, frame_id});
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
    const P8SlotToken token{edge_index_, slot_id, slot.generation, frame_id};
    EmitLocked("producer_acquired", token);
    return token;
  }

  const NDArray& ProducerView(const P8SlotToken& token, size_t tensor_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = ValidateTokenLocked(token, SlotState::kProducerWriting);
    return slot.tensors.at(tensor_index).producer_owner;
  }

  const NDArray& ProducerCPUView(const P8SlotToken& token, size_t tensor_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    ThrowIfAbortedLocked();
    Slot& slot = ValidateTokenLocked(token, SlotState::kProducerWriting);
    // The CPU owner aliases the same validated u-dma-buf allocation. A raw
    // external view has no DataBuffer header and cannot use ext_dev CopyTo.
    return slot.tensors.at(tensor_index).cpu_owner;
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
    ThrowIfAbortedLocked();
    Slot& slot = ValidateTokenLocked(token, SlotState::kProducerWriting);
    slot.state = SlotState::kReady;
    EmitLocked("published", token);
    condition_.notify_all();
  }

  void AcquireForConsumer(const P8SlotToken& token, double* wait_ms) {
    const double start_ms = NowMillis();
    std::unique_lock<std::mutex> lock(mutex_);
    ThrowIfAbortedLocked();
    if (token.edge_index != edge_index_ || token.slot_id >= slots_.size()) {
      throw std::runtime_error("P8 slot token references the wrong edge or slot");
    }
    const Slot& initial = slots_[token.slot_id];
    if (initial.generation != token.generation || initial.frame_id != token.frame_id) {
      throw std::runtime_error("P8 stale slot token generation/owner mismatch");
    }
    const auto ready = [&]() {
      if (aborted_) return true;
      const Slot& slot = slots_[token.slot_id];
      // Wake on identity invalidation too; ValidateTokenLocked rejects it below.
      return slot.generation != token.generation || slot.frame_id != token.frame_id ||
             slot.state == SlotState::kReady;
    };
    if (!ready()) EmitLocked("consumer_wait_ready", token);
    if (!condition_.wait_for(lock, timeout_, ready)) {
      throw std::runtime_error("P8 consumer timed out waiting for a READY slot on edge " +
                               std::to_string(edge_index_));
    }
    ThrowIfAbortedLocked();
    Slot& slot = ValidateTokenLocked(token, SlotState::kReady);
    slot.state = SlotState::kConsumerUsing;
    EmitLocked("consumer_acquired", token);
    *wait_ms = NowMillis() - start_ms;
  }

  void Release(const P8SlotToken& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    Slot& slot = ValidateTokenLocked(token, SlotState::kConsumerUsing);
    slot.state = SlotState::kFree;
    slot.frame_id = -1;
    EmitLocked("released", token);
    condition_.notify_all();
  }

  void Abort(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    abort_reason_ = reason;
    EmitLocked("aborted", P8SlotToken{edge_index_, slots_.size(), 0, -1});
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

  void EmitLocked(const char* event, const P8SlotToken& token) {
    if (observer_) observer_(event, token);
  }

  size_t edge_index_{0};
  std::chrono::seconds timeout_;
  std::vector<Slot> slots_;
  size_t next_slot_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
  bool aborted_{false};
  std::string abort_reason_;
  std::function<void(const char*, const P8SlotToken&)> observer_;
};

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity, std::function<void(const char*)> observer = {},
                        bool queue_timing = false)
      : capacity_(capacity), observer_(std::move(observer)), queue_timing_(queue_timing) {}

  bool Push(T value, std::shared_ptr<QueueTiming> timing = nullptr) {
    if (queue_timing_ && !timing) throw std::runtime_error("missing queue timing record");
    if (queue_timing_) timing->PushRequested(NowMillis());
    std::unique_lock<std::mutex> lock(mutex_);
    if (!closed_ && queue_.size() >= capacity_ && observer_) observer_("queue_wait_space");
    not_full_.wait(lock, [&]() { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    queue_.push_back(std::move(value));
    if (queue_timing_) {
      try {
        timings_.push_back(timing);
      } catch (...) {
        queue_.pop_back();  // Keep the value and timing queues aligned on allocation failure.
        throw;
      }
      timing->Enqueued(NowMillis());  // Queue ownership commits while holding its lock.
    }
    not_empty_.notify_one();
    if (queue_timing_) {
      lock.unlock();
      timing->PushReturned(NowMillis());  // Recorded after releasing the queue lock.
    }
    return true;
  }

  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!closed_ && queue_.empty() && observer_) observer_("queue_wait_input");
    not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    *value = std::move(queue_.front());
    queue_.pop_front();
    if (queue_timing_) {
      timings_.front()->Dequeued(NowMillis());  // Dequeue is also recorded under the lock.
      timings_.pop_front();
    }
    not_full_.notify_one();
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_ && observer_) observer_("queue_closed");
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
  std::function<void(const char*)> observer_;
  bool queue_timing_{false};
  std::deque<std::shared_ptr<QueueTiming>> timings_;
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
  // Leave the legacy output schema unchanged when both switches are disabled.
  if (args.queue_timing || args.defer_result_write) {
    os << ",\"queue_timing_enabled\":" << (args.queue_timing ? "true" : "false")
       << ",\"defer_result_write_enabled\":" << (args.defer_result_write ? "true" : "false");
  }
  if (args.cpu_stage_gate != "off") {
    os << ",\"cpu_stage_gate_mode\":\"" << args.cpu_stage_gate << "\""
       << ",\"cpu_stage_gate_scope\":\"CPU set/run/get and output snapshot; wait outside stage service\"";
    for (size_t i = 0; i < frame.stage_timings.size(); ++i) {
      const auto& timing = frame.stage_timings[i];
      for (const auto& point : std::vector<std::pair<const char*, double>>{
               {"request", timing.cpu_gate_request_ms},
               {"acquire", timing.cpu_gate_acquire_ms},
               {"release", timing.cpu_gate_release_ms}}) {
        os << ",\"stage" << i << "_cpu_gate_" << point.first << "_ms\":";
        if (timing.cpu_gate_timing_valid) os << rel(point.second);
        else os << "null";
      }
      os << ",\"stage" << i << "_cpu_gate_wait_ms\":";
      if (timing.cpu_gate_timing_valid)
        os << timing.cpu_gate_acquire_ms - timing.cpu_gate_request_ms;
      else os << "null";
    }
  }
  if (args.queue_timing) {
    os << ",\"queue_timings\":[";
    for (size_t edge_index = 0; edge_index < frame.queue_timings.size(); ++edge_index) {
      // In particular, wait for the final stage's Push return publication. The
      // output queue can wake us before that stage records its return time.
      const auto timing = frame.queue_timings[edge_index]->CompletedSnapshot();
      if (edge_index != 0) os << ",";
      os << "{\"edge_index\":" << edge_index
         << ",\"push_request_ms\":" << rel(timing.push_request_ms)
         << ",\"enqueue_ms\":" << rel(timing.enqueue_ms)
         << ",\"dequeue_ms\":" << rel(timing.dequeue_ms)
         << ",\"push_return_ms\":" << rel(timing.push_return_ms) << "}";
    }
    os << "]";
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
       << ",\"stage" << i << "_binding_done_ms\":" << rel(timing.set_end_ms)
       << ",\"stage" << i << "_run_start_ms\":" << rel(timing.run_start_ms)
       << ",\"stage" << i << "_run_end_ms\":" << rel(timing.get_start_ms)
       << ",\"stage" << i << "_end_ms\":" << rel(timing.end_ms)
       << ",\"stage" << i << "_queue_pop_ms\":";
    if (timing.queue_pop_timing_valid) {
      os << rel(timing.queue_pop_ms);
    } else {
      os << "null";
    }
    os << ",\"stage" << i << "_slots_acquired_ms\":";
    if (timing.slots_acquired_timing_valid) {
      os << rel(timing.slots_acquired_ms);
    } else {
      os << "null";
    }
    os << ",\"stage" << i << "_process_cpu_ms\":";
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
    size_t live_input_copy_bytes = 0;
    for (const auto& boundary : frame.p8_boundaries) {
      if (boundary.valid) live_input_copy_bytes += boundary.live_input_copy_bytes;
    }
    os << ",\"p8_managed_slot_count\":" << args.p8_managed_slots
       << ",\"p8_framework_materialization_bytes\":" << live_input_copy_bytes
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
         << ",\"consumer_wait_ms\":" << boundary.consumer_wait_ms
         << ",\"live_input_copy_bytes\":" << boundary.live_input_copy_bytes << "}";
    }
    os << "]"
       << ",\"p8_producer_slot_wait_ms\":" << total_producer_wait_ms
       << ",\"p8_consumer_slot_wait_ms\":" << total_consumer_wait_ms
       << ",\"p8_total_slot_wait_ms\":"
       << (total_producer_wait_ms + total_consumer_wait_ms);
  }
  if (args.c1_memory_audit) {
    uint64_t materialized = 0;
    if (args.p8_managed_slots > 0) {
      for (const auto& edge : frame.p8_boundaries)
        if (edge.valid) materialized += edge.live_input_copy_bytes;
    } else {
      for (size_t i = 0; i < frame.stage_timings.size(); ++i) {
        if (i > 0) materialized += frame.stage_timings[i].c1_input_copy_bytes;
        if (i + 1 < frame.stage_timings.size())
          materialized += frame.stage_timings[i].c1_output_snapshot_bytes;
      }
    }
    os << ",\"c1_framework_materialization_bytes\":" << materialized;
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

void ConfigureComponentTransition(const Args& args, const StageArgs& target) {
  const std::string& mode = args.component_thread_transition;
  if (mode.empty() || mode == "fixed") return;
  const std::vector<unsigned int> prefix3{0, 1, 2};
  auto configure_target = [&] {
    ConfigureThreadPool(target.runtime_num_threads, target.cpu_affinity,
                        "serial.component.target");
  };
  if (mode == "4-4") {
    configure_target();
    configure_target();
    return;
  }
  if (mode == "1-4") {
    ConfigureThreadPool(1, target.cpu_affinity, "serial.component.prelude1");
    configure_target();
    return;
  }
  if (mode == "3-1-4" || mode == "3-1-wait-4") {
    ConfigureThreadPool(3, prefix3, "serial.component.prelude3");
    ConfigureThreadPool(1, target.cpu_affinity, "serial.component.prelude1");
    if (mode == "3-1-wait-4" && args.transition_wait_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(args.transition_wait_us));
    }
    configure_target();
    return;
  }
  throw std::runtime_error("unhandled component thread transition mode");
}

}  // namespace

managed_lifecycle::Options LoadManagedLifecycleOptions(const Args& args) {
  using namespace shared_buffer;
  const JSON plan = Parse(Read(args.managed_lifecycle_plan));
  Require(Text(Field(plan, "schema")) == "managed-lifecycle-control-v1", "unknown lifecycle schema");
  managed_lifecycle::Options options;
  options.trace_path = Text(Field(plan, "trace_path"));
  options.command_fifo = Text(Field(plan, "command_fifo"));
  Require(std::filesystem::path(options.trace_path).is_absolute() &&
          std::filesystem::path(options.command_fifo).is_absolute() &&
          options.trace_path.find('\0') == std::string::npos &&
          options.command_fifo.find('\0') == std::string::npos, "absolute lifecycle paths required");
  uint64_t timeout = Number(Field(plan, "timeout_seconds"));
  Require(timeout > 0 && timeout <= 60, "lifecycle timeout outside functional-test contract");
  options.timeout_seconds = static_cast<int>(timeout);
  const std::vector<std::string> points{"consumer_acquired", "producer_acquired", "before_run",
                                        "after_run", "before_publish", "before_release"};
  for (const auto& gate : As<Array>(Field(plan, "gates"))) {
    const auto id = Text(Field(gate, "id")); const auto point = Text(Field(gate, "point"));
    uint64_t stage = Number(Field(gate, "stage")), frame = Number(Field(gate, "frame"));
    Require(!id.empty() && id.size() <= 80 &&
            id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos,
            "invalid lifecycle gate id");
    Require(std::find(points.begin(), points.end(), point) != points.end() &&
            stage < args.stages.size() && frame < static_cast<uint64_t>(args.runs),
            "lifecycle selector outside actual stage/frame contract");
    Require((point != "consumer_acquired" && point != "before_release") || stage > 0,
            "consumer gate on stage without incoming edge");
    Require((point != "producer_acquired" && point != "before_publish") || stage + 1 < args.stages.size(),
            "producer gate on stage without outgoing edge");
    for (const auto& prior : options.gates) {
      Require(prior.id != id && !(prior.point == point && prior.stage == static_cast<int>(stage) &&
                                 prior.frame == static_cast<int>(frame)), "duplicate lifecycle id or selector");
    }
    options.gates.push_back({id, point, static_cast<int>(stage), static_cast<int>(frame)});
  }
  return options;
}

// No u-dma-buf allocation and no per-frame hot-path logging. This optional
// diagnostic requires the explicitly deployed driver snapshot ABI.
void WriteAllocationSnapshot(const Args& args, const char* phase) {
  if (args.allocation_snapshot.empty()) return;
  using Snapshot = int (*)(uint64_t*, size_t);
  auto snapshot = reinterpret_cast<Snapshot>(dlsym(RTLD_DEFAULT, "VTAMemAllocationSnapshotV1"));
  uint64_t fields[8] = {};
  if (snapshot == nullptr || snapshot(fields, 8) != 0) {
    throw std::runtime_error("Allocation snapshot requires diagnostic AXU5EVB driver ABI v1");
  }
  std::ofstream out(args.allocation_snapshot, std::ios::app);
  if (!out) throw std::runtime_error("Cannot open allocation snapshot output");
  out << "{\"phase\":\"" << phase << "\",\"initialized\":" << fields[0]
      << ",\"capacity_bytes\":" << fields[1] << ",\"high_water_bytes\":" << fields[2]
      << ",\"requested_bytes\":" << fields[3] << ",\"padding_bytes\":" << fields[4]
      << ",\"allocation_count\":" << fields[5] << ",\"virtual_base\":" << fields[6]
      << ",\"physical_base\":" << fields[7] << "}\n";
  if (!out) throw std::runtime_error("Cannot write allocation snapshot");
}

#ifndef VTA_STAGE_PIPELINE_NO_MAIN
#if defined(__linux__)
void ProcessDeadlineExpired(int) {
  // Only async-signal-safe operations: a timed-out process is never qualified.
  const char message[] = "[C1_PROCESS_DEADLINE_EXPIRED]\n";
  const ssize_t ignored = ::write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)ignored;
  ::_exit(124);
}
#endif

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);
    if (args.c1_memory_audit) {
      // DumpInsn uses stdio, whereas driver completion records use stderr.
      // Preserve whole instruction lines in the combined diagnostic archive.
      std::setvbuf(stdout, nullptr, _IONBF, 0);
    }
    if (args.process_timeout_s > 0) {
#if defined(__linux__)
      std::signal(SIGALRM, ProcessDeadlineExpired);
      ::alarm(static_cast<unsigned int>(args.process_timeout_s));
#else
      throw std::runtime_error("process timeout requires Linux");
#endif
    }
    // Hash subprocesses must run before ConfigureThreadPool starts any threads.
    command_capacity::VerifiedCertificate command_capacity_certificate =
        ValidateCommandCapacityCertificate(args);
    shared_buffer::JSON shared_plan = ValidateSharedBufferPlan(args);
    uint64_t expected_slot_bytes = 0;
    if (args.shared_slot_budget_bytes >= 0 && args.p8_managed_slots > 0) {
      using namespace shared_buffer;
      Require(!args.shared_buffer_plan.empty(), "slot capacity requires an exact shared plan");
      for (const auto& edge : As<Array>(Field(shared_plan, "edges")))
        expected_slot_bytes += Number(Field(edge, "external_k2_bytes"));
      uint64_t saved = Number(Field(shared_plan, "predicted_saved_bytes"));
      Require(saved <= expected_slot_bytes, "invalid shared plan saved bytes");
      expected_slot_bytes -= saved;
      if (expected_slot_bytes > static_cast<uint64_t>(args.shared_slot_budget_bytes)) {
        std::cerr << "[C1_SLOT_CAPACITY_REJECTED] required=" << expected_slot_bytes
                  << " budget=" << args.shared_slot_budget_bytes << " before_graph_load\n";
        return 3;
      }
    }
    managed_lifecycle::Options lifecycle_options;
    std::string lifecycle_plan_hash;
    if (!args.managed_lifecycle_plan.empty()) {
      lifecycle_options = LoadManagedLifecycleOptions(args);
      lifecycle_plan_hash = shared_buffer::FileSHA256(args.managed_lifecycle_plan);
    }
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

    WriteAllocationSnapshot(args, "before_graph_load");
    std::vector<StageExecutor> stages;
    stages.reserve(args.stages.size());
    for (const StageArgs& stage_arg : args.stages) {
      stages.push_back(LoadStage(stage_arg.name, stage_arg.device, stage_arg));
    }

    WriteAllocationSnapshot(args, "graph_and_params_loaded");
    std::vector<InputRecord> inputs = LoadInputs(args, &stages[0]);
    // Declare before managers so their observers cannot outlive this controller.
    std::unique_ptr<managed_lifecycle::Control> lifecycle;
    if (!args.managed_lifecycle_plan.empty()) {
      lifecycle = std::make_unique<managed_lifecycle::Control>(std::move(lifecycle_options));
      lifecycle->Emit("lifecycle_contract_loaded", -1, -1, -1, -1, 0, lifecycle_plan_hash);
    }
    std::vector<std::unique_ptr<BoundarySlotManager>> p8_slot_managers;
    std::vector<std::pair<uint64_t, uint64_t>> anchor_ranges;
    using C1Snapshot = int (*)(uint64_t*, size_t);
    auto c1_snapshot = reinterpret_cast<C1Snapshot>(dlsym(RTLD_DEFAULT, "VTAMemAllocationSnapshotV1"));
    uint64_t c1_before[8]{};
    if (args.shared_slot_budget_bytes >= 0 && args.p8_managed_slots > 0 &&
        (c1_snapshot == nullptr || c1_snapshot(c1_before, 8) != 0))
      throw std::runtime_error("slot capacity requires actual allocation snapshot ABI");
    if (args.p8_managed_slots > 0) {
      p8_slot_managers.reserve(stages.size() - 1);
      for (size_t edge_index = 0; edge_index + 1 < stages.size(); ++edge_index) {
        std::function<void(const char*, const P8SlotToken&)> observer;
        if (lifecycle) observer = [&](const char* point, const P8SlotToken& token) {
          const std::string event(point);
          const int stage = static_cast<int>(token.edge_index) +
              (event == "consumer_acquired" || event == "consumer_wait_ready" || event == "released");
          lifecycle->Emit(event, stage, token.frame_id, static_cast<int64_t>(token.edge_index),
                          token.slot_id < 2 ? static_cast<int64_t>(token.slot_id) : -1, token.generation);
        };
        p8_slot_managers.emplace_back(std::make_unique<BoundarySlotManager>(
            &stages, edge_index, static_cast<size_t>(args.p8_managed_slots),
            args.p8_slot_timeout_s,
            PrepareSharedBufferAnchor(&stages, args, shared_plan, edge_index, &anchor_ranges), observer));
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
    if (args.shared_slot_budget_bytes >= 0 && args.p8_managed_slots > 0) {
      uint64_t after[8]{};
      if (c1_snapshot(after, 8) != 0 || after[3] < c1_before[3] ||
          after[3] - c1_before[3] != expected_slot_bytes ||
          after[3] - c1_before[3] > static_cast<uint64_t>(args.shared_slot_budget_bytes))
        throw std::runtime_error("actual slot allocation/anchor realization differs from capacity contract");
      std::cout << "[C1_SLOT_CAPACITY] actual=" << (after[3] - c1_before[3])
                << " budget=" << args.shared_slot_budget_bytes << " before_execution\n";
    }
    WriteAllocationSnapshot(args, "slots_allocated");
    std::mutex vta_run_mutex;
    std::vector<ThreadAffinityRecord> serial_affinities(stages.size());
    bool serial_affinity_report_written = false;

    auto capture_serial_affinity = [&](size_t stage_index, int frame_id) {
      if (args.thread_affinity_report.empty() || frame_id != 0) return;
      // Serial stages reuse the main thread's pool; these are snapshots after
      // each stage's configuration, rather than independent worker ownership.
      serial_affinities[stage_index].thread_ids = ProcessThreadIds();
      const PackedFunc* count = Registry::Get("runtime.NumThreads");
      serial_affinities[stage_index].active_runtime_threads = count ? (*count)().operator int() : 0;
#if defined(__linux__)
      serial_affinities[stage_index].caller_tid = static_cast<long>(syscall(SYS_gettid));
#endif
      CaptureThreadAffinities(&serial_affinities[stage_index]);
    };

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
      StageRunHooks hooks;
      if (lifecycle) {
        auto pause = [&, stage_index](const char* point) {
          lifecycle->Pause(point, static_cast<int>(stage_index), frame->frame_id);
        };
        hooks.after_consumer_acquire = [pause] { pause("consumer_acquired"); };
        hooks.after_producer_acquire = [pause] { pause("producer_acquired"); };
        hooks.before_run = [pause] { pause("before_run"); };
        hooks.after_run = [pause] { pause("after_run"); };
        hooks.before_publish = [pause] { pause("before_publish"); };
        hooks.before_release = [pause] { pause("before_release"); };
        hooks.started = [&, stage_index](double) {
          lifecycle->Emit("stage_run_started", static_cast<int>(stage_index), frame->frame_id);
        };
        hooks.finished = [&, stage_index](double) {
          lifecycle->Emit("stage_run_finished", static_cast<int>(stage_index), frame->frame_id);
        };
      }

      if (has_incoming) {
        const size_t edge_index = stage_index - 1;
        P8BoundaryObservation& observation = frame->p8_boundaries.at(edge_index);
        p8_slot_managers.at(edge_index)->AcquireForConsumer(
            frame->p8_edge_tokens.at(edge_index), &observation.consumer_wait_ms);
        if (hooks.after_consumer_acquire) hooks.after_consumer_acquire();
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
        if (hooks.after_producer_acquire) hooks.after_producer_acquire();
      }

      timing.slots_acquired_ms = NowMillis();
      timing.slots_acquired_timing_valid = true;
      timing.set_start_ms = NowMillis();
      std::vector<NDArray> bound_inputs;
      if (has_incoming) {
        const size_t edge_index = stage_index - 1;
        if (stage.input_names.size() != p8_slot_managers.at(edge_index)->tensor_count()) {
          throw std::runtime_error("P8 managed consumer input arity changed at runtime");
        }
        for (size_t tensor_index = 0; tensor_index < stage.input_names.size();
             ++tensor_index) {
          bound_inputs.push_back(p8_slot_managers.at(edge_index)->ConsumerView(
              frame->p8_edge_tokens.at(edge_index), tensor_index));
          stage.set_input_zero_copy(
              stage.input_names[tensor_index], bound_inputs.back());
        }
      } else {
        const std::vector<NDArray> first_inputs = ResolveStageInputs(args, *frame, stage_index);
        bound_inputs = first_inputs;
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
          if (stage.output_live_input_indices.at(tensor_index) >= 0) continue;
          stage.set_output_zero_copy(
              static_cast<int>(tensor_index),
              p8_slot_managers.at(stage_index)->ProducerView(token, tensor_index));
        }
      }
      timing.set_end_ms = NowMillis();
      if (hooks.before_run) hooks.before_run();
      if (stage.device == "vta") {
        timing.vta_mutex_timing_valid = true;
        timing.vta_mutex_request_ms = NowMillis();
        std::unique_lock<std::mutex> lock(vta_run_mutex);
        timing.vta_mutex_acquire_ms = NowMillis();
        timing.run_start_ms = NowMillis();
        timing.vta_run_call_start_ms = timing.run_start_ms;
        if (hooks.started) hooks.started(timing.run_start_ms);
        C1MemoryAudit(args, frame->frame_id, stage_index, true);
        stage.run();
        C1MemoryAudit(args, frame->frame_id, stage_index, false);
        timing.vta_run_call_end_ms = NowMillis();
        timing.get_start_ms = timing.vta_run_call_end_ms;
        if (hooks.finished) hooks.finished(timing.get_start_ms);
        timing.vta_mutex_release_ms = NowMillis();
        lock.unlock();
      } else {
        timing.run_start_ms = NowMillis();
        if (hooks.started) hooks.started(timing.run_start_ms);
        stage.run();
        timing.get_start_ms = NowMillis();
        if (hooks.finished) hooks.finished(timing.get_start_ms);
      }
      if (hooks.after_run) hooks.after_run();
      CheckStableStageOutputs(&stage, [&](int i) {
        if (has_outgoing && stage.output_live_input_indices.at(static_cast<size_t>(i)) < 0)
          return p8_slot_managers.at(stage_index)->ProducerCPUView(
              frame->p8_edge_tokens.at(stage_index), static_cast<size_t>(i));
        return ResolveStageOutput(&stage, bound_inputs, i);
      });

      if (has_outgoing) {
        const P8SlotToken& token = frame->p8_edge_tokens.at(stage_index);
        // A Var-backed output cannot be redirected by SetOutputZeroCopy. Copy the
        // live input into the complete outgoing bundle before its single Publish;
        // keep the incoming token held until the copy has completed.
        for (size_t output_index = 0; output_index < stage.output_live_input_indices.size();
             ++output_index) {
          const int live_input = stage.output_live_input_indices[output_index];
          if (live_input < 0) continue;
          NDArray destination = p8_slot_managers.at(stage_index)->ProducerView(token, output_index);
          const NDArray& source = bound_inputs.at(static_cast<size_t>(live_input));
          if (!SameDType(destination->dtype, source->dtype) ||
              !SameShape(destination.operator->(), source.operator->())) {
            throw std::runtime_error("P8 live-input bundle copy shape/dtype mismatch");
          }
          destination.CopyFrom(source);
          frame->p8_boundaries.at(stage_index).live_input_copy_bytes += TensorNBytes(source.operator->());
        }
        const bool retain_audit_snapshots = args.c1_memory_audit &&
                                            args.output_mode == "raw_all_stages";
        if (retain_audit_snapshots) {
          // Qualification snapshots are owned CPU copies, taken while this
          // frame still owns both slot tokens. Never retain a borrowed view
          // after Publish/Release permits another frame to reuse the slot.
          auto& snapshots = frame->stage_outputs.at(stage_index);
          snapshots.clear();
          for (size_t tensor_index = 0; tensor_index <
               p8_slot_managers.at(stage_index)->tensor_count(); ++tensor_index) {
            snapshots.push_back(p8_slot_managers.at(stage_index)
                ->ProducerCPUView(token, tensor_index).CopyTo(tvm::Device{kDLCPU, 0}));
          }
        }
        if (hooks.before_publish) hooks.before_publish();
        p8_slot_managers.at(stage_index)->Publish(token);
        // Managed queues carry only the slot token.  The manager owns the shared
        // views, so no borrowed intermediate NDArray escapes into the frame queue.
        if (!retain_audit_snapshots) frame->stage_outputs.at(stage_index).clear();
      } else {
        const int num_outputs = stage.NumOutputs();
        frame->stage_outputs.at(stage_index).reserve(static_cast<size_t>(num_outputs));
        for (int output_index = 0; output_index < num_outputs; ++output_index) {
          NDArray output = ResolveStageOutput(&stage, bound_inputs, output_index);
          frame->stage_outputs.at(stage_index).push_back(
              output.CopyTo(tvm::Device{kDLCPU, 0}));
        }
      }
      if (has_incoming) {
        if (hooks.before_release) hooks.before_release();
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
        if (!args.component_thread_transition.empty()) {
          ConfigureComponentTransition(args, args.stages[stage_index]);
        } else {
          ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                              args.stages[stage_index].cpu_affinity,
                              "serial.stage" + std::to_string(stage_index));
        }
        ConfigureProcessAffinity(args.process_cpu_affinity);
        capture_serial_affinity(stage_index, frame_id);
        const std::vector<NDArray> current_inputs = ResolveStageInputs(args, frame, stage_index);
        frame.stage_outputs[stage_index] =
            RunStage(&executors->at(stage_index), current_inputs,
                     &frame.stage_timings[stage_index],
                     measure_process_cpu_time);
      }
      frame.done_ms = NowMillis();
      frame.top1 = ResultTop1(args, frame.stage_outputs.back()[0]);
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
        capture_serial_affinity(stage_index, frame_id);
        run_managed_stage(&frame, stage_index);
      }
      frame.done_ms = NowMillis();
      frame.top1 = ResultTop1(args, frame.stage_outputs.back()[0]);
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
            if (stage.output_live_input_indices.at(0) >= 0) {
              NDArray destination = shared.producer_owner;
              destination.CopyFrom(current_inputs.at(
                  static_cast<size_t>(stage.output_live_input_indices.at(0))));
            }
            frame.stage_outputs[stage_index] = {shared.consumer_owner};
          } else {
            const int num_outputs = stage.NumOutputs();
            frame.stage_outputs[stage_index].reserve(static_cast<size_t>(num_outputs));
            for (int output_index = 0; output_index < num_outputs; ++output_index) {
              NDArray output = ResolveStageOutput(&stage, current_inputs, output_index);
              frame.stage_outputs[stage_index].push_back(
                  output.CopyTo(tvm::Device{kDLCPU, 0}));
            }
          }
          timing.end_ms = NowMillis();
        }
        frame.done_ms = NowMillis();
        frame.top1 = ResultTop1(args, frame.stage_outputs.back()[0]);
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

    WriteAllocationSnapshot(args, "after_warmup");
    if (!args.barrier_each_run) SignalReadyAndWait(args);
    ProcessPmuCounters process_pmu(args.profile_pmu);

    std::ofstream result(args.output_jsonl, std::ios::out);
    if (!result) {
      throw std::runtime_error("Unable to write " + args.output_jsonl);
    }

    const double benchmark_start_ms = NowMillis();
    int completed = 0;
    std::vector<std::string> deferred_results;
    if (args.defer_result_write) deferred_results.reserve(static_cast<size_t>(args.runs));
    struct DeferredWriteGuard {
      std::ofstream* result;
      std::vector<std::string>* lines;
      bool pending;
      ~DeferredWriteGuard() {
        if (!pending) return;
        // Preserve already serialized rows if a later inference fails. This
        // still retains only strings, and does not change the failure status.
        try {
          for (const auto& line : *lines) *result << line << "\n";
          result->flush();
        } catch (...) {
        }
      }
    } deferred_write_guard{&result, &deferred_results, args.defer_result_write};

    auto write_completed = [&](const Frame& frame, const std::string& mode) {
      if (args.serial && !args.thread_affinity_report.empty() && !serial_affinity_report_written) {
        WriteThreadAffinityReport(args, serial_affinities);
        serial_affinity_report_written = true;
      }
      if (args.defer_result_write) {
        // Serialize now: retaining frames here would also retain all snapshots.
        deferred_results.push_back(ResultJSON(args, frame, completed, mode, benchmark_start_ms));
      } else {
        result << ResultJSON(args, frame, completed, mode, benchmark_start_ms) << "\n";
        result.flush();
      }
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
      if (args.component_thread_transition == "fixed") {
        ConfigureThreadPool(args.stages[0].runtime_num_threads,
                            args.stages[0].cpu_affinity,
                            "serial.component.fixed");
      }
      vta_deploy::SessionProfile session_profile(args.session_profile_json);
      session_profile.Start();
      std::vector<std::string> block_profile_paths;
      std::vector<std::pair<int, int>> block_profile_ranges;
      std::unique_ptr<vta_deploy::SessionProfile> block_profile;
      for (int i = 0; i < args.runs; ++i) {
        if (!args.block_profile_json.empty() && i % args.block_profile_frames == 0) {
          std::ostringstream path;
          path << args.block_profile_json << ".block" << std::setw(4) << std::setfill('0')
               << block_profile_paths.size() << ".json";
          block_profile_paths.push_back(path.str());
          block_profile_ranges.emplace_back(i, std::min(args.runs, i + args.block_profile_frames));
          block_profile = std::make_unique<vta_deploy::SessionProfile>(path.str());
          block_profile->Start();
        }
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
        if (block_profile &&
            ((i + 1) % args.block_profile_frames == 0 || i + 1 == args.runs)) {
          block_profile->Stop();
          block_profile.reset();
        }
      }
      session_profile.Stop();
      if (!args.block_profile_json.empty()) {
        std::ostringstream manifest;
        manifest << "{\"schema\":\"vta_block_profile_v1\",\"block_frames\":"
                 << args.block_profile_frames << ",\"blocks\":[";
        for (size_t i = 0; i < block_profile_paths.size(); ++i) {
          if (i != 0) manifest << ",";
          manifest << "{\"index\":" << i << ",\"frame_start\":"
                   << block_profile_ranges[i].first << ",\"frame_end_exclusive\":"
                   << block_profile_ranges[i].second << ",\"profile_path\":\""
                   << JsonEscape(block_profile_paths[i]) << "\"}";
        }
        manifest << "]}";
        WriteFile(args.block_profile_json, manifest.str());
      }
    } else {
      std::vector<std::unique_ptr<BoundedQueue<std::shared_ptr<Frame>>>> queues;
      queues.reserve(stages.size() + 1);
      for (size_t i = 0; i <= stages.size(); ++i) {
        std::function<void(const char*)> observer;
        if (lifecycle) observer = [&, i](const char* event) {
          lifecycle->Emit(event, static_cast<int>(i));
        };
        queues.emplace_back(
            std::make_unique<BoundedQueue<std::shared_ptr<Frame>>>(args.queue_depth, observer,
                                                                args.queue_timing));
      }
      std::mutex error_mutex;
      vta_deploy::CpuStageGate cpu_stage_gate;
      std::vector<std::vector<unsigned int>> resource_masks;
      if (args.cpu_stage_gate == "resource_group") {
        for (const auto& stage : args.stages) {
          if (stage.device == "cpu" &&
              (stage.cpu_affinity.empty() || stage.runtime_num_threads !=
                  static_cast<int>(stage.cpu_affinity.size()))) {
            throw std::invalid_argument("resource_group requires explicit CPU masks and matching threads");
          }
          resource_masks.push_back(stage.device == "cpu" ? stage.cpu_affinity
                                                          : std::vector<unsigned int>{});
        }
      }
      vta_deploy::CpuResourceGroups cpu_resource_groups(resource_masks);
      std::exception_ptr first_error = nullptr;

      auto record_error = [&](std::exception_ptr err) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = err;
          cpu_stage_gate.Cancel();
          cpu_resource_groups.Cancel();
          if (lifecycle) lifecycle->Cancel();
          for (auto& manager : p8_slot_managers) {
            manager->Abort("pipeline worker failure");
          }
          for (auto& queue : queues) {
            queue->Close();
          }
        }
      };

      if (lifecycle) lifecycle->Start([&] {
        record_error(std::make_exception_ptr(std::runtime_error("managed lifecycle command aborted pipeline")));
      });

      std::vector<std::thread> stage_threads;
      stage_threads.reserve(stages.size());
      std::vector<ThreadAffinityRecord> stage_affinities(stages.size());
      std::mutex initialization_mutex;
      std::condition_variable initialization_ready;
      std::mutex session_profile_mutex;
      std::condition_variable session_profile_stopped;
      bool session_profile_may_exit = args.session_profile_json.empty();
      std::vector<bool> initialization_finished(stages.size(), false);
      auto publish_initialization = [&](size_t stage_index) {
        if (args.thread_affinity_report.empty() && args.session_profile_json.empty()) return;
        {
          std::lock_guard<std::mutex> lock(initialization_mutex);
          initialization_finished[stage_index] = true;
        }
        initialization_ready.notify_one();
      };
      for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
        stage_threads.emplace_back([&, stage_index]() {
          try {
            std::vector<long> threads_before;
            if (!args.thread_affinity_report.empty()) threads_before = ProcessThreadIds();
            ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                                args.stages[stage_index].cpu_affinity,
                                "pipeline.stage" + std::to_string(stage_index));
            ConfigureProcessAffinity(args.process_cpu_affinity);
            if (!args.thread_affinity_report.empty()) {
              const auto threads_after = ProcessThreadIds();
              auto& owned = stage_affinities[stage_index].thread_ids;
              const PackedFunc* count = Registry::Get("runtime.NumThreads");
              stage_affinities[stage_index].active_runtime_threads = count ? (*count)().operator int() : 0;
              std::set_difference(threads_after.begin(), threads_after.end(),
                                  threads_before.begin(), threads_before.end(),
                                  std::back_inserter(owned));
#if defined(__linux__)
              stage_affinities[stage_index].caller_tid = static_cast<long>(syscall(SYS_gettid));
              owned.push_back(stage_affinities[stage_index].caller_tid);
#endif
              std::sort(owned.begin(), owned.end());
            }
            publish_initialization(stage_index);
            std::shared_ptr<Frame> frame;
            while (queues[stage_index]->Pop(&frame)) {
              frame->stage_timings[stage_index].queue_pop_ms = NowMillis();
              frame->stage_timings[stage_index].queue_pop_timing_valid = true;
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
                  C1MemoryAudit(args, frame->frame_id, stage_index, true);
                  frame->stage_outputs[stage_index] = RunStage(
                      &stages[stage_index], stage_inputs, &timing, false, nullptr, args.c1_memory_audit);
                  C1MemoryAudit(args, frame->frame_id, stage_index, false);
                  timing.vta_run_call_start_ms = timing.run_start_ms;
                  timing.vta_run_call_end_ms = timing.get_start_ms;
                  timing.vta_mutex_release_ms = NowMillis();
                  lock.unlock();
                } else {
                  auto& timing = frame->stage_timings[stage_index];
                  std::unique_ptr<vta_deploy::CpuStageGate::Lease> cpu_gate_lease;
                  if (args.cpu_stage_gate != "off") {
                    timing.cpu_gate_timing_valid = true;
                    timing.cpu_gate_request_ms = NowMillis();
                    if (args.cpu_stage_gate == "exclusive")
                      cpu_gate_lease = cpu_stage_gate.Acquire();
                    if (args.cpu_stage_gate == "resource_group")
                      cpu_gate_lease = cpu_resource_groups.Acquire(stage_index);
                    timing.cpu_gate_acquire_ms = NowMillis();
                  }
                  frame->stage_outputs[stage_index] = RunStage(
                      &stages[stage_index], stage_inputs,
                      &timing, false, nullptr, args.c1_memory_audit);
                  if (args.cpu_stage_gate != "off") {
                    timing.cpu_gate_release_ms = NowMillis();
                    cpu_gate_lease.reset();
                  }
                }
              }
              if (stage_index + 1 == stages.size()) {
                frame->done_ms = NowMillis();
                frame->top1 = ResultTop1(args, frame->stage_outputs[stage_index][0]);
              }
              if (!queues[stage_index + 1]->Push(frame, frame->QueueTimingFor(stage_index + 1))) {
                break;
              }
            }
            queues[stage_index + 1]->Close();
            if (!args.session_profile_json.empty()) {
              std::unique_lock<std::mutex> lock(session_profile_mutex);
              session_profile_stopped.wait(lock, [&] { return session_profile_may_exit; });
            }
          } catch (...) {
            record_error(std::current_exception());
            publish_initialization(stage_index);
          }
        });
        if (!args.thread_affinity_report.empty() || !args.session_profile_json.empty()) {
          // Do not create the next actor until this actor has registered its
          // workers. Otherwise the /proc delta can include another stage's TID.
          std::unique_lock<std::mutex> lock(initialization_mutex);
          initialization_ready.wait(lock, [&] { return initialization_finished[stage_index]; });
        }
      }

      if (!args.thread_affinity_report.empty()) {
        try {
          bool initialization_failed;
          {
            std::lock_guard<std::mutex> lock(error_mutex);
            initialization_failed = first_error != nullptr;
          }
          if (!initialization_failed) {
            // All actors remain alive, waiting on empty queues. Capture final
            // masks so later configurations cannot hide an earlier overwrite.
            for (auto& record : stage_affinities) CaptureThreadAffinities(&record);
            WriteThreadAffinityReport(args, stage_affinities);
          }
        } catch (...) {
          record_error(std::current_exception());
        }
      }

      vta_deploy::SessionProfile session_profile(args.session_profile_json);
      if (args.session_profile_first_completion < 0) session_profile.Start();
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
            if (args.queue_timing) {
              frame->queue_timings.reserve(stages.size() + 1);
              for (size_t edge_index = 0; edge_index <= stages.size(); ++edge_index) {
                frame->queue_timings.push_back(std::make_shared<QueueTiming>());
              }
            }
            if (args.p8_managed_slots > 0) prepare_managed_frame(frame.get());
            if (!queues[0]->Push(frame, frame->QueueTimingFor(0))) {
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
      int profile_completion_index = 0;
      while (queues.back()->Pop(&frame)) {
        if (profile_completion_index == args.session_profile_first_completion) session_profile.Start();
        if (args.session_profile_first_completion >= 0 &&
            profile_completion_index == args.session_profile_first_completion +
                                            args.session_profile_completion_count - 1) {
          session_profile.Stop();
        }
        write_completed(*frame, "pipeline");
        ++profile_completion_index;
      }

      producer_thread.join();
      session_profile.Stop();
      {
        std::lock_guard<std::mutex> lock(session_profile_mutex);
        session_profile_may_exit = true;
      }
      session_profile_stopped.notify_all();
      for (std::thread& stage_thread : stage_threads) {
        stage_thread.join();
      }
      if (lifecycle) lifecycle->Stop();

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

    if (args.defer_result_write) {
      deferred_write_guard.pending = false;
      for (const auto& line : deferred_results) result << line << "\n";
      result.flush();
      if (!result) throw std::runtime_error("failed to write deferred result JSONL");
    }

    WriteAllocationSnapshot(args, "after_steady_state");

    DumpProfileSnapshot(args.profile_dir, "benchmark_totals", args.runs - 1, args.events_limit,
                        profiler_status, profiler_events);
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "vta_stage_pipeline_runner error: " << err.what() << "\n";
    return 1;
  }
}
#endif  // VTA_STAGE_PIPELINE_NO_MAIN
