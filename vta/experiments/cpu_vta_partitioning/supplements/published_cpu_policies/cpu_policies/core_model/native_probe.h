/* Runner-only diagnostic counters. The TVM runtime and kernels are unchanged. */
#ifndef TVM_VTA_CORE_MODEL_NATIVE_PROBE_H_
#define TVM_VTA_CORE_MODEL_NATIVE_PROBE_H_

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace vta_run_scope {

inline double NowMillis() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Counters {
  uint64_t runtime_ns{0};
  uint64_t runqueue_wait_ns{0};
  uint64_t timeslices{0};
};

// No stream/locale allocation on the sampling path. Reject truncated fields,
// signs, extra tokens and overflow rather than filling an unavailable value.
inline bool ParseCounters(const char* bytes, size_t length, Counters* output) {
  size_t cursor = 0;
  auto whitespace = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  uint64_t values[3]{};
  for (size_t index = 0; index < 3; ++index) {
    while (cursor < length && whitespace(bytes[cursor])) ++cursor;
    if (cursor == length || bytes[cursor] < '0' || bytes[cursor] > '9') return false;
    while (cursor < length && bytes[cursor] >= '0' && bytes[cursor] <= '9') {
      const unsigned digit = static_cast<unsigned>(bytes[cursor++] - '0');
      if (values[index] > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
      values[index] = values[index] * 10 + digit;
    }
    if (cursor < length && !whitespace(bytes[cursor])) return false;
  }
  while (cursor < length && whitespace(bytes[cursor])) ++cursor;
  if (cursor != length) return false;
  *output = Counters{values[0], values[1], values[2]};
  return true;
}

enum class Failure { None, Open, Read, Incomplete, Parse, Unsupported };
inline const char* FailureName(Failure value) {
  switch (value) {
    case Failure::None: return "";
    case Failure::Open: return "schedstat_open_failed";
    case Failure::Read: return "schedstat_read_failed";
    case Failure::Incomplete: return "schedstat_line_incomplete";
    case Failure::Parse: return "schedstat_parse_failed";
    case Failure::Unsupported: return "unsupported_platform";
  }
  return "unknown_failure";
}

struct Sample {
  bool available{false};
  Counters counters;
  double read_start_ms{0.0};
  double read_end_ms{0.0};
  Failure failure{Failure::None};
  int error_number{0};
};

class CachedCounter {
 public:
  CachedCounter() = default;
  explicit CachedCounter(long tid, std::string path = {}) { Open(tid, std::move(path)); }
  ~CachedCounter() { Close(); }
  CachedCounter(const CachedCounter&) = delete;
  CachedCounter& operator=(const CachedCounter&) = delete;
  CachedCounter(CachedCounter&& other) noexcept
      : fd_(other.fd_), path_(std::move(other.path_)), open_error_(other.open_error_) { other.fd_ = -1; }
  CachedCounter& operator=(CachedCounter&& other) noexcept {
    if (this != &other) {
      Close(); fd_ = other.fd_; path_ = std::move(other.path_); open_error_ = other.open_error_;
      other.fd_ = -1;
    }
    return *this;
  }

  void Open(long tid, std::string path = {}) {
    Close();
    path_ = path.empty() ? "/proc/self/task/" + std::to_string(tid) + "/schedstat" : std::move(path);
#if defined(__linux__)
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    open_error_ = fd_ < 0 ? errno : 0;
#endif
  }

  Sample Read() const {
    Sample sample;
    sample.read_start_ms = NowMillis();
#if defined(__linux__)
    if (fd_ < 0) {
      sample.failure = Failure::Open; sample.error_number = open_error_;
    } else {
      // Explicit offset zero is essential: the same descriptor is reused, but
      // the proc seq-file must generate a fresh snapshot at every boundary.
      char bytes[256];
      size_t used = 0;
      bool newline = false;
      while (used < sizeof(bytes) && !newline) {
        const ssize_t n = ::pread(fd_, bytes + used, sizeof(bytes) - used,
                                  static_cast<off_t>(used));
        if (n < 0) {
          if (errno == EINTR) continue;
          sample.failure = Failure::Read; sample.error_number = errno; break;
        }
        if (n == 0) break;
        const size_t previous = used;
        used += static_cast<size_t>(n);
        for (size_t i = previous; i < used; ++i) if (bytes[i] == '\n') newline = true;
      }
      if (sample.failure == Failure::None) {
        if (!newline) sample.failure = Failure::Incomplete;
        else if (!ParseCounters(bytes, used, &sample.counters)) sample.failure = Failure::Parse;
        else sample.available = true;
      }
    }
#else
    sample.failure = Failure::Unsupported;
#endif
    sample.read_end_ms = NowMillis();
    return sample;
  }

  Sample ReadFresh() const {
    // Qualification only. It is deliberately not used by performance probes.
    CachedCounter fresh(0, path_);
    return fresh.Read();
  }

 private:
  void Close() {
#if defined(__linux__)
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
  }
  int fd_{-1};
  std::string path_;
  int open_error_{0};
};

struct ThreadObservation {
  long tid{0};
  int logical_worker{-1};
  int core{-1};
  bool caller{false};
  Sample before, after;
  bool fresh_performed{false};
  Sample fresh;
};

struct Record {
  int frame_id{-1};
  size_t stage_index{0};
  std::string program;
  bool filled{false};
  double before_start_ms{0}, before_end_ms{0}, run_start_ms{0}, run_end_ms{0};
  double after_start_ms{0}, after_end_ms{0};
  std::vector<ThreadObservation> threads;
};

inline bool IsTarget(const std::string& program) {
  return program == "cpu_00_02" || program == "cpu_05_07";
}

inline std::vector<int> ParseFrames(const std::string& text) {
  std::vector<int> result;
  size_t begin = 0;
  if (text.empty()) throw std::invalid_argument("run counter frames are empty");
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    const std::string value = text.substr(begin, end == std::string::npos ? end : end - begin);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("run counter frame must be a nonnegative integer");
    errno = 0;
    char* tail = nullptr;
    const long number = std::strtol(value.c_str(), &tail, 10);
    if (errno || *tail || number > std::numeric_limits<int>::max())
      throw std::invalid_argument("run counter frame overflows");
    result.push_back(static_cast<int>(number));
    if (end == std::string::npos) break;
    begin = end + 1;
    if (begin == text.size()) throw std::invalid_argument("empty run counter frame");
  }
  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    throw std::invalid_argument("duplicate run counter frame");
  return result;
}

// This order is separate from the sorted affinity resource set.  Slot zero
// is the stage caller; the remaining entries are TVM's logical worker slots.
inline std::vector<unsigned int> ParseWorkerMap(const std::string& text) {
  if (text.empty()) throw std::invalid_argument("CPU worker map is empty");
  std::vector<unsigned int> result;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    const std::string token = text.substr(begin, end == std::string::npos ? end : end - begin);
    if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("CPU worker map needs unsigned core IDs");
    errno = 0;
    char* tail = nullptr;
    const unsigned long value = std::strtoul(token.c_str(), &tail, 10);
    if (errno || *tail || value > std::numeric_limits<unsigned int>::max())
      throw std::invalid_argument("CPU worker core overflows");
#if defined(__linux__)
    if (value >= CPU_SETSIZE) throw std::invalid_argument("CPU worker core exceeds affinity capacity");
#endif
    if (std::find(result.begin(), result.end(), static_cast<unsigned int>(value)) != result.end())
      throw std::invalid_argument("CPU worker map contains a duplicate core");
    result.push_back(static_cast<unsigned int>(value));
    if (end == std::string::npos) break;
    begin = end + 1;
    if (begin == text.size()) throw std::invalid_argument("CPU worker map has an empty core");
  }
  return result;
}

inline const std::vector<unsigned int>& WorkerCoreMap(
    const std::vector<unsigned int>& affinity, const std::vector<unsigned int>& ordered_map) {
  return ordered_map.empty() ? affinity : ordered_map;
}

inline void ValidateWorkerCoreMap(const std::vector<unsigned int>& affinity,
                                  const std::vector<unsigned int>& ordered_map,
                                  int threads, bool pinned, bool cpu_stage) {
  if (ordered_map.empty()) return;  // Preserve the original default exactly.
  if (!cpu_stage || !pinned)
    throw std::runtime_error("explicit CPU worker map requires a pinned CPU stage");
  if (threads <= 0 || ordered_map.size() != static_cast<size_t>(threads))
    throw std::runtime_error("CPU worker map count must equal configured threads");
  auto sorted = ordered_map;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() || sorted != affinity)
    throw std::runtime_error("CPU worker map must be a permutation of the affinity mask");
#if defined(__linux__)
  if (sorted.back() >= CPU_SETSIZE)
    throw std::runtime_error("CPU worker core exceeds affinity capacity");
#endif
}

class StageProbe {
 public:
  void Initialize(size_t stage_index, const std::string& program,
                  const std::vector<unsigned int>& worker_core_map, const std::vector<long>& tids,
                  long caller_tid, const std::vector<int>& frames, bool qualification) {
    if (!IsTarget(program)) return;
    if (initialized_) throw std::runtime_error("run probe initialized twice");
    if (tids.size() != worker_core_map.size() || tids.empty())
      throw std::runtime_error("run probe stage pool size differs");
    qualification_ = qualification;
    std::vector<ThreadObservation> workers;
    std::vector<int> slots;
    for (long tid : tids) {
#if defined(__linux__)
      cpu_set_t cpus; CPU_ZERO(&cpus);
      if (::sched_getaffinity(static_cast<pid_t>(tid), sizeof(cpus), &cpus) != 0 || CPU_COUNT(&cpus) != 1)
        throw std::runtime_error("run probe requires one pinned core per compute TID");
      int core = -1;
      for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &cpus)) core = c;
      const auto found = std::find(worker_core_map.begin(), worker_core_map.end(), static_cast<unsigned>(core));
      if (found == worker_core_map.end()) throw std::runtime_error("run probe compute TID escaped mask");
      const int logical = static_cast<int>(found - worker_core_map.begin());
      workers.push_back(ThreadObservation{});
      auto& worker = workers.back();
      worker.tid = tid; worker.logical_worker = logical; worker.core = core; worker.caller = tid == caller_tid;
      if (worker.caller != (logical == 0)) throw std::runtime_error("run probe caller is not worker zero");
      slots.push_back(logical);
#else
      throw std::runtime_error("run probes require Linux");
#endif
    }
    std::sort(slots.begin(), slots.end());
    for (size_t i = 0; i < slots.size(); ++i)
      if (slots[i] != static_cast<int>(i)) throw std::runtime_error("run probe logical map is not bijective");
    std::sort(workers.begin(), workers.end(), [](const ThreadObservation& a, const ThreadObservation& b) {
      return a.logical_worker < b.logical_worker;
    });
    counters_.reserve(workers.size());
    for (const auto& worker : workers) counters_.emplace_back(worker.tid);
    records_.resize(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      records_[i].frame_id = frames[i]; records_[i].stage_index = stage_index;
      records_[i].program = program; records_[i].threads = workers;
    }
    initialized_ = true;
  }

  Record* Find(int frame) {
    for (auto& record : records_) if (record.frame_id == frame) return &record;
    return nullptr;
  }

  void Before(Record* record) {
    if (record->filled) throw std::runtime_error("duplicate run probe frame");
    record->filled = true;
    record->before_start_ms = NowMillis();
    // Helpers first, caller last: do not charge the caller for reading all of
    // its helpers before the measured run. Samples are not simultaneous.
    for (size_t i = 1; i < counters_.size(); ++i) record->threads[i].before = counters_[i].Read();
    record->threads[0].before = counters_[0].Read();
    record->before_end_ms = NowMillis();
  }
  void RunStarted(Record* record) { record->run_start_ms = NowMillis(); }
  void RunFinished(Record* record) { record->run_end_ms = NowMillis(); }
  void After(Record* record) {
    record->after_start_ms = NowMillis();
    // Caller first for the symmetric reason. Helper task epilogues after the
    // synchronous launch may still be observed; keep the full read envelope.
    record->threads[0].after = counters_[0].Read();
    for (size_t i = 1; i < counters_.size(); ++i) record->threads[i].after = counters_[i].Read();
    if (qualification_) {
      for (size_t i = 0; i < counters_.size(); ++i) {
        record->threads[i].fresh_performed = true;
        record->threads[i].fresh = counters_[i].ReadFresh();
      }
    }
    record->after_end_ms = NowMillis();
  }
  const std::vector<Record>& records() const { return records_; }

 private:
  bool initialized_{false}, qualification_{false};
  std::vector<CachedCounter> counters_;
  std::vector<Record> records_;
};

inline void JsonString(std::ostream& out, const std::string& value) {
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c < 0x20) out << '?';
    else out << c;
  }
  out << '"';
}

inline void WriteSample(std::ostream& out, const Sample& sample, double origin) {
  out << "{\"available\":" << (sample.available ? "true" : "false")
      << ",\"read_start_ms\":" << sample.read_start_ms - origin
      << ",\"read_end_ms\":" << sample.read_end_ms - origin;
  const uint64_t values[] = {sample.counters.runtime_ns, sample.counters.runqueue_wait_ns, sample.counters.timeslices};
  const char* names[] = {"runtime_ns", "runqueue_wait_ns", "timeslices"};
  for (int i = 0; i < 3; ++i) {
    out << ",\"" << names[i] << "\":";
    if (sample.available) out << values[i]; else out << "null";
  }
  out << ",\"reason\":";
  if (sample.available) out << "null"; else JsonString(out, FailureName(sample.failure));
  out << ",\"errno\":" << sample.error_number << '}';
}

inline bool Monotonic(const ThreadObservation& thread) {
  return thread.after.counters.runtime_ns >= thread.before.counters.runtime_ns &&
         thread.after.counters.runqueue_wait_ns >= thread.before.counters.runqueue_wait_ns &&
         thread.after.counters.timeslices >= thread.before.counters.timeslices;
}

inline void WriteTrace(const std::string& path, double origin, const std::vector<int>& frames,
                       const std::vector<StageProbe>& probes) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write run thread counter trace");
  out << std::fixed << std::setprecision(6)
      << "{\"schema\":\"vta_run_scope_v1\",\"clock\":\"steady_clock_ms_relative_to_benchmark_origin\""
      << ",\"benchmark_origin_ms\":" << origin << ",\"requested_frames\":[";
  for (size_t i = 0; i < frames.size(); ++i) { if (i) out << ','; out << frames[i]; }
  out << "],\"records\":[";
  bool first = true;
  for (const auto& probe : probes) for (const auto& record : probe.records()) {
    if (!record.filled || record.run_end_ms < record.run_start_ms || record.run_start_ms == 0)
      throw std::runtime_error("incomplete selected run counter record");
    if (!first) out << ','; first = false;
    out << "{\"frame_id\":" << record.frame_id << ",\"stage_index\":" << record.stage_index << ",\"program\":";
    JsonString(out, record.program);
    const double points[] = {record.before_start_ms, record.before_end_ms, record.run_start_ms,
                             record.run_end_ms, record.after_start_ms, record.after_end_ms};
    const char* names[] = {"before_start_ms", "before_end_ms", "run_start_ms", "run_end_ms", "after_start_ms", "after_end_ms"};
    for (int i = 0; i < 6; ++i) out << ",\"" << names[i] << "\":" << points[i] - origin;
    bool all_available = true, growth = false;
    out << ",\"threads\":[";
    for (size_t i = 0; i < record.threads.size(); ++i) {
      const auto& thread = record.threads[i];
      if (i) out << ',';
      const bool available = thread.before.available && thread.after.available;
      const bool monotonic = available && Monotonic(thread);
      all_available &= monotonic;
      growth |= monotonic && thread.after.counters.runtime_ns > thread.before.counters.runtime_ns;
      out << "{\"tid\":" << thread.tid << ",\"logical_worker\":" << thread.logical_worker
          << ",\"core\":" << thread.core << ",\"role\":\"" << (thread.caller ? "caller" : "worker")
          << "\",\"before\":"; WriteSample(out, thread.before, origin);
      out << ",\"after\":"; WriteSample(out, thread.after, origin);
      const uint64_t before[] = {thread.before.counters.runtime_ns, thread.before.counters.runqueue_wait_ns, thread.before.counters.timeslices};
      const uint64_t after[] = {thread.after.counters.runtime_ns, thread.after.counters.runqueue_wait_ns, thread.after.counters.timeslices};
      const char* deltas[] = {"runtime_delta_ns", "runqueue_wait_delta_ns", "timeslices_delta"};
      for (int j = 0; j < 3; ++j) {
        out << ",\"" << deltas[j] << "\":";
        if (monotonic) out << after[j] - before[j]; else out << "null";
      }
      out << ",\"monotonic\":";
      if (available) out << (monotonic ? "true" : "false"); else out << "null";
      out << ",\"fresh_check\":{\"performed\":" << (thread.fresh_performed ? "true" : "false")
          << ",\"sample\":";
      if (thread.fresh_performed) WriteSample(out, thread.fresh, origin); else out << "null";
      out << ",\"cached_not_stale\":";
      if (thread.fresh_performed && thread.fresh.available && thread.after.available)
        out << (thread.fresh.counters.runtime_ns >= thread.after.counters.runtime_ns ? "true" : "false");
      else out << "null";
      out << "}}";
    }
    out << "],\"runtime_growth_observed\":";
    if (all_available) out << (growth ? "true" : "false"); else out << "null";
    out << '}';
  }
  out << "]}\n";
  out.flush();
  if (!out) throw std::runtime_error("run counter trace write failed");
}

}  // namespace vta_run_scope
#endif
