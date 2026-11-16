// Functional-test gates and native event trace. Disabled unless explicitly configured.
#ifndef VTA_APPS_NATIVE_DEPLOY_VTA_MANAGED_LIFECYCLE_CONTROL_H_
#define VTA_APPS_NATIVE_DEPLOY_VTA_MANAGED_LIFECYCLE_CONTROL_H_
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

namespace managed_lifecycle {
struct Gate { std::string id, point; int stage, frame; };
struct Options {
  std::string trace_path, command_fifo;
  int timeout_seconds{60};
  std::vector<Gate> gates;
};

inline std::string Escape(const std::string& text) {
  std::string result;
  const char* hex = "0123456789abcdef";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
    else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
    else result += char(c);
  }
  return result;
}

class Control {
 public:
  explicit Control(Options options) : options_(std::move(options)),
      entered_(options_.gates.size(), false), released_(options_.gates.size(), false) {
    if (options_.timeout_seconds <= 0) throw std::invalid_argument("positive lifecycle timeout required");
    for (size_t i = 0; i < options_.gates.size(); ++i) {
      if (options_.gates[i].id.empty()) throw std::invalid_argument("empty gate id");
      for (size_t j = 0; j < i; ++j) {
        const auto& a = options_.gates[i]; const auto& b = options_.gates[j];
        if (a.id == b.id || (a.point == b.point && a.stage == b.stage && a.frame == b.frame))
          throw std::invalid_argument("duplicate gate id or selector");
      }
    }
    if (!options_.trace_path.empty()) {
      trace_fd_ = open(options_.trace_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
      if (trace_fd_ < 0) throw std::runtime_error("cannot exclusively create lifecycle trace");
    }
    if (!options_.command_fifo.empty()) {
      if (mkfifo(options_.command_fifo.c_str(), 0600) != 0) {
        if (trace_fd_ >= 0) close(trace_fd_);
        trace_fd_ = -1;
        throw std::runtime_error("cannot exclusively create lifecycle command FIFO");
      }
      fifo_fd_ = open(options_.command_fifo.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
      if (fifo_fd_ < 0) {
        unlink(options_.command_fifo.c_str());
        if (trace_fd_ >= 0) close(trace_fd_);
        trace_fd_ = -1;
        throw std::runtime_error("cannot open lifecycle command FIFO");
      }
    }
  }
  Control(const Control&) = delete;
  Control& operator=(const Control&) = delete;
  ~Control() {
    Stop();
    if (trace_fd_ >= 0) close(trace_fd_);
    if (fifo_fd_ >= 0) { close(fifo_fd_); unlink(options_.command_fifo.c_str()); }
  }

  void Emit(const std::string& point, int stage = -1, int frame = -1,
            int64_t edge = -1, int64_t slot = -1, uint64_t generation = 0,
            const std::string& detail = "") {
    if (trace_fd_ < 0) return;
    std::lock_guard<std::mutex> lock(trace_mutex_);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::ostringstream out;
    out << "{\"seq\":" << ++sequence_ << ",\"pid\":" << getpid()
        << ",\"monotonic_ns\":" << ns << ",\"point\":\"" << Escape(point)
        << "\",\"stage\":" << stage << ",\"frame\":" << frame
        << ",\"edge\":" << edge << ",\"slot\":" << slot
        << ",\"generation\":" << generation << ",\"detail\":\"" << Escape(detail) << "\"}\n";
    const auto text = out.str(); size_t offset = 0;
    while (offset < text.size()) {
      ssize_t n = write(trace_fd_, text.data() + offset, text.size() - offset);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("lifecycle trace write failed");
      offset += static_cast<size_t>(n);
    }
  }

  void Pause(const std::string& point, int stage, int frame) {
    Emit(point, stage, frame);
    std::unique_lock<std::mutex> lock(mutex_);
    ThrowIfCancelled();
    for (size_t i = 0; i < options_.gates.size(); ++i) {
      const auto& gate = options_.gates[i];
      if (gate.point != point || gate.stage != stage || gate.frame != frame) continue;
      if (entered_[i]) throw std::runtime_error("lifecycle selector entered twice");
      entered_[i] = true;
      Emit("latch_entered", stage, frame, -1, -1, 0, gate.id);
      condition_.notify_all();
      if (!condition_.wait_for(lock, std::chrono::seconds(options_.timeout_seconds),
                               [&] { return cancelled_ || released_[i]; })) {
        cancelled_ = true;
        condition_.notify_all();
        throw std::runtime_error("managed lifecycle latch timed out");
      }
      ThrowIfCancelled();
      Emit("latch_resumed", stage, frame, -1, -1, 0, gate.id);
      return;
    }
  }

  void Release(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    ThrowIfCancelled();
    for (size_t i = 0; i < options_.gates.size(); ++i) {
      if (options_.gates[i].id != id) continue;
      if (!entered_[i] || released_[i]) throw std::runtime_error("premature or duplicate latch release");
      released_[i] = true;
      Emit("latch_released", options_.gates[i].stage, options_.gates[i].frame, -1, -1, 0, id);
      condition_.notify_all(); return;
    }
    throw std::runtime_error("unknown lifecycle latch id");
  }

  bool WaitEntered(const std::string& id, int seconds) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (size_t i = 0; i < options_.gates.size(); ++i)
      if (options_.gates[i].id == id)
        return condition_.wait_for(lock, std::chrono::seconds(seconds), [&] { return entered_[i]; });
    throw std::runtime_error("unknown lifecycle latch id");
  }
  void Cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true; condition_.notify_all();
  }
  void Start(std::function<void()> abort_handler) {
    if (fifo_fd_ < 0 || watcher_.joinable()) throw std::runtime_error("invalid lifecycle watcher start");
    watcher_ = std::thread([this, abort_handler] {
      try {
        std::string pending;
        while (!stopping_) {
          pollfd descriptor{fifo_fd_, POLLIN, 0};
          int ready = poll(&descriptor, 1, 50);
          if (ready < 0 && errno == EINTR) continue;
          if (ready < 0 || descriptor.revents & (POLLERR | POLLNVAL))
            throw std::runtime_error("lifecycle FIFO poll failed");
          if (!(descriptor.revents & POLLIN)) continue;
          char buffer[256]; ssize_t count = read(fifo_fd_, buffer, sizeof(buffer));
          if (count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
          if (count <= 0) throw std::runtime_error("lifecycle FIFO read failed");
          pending.append(buffer, static_cast<size_t>(count));
          if (pending.size() > 4096) throw std::runtime_error("lifecycle command too long");
          size_t newline;
          while ((newline = pending.find('\n')) != std::string::npos) {
            const auto command = pending.substr(0, newline); pending.erase(0, newline + 1);
            Emit("command_received", -1, -1, -1, -1, 0, command);
            if (command == "abort") { Cancel(); abort_handler(); return; }
            if (command.rfind("release ", 0) != 0) throw std::runtime_error("invalid lifecycle command");
            Release(command.substr(8));
          }
        }
      } catch (...) { Cancel(); if (!stopping_) abort_handler(); }
    });
  }
  void Stop() {
    stopping_ = true;
    Cancel();
    if (watcher_.joinable()) watcher_.join();
  }

 private:
  void ThrowIfCancelled() const {
    if (cancelled_) throw std::runtime_error("managed lifecycle control cancelled");
  }
  Options options_;
  int trace_fd_{-1}, fifo_fd_{-1};
  std::mutex mutex_, trace_mutex_;
  std::condition_variable condition_;
  std::vector<bool> entered_, released_;
  bool cancelled_{false};
  std::atomic<bool> stopping_{false};
  std::thread watcher_;
  uint64_t sequence_{0};
};
}  // namespace managed_lifecycle
#endif
