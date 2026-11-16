// Device serialization never holds the CPU completion/initialization mutex.
#ifndef CONTROLLED_CONTENTION_TURN_GATE_H_
#define CONTROLLED_CONTENTION_TURN_GATE_H_
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>
#include <utility>

namespace controlled_contention {
class TurnGate {
 public:
  explicit TurnGate(std::vector<size_t> stages): stages_(std::move(stages)) {}
  template<typename Function>
  bool Execute(size_t stage, const std::atomic<bool>& stop, Function run) {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock,[&]{return stop.load()||(!stages_.empty()&&stages_[turn_]==stage);});
    if (stop.load()) return false;
    run();
    turn_=(turn_+1)%stages_.size();
    lock.unlock(); ready_.notify_all();
    return true;
  }
  void NotifyStop() {
    // Synchronize with predicate evaluation to avoid a lost cancellation wakeup.
    { std::lock_guard<std::mutex> lock(mutex_); }
    ready_.notify_all();
  }
 private:
  std::vector<size_t> stages_;
  std::mutex mutex_;
  std::condition_variable ready_;
  size_t turn_{0};
};
}
#endif
