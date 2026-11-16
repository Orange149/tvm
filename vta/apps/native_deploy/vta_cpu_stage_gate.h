/* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See NOTICE for additional information.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef TVM_VTA_CPU_STAGE_GATE_H_
#define TVM_VTA_CPU_STAGE_GATE_H_

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace vta_deploy {

// FIFO admission prevents a fast upstream actor from starving the tail actor.
// Cancellation wakes all pending admissions on pipeline failure. A live lease
// releases admission even if graph execution throws.
class CpuStageGate {
 public:
  class Lease {
   public:
    explicit Lease(CpuStageGate* gate) : gate_(gate) {}
    ~Lease() { gate_->Release(); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
   private:
    CpuStageGate* gate_;
  };

  std::unique_ptr<Lease> Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    const uint64_t ticket = next_ticket_++;
    ready_.wait(lock, [&] { return cancelled_ || ticket == serving_ticket_; });
    if (cancelled_) throw std::runtime_error("CPU stage gate cancelled");
    return std::unique_ptr<Lease>(new Lease(this));
  }

  void Cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
    }
    ready_.notify_all();
  }

 private:
  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++serving_ticket_;
    }
    ready_.notify_all();
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  uint64_t next_ticket_{0};
  uint64_t serving_ticket_{0};
  bool cancelled_{false};
};

}  // namespace vta_deploy
#endif  // TVM_VTA_CPU_STAGE_GATE_H_
