#include "vta_cpu_stage_gate.h"
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
  vta_deploy::CpuStageGate gate;
  std::atomic<int> inside{0};
  std::atomic<bool> violated{false};
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&] {
      for (int j = 0; j < 2000; ++j) {
        auto lease = gate.Acquire();
        if (inside.fetch_add(1) != 0) violated = true;
        std::this_thread::yield();
        if (inside.fetch_sub(1) != 1) violated = true;
      }
    });
  }
  for (auto& worker : workers) worker.join();
  if (violated || inside != 0) return 1;
  try {
    auto lease = gate.Acquire();
    throw std::runtime_error("injected leased execution failure");
  } catch (const std::runtime_error&) {}
  auto owner = gate.Acquire();
  std::promise<void> entered;
  auto pending = std::async(std::launch::async, [&] {
    entered.set_value();
    try {
      auto lease = gate.Acquire();
      return false;
    } catch (const std::runtime_error&) { return true; }
  });
  entered.get_future().wait();
  if (pending.wait_for(std::chrono::milliseconds(10)) != std::future_status::timeout) return 2;
  gate.Cancel();
  if (pending.wait_for(std::chrono::seconds(1)) != std::future_status::ready || !pending.get()) return 3;
  owner.reset();
  try {
    auto forbidden = gate.Acquire();
    return 4;
  } catch (const std::runtime_error&) {}
  return 0;
}
