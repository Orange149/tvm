// Host-only probes of the verbatim production BoundarySlotManager included below.
// Allocation/view adapters do not model VTA DMA, GraphExecutor, or bundle validation.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using NDArray = std::shared_ptr<std::vector<uint8_t>>;
struct StageExecutor { size_t tensors{1}; };
enum DLDeviceType { kDLCPU = 1, kDLExtDev = 12 };
double NowMillis() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::mutex event_mutex, witness_mutex;
std::condition_variable witness_cv;
std::string current_case, witnessed_role;
uint64_t sequence = 0;
bool witnessed_wait = false;
void Event(const std::string& kind, const std::string& detail = "") {
  std::lock_guard<std::mutex> lock(event_mutex);
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  // Fixed internal messages contain no JSON metacharacters.
  std::cout << "{\"seq\":" << ++sequence << ",\"host_monotonic_ns\":" << ns
            << ",\"case\":\"" << current_case << "\",\"event\":\"" << kind
            << "\",\"detail\":\"" << detail << "\"}\n" << std::flush;
}
void HostWaitWitness(const char* role, bool available) {
  if (available) return;
  std::lock_guard<std::mutex> lock(witness_mutex);
  if (witnessed_role == role) {
    witnessed_wait = true;
    Event("wait_predicate_false", role);
    witness_cv.notify_all();
  }
}
void Require(bool value, const char* why) {
  if (!value) throw std::runtime_error(why);
}
void Arm(const char* role) {
  std::lock_guard<std::mutex> lock(witness_mutex);
  witnessed_wait = false;
  witnessed_role = role;
}
void WaitWitness() {
  std::unique_lock<std::mutex> lock(witness_mutex);
  Require(witness_cv.wait_for(lock, std::chrono::seconds(3), [] { return witnessed_wait; }),
          "wait predicate was never observed false");
}

// Generated at execution time from the frozen runner bytes; only two predicate
// observers are inserted, and removal must recover the exact production class.
struct P8SlotToken;
void HostProductionObserver(const char*, const P8SlotToken&);
#include "manager_under_test.inc"

void HostProductionObserver(const char* point, const P8SlotToken& token) {
  Event(std::string("production_") + point,
        std::to_string(token.edge_index) + "," + std::to_string(token.slot_id) + "," +
        std::to_string(token.generation) + "," + std::to_string(token.frame_id));
  if (std::string(point) == "producer_wait_free") HostWaitWitness("producer", false);
  if (std::string(point) == "consumer_wait_ready") HostWaitWitness("consumer", false);
}

std::vector<P8SharedEdge> AllocateP8HeterogeneousEdgeBundle(
    std::vector<StageExecutor>* stages, size_t edge_index) {
  std::vector<P8SharedEdge> result;
  for (size_t i = 0; i < stages->at(edge_index).tensors; ++i) {
    P8SharedEdge edge;
    edge.nbytes = i == 0 ? 64 : 32;
    edge.direction = "host-adapter";
    edge.producer_owner = std::make_shared<std::vector<uint8_t>>(edge.nbytes, 0xA5);
    edge.consumer_owner = edge.producer_owner;
    result.push_back(std::move(edge));
  }
  return result;
}

template<class F> void Reject(F operation, const std::string& expected = "") {
  bool rejected = false;
  try { operation(); }
  catch (const std::exception& error) {
    rejected = true;
    Event("rejected", error.what());
    Require(expected.empty() || std::string(error.what()).find(expected) != std::string::npos,
            "unexpected rejection category");
  }
  Require(rejected, "operation unexpectedly succeeded");
}
void Fill(BoundarySlotManager& manager, const P8SlotToken& token) {
  for (size_t i = 0; i < manager.tensor_count(); ++i) {
    auto& bytes = *manager.ProducerView(token, i);
    std::fill(bytes.begin(), bytes.end(), uint8_t(token.frame_id + i));
  }
  Event("producer_filled", std::to_string(token.frame_id));
}
void Consume(BoundarySlotManager& manager, const P8SlotToken& token) {
  double wait;
  manager.AcquireForConsumer(token, &wait);
  for (size_t i = 0; i < manager.tensor_count(); ++i) {
    const auto& bytes = *manager.ConsumerView(token, i);
    Require(std::all_of(bytes.begin(), bytes.end(), [&](uint8_t v) {
      return v == uint8_t(token.frame_id + i);
    }), "host payload overwritten or incomplete");
  }
  Event("consumer_checked", std::to_string(token.frame_id));
  manager.Release(token);
}

int main(int argc, char** argv) {
  const bool hardened = argc == 2 && std::string(argv[1]) == "hardened";
  int failed = 0, total = 0;
  auto Run = [&](std::string name, auto operation) {
    current_case = std::move(name);
    ++total;
    Event("case_start");
    try { operation(); Event("case_pass"); }
    catch (const std::exception& error) { ++failed; Event("case_fail", error.what()); }
  };
  for (int repetition = 0; repetition < 10; ++repetition) {
    const std::string suffix = "_" + std::to_string(repetition);
    for (size_t arity : {size_t(1), size_t(2)}) {
      Run("normal_bundle" + std::to_string(arity) + suffix, [&] {
        std::vector<StageExecutor> stages{{arity}};
        BoundarySlotManager manager(&stages, 0, 2, 1);
        uint64_t generations[2]{};
        for (int frame = 0; frame < 64; ++frame) {
          double wait;
          auto token = manager.AcquireForProducer(frame, &wait);
          Require(token.generation == ++generations[token.slot_id], "generation not monotonic");
          Fill(manager, token); manager.Publish(token); Consume(manager, token);
        }
        manager.VerifyAllFree();
      });
      Run("full_slots" + std::to_string(arity) + suffix, [&] {
        std::vector<StageExecutor> stages{{arity}};
        BoundarySlotManager manager(&stages, 0, 2, 1);
        double wait;
        auto first = manager.AcquireForProducer(1, &wait);
        Fill(manager, first); manager.Publish(first);
        manager.AcquireForConsumer(first, &wait); // Slow consumer owns slot0.
        auto second = manager.AcquireForProducer(2, &wait);
        Fill(manager, second); manager.Publish(second); // slot1 is READY.
        Arm("producer");
        auto future = std::async(std::launch::async, [&] {
          double blocked;
          return manager.AcquireForProducer(3, &blocked);
        });
        WaitWitness(); // Observer executes while the production mutex is held.
        Require(future.wait_for(std::chrono::seconds(0)) == std::future_status::timeout,
                "third producer did not block with two busy slots");
        for (size_t i = 0; i < arity; ++i)
          for (auto value : *manager.ConsumerView(first, i))
            Require(value == uint8_t(1 + i), "held consumer payload overwritten");
        Event("release_latch"); manager.Release(first);
        auto third = future.get();
        Require(third.slot_id == first.slot_id && third.generation == first.generation + 1,
                "unexpected slot reuse");
        Fill(manager, third); manager.Publish(third);
        Consume(manager, second); Consume(manager, third); manager.VerifyAllFree();
      });
    }
    for (const std::string fault : {"stale_generation", "wrong_frame", "wrong_edge", "wrong_slot"})
      Run(fault + suffix, [&] {
        std::vector<StageExecutor> stages{{1}};
        BoundarySlotManager manager(&stages, 0, 2, 1);
        double wait;
        auto token = manager.AcquireForProducer(10, &wait); Fill(manager, token);
        manager.Publish(token);
        auto bad = token;
        if (fault == "stale_generation") --bad.generation;
        if (fault == "wrong_frame") ++bad.frame_id;
        if (fault == "wrong_edge") ++bad.edge_index;
        if (fault == "wrong_slot") bad.slot_id = 99;
        const std::string category = (fault == "stale_generation" || fault == "wrong_frame")
            ? (hardened ? "stale" : "timed out") : "";
        Reject([&] { manager.AcquireForConsumer(bad, &wait); }, category);
        Consume(manager, token); manager.VerifyAllFree();
      });
    for (const std::string fault : {"duplicate_publish", "early_release", "wrong_release"})
      Run(fault + suffix, [&] {
        std::vector<StageExecutor> stages{{1}};
        BoundarySlotManager manager(&stages, 0, 2, 1);
        double wait;
        auto token = manager.AcquireForProducer(20, &wait); Fill(manager, token);
        if (fault == "early_release") Reject([&] { manager.Release(token); }, "transition");
        manager.Publish(token);
        if (fault == "duplicate_publish") Reject([&] { manager.Publish(token); }, "transition");
        if (fault == "wrong_release") {
          auto bad = token; ++bad.generation;
          Reject([&] { manager.Release(bad); }, "stale");
        }
        Consume(manager, token); manager.VerifyAllFree();
      });
    Run("abort_publish" + suffix, [&] {
      std::vector<StageExecutor> stages{{1}};
      BoundarySlotManager manager(&stages, 0, 2, 1);
      double wait; auto token = manager.AcquireForProducer(30, &wait);
      manager.Abort("host fault");
      Reject([&] { manager.Publish(token); }, "aborted");
      Reject([&] { manager.VerifyAllFree(); }, "not released");
    });
    Run("abort_new_acquire" + suffix, [&] {
      std::vector<StageExecutor> stages{{1}};
      BoundarySlotManager manager(&stages, 0, 2, 1);
      manager.Abort("host fault"); double wait;
      Reject([&] { manager.AcquireForProducer(1, &wait); }, "aborted");
    });
    Run("abort_held_release" + suffix, [&] {
      std::vector<StageExecutor> stages{{1}};
      BoundarySlotManager manager(&stages, 0, 2, 1);
      double wait; auto token = manager.AcquireForProducer(40, &wait);
      Fill(manager, token); manager.Publish(token); manager.AcquireForConsumer(token, &wait);
      manager.Abort("host fault");
      Reject([&] { manager.VerifyAllFree(); }, "not released");
      manager.Release(token); manager.VerifyAllFree();
    });
    for (const std::string role : {"producer", "consumer"})
      Run("abort_wait_" + role + suffix, [&] {
        std::vector<StageExecutor> stages{{1}};
        BoundarySlotManager manager(&stages, 0, 2, 1);
        double wait; auto token = manager.AcquireForProducer(50, &wait);
        if (role == "producer") manager.AcquireForProducer(51, &wait);
        Arm(role.c_str());
        auto future = std::async(std::launch::async, [&] {
          double blocked;
          Reject([&] {
            if (role == "producer") manager.AcquireForProducer(52, &blocked);
            else manager.AcquireForConsumer(token, &blocked);
          }, "aborted");
        });
        WaitWitness(); Event("abort_latch"); manager.Abort("host fault"); future.get();
        Reject([&] { manager.VerifyAllFree(); }, "not released");
      });
  }
  current_case = "total";
  Event("totals", std::to_string(total) + " cases; " + std::to_string(failed) + " failures");
  return failed ? 1 : 0;
}
