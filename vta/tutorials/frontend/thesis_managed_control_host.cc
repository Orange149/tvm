// Same production manager/queue core, host allocation adapter, real FIFO/CV gates.
#include <deque>
#define main ThesisPreviousManagerFixtureMain
#include "thesis_c2_host_lifecycle.cc"
#undef main
#include "vta_managed_lifecycle_control.h"
#include "queue_under_test.inc"

void Command(const std::string& fifo, const std::string& text) {
  int descriptor = open(fifo.c_str(), O_WRONLY | O_NONBLOCK);
  Require(descriptor >= 0, "cannot open command FIFO");
  const std::string bytes = text + "\n";
  const auto count = write(descriptor, bytes.data(), bytes.size()); close(descriptor);
  Require(count == static_cast<ssize_t>(bytes.size()), "cannot write complete command");
}
void Wait(std::future<void>& ready) {
  Require(ready.wait_for(std::chrono::seconds(3)) == std::future_status::ready,
          "production wait event or abort callback not observed");
  ready.get();
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string folder(argv[1]);
  int total = 0, failed = 0;
  auto Run = [&](const std::string& name, auto operation) {
    current_case = name; ++total; Event("control_case_start");
    try { operation(); Event("control_case_pass"); }
    catch (const std::exception& error) { ++failed; Event("control_case_fail", error.what()); }
  };
  auto Options = [&](bool fifo = false) {
    managed_lifecycle::Options options;
    options.trace_path = folder + "/" + current_case + ".jsonl";
    if (fifo) options.command_fifo = folder + "/" + current_case + ".fifo";
    options.timeout_seconds = 1;
    options.gates = {{"hold", "consumer_acquired", 1, 0}};
    return options;
  };
  for (int repetition = 0; repetition < 10; ++repetition) {
    const auto suffix = "_" + std::to_string(repetition);
    Run("direct_release" + suffix, [&] {
      managed_lifecycle::Control control(Options());
      auto worker = std::async(std::launch::async, [&] { control.Pause("consumer_acquired", 1, 0); });
      Require(control.WaitEntered("hold", 3), "gate not entered");
      Require(worker.wait_for(std::chrono::seconds(0)) == std::future_status::timeout, "gate did not hold");
      control.Release("hold"); worker.get(); control.Pause("before_run", 1, 1);
    });
    Run("direct_cancel" + suffix, [&] {
      auto options = Options(); options.gates.push_back({"other", "producer_acquired", 0, 0});
      managed_lifecycle::Control control(options);
      auto a = std::async(std::launch::async, [&] { Reject([&] { control.Pause("consumer_acquired", 1, 0); }, "cancelled"); });
      auto b = std::async(std::launch::async, [&] { Reject([&] { control.Pause("producer_acquired", 0, 0); }, "cancelled"); });
      Require(control.WaitEntered("hold", 3) && control.WaitEntered("other", 3), "two gates not entered");
      control.Cancel(); a.get(); b.get();
    });
    for (const std::string mode : {"release", "abort", "bad"})
      Run("fifo_" + mode + suffix, [&] {
        auto options = Options(true); managed_lifecycle::Control control(options);
        std::promise<void> aborted; auto abort_event = aborted.get_future();
        std::atomic<int> aborts{0};
        control.Start([&] { ++aborts; aborted.set_value(); });
        auto worker = std::async(std::launch::async, [&] {
          if (mode == "release") control.Pause("consumer_acquired", 1, 0);
          else Reject([&] { control.Pause("consumer_acquired", 1, 0); }, "cancelled");
        });
        Require(control.WaitEntered("hold", 3), "FIFO gate not entered");
        Command(options.command_fifo, mode == "release" ? "release hold" : mode == "abort" ? "abort" : "invalid");
        if (mode != "release") Wait(abort_event);
        worker.get(); control.Stop();
        Require(aborts == (mode == "release" ? 0 : 1), "unexpected abort callback count");
      });
    Run("premature_release" + suffix, [&] {
      managed_lifecycle::Control control(Options()); Reject([&] { control.Release("hold"); }, "premature");
    });
    Run("duplicate_release" + suffix, [&] {
      managed_lifecycle::Control control(Options());
      auto worker = std::async(std::launch::async, [&] { control.Pause("consumer_acquired", 1, 0); });
      Require(control.WaitEntered("hold", 3), "gate not entered");
      control.Release("hold"); worker.get(); Reject([&] { control.Release("hold"); }, "duplicate");
      Reject([&] { control.Pause("consumer_acquired", 1, 0); }, "twice");
    });
    Run("unknown_release" + suffix, [&] {
      managed_lifecycle::Control control(Options()); Reject([&] { control.Release("unknown"); }, "unknown");
    });
    Run("timeout" + suffix, [&] {
      managed_lifecycle::Control control(Options());
      Reject([&] { control.Pause("consumer_acquired", 1, 0); }, "timed out");
      Reject([&] { control.Pause("before_run", 1, 1); }, "cancelled");
    });
    Run("trace_exclusive" + suffix, [&] {
      auto options = Options(); managed_lifecycle::Control control(options);
      control.Emit("preserved_before");
      Reject([&] { managed_lifecycle::Control duplicate(options); }, "exclusively");
      control.Emit("preserved_after");
    });
    Run("duplicate_selector" + suffix, [&] {
      auto options = Options(); options.gates.push_back({"other", "consumer_acquired", 1, 0});
      Reject([&] { managed_lifecycle::Control duplicate(options); }, "duplicate");
    });
    Run("trace_escape" + suffix, [&] {
      managed_lifecycle::Control control(Options()); control.Emit("escape", -1, -1, -1, -1, 0, "quote\"slash\\newline\n");
    });
    Run("queue_producer_close" + suffix, [&] {
      std::promise<void> waiting; auto wait_event = waiting.get_future();
      BoundedQueue<int> queue(1, [&](const char* event) {
        if (std::string(event) == "queue_wait_space") waiting.set_value();
      });
      Require(queue.Push(1), "first push failed");
      auto worker = std::async(std::launch::async, [&] { return queue.Push(2); });
      Wait(wait_event); Require(worker.wait_for(std::chrono::seconds(0)) == std::future_status::timeout, "queue producer not held");
      queue.Close(); Require(!worker.get(), "closed queue admitted producer");
      int value; Require(queue.Pop(&value) && value == 1 && !queue.Pop(&value), "closed queue drain changed");
    });
    Run("queue_consumer_close" + suffix, [&] {
      std::promise<void> waiting; auto wait_event = waiting.get_future();
      BoundedQueue<int> queue(1, [&](const char* event) {
        if (std::string(event) == "queue_wait_input") waiting.set_value();
      });
      auto worker = std::async(std::launch::async, [&] { int value; return queue.Pop(&value); });
      Wait(wait_event); queue.Close(); Require(!worker.get(), "empty closed queue supplied input");
    });
    Run("fifo_queue_abort" + suffix, [&] {
      auto options = Options(true); managed_lifecycle::Control control(options);
      std::promise<void> waiting; auto wait_event = waiting.get_future();
      BoundedQueue<int> queue(1, [&](const char* event) {
        control.Emit(event, 1);
        if (std::string(event) == "queue_wait_input") waiting.set_value();
      });
      std::promise<void> aborted; auto abort_event = aborted.get_future();
      control.Start([&] { queue.Close(); aborted.set_value(); });
      auto worker = std::async(std::launch::async, [&] { int value; return queue.Pop(&value); });
      Wait(wait_event); Command(options.command_fifo, "abort"); Wait(abort_event);
      Require(!worker.get(), "FIFO abort did not release actual queue consumer"); control.Stop();
    });
    Run("fifo_full_manager_abort" + suffix, [&] {
      auto options = Options(true); managed_lifecycle::Control control(options);
      std::promise<void> waiting; auto wait_event = waiting.get_future();
      std::vector<StageExecutor> stages{{2}};
      BoundarySlotManager manager(&stages, 0, 2, 1, {}, [&](const char* event, const P8SlotToken& token) {
        control.Emit(event, 0, token.frame_id, token.edge_index, token.slot_id, token.generation);
        if (std::string(event) == "producer_wait_free") waiting.set_value();
      });
      double wait; auto a = manager.AcquireForProducer(0, &wait);
      Fill(manager, a); manager.Publish(a); manager.AcquireForConsumer(a, &wait);
      auto b = manager.AcquireForProducer(1, &wait); Fill(manager, b); manager.Publish(b);
      std::promise<void> aborted; auto abort_event = aborted.get_future();
      control.Start([&] { manager.Abort("host FIFO abort"); aborted.set_value(); });
      auto worker = std::async(std::launch::async, [&] {
        double blocked; Reject([&] { manager.AcquireForProducer(2, &blocked); }, "aborted");
      });
      Wait(wait_event); Command(options.command_fifo, "abort"); Wait(abort_event); worker.get();
      Reject([&] { manager.VerifyAllFree(); }, "not released"); manager.Release(a);
      Reject([&] { manager.VerifyAllFree(); }, "not released"); control.Stop();
    });
  }
  current_case = "control_total";
  Event("control_totals", std::to_string(total) + " cases; " + std::to_string(failed) + " failures");
  return failed ? 1 : 0;
}
