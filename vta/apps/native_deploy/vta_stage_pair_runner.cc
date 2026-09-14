// Controlled G0 component experiment, not a production pipeline controller.
// Reuse the frozen stage loader, input routing and invocation code in one TU.
#define VTA_STAGE_PIPELINE_NO_MAIN
#include "vta_stage_pipeline_runner.cc"

#include <array>
#include <cmath>

namespace {
struct PairState {
  std::mutex mutex;
  std::condition_variable changed;
  bool abort{false};
  std::exception_ptr error;
  int round{-1};
  std::array<bool, 2> ready{}, released{}, started{}, done{}, completed{};
  std::array<double, 2> binding{}, release{}, start{}, end{}, set_ms{};
  std::array<std::string, 2> affinity;
};

template <typename Predicate>
void PairWait(PairState* state, std::unique_lock<std::mutex>* lock, Predicate predicate) {
  if (!state->changed.wait_for(*lock, std::chrono::seconds(30),
                               [&]() { return state->abort || predicate(); })) {
    throw std::runtime_error("G0 pair synchronization timeout");
  }
  if (state->abort) throw std::runtime_error("G0 pair aborted");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::array<int, 2> selected{-1, -1};
    double offset_ms = -1;
    std::vector<char*> base_args{argv[0]};
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--pair-active" || key == "--pair-ready" || key == "--pair-offset-ms") {
        if (++i == argc) throw std::runtime_error("missing pair argument");
        if (key == "--pair-offset-ms") offset_ms = std::stod(argv[i]);
        else selected[key == "--pair-active" ? 0 : 1] = std::stoi(argv[i]);
      } else {
        base_args.push_back(argv[i]);
      }
    }
    Args args = ParseArgs(static_cast<int>(base_args.size()), base_args.data());
    if (!std::isfinite(offset_ms) || offset_ms < 0 || offset_ms > 1000 ||
        selected[0] < 0 || selected[1] < 0 || selected[0] == selected[1] ||
        static_cast<size_t>(std::max(selected[0], selected[1])) >= args.stages.size()) {
      throw std::runtime_error("invalid pair selection or offset");
    }
    if (!args.serial || args.p8_managed_slots || args.p8a_edge >= 0 ||
        args.profile_pmu || !args.profile_dir.empty() || !args.ready_file.empty() ||
        !args.process_cpu_affinity.empty() || args.runs > 1000 || args.warmup_runs > 100) {
      throw std::runtime_error("pair requires serial input routing, no managed slots/profiler/barrier");
    }
    const auto& a = args.stages[selected[0]];
    const auto& b = args.stages[selected[1]];
    if (!((a.device == "cpu" && b.device == "vta") ||
          (a.device == "vta" && b.device == "cpu"))) {
      throw std::runtime_error("pair requires exactly one CPU and one VTA");
    }
    ResolveStageThreadDefaults(&args);
    ValidateStageAffinities(args);
    int max_threads = args.runtime_num_threads;
    for (const auto& stage : args.stages) max_threads = std::max(max_threads, stage.runtime_num_threads);
    setenv("TVM_NUM_THREADS", std::to_string(max_threads).c_str(), 1);
    ConfigureThreadPool(args.runtime_num_threads, {}, "pair.prepare");
    std::vector<StageExecutor> stages;
    for (const auto& stage : args.stages) stages.push_back(LoadStage(stage.name, stage.device, stage));
    auto inputs = LoadInputs(args, &stages[0]);
    if (inputs.size() != 1) throw std::runtime_error("pair requires one frozen input");
    Frame reference;
    reference.stage_inputs = inputs[0].data;
    reference.stage_outputs.resize(stages.size());
    std::vector<std::vector<NDArray>> frozen_inputs(stages.size());
    std::vector<std::string> expected(stages.size());
    for (size_t i = 0; i < stages.size(); ++i) {
      ConfigureThreadPool(args.stages[i].runtime_num_threads, args.stages[i].cpu_affinity,
                          "pair.reference" + std::to_string(i));
      frozen_inputs[i] = ResolveStageInputs(args, reference, i);
      StageTiming timing;
      reference.stage_outputs[i] = RunStage(&stages[i], frozen_inputs[i], &timing, false);
      expected[i] = RawOutputsJSON(reference.stage_outputs[i]);
    }
    std::cout << "[REFERENCE] " << expected.back() << "\n";
    PairState state;
    const int warmup_count = 2 * args.warmup_runs;
    const int count = warmup_count + 4 * args.runs;  // runs = number of ABBA blocks
    std::vector<std::thread> workers;
    auto abort = [&](std::exception_ptr error) {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (!state.error) state.error = error;
      state.abort = true;
      state.changed.notify_all();
    };
    auto worker = [&](int side) {
      try {
        const int stage = selected[side];
        ConfigureThreadPool(args.stages[stage].runtime_num_threads, args.stages[stage].cpu_affinity,
                            "pair.worker" + std::to_string(side));
        if (args.stages[stage].device == "vta") {
          // Freeze the single VTA host thread; do not repin the other CPU workers.
          cpu_set_t mask;
          CPU_ZERO(&mask);
          CPU_SET(3, &mask);
          if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
            throw std::runtime_error("cannot pin pair VTA host to CPU3");
          }
        }
        { std::lock_guard<std::mutex> lock(state.mutex);
          state.affinity[side] = CurrentThreadAffinity(); }
        for (int sample = 0; sample < count; ++sample) {
          { std::unique_lock<std::mutex> lock(state.mutex);
            PairWait(&state, &lock, [&]() { return state.round == sample; }); }
          StageTiming timing;
          StageRunHooks hooks;
          hooks.before_run = [&]() {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.binding[side] = timing.set_end_ms;
            state.set_ms[side] = timing.SetMs();
            state.ready[side] = true;
            state.changed.notify_all();
            PairWait(&state, &lock, [&]() { return state.released[side]; });
          };
          hooks.started = [&](double time) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.start[side] = time;
            state.started[side] = true;
            state.changed.notify_all();
          };
          hooks.finished = [&](double time) {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.end[side] = time;
            state.done[side] = true;
            state.changed.notify_all();
            // Exclude get_output copies from the opposite graph-run measurement.
            PairWait(&state, &lock, [&]() { return state.done[0] && state.done[1]; });
          };
          const auto output = RunStage(&stages[stage], frozen_inputs[stage], &timing, false, &hooks);
          if (RawOutputsJSON(output) != expected[stage]) {
            throw std::runtime_error("pair output differs from isolated reference");
          }
          { std::lock_guard<std::mutex> lock(state.mutex);
            state.completed[side] = true;
            state.changed.notify_all(); }
        }
      } catch (...) { abort(std::current_exception()); }
    };
    std::ostringstream records;
    records << std::fixed << std::setprecision(6);
    try {
      workers.emplace_back(worker, 0);
      workers.emplace_back(worker, 1);
      for (int sample = 0; sample < count; ++sample) {
        const bool warmup = sample < warmup_count;
        const int pos = warmup ? sample % 2 : (sample - warmup_count) % 4;
        const bool wait = warmup ? pos == 1 : pos == 1 || pos == 2;
        std::unique_lock<std::mutex> lock(state.mutex);
        state.ready = {}; state.released = {}; state.started = {}; state.done = {};
        state.completed = {}; state.round = sample;
        state.changed.notify_all();
        PairWait(&state, &lock, [&]() { return state.ready[0] && state.ready[1]; });
        state.release[0] = NowMillis(); state.released[0] = true;
        state.changed.notify_all();
        PairWait(&state, &lock, [&]() { return state.started[0]; });
        const auto deadline = std::chrono::steady_clock::time_point(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double, std::milli>(state.start[0] + offset_ms)));
        lock.unlock(); std::this_thread::sleep_until(deadline); lock.lock();
        const double eligible = NowMillis();
        const bool reachable = !state.done[0];
        if (wait) PairWait(&state, &lock, [&]() { return state.done[0]; });
        state.release[1] = NowMillis(); state.released[1] = true;
        state.changed.notify_all();
        PairWait(&state, &lock, [&]() { return state.completed[0] && state.completed[1]; });
        records << "{\"sample\":" << sample << ",\"warmup\":" << (warmup ? "true" : "false")
                << ",\"block\":" << (warmup ? -1 : (sample - warmup_count) / 4)
                << ",\"policy\":\"" << (wait ? "wait" : "allow") << "\""
                << ",\"active_stage\":" << selected[0] << ",\"ready_stage\":" << selected[1]
                << ",\"offset_ms\":" << offset_ms << ",\"eligible_ms\":" << eligible
                << ",\"active_at_eligible\":" << (reachable ? "true" : "false")
                << ",\"makespan_ms\":" << std::max(state.end[0], state.end[1]) - state.start[0]
                << ",\"outputs_match_reference\":true,\"sides\":[";
        for (int side = 0; side < 2; ++side) {
          if (side) records << ",";
          records << "{\"binding_ms\":" << state.binding[side]
                  << ",\"set_ms\":" << state.set_ms[side]
                  << ",\"release_ms\":" << state.release[side]
                  << ",\"start_ms\":" << state.start[side]
                  << ",\"end_ms\":" << state.end[side]
                  << ",\"affinity\":\"" << state.affinity[side] << "\"}";
        }
        records << "]}\n";
      }
    } catch (...) { abort(std::current_exception()); }
    for (auto& thread : workers) thread.join();
    // Persist completed samples even on failure, without file I/O in measured cells.
    WriteFile(args.output_jsonl, records.str());
    if (state.error) std::rethrow_exception(state.error);
    std::cout << "[PAIR_OK] " << count << " samples; all stage outputs match reference\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vta_stage_pair_runner error: " << error.what() << "\n";
    return 1;
  }
}
