// CPU-only replay of the frozen runner's actual ResultJSON implementation.
// The legacy entry point is never invoked. No graph, stage, VTA device or
// parallel computation is initialized by this entry point.
#define main UnusedFrozenPipelineMain
#include "vta_stage_pipeline_runner_phase.cc"
#undef main

static uint64_t ControlClock(clockid_t clock) {
  timespec ts{};
  if (clock_gettime(clock, &ts)) throw std::runtime_error("clock_gettime failed");
  return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: output_control_profile input.txt ring_count");
    const int ring_count = std::stoi(argv[2]);
    if (ring_count != 1 && ring_count != 4) throw std::runtime_error("ring count must be 1 or 4");
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    for (int core = 0; core != 4; ++core) CPU_SET(core, &allowed);
    if (sched_setaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("affinity failed");
    std::ifstream manifest(argv[1]);
    int stage_count = 0, output_count = 0;
    manifest >> stage_count >> output_count;
    if (!manifest || stage_count < 1 || stage_count > 7 || (output_count != 1 && output_count != 8))
      throw std::runtime_error("invalid reference manifest");
    const auto base = std::filesystem::path(argv[1]).parent_path();
    std::vector<Frame> ring(static_cast<size_t>(ring_count));
    for (Frame& f : ring) {
      f.input_index = 0;
      f.input_file = "inputs/input_000.bin";
      f.stage_timings.resize(static_cast<size_t>(stage_count));
      f.stage_outputs.resize(static_cast<size_t>(stage_count));
    }
    size_t total_bytes = 0;
    for (int out = 0; out != output_count; ++out) {
      int code, bits, ndim;
      manifest >> code >> bits >> ndim;
      if (!manifest || (code != kDLFloat && code != kDLInt) || bits != 32 || ndim < 1 || ndim > 4)
        throw std::runtime_error("invalid reference tensor dtype/shape");
      std::vector<int64_t> shape(static_cast<size_t>(ndim));
      for (auto& dim : shape) {
        manifest >> dim;
        if (!manifest || dim < 1) throw std::runtime_error("invalid reference dimension");
      }
      std::string relative;
      manifest >> relative;
      if (!manifest || relative.find("..") != std::string::npos || relative[0] == '/')
        throw std::runtime_error("invalid reference path");
      const std::string bytes = ReadFile((base / relative).string(), true);
      for (Frame& f : ring) {
        NDArray value = NDArray::Empty(tvm::runtime::ShapeTuple(shape),
            DLDataType{static_cast<uint8_t>(code), 32, 1}, tvm::Device{kDLCPU, 0});
        if (bytes.size() != TensorNBytes(value.operator->())) throw std::runtime_error("reference bytes differ");
        std::memcpy(value->data, bytes.data(), bytes.size());
        f.stage_outputs.back().push_back(std::move(value));
      }
      total_bytes += bytes.size();
    }
    const auto initial_hashes = OutputHashes(ring.front().stage_outputs.back());
    const std::string summary = RawOutputsJSON(ring.front().stage_outputs.back());
    Args args;
    args.output_mode = "raw";
    args.defer_result_write = true;
    args.cpu_stage_gate = "off";
    std::vector<std::string> deferred;
    deferred.reserve(10);
    std::vector<std::tuple<uint64_t, uint64_t, size_t>> samples;
    for (int i = 0; i != 10; ++i) {
      Frame& f = ring[static_cast<size_t>(i % ring_count)];
      f.frame_id = i;
      // Metadata and buffers are ready before timing. Stage computation,
      // frame allocation/destruction, queue operations and disk I/O are absent.
      const auto wall_start = ControlClock(CLOCK_MONOTONIC);
      const auto cpu_start = ControlClock(CLOCK_THREAD_CPUTIME_ID);
      deferred.push_back(ResultJSON(args, f, i, "pipeline", 0.0));
      const auto cpu_end = ControlClock(CLOCK_THREAD_CPUTIME_ID);
      const auto wall_end = ControlClock(CLOCK_MONOTONIC);
      if (i >= 2) samples.emplace_back(wall_end-wall_start, cpu_end-cpu_start, deferred.back().size());
    }
    for (const Frame& f : ring)
      if (OutputHashes(f.stage_outputs.back()) != initial_hashes) throw std::runtime_error("reference mutated");
    std::cout << "{\"kind\":\"isolated_output_serialization\",\"DNN_inferences\":0"
              << ",\"FPGA_calls\":0,\"warmup\":2,\"scored\":8,\"ring_count\":" << ring_count
              << ",\"stage_count\":" << stage_count << ",\"output_bytes\":" << total_bytes
              << ",\"affinity\":\"" << CurrentThreadAffinity() << "\",\"raw_outputs\":" << summary
              << ",\"samples\":[";
    for (size_t i = 0; i != samples.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << "{\"wall_ns\":" << std::get<0>(samples[i]) << ",\"CPU_ns\":" << std::get<1>(samples[i])
                << ",\"JSON_bytes\":" << std::get<2>(samples[i]) << '}';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
