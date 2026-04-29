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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tvm::runtime::Module;
using tvm::runtime::NDArray;
using tvm::runtime::PackedFunc;
using tvm::runtime::Registry;

struct Args {
  std::string graph = "graph.json";
  std::string lib = "graphlib.so";
  std::string params = "params.params";
  std::string input = "input.bin";
  std::string input_list;
  std::string input_name = "data";
  std::string output_jsonl = "native_result.jsonl";
  std::string profile_dir;
  int runs = 1;
  int events_limit = 200;
  int checkpoint_every = 0;
  bool pipeline = false;
  int max_inflight = 2;
  int pipeline_window = 2;
};

double NowMillis() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(now).count();
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

void LoadParamsBlob(const PackedFunc& load_params, const std::string& params_data) {
  TVMByteArray params;
  params.data = params_data.data();
  params.size = params_data.size();
  load_params(params);
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

std::string MetaJSON(const std::string& snapshot, int run_index, int events_limit) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"snapshot\": \"" << JsonEscape(snapshot) << "\",\n";
  os << "  \"run_index\": " << run_index << ",\n";
  os << "  \"events_limit\": " << events_limit << "\n";
  os << "}";
  return os.str();
}

void Usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --graph PATH\n"
      << "  --lib PATH\n"
      << "  --params PATH\n"
      << "  --input PATH\n"
      << "  --input-list PATH\n"
      << "  --input-name NAME\n"
      << "  --runs N\n"
      << "  --output-jsonl PATH\n"
      << "  --pipeline\n"
      << "  --max-inflight N\n"
      << "  --pipeline-window N\n"
      << "  --vta-runtime-profile-dir DIR\n"
      << "  --vta-runtime-profile-events-limit N\n"
      << "  --vta-runtime-profile-checkpoint-every N\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string& opt) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + opt);
      }
      return argv[++i];
    };
    if (key == "--graph") {
      args.graph = need_value(key);
    } else if (key == "--lib") {
      args.lib = need_value(key);
    } else if (key == "--params") {
      args.params = need_value(key);
    } else if (key == "--input") {
      args.input = need_value(key);
    } else if (key == "--input-list") {
      args.input_list = need_value(key);
    } else if (key == "--input-name") {
      args.input_name = need_value(key);
    } else if (key == "--runs") {
      args.runs = std::stoi(need_value(key));
    } else if (key == "--output-jsonl") {
      args.output_jsonl = need_value(key);
    } else if (key == "--vta-runtime-profile-dir") {
      args.profile_dir = need_value(key);
    } else if (key == "--vta-runtime-profile-events-limit") {
      args.events_limit = std::stoi(need_value(key));
    } else if (key == "--vta-runtime-profile-checkpoint-every") {
      args.checkpoint_every = std::stoi(need_value(key));
    } else if (key == "--pipeline") {
      args.pipeline = true;
    } else if (key == "--max-inflight") {
      args.max_inflight = std::stoi(need_value(key));
    } else if (key == "--pipeline-window") {
      args.pipeline_window = std::stoi(need_value(key));
    } else if (key == "--help" || key == "-h") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option " + key);
    }
  }
  if (args.runs <= 0) {
    throw std::runtime_error("--runs must be positive");
  }
  if (args.events_limit < 0) {
    throw std::runtime_error("--vta-runtime-profile-events-limit must be non-negative");
  }
  if (args.checkpoint_every < 0) {
    throw std::runtime_error("--vta-runtime-profile-checkpoint-every must be non-negative");
  }
  if (args.max_inflight <= 0) {
    throw std::runtime_error("--max-inflight must be positive");
  }
  if (args.pipeline_window <= 0) {
    throw std::runtime_error("--pipeline-window must be positive");
  }
  if (args.pipeline_window > args.max_inflight) {
    throw std::runtime_error("--pipeline-window must be <= --max-inflight");
  }
  return args;
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

NDArray MakeCPUInputLike(DLTensor* input_tensor, const std::string& input_bytes) {
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

std::string TrimLine(const std::string& line) {
  const size_t begin = line.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const size_t end = line.find_last_not_of(" \t\r\n");
  return line.substr(begin, end - begin + 1);
}

struct InputRecord {
  int input_index{0};
  std::string input_file;
  NDArray data;
};

std::vector<InputRecord> LoadInputs(const Args& args, DLTensor* input_tensor) {
  std::vector<InputRecord> inputs;
  if (args.input_list.empty()) {
    inputs.push_back(InputRecord{0, args.input,
                                 MakeCPUInputLike(input_tensor, ReadFile(args.input, true))});
    return inputs;
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
    std::filesystem::path input_path(rel);
    std::filesystem::path resolved = input_path.is_absolute() ? input_path : base / input_path;
    inputs.push_back(InputRecord{input_index, rel,
                                 MakeCPUInputLike(input_tensor, ReadFile(resolved.string(), true))});
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

std::string ResultJSON(int run_idx, const InputRecord& input, double set_input_ms, double run_ms,
                       double get_output_ms, double total_ms, int top1) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(6);
  os << "{\"run\":" << run_idx << ",\"input_index\":" << input.input_index
     << ",\"input_file\":\"" << JsonEscape(input.input_file) << "\""
     << ",\"set_input_ms\":" << set_input_ms
     << ",\"run_ms\":" << run_ms << ",\"get_output_ms\":" << get_output_ms
     << ",\"total_ms\":" << total_ms << ",\"top1\":" << top1 << "}";
  return os.str();
}

std::string PipelineResultJSON(int run_idx, int64_t request_id, const InputRecord& input,
                               double submit_ms, double wait_ms, double total_ms, int top1,
                               const std::string& slot_status) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(6);
  os << "{\"run\":" << run_idx << ",\"request_id\":" << request_id
     << ",\"input_index\":" << input.input_index << ",\"input_file\":\""
     << JsonEscape(input.input_file) << "\""
     << ",\"submit_ms\":" << submit_ms << ",\"wait_ms\":" << wait_ms
     << ",\"total_ms\":" << total_ms << ",\"top1\":" << top1 << ",\"slot_status\":\""
     << JsonEscape(slot_status) << "\"}";
  return os.str();
}

int Top1FromOutputs(const tvm::runtime::Array<NDArray>& outputs) {
  if (outputs.size() == 0) {
    throw std::runtime_error("pipeline returned no outputs");
  }
  return Top1Float32(outputs[0]);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);

    const std::string graph_json = ReadFile(args.graph);
    const std::string params_data = ReadFile(args.params, true);
    Module lib = Module::LoadFromFile(args.lib);
    const PackedFunc* create = Registry::Get("tvm.graph_executor.create");
    if (create == nullptr) {
      throw std::runtime_error("Missing registry function tvm.graph_executor.create");
    }
    Module graph = (*create)(graph_json, lib, static_cast<int>(kDLExtDev), 0,
                             static_cast<int>(kDLCPU), 0);

    PackedFunc load_params = graph.GetFunction("load_params");
    PackedFunc get_input = graph.GetFunction("get_input");
    PackedFunc set_input = graph.GetFunction("set_input");
    PackedFunc run = graph.GetFunction("run");
    PackedFunc get_output = graph.GetFunction("get_output");
    PackedFunc pipeline_init = graph.GetFunction("pipeline_init", false);
    PackedFunc pipeline_submit = graph.GetFunction("pipeline_submit", false);
    PackedFunc pipeline_wait = graph.GetFunction("pipeline_wait", false);
    PackedFunc pipeline_poll = graph.GetFunction("pipeline_poll", false);
    PackedFunc pipeline_close = graph.GetFunction("pipeline_close", false);
    PackedFunc pipeline_stats = graph.GetFunction("pipeline_stats", false);

    LoadParamsBlob(load_params, params_data);
    DLTensor* input_tensor = get_input(args.input_name);
    if (input_tensor == nullptr) {
      throw std::runtime_error("Unable to resolve input tensor " + args.input_name);
    }
    std::vector<InputRecord> inputs = LoadInputs(args, input_tensor);

    const PackedFunc* profiler_clear = Registry::Get("vta.runtime.profiler_clear");
    const PackedFunc* profiler_status = Registry::Get("vta.runtime.profiler_status");
    const PackedFunc* profiler_events = Registry::Get("vta.runtime.profiler_events");
    if (!args.profile_dir.empty()) {
      EnsureDir(args.profile_dir);
      if (profiler_clear != nullptr) {
        (*profiler_clear)();
      }
    }

    std::ofstream result(args.output_jsonl, std::ios::out);
    if (!result) {
      throw std::runtime_error("Unable to write " + args.output_jsonl);
    }

    if (!args.pipeline) {
      for (int i = 0; i < args.runs; ++i) {
        const InputRecord& input = inputs[static_cast<size_t>(i) % inputs.size()];
        const double total_start = NowMillis();
        const double set_start = NowMillis();
        set_input(args.input_name, input.data);
        const double set_end = NowMillis();

        run();
        const double run_end = NowMillis();

        NDArray output = get_output(0).operator NDArray();
        NDArray output_cpu = output.CopyTo(tvm::Device{kDLCPU, 0});
        const int top1 = Top1Float32(output_cpu);
        const double output_end = NowMillis();

        result << ResultJSON(i, input, set_end - set_start, run_end - set_end,
                             output_end - run_end, output_end - total_start, top1)
               << "\n";
        result.flush();

        if (!args.profile_dir.empty() && i == 0) {
          DumpProfileSnapshot(args.profile_dir, "single_run", i, args.events_limit, profiler_status,
                              profiler_events);
        }
        if (!args.profile_dir.empty() && args.checkpoint_every > 0 &&
            (i + 1) % args.checkpoint_every == 0) {
          std::ostringstream name;
          name << "checkpoints/run_" << std::setw(4) << std::setfill('0') << (i + 1);
          const std::filesystem::path full = std::filesystem::path(args.profile_dir) / name.str();
          EnsureDir(full.parent_path().string());
          DumpProfileSnapshot(args.profile_dir, name.str(), i, args.events_limit, profiler_status,
                              profiler_events);
        }
      }
    } else {
      if (pipeline_init == nullptr || pipeline_submit == nullptr || pipeline_wait == nullptr) {
        throw std::runtime_error("GraphExecutor pipeline functions are unavailable");
      }
      struct PendingRequest {
        int run_idx;
        size_t input_slot;
        int64_t request_id;
        double total_start_ms;
        double submit_ms;
      };
      std::deque<PendingRequest> pending;
      int completed = 0;
      pipeline_init(args.max_inflight);

      auto wait_one = [&]() {
        PendingRequest req = pending.front();
        pending.pop_front();
        const InputRecord& input = inputs[req.input_slot];
        const double wait_start = NowMillis();
        std::string slot_status = "unknown";
        if (pipeline_poll != nullptr) {
          slot_status = pipeline_poll(req.request_id).operator std::string();
        }
        tvm::runtime::Array<NDArray> outputs =
            pipeline_wait(req.request_id, static_cast<int>(kDLCPU), 0).operator tvm::runtime::Array<NDArray>();
        const double wait_end = NowMillis();
        const int top1 = Top1FromOutputs(outputs);
        result << PipelineResultJSON(req.run_idx, req.request_id, input, req.submit_ms,
                                     wait_end - wait_start, wait_end - req.total_start_ms, top1,
                                     slot_status)
               << "\n";
        result.flush();
        ++completed;
        if (!args.profile_dir.empty() && completed == 1) {
          DumpProfileSnapshot(args.profile_dir, "single_run", req.run_idx, args.events_limit,
                              profiler_status, profiler_events);
        }
        if (!args.profile_dir.empty() && args.checkpoint_every > 0 &&
            completed % args.checkpoint_every == 0) {
          std::ostringstream name;
          name << "checkpoints/run_" << std::setw(4) << std::setfill('0') << completed;
          const std::filesystem::path full = std::filesystem::path(args.profile_dir) / name.str();
          EnsureDir(full.parent_path().string());
          DumpProfileSnapshot(args.profile_dir, name.str(), req.run_idx, args.events_limit,
                              profiler_status, profiler_events);
        }
      };

      for (int i = 0; i < args.runs; ++i) {
        while (static_cast<int>(pending.size()) >= args.pipeline_window) {
          wait_one();
        }
        const double total_start = NowMillis();
        const double submit_start = NowMillis();
        const size_t input_slot = static_cast<size_t>(i) % inputs.size();
        int64_t request_id =
            pipeline_submit(args.input_name, inputs[input_slot].data).operator int64_t();
        const double submit_end = NowMillis();
        pending.push_back(PendingRequest{i, input_slot, request_id, total_start,
                                         submit_end - submit_start});
      }
      while (!pending.empty()) {
        wait_one();
      }
      if (pipeline_stats != nullptr) {
        WriteFile("pipeline_stats.json", pipeline_stats().operator std::string());
      }
      if (pipeline_close != nullptr) {
        pipeline_close();
      }
    }

    DumpProfileSnapshot(args.profile_dir, "benchmark_totals", args.runs - 1, args.events_limit,
                        profiler_status, profiler_events);
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "vta_native_runner error: " << err.what() << "\n";
    return 1;
  }
}
