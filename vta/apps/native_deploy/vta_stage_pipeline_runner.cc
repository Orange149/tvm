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
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using tvm::runtime::Module;
using tvm::runtime::NDArray;
using tvm::runtime::PackedFunc;
using tvm::runtime::Registry;

struct StageArgs {
  std::string name;
  std::string device;
  std::string graph;
  std::string lib;
  std::string params;
  std::vector<std::string> input_names{"data0"};
  std::vector<std::string> input_sources;
  int runtime_num_threads = -1;
};

struct Args {
  std::vector<StageArgs> stages;
  std::string input;
  std::string input_list;
  std::string output_jsonl = "native_result.jsonl";
  std::string output_mode = "classification";
  std::string output_dump_dir;
  std::string profile_dir;
  int runs = 1;
  int queue_depth = 2;
  int runtime_num_threads = 4;
  int events_limit = 200;
  int checkpoint_every = 0;
  bool serial = false;
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

std::string TrimLine(const std::string& line) {
  const size_t begin = line.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const size_t end = line.find_last_not_of(" \t\r\n");
  return line.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitCSV(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = TrimLine(item);
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  if (items.empty()) {
    throw std::runtime_error("empty comma-separated argument");
  }
  return items;
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

void Usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --stageN-graph PATH --stageN-lib PATH --stageN-params PATH\n"
      << "  --stageN-input-names data0[,data1]\n"
      << "  --stageN-input-sources input:0|stageM:K[,stageM:K]\n"
      << "  --stageN-device cpu|vta\n"
      << "  --stageN-name NAME\n"
      << "  --input PATH | --input-list PATH\n"
      << "  --runs N\n"
      << "  --queue-depth N\n"
      << "  --runtime-num-threads N\n"
      << "  --stageN-runtime-num-threads N\n"
      << "  --serial\n"
      << "  --output-mode classification|raw|raw_all_stages\n"
      << "  --output-jsonl PATH\n"
      << "  --output-dump-dir DIR\n"
      << "  --vta-runtime-profile-dir DIR\n"
      << "  --vta-runtime-profile-events-limit N\n"
      << "  --vta-runtime-profile-checkpoint-every N\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  auto ensure_stage = [&](size_t index) -> StageArgs& {
    if (args.stages.size() <= index) {
      args.stages.resize(index + 1);
    }
    return args.stages[index];
  };
  auto parse_stage_key = [](const std::string& key, size_t* index,
                            std::string* suffix) -> bool {
    const std::string prefix = "--stage";
    if (key.rfind(prefix, 0) != 0) {
      return false;
    }
    size_t pos = prefix.size();
    if (pos >= key.size() || !std::isdigit(static_cast<unsigned char>(key[pos]))) {
      return false;
    }
    size_t value = 0;
    while (pos < key.size() && std::isdigit(static_cast<unsigned char>(key[pos]))) {
      value = value * 10 + static_cast<size_t>(key[pos] - '0');
      ++pos;
    }
    if (pos >= key.size() || key[pos] != '-') {
      return false;
    }
    *index = value;
    *suffix = key.substr(pos + 1);
    return true;
  };
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const std::string& opt) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + opt);
      }
      return argv[++i];
    };

    size_t stage_index = 0;
    std::string stage_suffix;
    if (parse_stage_key(key, &stage_index, &stage_suffix)) {
      StageArgs& stage = ensure_stage(stage_index);
      if (stage_suffix == "graph") {
        stage.graph = need_value(key);
      } else if (stage_suffix == "lib") {
        stage.lib = need_value(key);
      } else if (stage_suffix == "params") {
        stage.params = need_value(key);
      } else if (stage_suffix == "input-names") {
        stage.input_names = SplitCSV(need_value(key));
      } else if (stage_suffix == "input-sources") {
        stage.input_sources = SplitCSV(need_value(key));
      } else if (stage_suffix == "device") {
        stage.device = need_value(key);
      } else if (stage_suffix == "name") {
        stage.name = need_value(key);
      } else if (stage_suffix == "runtime-num-threads") {
        stage.runtime_num_threads = std::stoi(need_value(key));
      } else {
        throw std::runtime_error("Unknown stage option " + key);
      }
    } else if (key == "--input") {
      args.input = need_value(key);
    } else if (key == "--input-list") {
      args.input_list = need_value(key);
    } else if (key == "--runs") {
      args.runs = std::stoi(need_value(key));
    } else if (key == "--queue-depth") {
      args.queue_depth = std::stoi(need_value(key));
    } else if (key == "--runtime-num-threads") {
      args.runtime_num_threads = std::stoi(need_value(key));
    } else if (key == "--output-jsonl") {
      args.output_jsonl = need_value(key);
    } else if (key == "--output-mode") {
      args.output_mode = need_value(key);
    } else if (key == "--output-dump-dir") {
      args.output_dump_dir = need_value(key);
    } else if (key == "--serial") {
      args.serial = true;
    } else if (key == "--vta-runtime-profile-dir") {
      args.profile_dir = need_value(key);
    } else if (key == "--vta-runtime-profile-events-limit") {
      args.events_limit = std::stoi(need_value(key));
    } else if (key == "--vta-runtime-profile-checkpoint-every") {
      args.checkpoint_every = std::stoi(need_value(key));
    } else if (key == "--help" || key == "-h") {
      Usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option " + key);
    }
  }

  if (args.stages.empty()) {
    throw std::runtime_error("at least one stage is required");
  }
  for (size_t i = 0; i < args.stages.size(); ++i) {
    StageArgs& stage = args.stages[i];
    if (stage.name.empty()) {
      stage.name = "stage" + std::to_string(i);
    }
    if (stage.device.empty()) {
      stage.device = (i == 1 ? "vta" : "cpu");
    }
    if (stage.device != "cpu" && stage.device != "vta") {
      throw std::runtime_error("--stage" + std::to_string(i) + "-device must be cpu or vta");
    }
    if (stage.graph.empty() || stage.lib.empty() || stage.params.empty()) {
      throw std::runtime_error("all stage graph/lib/params arguments are required");
    }
    if (!stage.input_sources.empty() && stage.input_sources.size() != stage.input_names.size()) {
      throw std::runtime_error("--stage" + std::to_string(i) +
                               "-input-sources must match --stage-input-names count");
    }
    if (stage.runtime_num_threads < -1) {
      throw std::runtime_error("--stage*-runtime-num-threads must be >= -1");
    }
  }
  if (args.stages.size() > 1) {
    for (size_t i = 0; i < args.stages.size(); ++i) {
      if (args.stages[i].graph.empty() || args.stages[i].lib.empty() ||
          args.stages[i].params.empty()) {
        throw std::runtime_error("stage indices must be contiguous from stage0");
      }
    }
  }
  if (args.stages.empty()) {
    throw std::runtime_error("all stage graph/lib/params arguments are required");
  }
  if (!args.input.empty() && !args.input_list.empty()) {
    throw std::runtime_error("--input and --input-list are mutually exclusive");
  }
  if (args.input.empty() && args.input_list.empty()) {
    throw std::runtime_error("--input or --input-list is required");
  }
  if (args.runs <= 0) {
    throw std::runtime_error("--runs must be positive");
  }
  if (args.queue_depth <= 0) {
    throw std::runtime_error("--queue-depth must be positive");
  }
  if (args.runtime_num_threads < 0) {
    throw std::runtime_error("--runtime-num-threads must be non-negative");
  }
  if (args.output_mode != "classification" && args.output_mode != "raw" &&
      args.output_mode != "raw_all_stages") {
    throw std::runtime_error("--output-mode must be classification, raw, or raw_all_stages");
  }
  if (args.events_limit < 0) {
    throw std::runtime_error("--vta-runtime-profile-events-limit must be non-negative");
  }
  if (args.checkpoint_every < 0) {
    throw std::runtime_error("--vta-runtime-profile-checkpoint-every must be non-negative");
  }
  return args;
}

void ResolveStageThreadDefaults(Args* args) {
  for (size_t i = 0; i < args->stages.size(); ++i) {
    StageArgs& stage = args->stages[i];
    if (stage.runtime_num_threads >= 0) {
      continue;
    }
    if (args->serial) {
      stage.runtime_num_threads = stage.device == "cpu" ? args->runtime_num_threads : 1;
    } else {
      stage.runtime_num_threads = (i == 0 && stage.device == "cpu") ? 3 : 1;
    }
  }
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

size_t TensorNElements(const DLTensor* tensor) {
  size_t elems = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    elems *= static_cast<size_t>(tensor->shape[i]);
  }
  return elems;
}

const uint8_t* TensorDataBytes(const DLTensor* tensor) {
  return static_cast<const uint8_t*>(tensor->data) + tensor->byte_offset;
}

std::string DTypeString(const DLDataType& dtype) {
  std::string code = "unknown";
  if (dtype.code == kDLFloat) {
    code = "float";
  } else if (dtype.code == kDLInt) {
    code = "int";
  } else if (dtype.code == kDLUInt) {
    code = "uint";
  } else if (dtype.code == kDLBfloat) {
    code = "bfloat";
  }
  std::ostringstream os;
  os << code << static_cast<int>(dtype.bits);
  if (dtype.lanes != 1) {
    os << "x" << static_cast<int>(dtype.lanes);
  }
  return os.str();
}

uint64_t FNV1a64(const uint8_t* data, size_t nbytes) {
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < nbytes; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
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

struct InputRecord {
  int input_index{0};
  std::string input_file;
  NDArray data;
};

struct StageExecutor {
  std::string name;
  std::string device;
  std::vector<std::string> input_names;
  Module graph;
  PackedFunc set_input;
  PackedFunc run;
  PackedFunc get_output;
  PackedFunc get_input;
  PackedFunc get_num_outputs;

  int NumOutputs() const { return get_num_outputs().operator int(); }
};

StageExecutor LoadStage(const std::string& name, const std::string& device,
                        const StageArgs& stage_args) {
  const std::string graph_json = ReadFile(stage_args.graph);
  const std::string params_data = ReadFile(stage_args.params, true);
  Module lib = Module::LoadFromFile(stage_args.lib);
  const PackedFunc* create = Registry::Get("tvm.graph_executor.create");
  if (create == nullptr) {
    throw std::runtime_error("Missing registry function tvm.graph_executor.create");
  }

  Module graph;
  if (device == "vta") {
    graph = (*create)(graph_json, lib, static_cast<int>(kDLExtDev), 0,
                      static_cast<int>(kDLCPU), 0);
  } else {
    graph = (*create)(graph_json, lib, static_cast<int>(kDLCPU), 0);
  }

  PackedFunc load_params = graph.GetFunction("load_params");
  LoadParamsBlob(load_params, params_data);

  StageExecutor stage;
  stage.name = name;
  stage.device = device;
  stage.input_names = stage_args.input_names;
  stage.graph = graph;
  stage.set_input = graph.GetFunction("set_input");
  stage.run = graph.GetFunction("run");
  stage.get_output = graph.GetFunction("get_output");
  stage.get_input = graph.GetFunction("get_input");
  stage.get_num_outputs = graph.GetFunction("get_num_outputs");

  std::cout << "[LOAD] " << name << " device=" << device
            << " inputs=" << stage.input_names.size()
            << " outputs=" << stage.NumOutputs() << "\n";
  return stage;
}

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

std::string ShapeJSON(const DLTensor* tensor) {
  std::ostringstream os;
  os << "[";
  for (int i = 0; i < tensor->ndim; ++i) {
    if (i) os << ",";
    os << tensor->shape[i];
  }
  os << "]";
  return os.str();
}

std::string UInt64Hex(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setfill('0') << std::setw(16) << value;
  return os.str();
}

std::string TensorSummaryJSON(const NDArray& array, int index) {
  const DLTensor* tensor = array.operator->();
  const size_t nbytes = TensorNBytes(tensor);
  const size_t count = TensorNElements(tensor);
  const uint8_t* bytes = TensorDataBytes(tensor);

  double sum = 0.0;
  double min_value = 0.0;
  double max_value = 0.0;
  bool numeric = false;

  auto update_stats = [&](double value, size_t idx) {
    if (idx == 0) {
      min_value = value;
      max_value = value;
    } else {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    sum += value;
  };

  if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32) {
    const float* data = reinterpret_cast<const float*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(static_cast<double>(data[i]), i);
    }
    numeric = true;
  } else if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLFloat &&
             tensor->dtype.bits == 64) {
    const double* data = reinterpret_cast<const double*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(data[i], i);
    }
    numeric = true;
  } else if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLInt &&
             tensor->dtype.bits == 32) {
    const int32_t* data = reinterpret_cast<const int32_t*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(static_cast<double>(data[i]), i);
    }
    numeric = true;
  } else if (tensor->dtype.lanes == 1 && tensor->dtype.code == kDLInt &&
             tensor->dtype.bits == 64) {
    const int64_t* data = reinterpret_cast<const int64_t*>(bytes);
    for (size_t i = 0; i < count; ++i) {
      update_stats(static_cast<double>(data[i]), i);
    }
    numeric = true;
  }

  std::ostringstream os;
  os << std::fixed << std::setprecision(9);
  os << "{\"index\":" << index
     << ",\"shape\":" << ShapeJSON(tensor)
     << ",\"dtype\":\"" << JsonEscape(DTypeString(tensor->dtype)) << "\""
     << ",\"size\":" << count
     << ",\"nbytes\":" << nbytes
     << ",\"fnv1a64\":\"" << UInt64Hex(FNV1a64(bytes, nbytes)) << "\"";
  if (numeric) {
    os << ",\"min\":" << min_value
       << ",\"max\":" << max_value
       << ",\"mean\":" << (count ? sum / static_cast<double>(count) : 0.0)
       << ",\"sum\":" << sum;
  }
  os << "}";
  return os.str();
}

std::string RawOutputsJSON(const std::vector<NDArray>& outputs) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i) os << ",";
    os << TensorSummaryJSON(outputs[i], static_cast<int>(i));
  }
  os << "]";
  return os.str();
}

std::string DumpOutputsJSON(const std::string& output_dump_dir,
                            const std::vector<NDArray>& outputs,
                            int completion_index,
                            const std::string& tag) {
  if (output_dump_dir.empty()) {
    return "[]";
  }
  const std::filesystem::path run_dir =
      std::filesystem::path(output_dump_dir) /
      ("run_" + std::to_string(completion_index));
  EnsureDir(run_dir.string());
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < outputs.size(); ++i) {
    const DLTensor* tensor = outputs[i].operator->();
    const size_t nbytes = TensorNBytes(tensor);
    const uint8_t* bytes = TensorDataBytes(tensor);
    const std::string file_name = tag + "_output_" + std::to_string(i) + ".bin";
    const std::filesystem::path file_path = run_dir / file_name;
    std::ofstream out(file_path, std::ios::out | std::ios::binary);
    if (!out) {
      throw std::runtime_error("failed to open output dump " + file_path.string());
    }
    out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(nbytes));
    if (!out) {
      throw std::runtime_error("failed to write output dump " + file_path.string());
    }
    if (i) os << ",";
    os << "{\"index\":" << i
       << ",\"path\":\"" << JsonEscape(file_path.string()) << "\""
       << ",\"shape\":" << ShapeJSON(tensor)
       << ",\"dtype\":\"" << JsonEscape(DTypeString(tensor->dtype)) << "\""
       << ",\"nbytes\":" << nbytes
       << "}";
  }
  os << "]";
  return os.str();
}

struct StageTiming {
  double set_start_ms{0.0};
  double run_start_ms{0.0};
  double get_start_ms{0.0};
  double end_ms{0.0};

  double SetMs() const { return run_start_ms - set_start_ms; }
  double RunMs() const { return get_start_ms - run_start_ms; }
  double GetMs() const { return end_ms - get_start_ms; }
  double ServiceMs() const { return end_ms - set_start_ms; }
};

struct Frame {
  int frame_id{0};
  int input_index{0};
  std::string input_file;
  double enqueue_ms{0.0};
  std::vector<NDArray> stage_inputs;
  std::vector<std::vector<NDArray>> stage_outputs;
  std::vector<StageTiming> stage_timings;
  double done_ms{0.0};
  int top1{-1};
};

std::pair<int, int> ParseInputSource(const std::string& source) {
  const size_t colon = source.find(':');
  if (colon == std::string::npos) {
    throw std::runtime_error("input source must be input:0 or stageN:K, got " + source);
  }
  const std::string owner = source.substr(0, colon);
  const int index = std::stoi(source.substr(colon + 1));
  if (index < 0) {
    throw std::runtime_error("negative input source index in " + source);
  }
  if (owner == "input") {
    return {-1, index};
  }
  const std::string prefix = "stage";
  if (owner.rfind(prefix, 0) != 0 || owner.size() == prefix.size()) {
    throw std::runtime_error("input source must be input:0 or stageN:K, got " + source);
  }
  return {std::stoi(owner.substr(prefix.size())), index};
}

const NDArray& ResolveInputSource(const Frame& frame, const std::string& source) {
  const auto parsed = ParseInputSource(source);
  const int stage_index = parsed.first;
  const int output_index = parsed.second;
  if (stage_index < 0) {
    if (static_cast<size_t>(output_index) >= frame.stage_inputs.size()) {
      throw std::runtime_error("input source out of range: " + source);
    }
    return frame.stage_inputs[static_cast<size_t>(output_index)];
  }
  if (static_cast<size_t>(stage_index) >= frame.stage_outputs.size()) {
    throw std::runtime_error("stage source out of range: " + source);
  }
  const std::vector<NDArray>& outputs = frame.stage_outputs[static_cast<size_t>(stage_index)];
  if (static_cast<size_t>(output_index) >= outputs.size()) {
    throw std::runtime_error("stage output source out of range: " + source);
  }
  return outputs[static_cast<size_t>(output_index)];
}

std::vector<NDArray> ResolveStageInputs(const Args& args, const Frame& frame, size_t stage_index) {
  const StageArgs& stage = args.stages[stage_index];
  if (stage.input_sources.empty()) {
    if (stage_index == 0) {
      return frame.stage_inputs;
    }
    return frame.stage_outputs[stage_index - 1];
  }
  std::vector<NDArray> inputs;
  inputs.reserve(stage.input_sources.size());
  for (const std::string& source : stage.input_sources) {
    inputs.push_back(ResolveInputSource(frame, source));
  }
  return inputs;
}

std::vector<NDArray> RunStage(StageExecutor* stage, const std::vector<NDArray>& inputs,
                              StageTiming* timing) {
  if (inputs.size() != stage->input_names.size()) {
    std::ostringstream os;
    os << stage->name << " expected " << stage->input_names.size() << " input(s), got "
       << inputs.size();
    throw std::runtime_error(os.str());
  }
  timing->set_start_ms = NowMillis();
  for (size_t i = 0; i < inputs.size(); ++i) {
    stage->set_input(stage->input_names[i], inputs[i]);
  }
  timing->run_start_ms = NowMillis();
  stage->run();
  timing->get_start_ms = NowMillis();

  std::vector<NDArray> outputs;
  const int num_outputs = stage->NumOutputs();
  outputs.reserve(static_cast<size_t>(num_outputs));
  for (int i = 0; i < num_outputs; ++i) {
    NDArray output = stage->get_output(i).operator NDArray();
    outputs.push_back(output.CopyTo(tvm::Device{kDLCPU, 0}));
  }
  timing->end_ms = NowMillis();
  return outputs;
}

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

  bool Push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&]() { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    queue_.push_back(std::move(value));
    not_empty_.notify_one();
    return true;
  }

  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    *value = std::move(queue_.front());
    queue_.pop_front();
    not_full_.notify_one();
    return true;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  size_t capacity_;
  std::deque<T> queue_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool closed_{false};
};

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

std::string MetaJSON(const std::string& snapshot, int run_index, int events_limit) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"snapshot\": \"" << JsonEscape(snapshot) << "\",\n";
  os << "  \"run_index\": " << run_index << ",\n";
  os << "  \"events_limit\": " << events_limit << "\n";
  os << "}";
  return os.str();
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

std::string ResultJSON(const Args& args, const Frame& frame, int completion_index,
                       const std::string& mode, double benchmark_start_ms) {
  auto rel = [&](double value) { return value - benchmark_start_ms; };
  std::ostringstream os;
  os << std::fixed << std::setprecision(6);
  os << "{\"mode\":\"" << JsonEscape(mode) << "\""
     << ",\"completion_index\":" << completion_index
     << ",\"frame_id\":" << frame.frame_id
     << ",\"input_index\":" << frame.input_index
     << ",\"input_file\":\"" << JsonEscape(frame.input_file) << "\""
     << ",\"output_mode\":\"" << JsonEscape(args.output_mode) << "\"";
  if (args.output_mode == "classification") {
    os << ",\"top1\":" << frame.top1;
  } else {
    os << ",\"raw_outputs\":" << RawOutputsJSON(frame.stage_outputs.back());
    if (!args.output_dump_dir.empty()) {
      os << ",\"raw_output_files\":"
         << DumpOutputsJSON(args.output_dump_dir, frame.stage_outputs.back(),
                            completion_index, "final");
    }
    if (args.output_mode == "raw_all_stages") {
      os << ",\"stage_raw_outputs\":[";
      for (size_t i = 0; i < frame.stage_outputs.size(); ++i) {
        if (i != 0) os << ",";
        os << "{\"stage_index\":" << i << ",\"outputs\":" << RawOutputsJSON(frame.stage_outputs[i])
           << "}";
      }
      os << "]";
    }
  }
  os
     << ",\"total_latency_ms\":" << (frame.done_ms - frame.enqueue_ms)
     << ",\"stage_count\":" << frame.stage_timings.size();
  for (size_t i = 0; i < frame.stage_timings.size(); ++i) {
    const StageTiming& timing = frame.stage_timings[i];
    os << ",\"stage" << i << "_ms\":" << timing.ServiceMs()
       << ",\"stage" << i << "_set_ms\":" << timing.SetMs()
       << ",\"stage" << i << "_run_ms\":" << timing.RunMs()
       << ",\"stage" << i << "_get_ms\":" << timing.GetMs()
       << ",\"stage" << i << "_start_ms\":" << rel(timing.set_start_ms)
       << ",\"stage" << i << "_end_ms\":" << rel(timing.end_ms);
  }
  os << "}";
  return os.str();
}

void ConfigureThreadPool(int runtime_num_threads, const std::string& label = "") {
  if (runtime_num_threads <= 0) {
    return;
  }
  const PackedFunc* config = Registry::Get("runtime.config_threadpool");
  if (config != nullptr) {
    (*config)(1, runtime_num_threads);
  }
  const PackedFunc* num_threads = Registry::Get("runtime.NumThreads");
  if (num_threads != nullptr) {
    std::cout << "[THREADPOOL] " << (label.empty() ? "thread" : label)
              << " runtime.NumThreads=" << (*num_threads)().operator int() << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = ParseArgs(argc, argv);
    ResolveStageThreadDefaults(&args);
    int max_runtime_threads = args.runtime_num_threads;
    for (const StageArgs& stage : args.stages) {
      max_runtime_threads = std::max(max_runtime_threads, stage.runtime_num_threads);
    }
    if (max_runtime_threads > 0) {
      const std::string max_threads_env = std::to_string(max_runtime_threads);
      setenv("TVM_NUM_THREADS", max_threads_env.c_str(), 1);
    }
    ConfigureThreadPool(args.runtime_num_threads, "main");
    std::cout << "[THREADPOOL] resolved stage threads:";
    for (size_t i = 0; i < args.stages.size(); ++i) {
      std::cout << " stage" << i << "=" << args.stages[i].runtime_num_threads;
    }
    std::cout << "\n";

    std::vector<StageExecutor> stages;
    stages.reserve(args.stages.size());
    for (const StageArgs& stage_arg : args.stages) {
      stages.push_back(LoadStage(stage_arg.name, stage_arg.device, stage_arg));
    }

    DLTensor* input_tensor = stages[0].get_input(args.stages[0].input_names[0]);
    if (input_tensor == nullptr) {
      throw std::runtime_error("Unable to resolve first stage input tensor");
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

    const double benchmark_start_ms = NowMillis();
    int completed = 0;

    auto write_completed = [&](const Frame& frame, const std::string& mode) {
      result << ResultJSON(args, frame, completed, mode, benchmark_start_ms) << "\n";
      result.flush();
      ++completed;
      const bool allow_intermediate_profile = args.serial;
      if (allow_intermediate_profile && !args.profile_dir.empty() && completed == 1) {
        DumpProfileSnapshot(args.profile_dir, "single_run", frame.frame_id, args.events_limit,
                            profiler_status, profiler_events);
      }
      if (allow_intermediate_profile && !args.profile_dir.empty() && args.checkpoint_every > 0 &&
          completed % args.checkpoint_every == 0) {
        std::ostringstream name;
        name << "checkpoints/run_" << std::setw(4) << std::setfill('0') << completed;
        const std::filesystem::path full = std::filesystem::path(args.profile_dir) / name.str();
        EnsureDir(full.parent_path().string());
        DumpProfileSnapshot(args.profile_dir, name.str(), frame.frame_id, args.events_limit,
                            profiler_status, profiler_events);
      }
    };

    if (args.serial) {
      for (int i = 0; i < args.runs; ++i) {
        const InputRecord& input = inputs[static_cast<size_t>(i) % inputs.size()];
        Frame frame;
        frame.frame_id = i;
        frame.input_index = input.input_index;
        frame.input_file = input.input_file;
        frame.enqueue_ms = NowMillis();
        frame.stage_inputs = {input.data};
        frame.stage_outputs.resize(stages.size());
        frame.stage_timings.resize(stages.size());
        for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
          ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                              "serial.stage" + std::to_string(stage_index));
          const std::vector<NDArray> current_inputs = ResolveStageInputs(args, frame, stage_index);
          frame.stage_outputs[stage_index] =
              RunStage(&stages[stage_index], current_inputs, &frame.stage_timings[stage_index]);
        }
        frame.done_ms = NowMillis();
        if (args.output_mode == "classification") {
          frame.top1 = Top1Float32(frame.stage_outputs.back()[0]);
        }
        write_completed(frame, "serial");
      }
    } else {
      std::vector<std::unique_ptr<BoundedQueue<std::shared_ptr<Frame>>>> queues;
      queues.reserve(stages.size() + 1);
      for (size_t i = 0; i <= stages.size(); ++i) {
        queues.emplace_back(
            std::make_unique<BoundedQueue<std::shared_ptr<Frame>>>(args.queue_depth));
      }
      std::mutex error_mutex;
      std::mutex vta_run_mutex;
      std::exception_ptr first_error = nullptr;

      auto record_error = [&](std::exception_ptr err) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = err;
          for (auto& queue : queues) {
            queue->Close();
          }
        }
      };

      std::vector<std::thread> stage_threads;
      stage_threads.reserve(stages.size());
      for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
        stage_threads.emplace_back([&, stage_index]() {
          try {
            ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,
                                "pipeline.stage" + std::to_string(stage_index));
            std::shared_ptr<Frame> frame;
            while (queues[stage_index]->Pop(&frame)) {
              const std::vector<NDArray> stage_inputs = ResolveStageInputs(args, *frame, stage_index);
              if (stages[stage_index].device == "vta") {
                std::lock_guard<std::mutex> lock(vta_run_mutex);
                frame->stage_outputs[stage_index] = RunStage(
                    &stages[stage_index], stage_inputs, &frame->stage_timings[stage_index]);
              } else {
                frame->stage_outputs[stage_index] = RunStage(
                    &stages[stage_index], stage_inputs, &frame->stage_timings[stage_index]);
              }
              if (stage_index + 1 == stages.size()) {
                frame->done_ms = NowMillis();
                if (args.output_mode == "classification") {
                  frame->top1 = Top1Float32(frame->stage_outputs[stage_index][0]);
                }
              }
              if (!queues[stage_index + 1]->Push(frame)) break;
            }
            queues[stage_index + 1]->Close();
          } catch (...) {
            record_error(std::current_exception());
          }
        });
      }

      std::thread producer_thread([&]() {
        try {
          for (int i = 0; i < args.runs; ++i) {
            const InputRecord& input = inputs[static_cast<size_t>(i) % inputs.size()];
            auto frame = std::make_shared<Frame>();
            frame->frame_id = i;
            frame->input_index = input.input_index;
            frame->input_file = input.input_file;
            frame->enqueue_ms = NowMillis();
            frame->stage_inputs = {input.data};
            frame->stage_outputs.resize(stages.size());
            frame->stage_timings.resize(stages.size());
            if (!queues[0]->Push(frame)) {
              break;
            }
          }
          queues[0]->Close();
        } catch (...) {
          queues[0]->Close();
          record_error(std::current_exception());
        }
      });

      std::shared_ptr<Frame> frame;
      while (queues.back()->Pop(&frame)) {
        write_completed(*frame, "pipeline");
      }

      producer_thread.join();
      for (std::thread& stage_thread : stage_threads) {
        stage_thread.join();
      }

      if (first_error != nullptr) {
        std::rethrow_exception(first_error);
      }
      if (completed != args.runs) {
        std::ostringstream os;
        os << "Pipeline completed " << completed << " frame(s), expected " << args.runs;
        throw std::runtime_error(os.str());
      }
    }

    DumpProfileSnapshot(args.profile_dir, "benchmark_totals", args.runs - 1, args.events_limit,
                        profiler_status, profiler_events);
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "vta_stage_pipeline_runner error: " << err.what() << "\n";
    return 1;
  }
}
