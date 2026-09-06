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

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using tvm::runtime::Module;
using tvm::runtime::NDArray;
using tvm::runtime::PackedFunc;
using tvm::runtime::Registry;

struct Args {
  std::string lib;
  std::string function_name;
  std::string case_id;
  std::string access_kind = "contiguous";
  std::string output_jsonl = "dma_native_result.jsonl";
  int y = 1;
  int x = 1;
  int input_y = 0;
  int input_x = 0;
  int output_y = 0;
  int output_x = 0;
  int pad_top = 0;
  int pad_left = 0;
  int logical_input_y = 0;
  int logical_input_x = 0;
  int batch = 1;
  int block_out = 1;
  int alu_repeats = 1;
  std::string input_fill = "pattern";
  int output_bits = 8;
  int input_count = 1;
  int output_count = 1;
  int inner_repeats = 100;
  int warmup = 5;
  int runs = 20;
};

double NowMicros() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(Clock::now().time_since_epoch()).count();
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
      return argv[++i];
    };
    if (key == "--lib") {
      args.lib = value();
    } else if (key == "--function") {
      args.function_name = value();
    } else if (key == "--case-id") {
      args.case_id = value();
    } else if (key == "--access-kind") {
      args.access_kind = value();
    } else if (key == "--output-jsonl") {
      args.output_jsonl = value();
    } else if (key == "--y") {
      args.y = std::stoi(value());
    } else if (key == "--x") {
      args.x = std::stoi(value());
    } else if (key == "--input-y") {
      args.input_y = std::stoi(value());
    } else if (key == "--input-x") {
      args.input_x = std::stoi(value());
    } else if (key == "--output-y") {
      args.output_y = std::stoi(value());
    } else if (key == "--output-x") {
      args.output_x = std::stoi(value());
    } else if (key == "--pad-top") {
      args.pad_top = std::stoi(value());
    } else if (key == "--pad-left") {
      args.pad_left = std::stoi(value());
    } else if (key == "--logical-input-y") {
      args.logical_input_y = std::stoi(value());
    } else if (key == "--logical-input-x") {
      args.logical_input_x = std::stoi(value());
    } else if (key == "--batch") {
      args.batch = std::stoi(value());
    } else if (key == "--block-out") {
      args.block_out = std::stoi(value());
    } else if (key == "--alu-repeats") {
      args.alu_repeats = std::stoi(value());
    } else if (key == "--input-fill") {
      args.input_fill = value();
    } else if (key == "--output-bits") {
      args.output_bits = std::stoi(value());
    } else if (key == "--input-count") {
      args.input_count = std::stoi(value());
    } else if (key == "--output-count") {
      args.output_count = std::stoi(value());
    } else if (key == "--inner-repeats") {
      args.inner_repeats = std::stoi(value());
    } else if (key == "--warmup") {
      args.warmup = std::stoi(value());
    } else if (key == "--runs") {
      args.runs = std::stoi(value());
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (args.lib.empty() || args.function_name.empty() || args.case_id.empty()) {
    throw std::runtime_error("--lib, --function and --case-id are required");
  }
  if (args.input_y == 0) args.input_y = args.y;
  if (args.input_x == 0) args.input_x = args.x;
  if (args.output_y == 0) args.output_y = args.y;
  if (args.output_x == 0) args.output_x = args.x;
  if (args.logical_input_y == 0) args.logical_input_y = args.input_y;
  if (args.logical_input_x == 0) args.logical_input_x = args.input_x;
  if (args.access_kind != "contiguous" && args.access_kind != "strided" &&
      args.access_kind != "padded") {
    throw std::runtime_error("--access-kind must be contiguous, strided or padded");
  }
  if (args.input_fill != "pattern" && args.input_fill != "pre_padded") {
    throw std::runtime_error("--input-fill must be pattern or pre_padded");
  }
  if (args.y <= 0 || args.x <= 0 || (args.output_bits != 8 && args.output_bits != 32) ||
      (args.input_count != 1 && args.input_count != 2) ||
      (args.output_count != 1 && args.output_count != 2) ||
      args.batch <= 0 || args.block_out <= 0 || args.alu_repeats <= 0 ||
      args.logical_input_y <= 0 || args.logical_input_x <= 0 ||
      args.inner_repeats <= 0 || args.warmup < 0 || args.runs <= 0) {
    throw std::runtime_error("Invalid shape, dtype, warmup or runs");
  }
  return args;
}

size_t FlatIndex(int y, int x, int batch, int lane, int width, int batch_count,
                 int block_out) {
  return ((static_cast<size_t>(y) * width + x) * batch_count + batch) * block_out + lane;
}

void FillInput(const NDArray& array, const Args& args) {
  int32_t* data = static_cast<int32_t*>(array->data);
  for (int y = 0; y < args.input_y; ++y) {
    for (int x = 0; x < args.input_x; ++x) {
      for (int batch = 0; batch < args.batch; ++batch) {
        for (int lane = 0; lane < args.block_out; ++lane) {
          const size_t physical = FlatIndex(y, x, batch, lane, args.input_x,
                                            args.batch, args.block_out);
          if (args.input_fill == "pre_padded") {
            const int logical_y = y - args.pad_top;
            const int logical_x = x - args.pad_left;
            if (logical_y < 0 || logical_y >= args.logical_input_y || logical_x < 0 ||
                logical_x >= args.logical_input_x) {
              data[physical] = 0;
              continue;
            }
            const size_t logical = FlatIndex(logical_y, logical_x, batch, lane,
                                             args.logical_input_x, args.batch,
                                             args.block_out);
            data[physical] = static_cast<int32_t>((logical % 9) + 1);
          } else {
            data[physical] = static_cast<int32_t>((physical % 9) + 1);
          }
        }
      }
    }
  }
}

bool CheckOutput(const NDArray& array, const Args& args) {
  for (int y = 0; y < args.output_y; ++y) {
    for (int x = 0; x < args.output_x; ++x) {
      for (int batch = 0; batch < args.batch; ++batch) {
       for (int lane = 0; lane < args.block_out; ++lane) {
        int expected = 0;
        int source_y = y;
        int source_x = x;
        bool in_source = source_y < args.input_y && source_x < args.input_x;
        if (args.access_kind == "padded") {
          source_y -= args.pad_top;
          source_x -= args.pad_left;
          in_source = source_y >= 0 && source_y < args.input_y && source_x >= 0 &&
                      source_x < args.input_x;
        }
        if (in_source) {
          size_t source_index;
          if (args.input_fill == "pre_padded") {
            const int logical_y = y - args.pad_top;
            const int logical_x = x - args.pad_left;
            if (logical_y < 0 || logical_y >= args.logical_input_y || logical_x < 0 ||
                logical_x >= args.logical_input_x) {
              source_index = 0;
              expected = 0;
            } else {
              source_index = FlatIndex(logical_y, logical_x, batch, lane,
                                       args.logical_input_x, args.batch, args.block_out);
              expected = static_cast<int>((source_index % 9) + 1) * 2;
            }
          } else {
            source_index = FlatIndex(source_y, source_x, batch, lane, args.input_x,
                                     args.batch, args.block_out);
            expected = static_cast<int>((source_index % 9) + 1) * 2;
          }
          expected += args.alu_repeats - 1;
        }
        const size_t output_index = FlatIndex(y, x, batch, lane, args.output_x,
                                             args.batch, args.block_out);
        const int actual = args.output_bits == 8
                               ? static_cast<const int8_t*>(array->data)[output_index]
                               : static_cast<const int32_t*>(array->data)[output_index];
        if (actual != expected) {
          std::cerr << "output mismatch case=" << args.case_id << " y=" << y << " x=" << x
                    << " batch=" << batch << " lane=" << lane << " expected=" << expected
                    << " actual=" << actual << "\n";
          return false;
        }
       }
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const tvm::Device cpu{kDLCPU, 0};
    const tvm::Device ext{kDLExtDev, 0};
    const tvm::runtime::ShapeTuple input_shape(
        {args.input_y, args.input_x, args.batch, args.block_out});
    const tvm::runtime::ShapeTuple output_shape(
        {args.output_y, args.output_x, args.batch, args.block_out});
    const DLDataType input_dtype{kDLInt, 32, 1};
    const DLDataType output_dtype{kDLInt, static_cast<uint8_t>(args.output_bits), 1};

    Module module = Module::LoadFromFile(args.lib);
    PackedFunc function = module.GetFunction(args.function_name, true);
    NDArray input_cpu = NDArray::Empty(input_shape, input_dtype, cpu);
    NDArray input_ext = NDArray::Empty(input_shape, input_dtype, ext);
    NDArray input_ext_b;
    NDArray output_ext = NDArray::Empty(output_shape, output_dtype, ext);
    NDArray output_ext_b;
    FillInput(input_cpu, args);
    input_ext.CopyFrom(input_cpu);
    if (args.input_count == 2) {
      input_ext_b = NDArray::Empty(input_shape, input_dtype, ext);
      input_ext_b.CopyFrom(input_cpu);
    }
    if (args.output_count == 2) {
      output_ext_b = NDArray::Empty(output_shape, output_dtype, ext);
    }

    auto invoke = [&]() {
      if (args.input_count == 1 && args.output_count == 1) {
        function(input_ext, output_ext);
      } else if (args.input_count == 1 && args.output_count == 2) {
        function(input_ext, output_ext, output_ext_b);
      } else if (args.input_count == 2 && args.output_count == 1) {
        function(input_ext, input_ext_b, output_ext);
      } else {
        function(input_ext, input_ext_b, output_ext, output_ext_b);
      }
    };

    const PackedFunc* profiler_clear = Registry::Get("vta.runtime.profiler_clear");
    const PackedFunc* profiler_status = Registry::Get("vta.runtime.profiler_status");
    if (profiler_clear == nullptr || profiler_status == nullptr) {
      throw std::runtime_error("VTA runtime profiler functions are unavailable");
    }

    for (int i = 0; i < args.warmup; ++i) invoke();

    std::ofstream output(args.output_jsonl, std::ios::out);
    if (!output) throw std::runtime_error("Unable to write " + args.output_jsonl);
    bool all_correct = true;
    for (int i = 0; i < args.runs; ++i) {
      (*profiler_clear)();
      const double begin_us = NowMicros();
      for (int repeat = 0; repeat < args.inner_repeats; ++repeat) {
        invoke();
      }
      const double end_us = NowMicros();
      std::string profile = (*profiler_status)().operator std::string();
      NDArray output_cpu = output_ext.CopyTo(cpu);
      bool correct = CheckOutput(output_cpu, args);
      if (args.output_count == 2) {
        NDArray output_cpu_b = output_ext_b.CopyTo(cpu);
        correct = correct && CheckOutput(output_cpu_b, args);
      }
      all_correct = all_correct && correct;
      output << "{\"case_id\":\"" << args.case_id << "\",\"run\":" << i
             << ",\"inner_repeats\":" << args.inner_repeats
             << ",\"wall_us\":" << (end_us - begin_us) << ",\"correct\":"
             << (correct ? "true" : "false") << ",\"profile\":" << profile << "}\n";
      output.flush();
    }
    return all_correct ? 0 : 2;
  } catch (const std::exception& err) {
    std::cerr << "vta_dma_microbench_runner error: " << err.what() << "\n";
    return 1;
  }
}
