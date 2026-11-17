// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Independent integer scalar verification of selected execution paths.
// This is deliberately NOT a second global Top-K enumeration.
#define PICOJSON_USE_INT64
#include "picojson.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using I = std::int64_t;
using V = picojson::value;
using O = picojson::object;
using A = picojson::array;

I integer(const V& value) {
  if (!value.is<I>()) throw std::runtime_error("expected an exact int64 value");
  I out = value.get<I>();
  if (out < 0) throw std::runtime_error("negative resource input");
  return out;
}

I get(const O& object, const char* name) { return integer(object.at(name)); }

int bits(const V& value) {
  int result = 0;
  for (const auto& x : value.get<A>()) {
    I core = integer(x);
    if (core > 3 || (result & (1 << core))) throw std::runtime_error("invalid core set");
    result |= 1 << core;
  }
  if (!result) throw std::runtime_error("empty core set");
  return result;
}

std::vector<I> allocation(I total, const A& weights) {
  I denominator = 0;
  for (const auto& value : weights) denominator += integer(value);
  if (denominator <= 0) throw std::runtime_error("empty CPU weights");
  std::vector<I> result(weights.size()), remainder(weights.size());
  I sum = 0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    __int128 product = static_cast<__int128>(total) * integer(weights[i]);
    result[i] = static_cast<I>(product / denominator);
    remainder[i] = static_cast<I>(product % denominator);
    sum += result[i];
  }
  std::vector<std::size_t> order(weights.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return remainder[a] > remainder[b];
  });
  for (I j = 0; j < total - sum; ++j) ++result[order[j]];
  return result;
}

I score(const O& cfg) {
  const std::string method = cfg.at("method").get<std::string>();
  const std::string gate = cfg.at("gate").get<std::string>();
  if ((method != "complete" && method != "tarnawski") ||
      (gate != "off" && gate != "resource_group") ||
      (method == "tarnawski" && gate != "resource_group"))
    throw std::runtime_error("unsupported frozen model or gate");
  const int pool = bits(cfg.at("compute_pool")), host_set = bits(cfg.at("host_mask"));
  std::array<I, 4> demand{}, wall{}, team{};
  std::map<int, I> groups;
  I host = 0, vta = 0, ddr = 0, cpu_max = 0;
  I position = 0, islands = 0;
  bool want_cpu = true;
  for (const auto& value : cfg.at("stages").get<A>()) {
    const O& row = value.get<O>();
    const bool cpu = row.at("device").get<std::string>() == "cpu";
    if (cpu != want_cpu || get(row, "begin") != position || get(row, "end") <= position)
      throw std::runtime_error("invalid alternating execution path");
    position = get(row, "end");
    want_cpu = !want_cpu;
    const I service = get(row, "service_ns"), run = get(row, "run_ns"), boundary = get(row, "boundary_ns");
    ddr += get(row, "ddr_ns");
    int mask = bits(row.at("mask"));
    if (!cpu) {
      if (mask != host_set) throw std::runtime_error("VTA host placement mismatch");
      ++islands; vta += service; host += run + boundary; continue;
    }
    if ((mask & pool) != mask) throw std::runtime_error("CPU uses a reserved core");
    if (gate == "resource_group") {
      for (const auto& group : groups)
        if (group.first != mask && (group.first & mask))
          throw std::runtime_error("incompatible serial groups");
      groups[mask] += service;
    }
    const auto& weights = row.at("weights").get<A>();
    if (weights.size() != static_cast<std::size_t>(__builtin_popcount(mask)))
      throw std::runtime_error("logical CPU count differs from core set");
    if (method == "tarnawski" && weights.size() != 1)
      throw std::runtime_error("baseline requires physical single-core stages");
    auto shares = allocation(run, weights);
    std::size_t slot = 0;
    for (int core = 0; core < 4; ++core) if (mask & (1 << core)) {
      demand[core] += shares[slot++];
      team[core] += service;
      wall[core] += service;
    }
    demand[__builtin_ctz(mask)] += boundary;
    cpu_max = std::max(cpu_max, service);
  }
  if (position != get(cfg, "unit_count") || want_cpu || islands < 1 || islands > get(cfg, "max_vta_islands"))
    throw std::runtime_error("incomplete CPU-ended path");
  I answer = std::max(vta, ddr);
  if (method == "tarnawski")
    return std::max(answer, *std::max_element(wall.begin(), wall.end()));
  answer = std::max(answer, cpu_max);
  answer = std::max(answer, *std::max_element(team.begin(), team.end()));
  if (gate == "resource_group")
    for (const auto& group : groups) answer = std::max(answer, group.second);
  for (int core = 0; core < 4; ++core)
    answer = std::max(answer, demand[core] + ((host_set & (1 << core)) ? host : 0));
  for (int subset = 1; subset < 16; ++subset) {
    I work = 0;
    for (int core = 0; core < 4; ++core) if (subset & (1 << core)) work += demand[core];
    if ((subset & host_set) == host_set) work += host;
    int width = __builtin_popcount(subset);
    answer = std::max(answer, (work + width - 1) / width);
  }
  return answer;
}

int main() {
  try {
    std::string input((std::istreambuf_iterator<char>(std::cin)), {});
    V root;
    std::string error = picojson::parse(root, input);
    if (!error.empty()) throw std::runtime_error(error);
    A scores;
    for (const auto& cfg : root.get<O>().at("configurations").get<A>())
      scores.emplace_back(score(cfg.get<O>()));
    O result;
    result["scores_ns"] = V(scores);
    std::cout << V(result).serialize() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
