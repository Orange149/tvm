/* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See NOTICE for additional information.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef TVM_VTA_CPU_RESOURCE_GROUPS_H_
#define TVM_VTA_CPU_RESOURCE_GROUPS_H_

#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#include "vta_cpu_stage_gate.h"

namespace vta_deploy {
// Construct once before starting actors. Equal masks denote one device;
// disjoint masks denote independent devices. Partial overlap is not a fixed
// device partition and is rejected instead of silently serializing everything.
class CpuResourceGroups {
 public:
  explicit CpuResourceGroups(const std::vector<std::vector<unsigned int>>& masks) {
    for (const auto& mask : masks) {
      if (mask.empty()) {
        stages_.push_back(nullptr);  // accelerator actor
        continue;
      }
      for (const auto& item : groups_) {
        if (item.first == mask) continue;
        for (auto cpu : mask) {
          for (auto other : item.first) {
            if (cpu == other) throw std::invalid_argument("CPU resource groups partially overlap");
          }
        }
      }
      auto& gate = groups_[mask];
      if (!gate) gate.reset(new CpuStageGate);
      stages_.push_back(gate.get());
    }
  }

  std::unique_ptr<CpuStageGate::Lease> Acquire(size_t stage) {
    if (stage >= stages_.size() || !stages_[stage])
      throw std::invalid_argument("CPU group missing for stage");
    return stages_[stage]->Acquire();
  }

  void Cancel() {
    for (auto& item : groups_) item.second->Cancel();
  }

 private:
  std::map<std::vector<unsigned int>, std::unique_ptr<CpuStageGate>> groups_;
  std::vector<CpuStageGate*> stages_;
};
}  // namespace vta_deploy
#endif  // TVM_VTA_CPU_RESOURCE_GROUPS_H_
