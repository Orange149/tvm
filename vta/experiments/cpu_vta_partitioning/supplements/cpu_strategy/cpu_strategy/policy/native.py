"""Opt-in source overlay; original runner/runtime remain immutable."""
import difflib
import hashlib
import re

from ..runner_patch import patch_runner


HELPERS = r'''
struct PolicySlot {
  int slot{-1};
  long tid{0};
  std::vector<unsigned int> cpus;
};
thread_local bool policy_pin_cpu = false;
std::vector<PolicySlot> policy_control_slots(2);

void SetPolicyAffinity(const std::vector<unsigned int>& cpus) {
  if (cpus.empty()) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (unsigned int cpu : cpus) {
    if (cpu >= CPU_SETSIZE) throw std::runtime_error("policy CPU id exceeds CPU_SETSIZE");
    CPU_SET(cpu, &set);
  }
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
    throw std::runtime_error("policy current-thread affinity failed");
  if (ParseCPUList(CurrentThreadAffinity()) != cpus)
    throw std::runtime_error("policy affinity readback differs");
}

void CapturePolicyControl(int slot, const std::vector<unsigned int>& cpus) {
  SetPolicyAffinity(cpus);
  policy_control_slots.at(slot) = PolicySlot{slot, static_cast<long>(syscall(SYS_gettid)),
                                          ParseCPUList(CurrentThreadAffinity())};
}
'''

PROBE = r'''
int CapturePolicySlot(int slot, TVMParallelGroupEnv*, void* data) {
  try {
    auto* slots = static_cast<std::vector<PolicySlot>*>(data);
    slots->at(slot) = PolicySlot{slot, static_cast<long>(syscall(SYS_gettid)),
                                ParseCPUList(CurrentThreadAffinity())};
    return 0;
  } catch (...) { return -1; }
}

void ProbePolicySlots(ThreadAffinityRecord* record, int threads) {
  record->policy_slots.resize(threads);
  if (TVMBackendParallelLaunch(CapturePolicySlot, &record->policy_slots, threads) != 0)
    throw std::runtime_error("policy logical-worker probe failed");
}
'''


def patch(source):
    original=source
    source,_=patch_runner(source)
    def replace(before,after,n=1):
        nonlocal source
        if source.count(before)!=n:raise ValueError("policy patch anchor missing/ambiguous: "+before[:100])
        source=source.replace(before,after)
    replace('#include <tvm/runtime/module.h>', '#include <tvm/runtime/c_backend_api.h>\n#include <tvm/runtime/module.h>')
    replace('  bool allow_cpu_affinity_superset{false};',
            '  bool allow_cpu_affinity_superset{false};\n  bool cpu_worker_pinning{false};\n'
            '  bool resource_affinity_probe{false};\n  std::vector<unsigned int> control_cpu_affinity;')
    replace('    } else if (key == "--allow-cpu-affinity-superset") {',
            '    } else if (key == "--cpu-worker-pinning") {\n      args.cpu_worker_pinning = true;\n'
            '    } else if (key == "--resource-affinity-probe") {\n      args.resource_affinity_probe = true;\n'
            '    } else if (key == "--control-cpu-affinity") {\n      args.control_cpu_affinity = ParseCPUList(need_value(key));\n'
            '    } else if (key == "--allow-cpu-affinity-superset") {')
    replace('      << "  --allow-cpu-affinity-superset (opt-in threads <= allowed CPU cores)\\n"',
            '      << "  --cpu-worker-pinning (CPU worker0 and workers pinned one per core)\\n"\n'
            '      << "  --control-cpu-affinity CPU0[,CPU1]\\n"\n'
            '      << "  --resource-affinity-probe (qualification only)\\n"\n'
            '      << "  --allow-cpu-affinity-superset (opt-in threads <= allowed CPU cores)\\n"')
    replace('void ValidateStageAffinities(const Args& args) {',
            'void ValidateStageAffinities(const Args& args) {\n'
            '  if (args.cpu_worker_pinning) {\n'
            '    if (!args.process_cpu_affinity.empty()) throw std::runtime_error("pinned policy forbids process-wide affinity overwrite");\n'
            '    for (const auto& s : args.stages) if (s.device == "cpu" &&\n'
            '        (s.runtime_num_threads <= 0 || s.cpu_affinity.size() != static_cast<size_t>(s.runtime_num_threads)))\n'
            '      throw std::runtime_error("pinned policy needs one core for every CPU slot");\n'
            '  }')
    replace('struct ThreadAffinityRecord {', HELPERS+'\nstruct ThreadAffinityRecord {\n  std::vector<PolicySlot> policy_slots;')
    replace('void WriteThreadAffinityReport(const Args& args, const std::vector<ThreadAffinityRecord>& records) {',
            PROBE+'\nvoid WriteThreadAffinityReport(const Args& args, const std::vector<ThreadAffinityRecord>& records) {')
    replace('    os << "],\\\"effective_affinities\\\":[";',r'''    os << "],\"worker_slots\":[";
    for (size_t i = 0; i < record.policy_slots.size(); ++i) {
      if (i) os << ",";
      const auto& slot = record.policy_slots[i];
      os << "{\"slot\":" << slot.slot << ",\"tid\":" << slot.tid << ",\"cpus\":";
      cpus_json(slot.cpus); os << "}";
    }
    os << "],\"effective_affinities\":[";''')
    replace('  os << "]}\\n";\n  WriteFile(args.thread_affinity_report, os.str());',r'''  os << "],\"cpu_worker_pinning\":" << (args.cpu_worker_pinning ? "true" : "false")
     << ",\"control_threads\":[";
  bool first_control = true;
  for (const auto& slot : policy_control_slots) {
    if (slot.tid == 0) continue;
    if (!first_control) os << ",";
    first_control = false;
    os << "{\"role\":\"" << (slot.slot == 0 ? "collector" : "producer")
       << "\",\"tid\":" << slot.tid << ",\"cpus\":";
    cpus_json(slot.cpus); os << "}";
  }
  os << "]}\n";
  WriteFile(args.thread_affinity_report, os.str());''')
    replace('      (*config)(-3, runtime_num_threads, cpu_array);',
            '      (*config)(policy_pin_cpu ? -2 : -3, runtime_num_threads, cpu_array);\n'
            '      if (policy_pin_cpu) SetPolicyAffinity({cpu_affinity.at(0)});')
    replace('    ConfigureThreadPool(args.runtime_num_threads, {}, "main");',
            '    if (args.serial && args.cpu_worker_pinning && args.stages.at(0).device == "cpu") {\n'
            '      policy_pin_cpu = true;\n'
            '      ConfigureThreadPool(args.stages.at(0).runtime_num_threads, args.stages.at(0).cpu_affinity, "component.initial");\n'
            '    } else {\n      ConfigureThreadPool(args.runtime_num_threads, {}, "main");\n    }\n'
            '    if (!args.serial) CapturePolicyControl(0, args.control_cpu_affinity);')
    source, count = re.subn(r'(?m)^([ \t]*)ConfigureThreadPool\(args\.stages\[stage_index\]\.runtime_num_threads,',
        lambda m:m[1]+'policy_pin_cpu = args.cpu_worker_pinning && args.stages[stage_index].device == "cpu";\n'+m[0],source)
    if count != 4: raise ValueError("policy stage configuration coverage differs")
    replace('      CaptureThreadAffinities(&serial_affinities[stage_index]);',
            '      CaptureThreadAffinities(&serial_affinities[stage_index]);\n'
            '      if (args.resource_affinity_probe && args.stages[stage_index].device == "cpu")\n'
            '        ProbePolicySlots(&serial_affinities[stage_index], args.stages[stage_index].runtime_num_threads);')
    replace('            publish_initialization(stage_index);\n            std::shared_ptr<Frame> frame;',
            '            if (args.resource_affinity_probe && args.stages[stage_index].device == "cpu")\n'
            '              ProbePolicySlots(&stage_affinities[stage_index], args.stages[stage_index].runtime_num_threads);\n'
            '            publish_initialization(stage_index);\n            std::shared_ptr<Frame> frame;')
    replace('      std::thread producer_thread([&]() {\n        try {',
            '      std::thread producer_thread([&]() {\n        try {\n'
            '          CapturePolicyControl(1, args.control_cpu_affinity);')
    diff=''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile="original/runner.cc",tofile="policy/runner.cc"))
    return source,{"preimage_sha256":hashlib.sha256(original.encode()).hexdigest(),
                   "patched_sha256":hashlib.sha256(source.encode()).hexdigest(),
                   "patch_sha256":hashlib.sha256(diff.encode()).hexdigest(),"diff":diff}
