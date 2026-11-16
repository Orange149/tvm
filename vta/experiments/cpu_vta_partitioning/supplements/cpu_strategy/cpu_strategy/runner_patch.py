"""Deterministic overlay for a copied runner; never edits the historical source."""
import difflib
import hashlib


BASE_RUNNER_SHA256 = "017cc09fbd874b48acae27efec243c4676efea19f0680d0af29751931cfa1c97"
PATCH_VERSION = 1


def patch_runner(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_RUNNER_SHA256:
        raise ValueError("runner patch preimage changed; audit the source before updating the overlay")
    original = source

    def replace(before, after, count=1):
        nonlocal source
        if source.count(before) != count:
            raise ValueError("runner patch anchor missing or ambiguous: " + before[:90])
        source = source.replace(before, after)

    replace("  bool input_list_multi{false};", "  bool input_list_multi{false};\n  bool allow_cpu_affinity_superset{false};")
    replace('      << "  --stageN-cpu-affinity CPU0[,CPU1]\\n"',
            '      << "  --allow-cpu-affinity-superset (opt-in threads <= allowed CPU cores)\\n"\n'
            '      << "  --stageN-cpu-affinity CPU0[,CPU1]\\n"')
    replace('    } else if (key == "--thread-affinity-report") {',
            '    } else if (key == "--allow-cpu-affinity-superset") {\n'
            '      args.allow_cpu_affinity_superset = true;\n'
            '    } else if (key == "--thread-affinity-report") {')
    replace('    if (stage.device == "cpu" && !stage.cpu_affinity.empty() &&\n'
            '        stage.cpu_affinity.size() != static_cast<size_t>(stage.runtime_num_threads)) {',
            '    const bool invalid_cpu_mask = args.allow_cpu_affinity_superset\n'
            '        ? (stage.runtime_num_threads <= 0 ||\n'
            '           static_cast<size_t>(stage.runtime_num_threads) > stage.cpu_affinity.size())\n'
            '        : stage.cpu_affinity.size() != static_cast<size_t>(stage.runtime_num_threads);\n'
            '    if (stage.device == "cpu" && !stage.cpu_affinity.empty() && invalid_cpu_mask) {')
    replace('                               "-cpu-affinity count must equal runtime-num-threads");',
            '                               (args.allow_cpu_affinity_superset\n'
            '                                ? "-cpu-affinity count must cover positive runtime-num-threads"\n'
            '                                : "-cpu-affinity count must equal runtime-num-threads"));')
    replace('  double done_ms{0.0};',
            '  double done_ms{0.0};\n  double completion_process_cpu_ms{0.0};')
    replace('frame.done_ms = NowMillis();',
            'frame.done_ms = NowMillis();\n      frame.completion_process_cpu_ms = ProcessCpuMillis();', count=3)
    replace('frame->done_ms = NowMillis();',
            'frame->done_ms = NowMillis();\n                frame->completion_process_cpu_ms = ProcessCpuMillis();')
    replace('     << ",\\\"completion_ms\\\":" << rel(frame.done_ms)',
            '     << ",\\\"completion_ms\\\":" << rel(frame.done_ms)\n'
            '     << ",\\\"completion_process_cpu_ms\\\":" << frame.completion_process_cpu_ms')
    replace('  std::vector<std::vector<unsigned int>> effective_affinities;',
            '  std::vector<std::vector<unsigned int>> effective_affinities;\n'
            '  std::vector<double> cpu_start_ms;\n'
            '  std::vector<double> cpu_time_delta_ms;')
    replace('void CaptureThreadAffinities(ThreadAffinityRecord* record) {', r'''double ThreadCpuMillis(long tid) {
#if defined(__linux__)
  std::ifstream input("/proc/self/task/" + std::to_string(tid) + "/stat");
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("unable to read thread CPU time");
  const auto end = line.rfind(')');
  if (end == std::string::npos) throw std::runtime_error("malformed task stat");
  std::istringstream fields(line.substr(end + 2));
  std::string value;
  unsigned long long ticks = 0;
  for (int field = 3; field <= 15; ++field) {
    if (!(fields >> value)) throw std::runtime_error("short task stat");
    if (field == 14 || field == 15) ticks += std::stoull(value);
  }
  const long frequency = sysconf(_SC_CLK_TCK);
  if (frequency <= 0) throw std::runtime_error("invalid task CPU clock frequency");
  return static_cast<double>(ticks) * 1000.0 / frequency;
#else
  throw std::runtime_error("per-thread CPU counters require Linux");
#endif
}

void CaptureThreadAffinities(ThreadAffinityRecord* record) {''')
    replace('    record->effective_affinities.push_back(std::move(cpus));',
            '    record->effective_affinities.push_back(std::move(cpus));\n'
            '    if (record->cpu_start_ms.size() < record->thread_ids.size())\n'
            '      record->cpu_start_ms.push_back(ThreadCpuMillis(tid));')
    replace('void WriteThreadAffinityReport(const Args& args, const std::vector<ThreadAffinityRecord>& records) {', r'''void FinishThreadActivity(ThreadAffinityRecord* record) {
  CaptureThreadAffinities(record);
  record->cpu_time_delta_ms.clear();
  for (size_t i = 0; i < record->thread_ids.size(); ++i) {
    const double delta = ThreadCpuMillis(record->thread_ids[i]) - record->cpu_start_ms.at(i);
    if (delta < 0.0) throw std::runtime_error("thread CPU counter decreased");
    record->cpu_time_delta_ms.push_back(delta);
  }
}

void WriteThreadAffinityReport(const Args& args, const std::vector<ThreadAffinityRecord>& records) {''')
    replace('      cpus_json(record.effective_affinities.at(i));\n      os << "}";',
            '      cpus_json(record.effective_affinities.at(i));\n'
            '      os << ",\\\"cpu_time_delta_ms\\\":"\n'
            '         << (i < record.cpu_time_delta_ms.size() ? record.cpu_time_delta_ms[i] : 0.0)\n'
            '         << ",\\\"thread_role\\\":\\\""\n'
            '         << (record.thread_ids[i] == record.caller_tid ? "caller" : "worker") << "\\\"";\n'
            '      os << "}";')
    replace('     << "\\\",\\\"stages\\\":[";',
            '     << "\\\",\\\"cpu_time_delta_scope\\\":\\\"worker_session_including_warmup_and_tail\\\",\\\"stages\\\":[";')
    replace('                     measure_process_cpu_time);\n      }',
            '                     measure_process_cpu_time);\n'
            '        if (!args.thread_affinity_report.empty() && frame_id + 1 == args.runs)\n'
            '          FinishThreadActivity(&serial_affinities[stage_index]);\n'
            '      }')
    replace('            queues[stage_index + 1]->Close();\n            if (!args.session_profile_json.empty()) {',
            '            if (!args.thread_affinity_report.empty())\n'
            '              FinishThreadActivity(&stage_affinities[stage_index]);\n'
            '            queues[stage_index + 1]->Close();\n            if (!args.session_profile_json.empty()) {')
    replace('      for (std::thread& stage_thread : stage_threads) {\n        stage_thread.join();\n      }',
            '      for (std::thread& stage_thread : stage_threads) {\n        stage_thread.join();\n      }\n'
            '      if (!args.thread_affinity_report.empty())\n'
            '        WriteThreadAffinityReport(args, stage_affinities);')
    replace('    for (auto& manager : p8_slot_managers) {\n      manager->VerifyAllFree();',
            '    if (args.serial && !args.thread_affinity_report.empty())\n'
            '      WriteThreadAffinityReport(args, serial_affinities);\n\n'
            '    for (auto& manager : p8_slot_managers) {\n      manager->VerifyAllFree();')
    patch = "".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                                          fromfile="original/vta_stage_pipeline_runner.cc",
                                          tofile="cpu_strategy/vta_stage_pipeline_runner.cc"))
    return source, {"version": PATCH_VERSION, "preimage_sha256": BASE_RUNNER_SHA256,
                    "patched_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "unified_diff": patch}
