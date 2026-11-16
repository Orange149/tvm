"""Patch only the authenticated pinned-runner source in a private build tree."""
from hashlib import sha256
from pathlib import Path
import shutil

PINNED_SOURCE_SHA256 = '672710ad702b187225834949d233bf8ae917dae0750c230c60728d4646c87ca8'
RUNNER_NAME = 'vta_stage_pipeline_runner_run_scope'


def _replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError('runner patch anchor differs: ' + before[:90])
    return source.replace(before, after, 1)


def patch(source_text):
    if sha256(source_text.encode()).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError('run-scope patch requires the authenticated pinned source, not native_source legacy')
    source = _replace(source_text, '#include "vta_session_profile.h"',
                      '#include "vta_session_profile.h"\n#include "native_probe.h"')
    source = _replace(source, '  bool resource_affinity_probe{false};',
                      '  bool resource_affinity_probe{false};\n  std::string run_thread_counters_json;\n  std::vector<int> run_thread_counters_frames;')
    source = _replace(source, '      << "  --resource-affinity-probe (qualification only)\\n"',
                      '      << "  --run-thread-counters-json PATH --run-thread-counters-frames IDS (CPU run probes)\\n"\n'
                      '      << "  --resource-affinity-probe (qualification only)\\n"')
    source = _replace(source, '    } else if (key == "--resource-affinity-probe") {',
                      '    } else if (key == "--run-thread-counters-json") {\n'
                      '      args.run_thread_counters_json = need_value(key);\n'
                      '    } else if (key == "--run-thread-counters-frames") {\n'
                      '      args.run_thread_counters_frames = vta_run_scope::ParseFrames(need_value(key));\n'
                      '    } else if (key == "--resource-affinity-probe") {')
    validation = '''  if (args.run_thread_counters_json.empty() != args.run_thread_counters_frames.empty())
    throw std::runtime_error("run counters require both a path and selected frame IDs");
  if (!args.run_thread_counters_json.empty()) {
    if (!args.cpu_worker_pinning || args.thread_affinity_report.empty() || args.host_empty ||
        args.p8a_edge >= 0 || args.p8_managed_slots > 0 || args.warmup_runs != 0 ||
        args.queue_timing || !args.session_profile_json.empty() || args.profile_pmu ||
        !args.profile_dir.empty() || !args.block_profile_json.empty() ||
        !args.component_thread_transition.empty())
      throw std::runtime_error("run counters require ordinary fixed CPU pools without other diagnostic modes");
    for (int frame : args.run_thread_counters_frames)
      if (frame >= args.runs) throw std::runtime_error("selected run counter frame outside runs");
    bool target = false;
    for (const auto& stage : args.stages)
      target |= stage.device == "cpu" && vta_run_scope::IsTarget(stage.name);
    if (!target) throw std::runtime_error("run counters have no target CPU stage");
  }
'''
    source = _replace(source, '  const std::vector<std::string> transition_modes{',
                      validation + '  const std::vector<std::string> transition_modes{')
    source = _replace(source, '                              const StageRunHooks* hooks = nullptr, bool count_c1_copy_bytes = false) {',
                      '                              const StageRunHooks* hooks = nullptr, bool count_c1_copy_bytes = false,\n'
                      '                              vta_run_scope::StageProbe* run_probe = nullptr, int run_probe_frame = -1) {')
    source = _replace(source, '  stage->run();\n  if (measure_process_cpu_time) {', '''  auto* run_record = run_probe ? run_probe->Find(run_probe_frame) : nullptr;
  if (run_record) {
    run_probe->Before(run_record);
    run_probe->RunStarted(run_record);
    stage->run();
    run_probe->RunFinished(run_record);
    run_probe->After(run_record);
  } else {
    stage->run();
  }
  if (measure_process_cpu_time) {''')
    source = _replace(source, '    std::vector<ThreadAffinityRecord> serial_affinities(stages.size());',
                      '    std::vector<ThreadAffinityRecord> serial_affinities(stages.size());\n'
                      '    std::vector<vta_run_scope::StageProbe> run_scope_probes(stages.size());')
    source = _replace(source, '''      if (args.resource_affinity_probe && args.stages[stage_index].device == "cpu")
        ProbePolicySlots(&serial_affinities[stage_index], args.stages[stage_index].runtime_num_threads);
    };''', '''      if (args.resource_affinity_probe && args.stages[stage_index].device == "cpu")
        ProbePolicySlots(&serial_affinities[stage_index], args.stages[stage_index].runtime_num_threads);
      if (!args.run_thread_counters_json.empty() && args.stages[stage_index].device == "cpu")
        run_scope_probes[stage_index].Initialize(stage_index, args.stages[stage_index].name,
            args.stages[stage_index].cpu_affinity, serial_affinities[stage_index].thread_ids,
            serial_affinities[stage_index].caller_tid, args.run_thread_counters_frames,
            args.resource_affinity_probe);
    };''')
    source = _replace(source, '''                     &frame.stage_timings[stage_index],
                     measure_process_cpu_time);''', '''                     &frame.stage_timings[stage_index],
                     measure_process_cpu_time, nullptr, false,
                     args.run_thread_counters_json.empty() ? nullptr : &run_scope_probes[stage_index], frame_id);''')
    source = _replace(source, '''            if (args.resource_affinity_probe && args.stages[stage_index].device == "cpu")
              ProbePolicySlots(&stage_affinities[stage_index], args.stages[stage_index].runtime_num_threads);
            publish_initialization(stage_index);''', '''            if (args.resource_affinity_probe && args.stages[stage_index].device == "cpu")
              ProbePolicySlots(&stage_affinities[stage_index], args.stages[stage_index].runtime_num_threads);
            if (!args.run_thread_counters_json.empty() && args.stages[stage_index].device == "cpu")
              run_scope_probes[stage_index].Initialize(stage_index, args.stages[stage_index].name,
                  args.stages[stage_index].cpu_affinity, stage_affinities[stage_index].thread_ids,
                  stage_affinities[stage_index].caller_tid, args.run_thread_counters_frames,
                  args.resource_affinity_probe);
            publish_initialization(stage_index);''')
    source = _replace(source, '''                      &timing, false, nullptr, args.c1_memory_audit);
                  if (args.cpu_stage_gate != "off") {''', '''                      &timing, false, nullptr, args.c1_memory_audit,
                      args.run_thread_counters_json.empty() ? nullptr : &run_scope_probes[stage_index], frame->frame_id);
                  if (args.cpu_stage_gate != "off") {''')
    source = _replace(source, '    WriteAllocationSnapshot(args, "after_steady_state");',
                      '    if (!args.run_thread_counters_json.empty())\n'
                      '      vta_run_scope::WriteTrace(args.run_thread_counters_json, benchmark_start_ms,\n'
                      '                                args.run_thread_counters_frames, run_scope_probes);\n\n'
                      '    WriteAllocationSnapshot(args, "after_steady_state");')
    return source


apply = patch


def prepare_source(parent, dest):
    parent, dest = Path(parent).resolve(), Path(dest).resolve()
    source_root = parent / 'sources/vta/apps/native_deploy'
    source_path = source_root / 'vta_stage_pipeline_runner.cc'
    text = source_path.read_text()
    patched = patch(text)
    dest.mkdir(parents=True, exist_ok=True)
    for header in sorted(source_root.glob('*.h')):
        if header.is_symlink():
            raise ValueError('linked native source header')
        shutil.copy2(header, dest / header.name)
    shutil.copy2(source_path, dest / 'vta_stage_pipeline_runner_parent.cc')
    shutil.copy2(Path(__file__).with_name('native_probe.h'), dest / 'native_probe.h')
    target = dest / (RUNNER_NAME + '.cc')
    target.write_text(patched)
    files = {p.name: {'sha256': sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
             for p in sorted(dest.iterdir()) if p.is_file()}
    return {'source': str(target), 'source_path': str(target), 'header': str(dest / 'native_probe.h'),
            'runner_name': RUNNER_NAME, 'source_preimage_sha256': PINNED_SOURCE_SHA256,
            'source_patched_sha256': sha256(patched.encode()).hexdigest(), 'files': files,
            'runtime_changed': False, 'trace_schema': 'vta_run_scope_v1'}
