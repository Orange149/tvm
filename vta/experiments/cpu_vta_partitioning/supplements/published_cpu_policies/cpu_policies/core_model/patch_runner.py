"""Add ordered logical-worker placement to a private authenticated runner copy.

The affinity mask remains a sorted resource set.  The independent worker map
specifies the caller (slot zero) followed by TVM's logical helper slots.
"""
from hashlib import sha256
from pathlib import Path
import shutil

from ..run_scope.patch_runner import patch as run_scope_patch
from ..run_scope.patch_runner import PINNED_SOURCE_SHA256

RUNNER_NAME = 'vta_stage_pipeline_runner_core_model'


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError('core-model runner patch anchor differs: ' + before[:100])
    return source.replace(before, after)


def patch(source_text):
    """Apply the frozen run-scope patch, then preserve ordered worker maps."""
    source = run_scope_patch(source_text)
    source = _replace(source, '  std::vector<unsigned int> cpu_affinity;\n',
                      '  std::vector<unsigned int> cpu_affinity;\n'
                      '  std::vector<unsigned int> cpu_worker_map;\n')
    source = _replace(source, 'struct Args {', '''// Resource accounting uses cpu_affinity; runtime logical slots use this order.
const std::vector<unsigned int>& StageWorkerCPUs(const StageArgs& stage) {
  return vta_run_scope::WorkerCoreMap(stage.cpu_affinity, stage.cpu_worker_map);
}

struct Args {''')
    source = _replace(source, '      << "  --stageN-cpu-affinity CPU0[,CPU1]\\n"',
                      '      << "  --stageN-cpu-affinity CPU0[,CPU1]\\n"\n'
                      '      << "  --stageN-cpu-worker-map CPU0[,CPU1] (ordered caller then logical workers)\\n"')
    source = _replace(source, '''      } else if (stage_suffix == "cpu-affinity") {
        stage.cpu_affinity = ParseCPUList(need_value(key));''', '''      } else if (stage_suffix == "cpu-affinity") {
        stage.cpu_affinity = ParseCPUList(need_value(key));
      } else if (stage_suffix == "cpu-worker-map") {
        stage.cpu_worker_map = vta_run_scope::ParseWorkerMap(need_value(key));''')
    source = _replace(source, '    const StageArgs& stage = args.stages[i];\n',
                      '    const StageArgs& stage = args.stages[i];\n'
                      '    vta_run_scope::ValidateWorkerCoreMap(stage.cpu_affinity, stage.cpu_worker_map,\n'
                      '        stage.runtime_num_threads, args.cpu_worker_pinning, stage.device == "cpu");\n')

    # Preserve each original call's thread count, policy flag and label.  Only
    # stage pool creation receives the ordered mapping; masks in gates,
    # validation and affinity reports are deliberately unchanged.
    source = _replace(source, 'ConfigureThreadPool(target.runtime_num_threads, target.cpu_affinity,',
                      'ConfigureThreadPool(target.runtime_num_threads, StageWorkerCPUs(target),')
    source = _replace(source, 'ConfigureThreadPool(1, target.cpu_affinity,',
                      'ConfigureThreadPool(1, StageWorkerCPUs(target),', count=2)
    source = _replace(source,
                      'ConfigureThreadPool(args.stages.at(0).runtime_num_threads, args.stages.at(0).cpu_affinity,',
                      'ConfigureThreadPool(args.stages.at(0).runtime_num_threads, StageWorkerCPUs(args.stages.at(0)),')
    for indentation, count in [('                              ', 2),
                               ('                            ', 1),
                               ('                                ', 1)]:
        source = _replace(source,
                          'ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,\n'
                          + indentation + 'args.stages[stage_index].cpu_affinity,',
                          'ConfigureThreadPool(args.stages[stage_index].runtime_num_threads,\n'
                          + indentation + 'StageWorkerCPUs(args.stages[stage_index]),', count=count)
    source = _replace(source,
                      'ConfigureThreadPool(args.stages[0].runtime_num_threads,\n'
                      '                            args.stages[0].cpu_affinity,',
                      'ConfigureThreadPool(args.stages[0].runtime_num_threads,\n'
                      '                            StageWorkerCPUs(args.stages[0]),')
    source = _replace(source,
                      'args.stages[stage_index].cpu_affinity, serial_affinities[stage_index].thread_ids,',
                      'StageWorkerCPUs(args.stages[stage_index]), serial_affinities[stage_index].thread_ids,')
    source = _replace(source,
                      'args.stages[stage_index].cpu_affinity, stage_affinities[stage_index].thread_ids,',
                      'StageWorkerCPUs(args.stages[stage_index]), stage_affinities[stage_index].thread_ids,')
    source = _replace(source, '''    cpus_json(args.stages[stage].cpu_affinity);
    os << ",\\"thread_ids\\":[";''', '''    cpus_json(args.stages[stage].cpu_affinity);
    os << ",\\"requested_worker_core_map\\":";
    cpus_json(args.stages[stage].device == "cpu" ? StageWorkerCPUs(args.stages[stage])
                                              : std::vector<unsigned int>{});
    os << ",\\"thread_ids\\":[";''')
    return source


apply = patch


def prepare_source(parent, dest):
    parent, dest = Path(parent).resolve(), Path(dest).resolve()
    source_root = parent / 'sources/vta/apps/native_deploy'
    source_path = source_root / 'vta_stage_pipeline_runner.cc'
    original = source_path.read_text()
    patched = patch(original)
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
            'runtime_changed': False, 'trace_schema': 'vta_run_scope_v1',
            'worker_map_semantics': 'ordered_caller_then_logical_helpers'}
