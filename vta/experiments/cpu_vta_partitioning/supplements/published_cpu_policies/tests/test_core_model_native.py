"""Host checks for the actual low-overhead native reader and pinned-source patch."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from cpu_policies.core_model.patch_runner import patch, prepare_source, PINNED_SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'cpu_policies/core_model/native_probe.h'
PARENT = ROOT / 'artifacts/bundles/abbcccc62af230ee844745ac'


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('host C++ compiler unavailable')
    root = tmp_path_factory.mktemp('core-model-native')
    source = root / 'test.cc'
    source.write_text(r'''
#include "@HEADER@"
#include <cassert>
#include <atomic>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  const std::string mode = argv[1];
  using namespace vta_run_scope;
  if (mode == "parse") {
    Counters value;
    assert(ParseCounters("12 34 56\n", 9, &value));
    assert(value.runtime_ns == 12 && value.runqueue_wait_ns == 34 && value.timeslices == 56);
    for (const std::string bad : {"12 34\n", "12 -34 56\n", "12 34 56 78\n", "", "1 2 18446744073709551616\n"})
      assert(!ParseCounters(bad.data(), bad.size(), &value));
    auto frames = ParseFrames("8,16,12,20");
    assert((frames == std::vector<int>{8,12,16,20}));
    for (const std::string bad : {"", "1,", "1,,2", "-1", "1,1", "2147483648"}) {
      bool rejected = false;
      try { ParseFrames(bad); } catch (const std::invalid_argument&) { rejected = true; }
      assert(rejected);
    }
  } else if (mode == "worker_map") {
    auto map = ParseWorkerMap("2,0,3,1");
    assert((map == std::vector<unsigned int>{2,0,3,1}));
    std::vector<unsigned int> mask{0,1,2,3}, empty;
    assert(&WorkerCoreMap(mask, empty) == &mask);
    assert(&WorkerCoreMap(mask, map) == &map);
    ValidateWorkerCoreMap(mask, map, 4, true, true);
    ValidateWorkerCoreMap(mask, empty, 4, false, true);
    for (const std::string bad : {"", "1,", "1,,2", "-1", "+1", "1,1", "0,1024", "4294967296"}) {
      bool rejected = false;
      try { ParseWorkerMap(bad); } catch (const std::invalid_argument&) { rejected = true; }
      assert(rejected);
    }
    for (int invalid = 0; invalid < 5; ++invalid) {
      bool rejected = false;
      try {
        if (invalid == 0) ValidateWorkerCoreMap(mask, map, 3, true, true);
        if (invalid == 1) ValidateWorkerCoreMap(mask, map, 4, false, true);
        if (invalid == 2) ValidateWorkerCoreMap(mask, map, 4, true, false);
        if (invalid == 3) ValidateWorkerCoreMap(mask, {2,0,3,2}, 4, true, true);
        if (invalid == 4) ValidateWorkerCoreMap(mask, {2,0,3,4}, 4, true, true);
      } catch (const std::runtime_error&) { rejected = true; }
      assert(rejected);
    }
  } else if (mode == "ordered_trace") {
    cpu_set_t old; CPU_ZERO(&old); assert(sched_getaffinity(0, sizeof(old), &old) == 0);
    std::vector<unsigned int> allowed;
    for (unsigned c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &old)) allowed.push_back(c);
    if (allowed.size() < 2) { std::cout << "skip-single-core\n"; return 0; }
    const unsigned low = allowed[0], high = allowed[1];
    std::atomic<long> helper_tid{0}; std::atomic<bool> quit{false};
    std::thread helper([&] {
      cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(low, &mask);
      assert(sched_setaffinity(0, sizeof(mask), &mask) == 0);
      helper_tid.store(syscall(SYS_gettid), std::memory_order_release);
      volatile uint64_t work = 1;
      while (!quit.load(std::memory_order_acquire)) work = work * 1664525 + 1013904223;
    });
    while (helper_tid.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    cpu_set_t single; CPU_ZERO(&single); CPU_SET(high, &single);
    assert(sched_setaffinity(0, sizeof(single), &single) == 0);
    const long caller_tid = syscall(SYS_gettid);
    const double origin = NowMillis();
    std::vector<StageProbe> probes(1);
    // Supply TIDs in the opposite order: logical identity must come from map.
    probes[0].Initialize(0, "cpu_05_07", {high,low},
                         {helper_tid.load(),caller_tid}, caller_tid, {0}, true);
    auto* record = probes[0].Find(0);
    probes[0].Before(record); probes[0].RunStarted(record);
    volatile uint64_t work = 1; const double finish = NowMillis() + 35;
    do { for (int i = 0; i < 20000; ++i) work = work * 1664525 + 1013904223; } while (NowMillis() < finish);
    probes[0].RunFinished(record); probes[0].After(record);
    WriteTrace(argv[2], origin, {0}, probes);
    quit.store(true, std::memory_order_release); helper.join();
    assert(sched_setaffinity(0, sizeof(old), &old) == 0);
  } else if (mode == "fresh") {
    const std::string path = argv[2];
    { std::ofstream out(path); out << "100 20 3\n"; }
    CachedCounter counter(0, path);
    auto first = counter.Read(); assert(first.available && first.counters.runtime_ns == 100);
    { std::ofstream out(path); out << "900 35 8\n"; }
    auto second = counter.Read(); assert(second.available && second.counters.runtime_ns == 900);
    auto fresh = counter.ReadFresh(); assert(fresh.available && fresh.counters.runtime_ns == 900);
    { std::ofstream out(path); out << "123 45"; }
    auto partial = counter.Read(); assert(!partial.available && partial.failure == Failure::Incomplete);
    CachedCounter missing(0, path + ".missing"); assert(!missing.Read().available);
  } else if (mode == "trace") {
    cpu_set_t old; CPU_ZERO(&old); assert(sched_getaffinity(0, sizeof(old), &old) == 0);
    unsigned core = 0; while (core < CPU_SETSIZE && !CPU_ISSET(core, &old)) ++core;
    assert(core < CPU_SETSIZE);
    cpu_set_t single; CPU_ZERO(&single); CPU_SET(core, &single);
    assert(sched_setaffinity(0, sizeof(single), &single) == 0);
    const long tid = syscall(SYS_gettid);
    const double origin = NowMillis();
    std::vector<StageProbe> probes(1);
    probes[0].Initialize(0, "cpu_00_02", {core}, {tid}, tid, {0}, true);
    auto* record = probes[0].Find(0); assert(record != nullptr && probes[0].Find(1) == nullptr);
    probes[0].Before(record); probes[0].RunStarted(record);
    volatile uint64_t work = 1;
    const double finish = NowMillis() + 35;
    do { for (int i = 0; i < 20000; ++i) work = work * 1664525 + 1013904223; } while (NowMillis() < finish);
    probes[0].RunFinished(record); probes[0].After(record);
    bool duplicate = false;
    try { probes[0].Before(record); } catch (const std::runtime_error&) { duplicate = true; }
    assert(duplicate);
    WriteTrace(argv[2], origin, {0}, probes);
    assert(sched_setaffinity(0, sizeof(old), &old) == 0);
  } else return 2;
  std::cout << "ok\n";
}
'''.replace('@HEADER@', str(HEADER)))
    binary = root / 'test'
    result = subprocess.run([compiler, '-std=c++17', '-O2', '-pthread', str(source), '-o', str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


def test_native_parser_rejects_partial_signed_extra_and_overflow(native):
    subprocess.run([str(native), 'parse'], check=True, capture_output=True)


def test_cached_descriptor_restarts_offset_and_reports_unknown(native, tmp_path):
    subprocess.run([str(native), 'fresh', str(tmp_path / 'schedstat')], check=True, capture_output=True)


def test_real_thread_probe_freshness_monotonicity_and_envelopes(native, tmp_path):
    trace = tmp_path / 'trace.json'
    subprocess.run([str(native), 'trace', str(trace)], check=True, capture_output=True)
    data = json.loads(trace.read_text())
    assert data['schema'] == 'vta_run_scope_v1'
    assert data['requested_frames'] == [0]
    record, = data['records']
    assert record['runtime_growth_observed'] is True
    points = [record[k] for k in ['before_start_ms', 'before_end_ms', 'run_start_ms',
                                 'run_end_ms', 'after_start_ms', 'after_end_ms']]
    assert points == sorted(points) and record['run_end_ms'] - record['run_start_ms'] >= 30
    thread, = record['threads']
    assert thread['role'] == 'caller' and thread['logical_worker'] == 0
    assert thread['before']['available'] and thread['after']['available']
    assert thread['monotonic'] is True and thread['runtime_delta_ns'] > 0
    assert thread['fresh_check']['performed'] is True
    assert thread['fresh_check']['cached_not_stale'] is True


def test_patch_requires_authenticated_pinned_preimage_and_keeps_legacy_clocks(tmp_path):
    source = PARENT / 'sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc'
    if not source.is_file():
        pytest.skip('original frozen pinned source is unavailable')
    original = source.read_text()
    result = patch(original)
    assert result != original
    assert 'double RunMs() const { return get_start_ms - run_start_ms; }' in result
    assert 'double ServiceMs() const { return end_ms - set_start_ms; }' in result
    assert '--run-thread-counters-frames IDS' in result
    assert '--stageN-cpu-worker-map CPU0[,CPU1]' in result
    assert 'stage.cpu_worker_map = vta_run_scope::ParseWorkerMap' in result
    assert 'requested_worker_core_map' in result
    assert result.count('StageWorkerCPUs(') == 13
    assert 'args.stages[stage_index].cpu_affinity, stage_affinities' not in result
    assert 'args.stages[stage_index].cpu_affinity, serial_affinities' not in result
    # Resource-group construction must continue to use unordered masks.
    assert 'resource_masks.push_back(stage.device == "cpu" ? stage.cpu_affinity' in result
    assert 'vta_run_scope::WriteTrace' in result
    assert 'run_probe->RunStarted(run_record);' in result
    with pytest.raises(ValueError, match='authenticated pinned source'):
        patch(original + '\n')
    with pytest.raises(ValueError, match='authenticated pinned source'):
        patch((PARENT / 'sources/native_source/vta_stage_pipeline_runner.cc').read_text())
    prepared = prepare_source(PARENT, tmp_path / 'native')
    assert prepared['source_preimage_sha256'] == PINNED_SOURCE_SHA256
    assert Path(prepared['source_path']).read_text() == result
    assert (tmp_path / 'native/native_probe.h').is_file()
    assert source.read_text() == original


def test_ordered_worker_map_preserves_caller_and_rejects_invalid_mapping(native):
    subprocess.run([str(native), 'worker_map'], check=True, capture_output=True)


def test_probe_uses_ordered_map_for_real_nonascending_thread_slots(native, tmp_path):
    trace = tmp_path / 'ordered.json'
    result = subprocess.run([str(native), 'ordered_trace', str(trace)], check=True,
                            capture_output=True, text=True)
    if 'skip-single-core' in result.stdout:
        pytest.skip('host process has only one allowed core')
    record, = json.loads(trace.read_text())['records']
    caller, worker = record['threads']
    assert caller['logical_worker'] == 0 and caller['role'] == 'caller'
    assert worker['logical_worker'] == 1 and worker['role'] == 'worker'
    assert caller['core'] > worker['core']
    assert caller['tid'] != worker['tid']
    for item in (caller, worker):
        assert item['monotonic'] is True
        assert item['fresh_check']['performed'] is True
        assert item['fresh_check']['cached_not_stale'] is True
    assert record['runtime_growth_observed'] is True
