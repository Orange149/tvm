#!/usr/bin/env python3
"""Run host lifecycle probes against frozen, verbatim production manager code."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_supplement_audit import reconcile, seal

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parents[2]
BASE = FRONTEND / 'report_out/stage_tile_cotuning/thesis_supplement_experiments'
RUNNER = ROOT / 'vta/apps/native_deploy/vta_stage_pipeline_runner.cc'


def extract(text, marker):
    if text.count(marker) != 1:
        raise ValueError('Source marker missing or ambiguous: ' + marker)
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    if text[end] != ';':
        raise ValueError('Expected declaration terminator')
    return text[start:end + 1]


def prepare(directory):
    directory = Path(directory)
    text = (directory / 'raw/production_runner.cc').read_text()
    token = extract(text, 'struct P8SlotToken {')
    edge = extract(text, 'struct P8SharedEdge {')
    original = extract(text, 'class BoundarySlotManager {')
    replacements = [
        ('const auto ready = [&]() { return aborted_ || FindFreeSlotLocked() < slots_.size(); };',
         'const auto ready = [&]() { bool value = aborted_ || FindFreeSlotLocked() < slots_.size(); '
         'HostWaitWitness("producer", value); return value; };'),
        ('return slot.state == SlotState::kReady && slot.generation == token.generation &&\n'
         '             slot.frame_id == token.frame_id;',
         'bool value = slot.state == SlotState::kReady && slot.generation == token.generation &&\n'
         '             slot.frame_id == token.frame_id; HostWaitWitness("consumer", value); return value;'),
    ]
    if 'EmitLocked("producer_wait_free"' in original:
        replacements = [
            ('std::function<void(const char*, const P8SlotToken&)> observer = {}',
             'std::function<void(const char*, const P8SlotToken&)> observer = HostProductionObserver')]
    # Hardened source also wakes waiters on identity invalidation.
    elif replacements[1][0] not in original:
        replacements[1] = (
            'return slot.generation != token.generation || slot.frame_id != token.frame_id ||\n'
            '             slot.state == SlotState::kReady;',
            'bool value = slot.generation != token.generation || slot.frame_id != token.frame_id ||\n'
            '             slot.state == SlotState::kReady; HostWaitWitness("consumer", value); return value;',
        )
    instrumented = original
    for before, after in replacements:
        if instrumented.count(before) != 1:
            raise ValueError('Predicate observer insertion not uniquely applicable')
        instrumented = instrumented.replace(before, after)
    recovered = instrumented
    for before, after in reversed(replacements):
        recovered = recovered.replace(after, before)
    if recovered != original:
        raise ValueError('Observer removal did not restore production bytes')
    prototype = ('std::vector<P8SharedEdge> AllocateP8HeterogeneousEdgeBundle('
                 'std::vector<StageExecutor>*, size_t);\n')
    (directory / 'raw/manager_under_test.inc').write_text(
        token + '\n' + edge + '\n' + prototype + instrumented + '\n')
    (directory / 'raw/manager_verbatim.inc').write_text(original + '\n')
    write_new_json(directory / 'analysis/extraction.json', {
        'production_runner_sha256': sha256(directory / 'raw/production_runner.cc'),
        'production_class_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'observers_removed_exact_source_equal': recovered == original,
        'observer_insertions': replacements,
        'uses_actual_production_observer_api': 'EmitLocked("producer_wait_free"' in original,
        'adapters': 'CPU shared byte vectors; no TVM/DMA/physical address simulation',
    })


def run(variant, retry_of, runner_source=None):
    runner_source = Path(runner_source).resolve() if runner_source else RUNNER
    manager = extract(runner_source.read_text(), 'class BoundarySlotManager {')
    publish = manager.split('void Publish(const P8SlotToken& token) {', 1)[1].split('\n  }', 1)[0]
    has_abort_gate = 'ThrowIfAbortedLocked();' in publish
    if (variant == 'baseline' and has_abort_gate) or (variant == 'hardened' and not has_abort_gate):
        raise ValueError('Variant does not match source. To reproduce the old baseline, '
                         'use --runner-source <old-run>/raw/production_runner.cc; do not revert workspace code.')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_c2_host_' + uuid.uuid4().hex[:8]
    directory = BASE / 'results/03' / run_id
    campaign = json.loads((BASE / 'results/ACTIVE_CAMPAIGN.json').read_text())['campaign_id']
    contract = {
        'protocol_version': 'thesis_supplement_v1_20260916_independent_host_c2',
        'campaign_id': campaign, 'run_id': run_id, 'stage': '03', 'retry_of': retry_of,
        'purpose': '用户授权先开展别的实验：C2生产manager本机功能预资格',
        'scope': 'host_prequalification_only', 'variant': variant,
        'formal_phase03_completed': False, 'phase01_certificate': None,
        'phase02_real_data_qualification': None,
        'protocol_reference': str(BASE / '00_GLOBAL_PROTOCOL.md'),
        'protocol_sha256': sha256(BASE / '00_GLOBAL_PROTOCOL.md'),
        'plan_sha256': sha256(BASE / '03_C2_LIFETIME_AND_CAPACITY_PLAN.md'),
        'implementation_sha256': sha256(Path(__file__)),
        'runner_source_path': str(runner_source),
        'runner_sha256': sha256(runner_source),
        'harness_sha256': sha256(FRONTEND / 'thesis_c2_host_lifecycle.cc'),
        'inputs': 'deterministic host byte patterns frame_id + tensor_index; no model/real images',
        'seed': None, 'slot_count': 2, 'tensor_arities': [1, 2],
        'repeats_per_case': 10, 'normal_frames_per_case': 64, 'total_cases': 160,
        'condition_order': ['normal/full-slots arity1,arity2', 'four token faults',
                            'three transition faults', 'five abort cases'],
        'expected_abort_publish': 'reject; old baseline is anticipated to violate this protocol',
        'expected_stale_consumer': 'bounded timeout' if variant == 'baseline' else 'entry identity rejection',
        'manager_timeout_seconds': 1,
        'sanitizers': {'address': True, 'undefined_behavior': True, 'leak': False},
        'sanitizer_amendment_reason': 'LeakSanitizer在当前ptrace环境退出时致命失败；仅关闭leak扫描，不主张无泄漏',
        'timeout_amendment_reason': '本机无RPC，用1s缩短预定过期token负对照；正式板端仍60s',
        'action_timeout_seconds': {'extract': 30, 'compile': 60, 'native_syntax': 60, 'probe': 90},
        'known_baseline_failure_count': 10 if variant == 'baseline' else 0,
        'hardware_calls': 0, 'online_search': False, 'performance_labels': False,
        'wait_assertion': 'false production predicate observed under manager mutex; no sleep surrogate',
        'observer_amendment': 'New manager emits actual production state/wait events; host supplies observer callback as default constructor argument, restoring that argument restores exact source',
        'exceptions': 'independent fresh manager per case; terminate case on assertion failure; retain all failures',
        'certification_impact': 'runner changes require future native rebuild and affected phase02/03 qualification',
        'non_applicable': ['hardware affinity/clock/bitstream', 'quantization/accuracy',
                           'Q1 capacity peaks/certificate', 'GraphExecutor bundle binding', 'online T0'],
        'historical_knowledge': 'source inspection predicted abort Publish admission and stale-token timeout',
    }
    store = RunStore.create(directory, contract)
    for source, name in [(runner_source, 'production_runner.cc'),
                         (FRONTEND / 'thesis_c2_host_lifecycle.cc', 'harness.cc'),
                         (Path(__file__), 'controller.py'),
                         (FRONTEND / 'thesis_supplement_run.py', 'run_store.py'),
                         (FRONTEND / 'thesis_supplement_audit.py', 'audit.py')]:
        shutil.copyfile(source, directory / 'raw' / name)
    store.event('frozen_sources_copied', runner_sha256=sha256(directory / 'raw/production_runner.cc'))
    actions = []
    actions.append(store.execute([sys.executable, str(Path(__file__)), '--prepare', str(directory)],
                                 'extract_production_core', 'functional_qualification', 30))
    if actions[-1]['status'] == 'ok':
        actions.append(store.execute(['g++', '-std=c++17', '-O1', '-g', '-pthread',
                                     '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                                     str(directory / 'raw/harness.cc'), '-o', str(directory / 'raw/probe')],
                                    'compile_host_probe', 'functional_qualification', 60))
    probe = None
    if actions[-1]['status'] == 'ok':
        actions.append(store.execute(['g++', '-std=c++17', '-fsyntax-only',
                                     '-I', str(ROOT / 'include'),
                                     '-I', str(ROOT / '3rdparty/dlpack/include'),
                                     '-I', str(ROOT / '3rdparty/dmlc-core/include'),
                                     '-I', str(ROOT / '3rdparty/vta-hw/include'),
                                     str(directory / 'raw/production_runner.cc'),
                                     '-I', str(RUNNER.parent)],
                                    'native_runner_syntax', 'functional_qualification', 60))
    if actions[-1]['status'] == 'ok':
        probe = store.execute(['env', 'ASAN_OPTIONS=detect_leaks=0', str(directory / 'raw/probe'), variant],
                              'run_host_lifecycle', 'functional_qualification', 90)
        actions.append(probe)
    events = []
    if probe:
        for row in (directory / probe['stdout']).read_text().splitlines():
            events.append(json.loads(row))
    passed = [item['case'] for item in events if item['event'] == 'case_pass']
    failed = [item for item in events if item['event'] == 'case_fail']
    counts = Counter(item['event'] for item in events)
    expected = contract['known_baseline_failure_count']
    observed_expected = len(passed) + len(failed) == 160 and len(failed) == expected
    if variant == 'baseline':
        observed_expected &= all(item['case'].startswith('abort_publish_') for item in failed)
    success = observed_expected and variant == 'hardened' and all(a['status'] == 'ok' for a in actions)
    summary = {
        'status': 'HOST_PREQUALIFICATION_PASS' if success else 'FAILED_NO_FORMAL_CERTIFICATE',
        'variant': variant, 'cases_passed': len(passed), 'cases_failed': len(failed),
        'failed_cases': failed, 'observed_predeclared_outcome': observed_expected,
        'event_counts': dict(counts), 'actual_hardware_calls': 0,
        'formal_phase03_completed': False, 'full_goal_completed': False,
        'limits': ['host allocation adapter only', 'no actual DMA/Executor/task accuracy',
                   'no managed lifecycle latch/bundle/capacity or two-boot qualification'],
    }
    write_new_json(directory / 'summary.json', summary)
    write_new_json(directory / 'analysis/case_results.json', {'passed': passed, 'failed': failed})
    if not success:
        write_new_json(directory / 'invalid_session.json', {
            'reason': 'production baseline violates abort publication contract' if observed_expected
                      else 'unexpected extraction/compile/probe/assertion failure',
            'all_costs_preserved': True, 'retry_requires_new_run': True,
        })
    store.event('host_prequalification_audited', status=summary['status'], formal_phase03_completed=False)
    cost = reconcile(directory)
    write_new_json(directory / 'analysis/cost_audit.json', cost)
    (directory / 'REPORT.md').write_text(
        '# C2 本机生产核心生命周期实验\n\n'
        f'本轮为 `{variant}`，{len(passed)}/160 case通过，{len(failed)}个失败；'
        '正式03未完成，实际VTA调用0。\n\n'
        '生产manager控制逻辑逐字提取，只在两处等待predicate加入观察器；去除观察器后'
        '与冻结源码完全一致。分配及NDArray替换为CPU共享字节vector，不模拟DMA、'
        'GraphExecutor绑定或真实bundle校验。\n\n'
        '固定10重复，覆盖单/双tensor正常64帧、满两slot、过期/错误token、非法转换、'
        'abort后发布/新acquire/持有者release及两类等待唤醒。满slot依赖实际predicate事件屏障，'
        '没有用sleep证明反压。consumer等待READY仅为manager单元故障测试，'
        '生产pipeline正常投递必须先Publish再Push(frame)。\n\n'
        f'已测量动作成本{cost["measured_action_total_seconds"]:.6f}s；开发/交互成本未知为null，'
        '不是在线墙钟或性能数据。失败日志、原始事件及当前源码均保留。\n\n'
        '原始事件见raw中run_host_lifecycle的stdout，逐case结果见'
        '[case_results](analysis/case_results.json)，源码等价证据见'
        '[extraction](analysis/extraction.json)，成本见[cost_audit](analysis/cost_audit.json)。\n\n'
        '后续仍需managed真实生命周期latch、完整bundle入口、当前FINISH峰值和Q1认证、'
        '板卡两boot真实输出。任何native runner修改须重新构建并复验受影响02/03合同。\n')
    seal(directory)
    print(json.dumps({'run_dir': str(directory), 'summary': summary}, ensure_ascii=False))
    return 0 if success or (variant == 'baseline' and observed_expected) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare')
    parser.add_argument('--variant', choices=['baseline', 'hardened'])
    parser.add_argument('--retry-of')
    parser.add_argument('--runner-source', help='Frozen runner snapshot for baseline reproduction')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        if not args.variant:
            parser.error('--variant required')
        sys.exit(run(args.variant, args.retry_of, args.runner_source))
