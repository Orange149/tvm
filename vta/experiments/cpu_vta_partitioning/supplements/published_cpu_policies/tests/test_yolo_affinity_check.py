"""Mode identity, same executor, logical slot proof and replay failures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / 'src')]
from cpu_vta_partitioning.common import digest, read, write
from yolo_affinity_check.common import configurations, protocol, task_order, PARENT, validate_config
from yolo_affinity_check.runtime import make_task, native_command, affinity, summary
from yolo_affinity_check.campaign import accept
from yolo_affinity_check.analysis import aggregate


@pytest.fixture
def capsule(tmp_path):
    for leaf in ('inputs.json', 'programs.json', 'expected_outputs.json'):
        write(tmp_path / 'plans' / leaf, read(PARENT / 'plans' / leaf))
    write(tmp_path / 'plans/configurations.json', configurations())
    write(tmp_path / 'plans/protocol.json', protocol())
    write(tmp_path / 'plans/task_order.json', task_order())
    write(tmp_path / 'manifest.json', dict(bundle_id='fixture', runner_sha256='same-runner', runtime_sha256='same-runtime'))
    return tmp_path


def spec(scope='pipeline', case='overlap4_pinned', phase='measure', round=1):
    return dict(scope=scope, case=case, phase=phase, round=round)


def report(cfg):
    rows = []
    tid = 100
    for stage in cfg['stages']:
        count = stage['threads'] if stage['device'] == 'cpu' else 1
        entries = [dict(tid=tid+i, cpus=[stage['mask'][i]] if cfg['affinity_mode'] == 'pinned'
            and stage['device'] == 'cpu' else stage['mask'], cpu_time_delta_ms=float(i)) for i in range(count)]
        slots = [dict(slot=i, tid=e['tid'], cpus=e['cpus']) for i, e in enumerate(entries)] if stage['device'] == 'cpu' else []
        rows.append(dict(requested_cpus=stage['mask'], active_runtime_threads=stage['threads'],
            caller_tid=tid, thread_ids=[e['tid'] for e in entries], effective_affinities=entries, worker_slots=slots))
        tid += 10
    return dict(cpu_worker_pinning=cfg['affinity_mode'] == 'pinned', stages=rows,
        control_threads=[dict(role=role, tid=tid+i, cpus=[0,1,2,3]) for i, role in enumerate(('collector', 'producer'))])


def test_matched_pairs_only_change_affinity_semantics():
    configs = configurations()
    assert len({c['id'] for c in configs['pipelines'].values()}) == 4
    programs = read(PARENT / 'plans/programs.json')
    for family in ('overlap4', 'split3'):
        fixed, group = (deepcopy(configs['pipelines'][family+'_'+m]) for m in ('pinned', 'group'))
        validate_config(fixed, programs)
        validate_config(group, programs)
        for cfg in (fixed, group):
            cfg.pop('id'); cfg.pop('affinity_mode')
            for stage in cfg['stages']:
                stage.pop('worker_core_map'); stage.pop('caller_core')
        assert fixed == group


def test_budget_order_and_reverse_second_round():
    order = task_order()
    assert order == task_order() and len(order) == 20
    assert [s['case'] for s in order[12:16]] == list(reversed([s['case'] for s in order[16:20]]))
    dnn = sum((1 if s['phase'] == 'qualify' else 104) for s in order if s['scope'] == 'pipeline')
    cpu = sum((1 if s['phase'] == 'qualify' else 10) for s in order if s['scope'] == 'component')
    assert (dnn, cpu) == (836, 44)


@pytest.mark.parametrize('case', list(configurations()['pipelines']))
def test_same_binary_only_pinning_flag_changes(capsule, case):
    doc = make_task(capsule, spec(case=case))
    argv, inputs = native_command(doc, capsule, '/new/bundle', '/new/job')
    assert '/new/bundle/runner/vta_stage_pipeline_runner' in argv
    assert ('--cpu-worker-pinning' in argv) == case.endswith('_pinned')
    assert '--resource-affinity-probe' in argv
    assert '--process-cpu-affinity' not in argv and '--queue-timing' not in argv
    assert next(v for v in argv if v.startswith('LD_PRELOAD=')) == 'LD_PRELOAD=/new/bundle/runtime/libvta.so'
    assert argv[argv.index('--runs')+1] == '104'
    assert inputs == '/new/bundle/inputs/000.bin\n'
    changed = deepcopy(doc); changed['runtime_sha256'] = 'different'
    with pytest.raises(ValueError): native_command(changed, capsule, '/new/bundle', '/new/job')


def test_rounds_have_distinct_tasks(capsule):
    a = make_task(capsule, spec(round=1))
    b = make_task(capsule, spec(round=2))
    assert a['task_id'] != b['task_id'] and a['configuration'] == b['configuration']


@pytest.mark.parametrize('case', list(configurations()['pipelines']))
def test_logical_slots_both_modes(case):
    cfg = configurations()['pipelines'][case]
    r = report(cfg)
    ledger, idle = affinity(r, cfg)
    assert len([x for x in ledger if x.get('program') == 'cpu_00_02']) == cfg['stages'][0]['threads']
    assert not idle
    bad = deepcopy(r); bad['stages'][0]['worker_slots'][0]['tid'] += 1
    with pytest.raises(ValueError): affinity(bad, cfg)
    bad = deepcopy(r); bad['stages'][0]['worker_slots'] = []
    with pytest.raises(ValueError): affinity(bad, cfg)
    bad = deepcopy(r); bad['control_threads'][0]['cpus'] = [0]
    with pytest.raises(ValueError): affinity(bad, cfg)


def test_migration_needs_entire_allowed_set_not_a_subset():
    cfg = configurations()['pipelines']['overlap4_group']
    r = report(cfg)
    r['stages'][0]['effective_affinities'][0]['cpus'] = [0]
    r['stages'][0]['worker_slots'][0]['cpus'] = [0]
    with pytest.raises(ValueError, match='allowed mask'): affinity(r, cfg)


def test_unused_helpers_separate_and_no_cross_stage_tid_reuse():
    cfg = configurations()['pipelines']['overlap4_group']
    r = report(cfg)
    row = r['stages'][2]
    row['thread_ids'].append(999)
    row['effective_affinities'].append(dict(tid=999, cpus=[1,2,3], cpu_time_delta_ms=0))
    _, idle = affinity(r, cfg)
    assert len(idle) == 1 and idle[0]['logical_worker'] is None
    row['effective_affinities'][-1]['cpu_time_delta_ms'] = 10
    with pytest.raises(ValueError, match='helper'): affinity(r, cfg)
    r = report(cfg); r['stages'][1]['thread_ids'][0] = 100
    with pytest.raises(ValueError, match='ownership'): affinity(r, cfg)


def test_fixed_slots_cannot_escape_or_duplicate_core():
    cfg = configurations()['pipelines']['overlap4_pinned']
    r = report(cfg)
    for entry in (r['stages'][0]['worker_slots'][1], r['stages'][0]['effective_affinities'][1]):
        entry['cpus'] = [0]
    with pytest.raises(ValueError, match='allowed mask'): affinity(r, cfg)


def test_summary_rejects_missing_or_duplicate_frames_before_any_cost(capsule, tmp_path):
    doc = make_task(capsule, spec())
    raw = tmp_path / 'raw'; raw.mkdir()
    (raw / 'result.jsonl').write_text('')
    with pytest.raises(ValueError, match='expected 104'): summary(raw, doc, capsule)
    rows = [dict(frame_id=i, completion_index=i, completion_ms=i+1) for i in range(104)]
    rows[-1]['frame_id'] = 0
    (raw / 'result.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='duplicate/missing'): summary(raw, doc, capsule)


def test_receipt_resume_does_not_reexecute_and_rejects_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setitem(accept.__globals__, 'summary', lambda *a: dict(correct=True))
    doc = dict(task_id='task')
    class Fake:
        calls = 0
        def measure(self, *args):
            self.calls += 1
            path = tmp_path/'raw'; path.mkdir()
            (path/'boot_id').write_text('A'); (path/'task_hash').write_text(digest(doc))
            (path/'result.jsonl').write_text('raw')
            return path
        def acknowledge(self, *args): pass
    ssh = Fake()
    a = accept(ssh, doc, 'bundle', 'remote', tmp_path, 'A')
    assert accept(ssh, doc, 'bundle', 'remote', tmp_path, 'A') == a and ssh.calls == 1
    (tmp_path/'raw/task_hash').write_text('wrong')
    with pytest.raises(ValueError): accept(ssh, doc, 'bundle', 'remote', tmp_path, 'A')


def test_aggregate_averages_intervals_not_FPS():
    isolated, pipes = {}, {}
    for case in configurations()['pipelines']:
        isolated[case] = dict(cost=dict(service_ms=100))
        pipes[case] = {r: dict(ii_ms=t, fps=1000/t, stage_service_ms=[150]*5,
                          process_core_ms_per_completion=600) for r, t in ((1,100),(2,200))}
    cases, _ = aggregate(isolated, pipes)
    assert all(c['aggregate_FPS'] == 1000/150 for c in cases)
    assert all(c['aggregate_FPS'] != (10+5)/2 for c in cases)


def test_board_requires_explicit_execute():
    from run_yolo_affinity_check import main
    with pytest.raises(SystemExit): main(['board', 'run'])
