#!/usr/bin/env python3
"""Recover audit provenance only; never re-execute or relabel board measurements."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import uuid
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_supplement_audit import reconcile, seal

CONTROL_PATHS = {'raw/c1_boot3_restore.sh', 'raw/stop_identified_idle_source_rpc.sh'}


def create(source):
    source = Path(source).resolve()
    store = RunStore(source)
    if (source / 'artifact_hashes.json').exists():
        raise ValueError('Source must be stopped and not yet sealed')
    invalid = json.loads((source / 'invalid_session.json').read_text())
    if invalid['operation'] != 'serial_supervisor':
        raise ValueError('Source was not stopped by the final audit supervisor')
    manifest = json.loads((source / 'analysis/imported_evidence.json').read_text())
    original = Path(manifest['source_run'])
    if sha256(original / 'artifact_hashes.json') != manifest['source_ledger_sha256']:
        raise ValueError('Original source ledger changed')
    timeline = [json.loads(line) for line in (source / 'timeline.jsonl').read_text().splitlines()]
    differences = []
    for item in manifest['imported_files']:
        path = source / item['path']
        current = sha256(path)
        if current == item['sha256']:
            continue
        if item['path'] not in CONTROL_PATHS or sha256(original / item['source_path']) != item['sha256']:
            raise ValueError('A noncontrol artifact drifted, or original control is unavailable')
        starts = [row for row in timeline if row['event'] == 'action_start' and row.get('stdin_sha256') == current]
        if len(starts) != 1 or not any(row['event'] == 'action_end' and row.get('action_id') == starts[0]['action_id'] and row['status'] == 'ok' for row in timeline):
            raise ValueError('Current control is not the exact successful execution input')
        differences.append({'path': item['path'], 'source_path': item['source_path'],
            'original_source_run': str(original), 'original_sha256': item['sha256'],
            'current_execution_sha256': current, 'current_action_id': starts[0]['action_id']})
    if {item['path'] for item in differences} != CONTROL_PATHS:
        raise ValueError('Recovery is limited to the two identified control namespace collisions')
    sessions = source / 'raw/native_sessions'
    if len(list(sessions.glob('b3_block*/metrics.json'))) != 24:
        raise ValueError('All 24 measured sessions must exist')
    for name in ('A', 'B', 'C', 'Ca', 'D', 'E'):
        for suffix in ('', '_pipeline'):
            if not (sessions / ('b3_' + name + '_integrated_correctness' + suffix) / 'metrics.json').exists():
                raise ValueError('All six conditions require both qualification sessions')
    write_new_json(source / 'analysis/provenance_reconciliation.json', {
        'audit_error': 'Imported evidence drifted: raw/c1_boot3_restore.sh',
        'differences': differences, 'all_other_imported_files_exact': True,
        'board_measurements_changed': False, 'new_board_calls_required': 0,
        'explanation': 'Imported controls reused filenames subsequently generated for current deployment; both original versions remain in the sealed predecessor. Current versions match actual successful action stdin hashes.'})
    store.event('provenance_control_collision_stop', changed_control_files=2,
                completed_performance_sessions=24, new_board_calls_required=0)
    costs = reconcile(source)
    if costs['unfinished_action_ids']:
        raise ValueError('Source has unfinished paid actions')
    write_new_json(source / 'analysis/cost_audit.json', costs)
    write_new_json(source / 'summary.json', {'status': 'STOPPED_PROVENANCE_CONTROL_NAMESPACE_COLLISION',
        'C1_completed_windows': 48, 'complete_correctness_outputs': 120,
        'completed_performance_sessions': 24, 'numerical_fault': False,
        'formal_stage_completed': False, 'accuracy_evidence': False, 'all_paid_costs_retained': True})
    with (source / 'REPORT.md').open('a') as stream:
        stream.write('\n## 最终审计停止\n\n六条件各10次串行和10次流水完整输出检查通过，24个正逆序性能会话全部完成。最终来源哈希检查发现两份导入控制脚本被当前部署同名脚本覆盖，停止签发结论。其余全部导入文件精确一致；两份当前脚本分别匹配本run实际成功执行action的stdin SHA，原版仍在已封存前驱run。计算artifact和原始测量未改动。\n\n本run停止并封存。另建仅审计目录归档两版控制和源执行合同，从raw重新验算；不新增板端调用、不把源测量重标为新测量。详见[来源核对](analysis/provenance_reconciliation.json)和[全部成本](analysis/cost_audit.json)。\n')
    seal(source)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_single_image_audit_' + uuid.uuid4().hex[:8]
    contract = dict(store.contract, run_id=run_id, retry_of=source.name,
        reused_evidence_run=str(source), audit_only_recovery=True,
        source_execution_run_id=source.name, new_board_calls=0,
        provenance_reconciliation_sha256=sha256(source / 'analysis/provenance_reconciliation.json'))
    new = RunStore.create(source.parent / run_id, contract)
    action_id = uuid.uuid4().hex
    begin = time.monotonic_ns()
    parent = new.event('action_start', action_id=action_id, action='audit_only_evidence_clone',
        cost_class='audit', argv=['internal verified filesystem clone', str(source)],
        timeout_seconds=None)
    ledger = json.loads((source / 'artifact_hashes.json').read_text())
    files = []
    excluded = {'analysis/cost_audit.json', 'analysis/retry_chain_cost_audit.json'}
    for relative, item in ledger.items():
        if sha256(source / relative) != item['sha256']:
            raise ValueError('Newly sealed execution source changed')
        if relative in excluded:
            continue
        if relative == 'analysis/imported_evidence.json':
            target_relative = 'raw/source_executions/' + source.name + '/imported_evidence.json'
        elif relative.startswith(('raw/', 'analysis/', 'figures/')):
            target_relative = relative
        else:
            target_relative = 'raw/source_executions/' + source.name + '/' + relative
        target = new.directory / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Audit import target collision')
        shutil.copy2(source / relative, target)
        files.append({'path': target_relative, 'source_path': relative,
            'source_run': source.name, 'sha256': item['sha256']})
    for item in differences:
        target_relative = 'raw/source_control_versions/' + original.name + '/' + item['path']
        target = new.directory / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / item['source_path'], target)
        if sha256(target) != item['original_sha256']:
            raise ValueError('Original control version copy changed')
        files.append({'path': target_relative, 'source_path': item['source_path'],
            'source_run': original.name, 'sha256': item['original_sha256']})
    first = json.loads((source / 'analysis/c1_boot1_windows.json').read_text())
    write_new_json(new.directory / 'analysis/imported_evidence.json', {
        'source_run': str(source), 'source_ledger_sha256': sha256(source / 'artifact_hashes.json'),
        'original_boot_id': first['boot_id'], 'imported_files': files,
        'source_status': 'all measured sessions completed; control provenance collision explicitly reconciled',
        'new_observations': False, 'new_board_calls': 0,
        'original_control_source_run': str(original),
        'original_control_source_ledger_sha256': manifest['source_ledger_sha256'],
        'scope': 'audit-only clone; raw outputs, timestamps, artifacts, executor identities and costs preserve source execution run and kernel boot'})
    new.event('audit_only_execution_evidence_imported', source_run=source.name,
        files=len(files), new_board_calls=0, source_ledger_sha256=sha256(source / 'artifact_hashes.json'))
    output = new.directory / 'raw/audit_only_evidence_clone.stdout'
    error = new.directory / 'raw/audit_only_evidence_clone.stderr'
    output.write_text(json.dumps({'files': len(files), 'source_run': source.name, 'new_board_calls': 0}) + '\n')
    error.write_text('')
    end = time.monotonic_ns()
    new.event('action_end', action_id=action_id, parent_event_id=parent,
        action='audit_only_evidence_clone', cost_class='audit', status='ok', returncode=0,
        wall_seconds=(end - begin) / 1e9, begin_monotonic_ns=begin, end_monotonic_ns=end,
        stdout=str(output.relative_to(new.directory)), stderr=str(error.relative_to(new.directory)),
        stdout_sha256=sha256(output), stderr_sha256=sha256(error))
    print(new.directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    create(parser.parse_args().source_run)
