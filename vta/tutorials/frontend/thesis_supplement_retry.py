#!/usr/bin/env python3
"""Retain a failed preparation run and create an explicitly related retry."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from thesis_supplement_run import RunStore, write_new_json
from thesis_supplement_audit import reconcile, seal


def retry(base, reason, contract_overrides=None):
    if any(key in ('run_id', 'stage', 'campaign_id', 'retry_of', 'created_utc')
           for key in (contract_overrides or {})):
        raise ValueError('Identity override is not permitted')
    pointer_path = base / 'results/ACTIVE_CAMPAIGN.json'
    pointer = json.loads(pointer_path.read_text())
    old = Path(pointer['run_dir'])
    run = RunStore(old)
    completed_calls = [json.loads(line) for path in old.glob('raw/boot*/calls.jsonl')
                       for line in path.read_text().splitlines()]
    executed = bool(completed_calls)
    failure_path = old / 'invalid_session.json'
    if failure_path.exists():
        failure_path = old / 'analysis/retry_disposition.json'
    write_new_json(failure_path, {
        'status': 'BOARD_SESSION_FAILED' if executed else 'PREPARATION_FAILED', 'reason': reason,
        'completed_full_calls_retained': len(completed_calls),
        'completed_scored_calls_retained': sum(item['scored'] for item in completed_calls),
        'costs_preserved': True,
    })
    write_new_json(old / 'analysis/cost_audit.json', reconcile(old))
    write_new_json(old / 'summary.json', {'status': 'FAILED_NO_CERTIFICATE',
                                        'hardware_experiment_started': executed,
                                        'completed_full_calls_retained': len(completed_calls)})
    (old / 'REPORT.md').open('x').write('# 失败运行\n\n' + reason + '\n\n保留完整调用记录' + str(len(completed_calls)) + '条。整个run未通过，不能签发完成证书，也不能拼接旧前缀到新run。原始日志、完整输出、派生产物和成本原样保留；新run关联retry_of。\n')
    run.event('run_invalid', reason=reason)
    seal(old)
    contract = run.contract
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_completion_' + uuid.uuid4().hex[:8]
    new = base / 'results' / contract['stage'] / identity
    contract.update(run_id=identity, created_utc=datetime.now(timezone.utc).isoformat(),
                    retry_of=old.name, retry_reason=reason)
    contract['completion']['build_tools'] = {'cmake': '/usr/bin/cmake', 'cmake_version': '4.3.2',
                                             'ninja': '/usr/bin/ninja', 'cpp_rpc_enabled': True,
                                             'alternative_linker': False}
    for key, value in (contract_overrides or {}).items():
        contract[key] = value
    RunStore.create(new, contract)
    (new / 'raw/candidate_definitions.json').write_bytes((old / 'raw/candidate_definitions.json').read_bytes())
    pointer.update(run_id=identity, run_dir=str(new))
    pointer_path.write_text(json.dumps(pointer, indent=2) + '\n')
    status = base / 'EXECUTION_STATUS.md'
    text = status.read_text().replace(old.name, identity, 1).replace(
        'results/01/' + old.name + '/', 'results/01/' + identity + '/', 1)
    text += '- 2026-09-16：01准备run ' + old.name + '失败，[' + '旧报告](results/01/' + old.name + '/REPORT.md)及成本保留；新run ' + identity + '关联retry_of。原因：' + reason + '\n'
    status.write_text(text)
    return new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reason')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / 'report_out/stage_tile_cotuning/thesis_supplement_experiments'
    print(retry(base, args.reason))
