"""Postprocess the completed, unchanged A/C attempt; no target execution.

The frozen original report entry mistakenly expected results.jsonl, while the
reused executor writes a JSON array in results.json. Preserve that entry point
and the already-written accounting record, and validate every original row.
"""
import argparse
import json
from pathlib import Path

from c3_tps_sync_release_local import read, write, sha, verify, require
from c3_yolo416_board_execute import accounting
from c3_yolo416_comparison_protocol import analyse

SLOTS = {'TH': 'C', 'P': 'A'}


def read_results(root):
    rows = read(root/'board/results.json')
    require(isinstance(rows, list), 'EXECUTOR_RESULTS_MUST_BE_JSON_ARRAY')
    return rows


def recover(root):
    require(not (root/'completion_manifest.json').exists() and not (root/'result.json').exists(),
            'REPORT_ALREADY_COMPLETED')
    freeze = read(root/'freeze.json')
    verify(freeze['bindings'])
    summary = read(root/'board/summary.json')
    require(summary['status'] == 'COMPLETE_TH_P_OPERATORS_AND_GRAPH_SINGLE_BOOT_ATTEMPT', 'EXECUTION_NOT_COMPLETE')
    account = accounting(root/'board')
    require(account == read(root/'call_accounting.json') == read(root/'board/call_accounting.json'),
            'EXISTING_ACCOUNTING_CHANGED')
    require(account['registered_attempts'] == account['certified'] ==
            account['helper_started_observed_lower_bound'] == account['helper_completed_observed_lower_bound'] == 240
            and not account['execution_unresolved_units'] and not account['invalid_records'],
            'EXECUTION_ACCOUNTING_INCOMPLETE')
    ledger = read(root/'call_manifest.json')
    rows = read_results(root)
    collected = []
    for i, session in enumerate(ledger['sessions'], 1):
        unit = read(root/'board'/('session_%03d' % i)/'summary.json')
        require(unit['status'] == 'COMPLETE_FRESH_UNIT_SUCCESS' and unit['session'] == session and
                len(unit['results']) == 6, 'FRESH_UNIT_NOT_COMPLETE')
        collected.extend(unit['results'])
    require(rows == collected, 'GLOBAL_RESULTS_DIFFER_FROM_ALL_ORIGINAL_UNIT_RESULTS')
    stats = analyse(ledger, rows)
    result = {'status': 'C_A_COMPARISON_COMPLETE', 'success': True, 'actual_variants': SLOTS,
              'no_TopHub_program_in_this_comparison': True, 'accounting': account,
              'historical_performance_samples_reused': 0, 'statistics_legacy_slot_names': stats,
              'comparisons': []}
    for item in stats['objects']:
        result['comparisons'].append({'object_id': item['object_id'],
            'T_ms': {SLOTS[k]: v for k, v in item['T_ms'].items()},
            'round_medians_ms': {SLOTS[k]: v for k, v in item['round_medians_ms'].items()},
            'A_over_C_gain_percent': item['Gain_P_over_TH_percent'],
            'A_over_C_round_gains_percent': item['round_gains_percent'],
            'A_strict_wins_over_C': item['strict_wins'],
            'A_saves_over_C_median_ms': item['median_absolute_ms'],
            'gain_MAD_percent': item['gain_MAD_percent']})
    write(root/'report_reader_failure.json', {
        'phase': 'postprocessing_after_all_240_calls_completed',
        'error': 'FileNotFoundError: frozen report entry expected board/results.jsonl; executor writes board/results.json',
        'original_entrypoint_sha256': sha(Path(__file__).with_name('c3_tps_broader_baseline_compare.py')),
        'actual_executor_results_sha256': sha(root/'board/results.json'),
        'recovery_entrypoint': str(Path(__file__).resolve()), 'recovery_sha256': sha(Path(__file__)),
        'execution_or_frozen_code_modified': False, 'new_target_calls': 0,
        'all_original_unit_results_compared_exactly': True})
    write(root/'result.json', result)
    write(root/'completion_manifest.json', {'status': result['status'], 'success': True, 'paused': True,
          'postprocess_reader_recovered': True,
          'artifact_sha256': {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}})
    print(json.dumps({'status': result['status'], 'comparisons': result['comparisons']}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    recover(p.parse_args().root.resolve())
