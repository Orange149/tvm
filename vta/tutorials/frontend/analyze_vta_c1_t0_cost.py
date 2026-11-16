#!/usr/bin/env python3
"""Read-only validation of the dedicated conditional-online T0 campaign."""
import argparse
import json
import math
from pathlib import Path

from analyze_vta_shared_memory_topk_board import tar_outputs
from measure_vta_c1_t0_cost import audit_protocol, qualifying, seal, verify_rows, verify_performance
import run_vta_shared_memory_topk_board_boot as native
from shared_memory_topk import read_json, sha256_file


def verify_events(arm):
    events = arm['events']
    expected_names = ['audit_and_order_recomputation_complete',
                      'deployment_and_hash_verification_complete'] + [
                          'candidate_{}_qualification_decision'.format(i + 1)
                          for i in range(len(arm['rows']))]
    if [e['name'] for e in events] != expected_names:
        raise ValueError('missing or reordered monotonic events')
    clocks = [arm['t0_monotonic_ns']] + [e['monotonic_ns'] for e in events]
    if any(b <= a for a, b in zip(clocks, clocks[1:])):
        raise ValueError('non-monotonic cost event')
    for event in events:
        elapsed = (event['monotonic_ns'] - arm['t0_monotonic_ns']) / 1e9
        if not math.isfinite(event['elapsed_s']) or abs(event['elapsed_s'] - elapsed) > 1e-9:
            raise ValueError('elapsed value not derived from observed monotonic clock')
    return events


def validate_campaign(protocol_path, campaign_path):
    protocol = read_json(protocol_path)
    audit_protocol(protocol)
    campaign = read_json(campaign_path)
    unsealed = dict(campaign)
    if unsealed.pop('artifact_sha256') != seal(unsealed):
        raise ValueError('campaign seal mismatch')
    if campaign['protocol_sha256'] != sha256_file(protocol_path):
        raise ValueError('protocol hash mismatch')
    if campaign['status'] != 'complete_one_cost_only_campaign':
        raise ValueError('campaign not complete')
    if campaign['boot_id'] != protocol['expected_boot_id']:
        raise ValueError('boot identity mismatch')
    root = Path(campaign_path).parent
    for name, digest in campaign['artifacts'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError('cost artifact mismatch: ' + name)
    if [r['arm'] for r in campaign['arms']] != protocol['arm_order']:
        raise ValueError('arm order mismatch')
    deployment = read_json(protocol['deployment_path'])
    summary = []
    for arm in campaign['arms']:
        name = arm['arm']
        order = protocol['strategy_orders'][name]
        rows = arm['rows']
        if not 1 <= len(rows) <= 5 or [r['configuration_id'] for r in rows] != order[:len(rows)]:
            raise ValueError('executed order is not frozen prefix')
        events = verify_events(arm)
        calls = 0
        for position, row in enumerate(rows):
            token = 'cfg{:02d}_{}'.format(position, native.sha256_text(row['configuration_id'])[:12])
            directory = root / name
            ref = [json.loads(line) for line in (directory / (token + '_reference.jsonl')).read_text().splitlines()]
            correct = [json.loads(line) for line in (directory / (token + '_correctness.jsonl')).read_text().splitlines()]
            perf = [json.loads(line) for line in (directory / (token + '_performance.jsonl')).read_text().splitlines()]
            verify_rows(ref, 3)
            verify_rows(correct, 3)
            if tar_outputs(directory / (token + '_reference_outputs.tar')) != tar_outputs(directory / (token + '_correctness_outputs.tar')):
                raise ValueError('full output mismatch')
            verify_performance(perf, correct, int(deployment['performance_protocol']['runs']))
            ii = native.completion_window_ii(perf, int(deployment['performance_protocol']['discard_first_scored_rows']))
            if abs(ii - row['ii_ms']) > 1e-9 or not row['correct']:
                raise ValueError('cost II inconsistent with raw data')
            if row['qualifies'] != qualifying(ii, protocol['threshold_ms']):
                raise ValueError('fixed threshold decision mismatch')
            if position < len(rows) - 1 and row['qualifies']:
                raise ValueError('did not stop at first qualification')
            if row['decision_elapsed_s'] != events[position + 2]['elapsed_s']:
                raise ValueError('decision clock mismatch')
            profile = read_json(directory / (token + '_profile.json'))
            if row['performance_driver_run_calls'] != profile['driver_run_calls']:
                raise ValueError('dispatch mismatch')
            calls += row['performance_driver_run_calls']
        hit = rows[-1]['qualifies']
        if not hit and len(rows) != 5:
            raise ValueError('early unexplained censoring')
        expected_rank = len(rows) if hit else None
        expected_wall = rows[-1]['decision_elapsed_s'] if hit else None
        if arm['first_hit_rank'] != expected_rank or arm['wallclock_to_first_qualifying_s'] != expected_wall:
            raise ValueError('first hit summary mismatch')
        if arm['observed_budget_wallclock_s'] != rows[-1]['decision_elapsed_s'] or arm['performance_driver_run_calls'] != calls:
            raise ValueError('budget summary mismatch')
        summary.append({'strategy': name, 'first_hit_rank': expected_rank,
                        'wallclock_to_first_qualifying_s': expected_wall,
                        'observed_budget_wallclock_s': arm['observed_budget_wallclock_s'],
                        'audit_and_order_recomputation_s': events[0]['elapsed_s'],
                        'deployment_and_preflight_and_hash_verification_s': events[1]['elapsed_s'] - events[0]['elapsed_s'],
                        'reference_subprocess_s': sum(r['reference_subprocess_s'] for r in rows),
                        'correctness_subprocess_s': sum(r['correctness_subprocess_s'] for r in rows),
                        'performance_subprocess_s': sum(r['performance_subprocess_s'] for r in rows),
                        'performance_driver_run_calls': calls,
                        'correct_candidates': len(rows), 'status': arm['status']})
    result = {'schema': 'vta_c1_dedicated_t0_analysis_v1',
              'status': 'complete_validated_cost_only_campaign',
              'boot_id': campaign['boot_id'], 'threshold_ms': protocol['threshold_ms'],
              'scope': protocol['scope'], 'included': protocol['included'], 'excluded': protocol['excluded'],
              'conditional_online_t0_observed': True, 'from_scratch_t0_observed': False,
              'new_full_compiles': 0, 'campaign_count': 1, 'independent_cost_boot_count': 1,
              'primary_three_boot_ranking_unchanged': True,
              'statistical_wallclock_saving_claim_allowed': False,
              'interpretation': protocol['interpretation'], 'strategies': summary,
              'sources': {'protocol': {'path': str(Path(protocol_path).resolve()), 'sha256': sha256_file(protocol_path)},
                          'campaign': {'path': str(Path(campaign_path).resolve()), 'sha256': sha256_file(campaign_path)},
                          'analyzer': {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(__file__)}}}
    result['artifact_sha256'] = seal(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('refusing to overwrite')
    result = validate_campaign(args.protocol, args.campaign)
    args.output_dir.mkdir(parents=True)
    native.write_json(args.output_dir / 'analysis.json', result)
    print(json.dumps(result['strategies'], indent=2))


if __name__ == '__main__':
    main()
