#!/usr/bin/env python3
"""Read-only cost accounting; never reconstruct unobserved end-to-end T0 time."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_vta_shared_memory_topk_board import (
    analyze, sha256_text, validate_boot_artifacts, validate_deployment,
)
from shared_memory_topk import read_json, sha256_file


def prefix_cost(order, first_hit, rows, profiles):
    if not isinstance(first_hit, int) or not 1 <= first_hit <= len(order):
        raise ValueError('invalid first-hit rank')
    cost = 0.0
    dispatches = 0
    for configuration_id in order[:first_hit]:
        row = rows[configuration_id]
        if not row['correct']:
            raise ValueError('incorrect configuration cannot contribute timing')
        for key in ('correctness_wall_clock_s', 'performance_wall_clock_s'):
            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid subprocess cost')
            cost += value
        calls = profiles[configuration_id]['driver_run_calls']
        if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
            raise ValueError('invalid driver dispatch count')
        dispatches += calls
    return {
        'prefix_configuration_count': first_hit,
        'observed_correctness_plus_performance_subprocess_cost_s': cost,
        'performance_driver_run_calls': dispatches,
    }


def build_accounting(contract_path, deployment_path, boot_paths):
    contract = read_json(contract_path)
    contract_hash = sha256_file(contract_path)
    deployment, deployment_hash = validate_deployment(deployment_path, contract_hash)
    boots = [validate_boot_artifacts(p, deployment, deployment_hash) for p in boot_paths]
    if len(boots) != 3 or {boot['boot_index'] for boot in boots} != {0, 1, 2}:
        raise ValueError('exactly three frozen boot indices are required')
    primary = analyze(contract, boots, contract_hash)
    profiles_by_boot = []
    profile_sources = []
    for path, boot in zip(boot_paths, boots):
        profiles = {}
        for position, configuration_id in enumerate(boot['configuration_order']):
            token = 'cfg{:02d}_{}'.format(position, sha256_text(configuration_id)[:12])
            profile_path = Path(path).parent / (token + '_profile.json')
            profiles[configuration_id] = read_json(profile_path)
            profile_sources.append({'path': str(profile_path.resolve()),
                                    'sha256': sha256_file(profile_path)})
        profiles_by_boot.append(profiles)
    strategies = {}
    for name, order in contract['strategies'].items():
        first = primary['strategy_metrics'][name]['first_oracle_plus_2pct_index']
        if first is None:
            strategies[name] = {'first_hit': None, 'cost_to_hit': None}
            continue
        observations = []
        for boot, profiles in zip(boots, profiles_by_boot):
            rows = {row['configuration_id']: row for row in boot['rows']}
            observations.append({'boot_id': boot['boot_id'],
                                 **prefix_cost(order, first, rows, profiles)})
        strategies[name] = {
            'first_hit': first,
            'per_boot': observations,
            'subprocess_prefix_cost_s_median': statistics.median(
                row['observed_correctness_plus_performance_subprocess_cost_s']
                for row in observations),
            'performance_driver_run_calls_median': statistics.median(
                row['performance_driver_run_calls'] for row in observations),
        }
    result = {
        'schema': 'vta_c1_topk_observed_budget_v1',
        'status': 'complete_read_only_partial_cost_accounting',
        'scope': 'post-label accounting of frozen orders; primary metrics unchanged',
        'new_board_runs': 0,
        'new_vta_full_compiles_in_frozen_board_protocol': 0,
        'full_end_to_end_t0_wallclock_observed': False,
        'full_end_to_end_t0_wallclock_s': None,
        'claim_boundary': (
            'Prefix sums replay recorded subprocess costs at the cross-boot median '
            'first-hit rank, not the actual interleaved union elapsed time or full T0. '
            'They exclude search, deployment, serial-reference preparation and result '
            'collection. driver_run_calls cover performance processes only, not '
            'correctness/reference dispatch or FPGA reprogramming. No end-to-end '
            'wallclock saving claim is admitted.'),
        'strategies': strategies,
        'sources': {
            'contract': {'path': str(Path(contract_path).resolve()), 'sha256': contract_hash},
            'deployment': {'path': str(Path(deployment_path).resolve()), 'sha256': deployment_hash},
            'boots': [{'path': str(Path(p).resolve()), 'sha256': sha256_file(p)} for p in boot_paths],
            'profiles': profile_sources,
            'accountant': {'path': str(Path(__file__).resolve()), 'sha256': sha256_file(__file__)},
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    result['artifact_sha256'] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--deployment-manifest', type=Path, required=True)
    parser.add_argument('--boots', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('refusing to overwrite ' + str(args.output_dir))
    result = build_accounting(args.contract, args.deployment_manifest, args.boots)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'accounting.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'strategies': result['strategies'],
                      'full_end_to_end_t0_wallclock_observed': False,
                      'artifact_sha256': result['artifact_sha256']}, indent=2))


if __name__ == '__main__':
    main()
