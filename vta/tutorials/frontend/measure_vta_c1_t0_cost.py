#!/usr/bin/env python3
"""One authorized, post-label cost campaign; never updates primary E3 labels.

T0 is conditional on frozen topology/profile/template inputs and precompiled SD
packages. It is NOT from-scratch Relay partitioning, DP profiling or AutoTune.
Each arm recomputes the original orders, deploys fresh diagnostic assets, prepares
references on demand and stops at a fixed, previously observed qualification II.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import run_vta_shared_memory_topk_board_boot as native
from analyze_vta_shared_memory_topk_board import validate_deployment
from shared_memory_topk import read_json, sha256_file


ROOT = Path(__file__).resolve().parent
ARMS = ['S0_m1_copy', 'S1_m1_payload', 'S2_m1_certified_shared']
SOURCE_FLAGS = ['topologies', 'edge_contexts', 'p8_summary', 'cross_boot',
                'anchor_summary', 'contention_summary', 'contention_multiboot_summary']


def seal(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_file(record):
    if sha256_file(record['path']) != record['sha256']:
        raise ValueError('frozen file changed: ' + record['path'])
    return read_json(record['path']) if record['path'].endswith('.json') else None


def qualifying(ii_ms, threshold):
    if not math.isfinite(ii_ms) or ii_ms <= 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('invalid II or threshold')
    return ii_ms <= threshold


def verify_rows(rows, count):
    if len(rows) != count or [int(r['input_index']) for r in rows] != list(range(count)):
        raise RuntimeError('three-input full-output gate is incomplete')


def verify_performance(rows, references, expected):
    if len(rows) != expected:
        raise RuntimeError('performance count mismatch')
    for row in rows:
        index = int(row['input_index'])
        if not 0 <= index < len(references):
            raise RuntimeError('performance input index out of range')
        if [r['fnv1a64'] for r in row['raw_outputs']] != [r['fnv1a64'] for r in references[index]['raw_outputs']]:
            raise RuntimeError('performance output drift')


def freeze(args):
    if args.output_dir.exists():
        raise FileExistsError('refusing to overwrite protocol directory')
    contract = read_json(args.contract)
    deployment, _ = validate_deployment(args.deployment_manifest, sha256_file(args.contract))
    analysis = checked_file(contract['frozen_sources']['analysis'])
    primary = read_json(args.primary_analysis)
    if primary['independent_boot_count'] != 3 or not primary['gates']['all_union_configurations_correct']:
        raise ValueError('primary three-boot evidence not closed')
    if primary['sources']['contract']['sha256'] != sha256_file(args.contract):
        raise ValueError('primary contract mismatch')
    oracle = float(primary['union_oracle_ii_ms'])
    qualifying(oracle, oracle)
    files = {str(Path(p).resolve()): sha256_file(p) for p in (
        args.contract, args.deployment_manifest, args.primary_analysis, args.known_hosts,
        Path(__file__), ROOT / 'run_vta_shared_memory_topk.py',
        ROOT / 'shared_memory_topk.py', ROOT / 'analyze_vta_shared_memory_topk_board.py',
        ROOT / 'run_vta_shared_memory_topk_board_boot.py',
    )}
    source_changes = []
    for record in list(contract['frozen_sources'].values()) + list(analysis['sources'].values()):
        current = sha256_file(record['path'])
        if current != record['sha256'] and record['path'].endswith('.py'):
            source_changes.append({'path': record['path'], 'primary_recorded_sha256': record['sha256'],
                                   'cost_current_sha256': current,
                                   'gate': 'all three recomputed Top-5 orders must exactly equal original freeze'})
        else:
            checked_file(record)
        files[record['path']] = current
    payload = {
        'schema': 'vta_c1_dedicated_t0_protocol_v1',
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization': 'user: 允许仅为补齐 T0 进行一次专门成本补测',
        'status': 'frozen_before_new_cost_labels', 'campaign_limit': 1,
        'arm_order': sorted(ARMS, key=lambda name: hashlib.sha256(name.encode()).hexdigest()),
        'strategy_orders': contract['strategies'], 'budget_per_arm': 5,
        'threshold_ms': oracle * 1.02, 'reference_oracle_ms': oracle,
        'stop_rule': 'first correct candidate with current-run II <= fixed reference threshold; else right-censored at 5; any failure stops campaign, no retry',
        'scope': 'conditional online T0: frozen inventory/profile/template/certificates and precompiled packages available; initialized single boot; fresh processes and per-arm /tmp directories; OS/SD page cache not cleared',
        'included': ['local source/context-inventory hash audit', 'original label-free order recomputation from frozen topology scores', 'board preflight', 'diagnostic/plan/input deployment', 'remote package/runtime hash verification', 'on-demand three-input serial references', 'three full-output correctness checks', '302-frame timing', 'result/profile/allocation retrieval', 'output drift and boot identity checks', 'qualification decision'],
        'excluded': ['offline AutoTune and 23 template calibrations', '64 blind validation builds', 'from-scratch Relay context extraction/partition enumeration and M1 DP/resource profiling', 'board reboot and FPGA/udmabuf initialization', 'post-decision archival and final audit'],
        'interpretation': 'post-label cost-only single-boot descriptive benchmark; no statistical wallclock superiority claim, no new ranking/regret labels, no retuning',
        'deployment_path': str(args.deployment_manifest.resolve()),
        'contract_path': str(args.contract.resolve()),
        'primary_analysis_path': str(args.primary_analysis.resolve()),
        'selection_sources': analysis['sources'],
        'selection_source_version_differences': source_changes,
        'band_fraction': analysis['model_semantics']['m1_band_fraction'],
        'capacity_bytes': analysis['model_semantics']['shared_capacity_bytes'],
        'host': args.host, 'expected_boot_id': args.expected_boot_id,
        'known_hosts': str(args.known_hosts.resolve()), 'identity': args.identity,
        'files': files,
        'new_full_compiles': 0,
    }
    if deployment['performance_protocol']['runs'] != 302 or contract['budget_per_strategy'] != 5:
        raise ValueError('unexpected frozen budget')
    with tempfile.TemporaryDirectory(prefix='c1_t0_local_order_check_') as directory:
        recompute_order(payload, Path(directory), ARMS[0])
    payload['artifact_sha256'] = seal(payload)
    args.output_dir.mkdir(parents=True)
    native.write_json(args.output_dir / 'protocol.json', payload)
    print(json.dumps({'protocol_sha256': sha256_file(args.output_dir / 'protocol.json'),
                      'arm_order': payload['arm_order'], 'threshold_ms': payload['threshold_ms']}))


def audit_protocol(protocol):
    unsealed = dict(protocol)
    digest = unsealed.pop('artifact_sha256')
    if seal(unsealed) != digest or protocol['status'] != 'frozen_before_new_cost_labels':
        raise ValueError('invalid protocol seal')
    for path, digest in protocol['files'].items():
        checked_file({'path': path, 'sha256': digest})


def recompute_order(protocol, arm_dir, arm):
    command = [sys.executable, str(ROOT / 'run_vta_shared_memory_topk.py')]
    for flag in SOURCE_FLAGS:
        if flag in protocol['selection_sources']:
            command += ['--' + flag.replace('_', '-'), protocol['selection_sources'][flag]['path']]
    command += ['--budget', '5', '--band-fraction', str(protocol['band_fraction']),
                '--output-dir', str(arm_dir / 'recomputed_selection')]
    if protocol['capacity_bytes'] is not None:
        command += ['--shared-capacity-bytes', str(protocol['capacity_bytes'])]
    proc = subprocess.run(command, capture_output=True, check=True)
    (arm_dir / 'selection.stdout').write_bytes(proc.stdout)
    (arm_dir / 'selection.stderr').write_bytes(proc.stderr)
    observed = read_json(arm_dir / 'recomputed_selection' / 'selection_contract.json')['strategies']
    if observed != protocol['strategy_orders']:
        raise RuntimeError('recomputed orders differ from primary freeze')
    return observed[arm]


def run_candidate(protocol, deployment, config, work, local, position):
    host = protocol['host']
    candidate = config['candidate_id']
    package = deployment['packages'][candidate]
    manifest = read_json(Path(package['source_package']) / 'manifest.json')
    token = 'cfg{:02d}_{}'.format(position, native.sha256_text(config['configuration_id'])[:12])
    runner, driver, inputs = work + '/vta_stage_pipeline_runner', work + '/libvta.so', work + '/inputs.txt'

    def execute(stem, count, serial=False, plan='', profile=False, dump=True):
        output = work + '/' + stem + '.jsonl'
        output_dump = work + '/' + stem + '_outputs' if dump else ''
        profile_dir = work + '/' + stem + '_profile' if profile else ''
        allocation = work + '/' + stem + '_allocation.jsonl' if profile else ''
        command = native.runner_args(manifest, package['desired_stage_runtime_threads'], inputs,
                                     count, output, output_dump, profile_dir, allocation,
                                     serial=serial, shared_plan=plan)
        shell, elapsed = native.run_native(host, package['remote_package'], runner, driver, command,
                                          local / (stem + '.stdout'), local / (stem + '.stderr'), timeout=1200)
        (local / (stem + '_command.sh')).write_text(shell + '\n')
        rows = native.remote_jsonl(host, output, local / (stem + '.jsonl'))
        outputs = None
        if dump:
            outputs = native.remote_archive(host, output_dump, local / (stem + '_outputs.tar'))
            if len(outputs) != 3:
                raise RuntimeError('incomplete full-output archive')
        return rows, outputs, elapsed, profile_dir, allocation

    reference, reference_outputs, reference_s, _, _ = execute(token + '_reference', 3, serial=True)
    verify_rows(reference, 3)
    plan = ''
    if config['policy'] == 'pool_anchor_k2_shared':
        plan = work + '/plan_' + deployment['shared_plans'][candidate]['sha256'][:12] + '.json'
    correctness, observed, correctness_s, _, _ = execute(token + '_correctness', 3, plan=plan)
    verify_rows(correctness, 3)
    if observed != reference_outputs:
        raise RuntimeError('full-output correctness failed; timing prohibited')
    runs = int(deployment['performance_protocol']['runs'])
    performance, _, performance_s, profile_dir, allocation = execute(
        token + '_performance', runs, plan=plan, profile=True, dump=False)
    verify_performance(performance, correctness, runs)
    profile = json.loads(native.remote(host, 'cat ' + shlex.quote(profile_dir + '/benchmark_totals_status.json')).stdout)
    native.write_json(local / (token + '_profile.json'), profile)
    native.remote_jsonl(host, allocation, local / (token + '_allocation.jsonl'))
    boot = native.remote(host, 'cat /proc/sys/kernel/random/boot_id').stdout.decode().strip()
    if boot != protocol['expected_boot_id']:
        raise RuntimeError('boot changed during cost campaign')
    ii = native.completion_window_ii(performance, int(deployment['performance_protocol']['discard_first_scored_rows']))
    return {'configuration_id': config['configuration_id'], 'correct': True, 'ii_ms': ii,
            'qualifies': qualifying(ii, protocol['threshold_ms']), 'reference_subprocess_s': reference_s,
            'correctness_subprocess_s': correctness_s, 'performance_subprocess_s': performance_s,
            'performance_driver_run_calls': int(profile['driver_run_calls']),
            'full_output_gate': 'byte-exact same-package serial reference; not canonical-model accuracy'}


def run(args):
    protocol = read_json(args.protocol)
    audit_protocol(protocol)
    output = args.protocol.parent / 'campaign'
    output.mkdir()  # single-use lock: failure or completion cannot be retried
    native.SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                          '-o', 'IdentitiesOnly=yes', '-i', protocol['identity'],
                          '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + protocol['known_hosts'],
                          '-o', 'GlobalKnownHostsFile=/dev/null', '-o', 'HostKeyAlgorithms=ssh-rsa',
                          '-o', 'PubkeyAcceptedAlgorithms=ssh-rsa']
    report = {'schema': 'vta_c1_dedicated_t0_campaign_v1', 'status': 'running',
              'protocol_sha256': sha256_file(args.protocol), 'arms': [],
              'boot_id': protocol['expected_boot_id'], 'primary_labels_modified': False,
              'statistical_wallclock_saving_claim_allowed': False,
              'from_scratch_t0_observed': False, 'conditional_online_t0_observed': False}
    try:
        for arm in protocol['arm_order']:
            local = output / arm
            local.mkdir()
            result = {'arm': arm, 'status': 'running', 'rows': [], 'events': [],
                      't0_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      't0_monotonic_ns': time.monotonic_ns()}
            report['arms'].append(result)
            def event(name):
                now = time.monotonic_ns()
                result['events'].append({'name': name, 'monotonic_ns': now,
                                         'elapsed_s': (now - result['t0_monotonic_ns']) / 1e9})
            audit_protocol(protocol)
            deployment, _ = validate_deployment(protocol['deployment_path'], sha256_file(protocol['contract_path']))
            order = recompute_order(protocol, local, arm)
            event('audit_and_order_recomputation_complete')
            board, raw = native.board_preflight(protocol['host'], deployment['board_requirements'])
            if board['BOOT'] != protocol['expected_boot_id']:
                raise RuntimeError('board identity differs from cost preregistration')
            (local / 'board_preflight.txt').write_bytes(raw)
            work = '/tmp/c1_t0_{}_{}_{}'.format(board['BOOT'][:8], report['protocol_sha256'][:8], arm)
            native.remote(protocol['host'], 'mkdir ' + shlex.quote(work))
            result['remote_work_dir'] = work
            native.deploy_assets(protocol['host'], deployment, work, local)
            hashes = native.verify_remote_hashes(protocol['host'], deployment['packages'], deployment['diagnostic_build'], work)
            (local / 'remote_artifact_sha256.txt').write_text(hashes)
            event('deployment_and_hash_verification_complete')
            configs = {r['configuration_id']: r for r in deployment['configurations']}
            for position, identity in enumerate(order):
                row = run_candidate(protocol, deployment, configs[identity], work, local, position)
                result['rows'].append(row)
                native.write_json(local / 'decision_checkpoint.json', result)
                event('candidate_{}_qualification_decision'.format(position + 1))
                row['decision_elapsed_s'] = result['events'][-1]['elapsed_s']
                print('[COST] {} {}/5 II={:.6f} elapsed={:.3f}s hit={}'.format(
                    arm, position + 1, row['ii_ms'], row['decision_elapsed_s'], row['qualifies']), flush=True)
                native.write_json(output / 'partial.json', report)
                if row['qualifies']:
                    break
            hit = result['rows'][-1]['qualifies']
            result.update({'status': 'qualified' if hit else 'right_censored_budget_exhausted',
                           'first_hit_rank': len(result['rows']) if hit else None,
                           'wallclock_to_first_qualifying_s': result['rows'][-1]['decision_elapsed_s'] if hit else None,
                           'observed_budget_wallclock_s': result['rows'][-1]['decision_elapsed_s'],
                           'performance_driver_run_calls': sum(r['performance_driver_run_calls'] for r in result['rows'])})
            native.write_json(local / 'arm.json', result)
        board, raw = native.board_preflight(protocol['host'], deployment['board_requirements'])
        if board['BOOT'] != protocol['expected_boot_id']:
            raise RuntimeError('boot changed at final audit')
        (output / 'board_after.txt').write_bytes(raw)
        report['status'] = 'complete_one_cost_only_campaign'
        report['conditional_online_t0_observed'] = True
        report['artifacts'] = {str(p.relative_to(output)): sha256_file(p)
                               for p in sorted(output.rglob('*')) if p.is_file()}
        report['artifact_sha256'] = seal(report)
        native.write_json(output / 'campaign.json', report)
        print('[DONE] single cost-only campaign', flush=True)
    except Exception as error:
        report['status'] = 'failed_no_retry'
        report['error'] = repr(error)
        native.write_json(output / 'failed.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    frozen = sub.add_parser('freeze')
    for name in ('contract', 'deployment-manifest', 'primary-analysis', 'known-hosts', 'output-dir'):
        frozen.add_argument('--' + name, type=Path, required=True)
    frozen.add_argument('--expected-boot-id', required=True)
    frozen.add_argument('--identity', default='/home/orange/.ssh/id_rsa')
    frozen.add_argument('--host', default='root@192.168.1.247')
    runner = sub.add_parser('run')
    runner.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    freeze(args) if args.mode == 'freeze' else run(args)


if __name__ == '__main__':
    main()
