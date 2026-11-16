#!/usr/bin/env python3
"""Resume with a separately frozen transport, reusing exact qualified binaries."""
import argparse
import json
from pathlib import Path
import shutil

from c3_tps_aggregation_board import sha, write, verify_bindings
from c3_tps_aggregation_protocol import manifest, _check_manifest
from c3_tps_aggregation_transport_v2 import transport_contract, smoke
from c3_tps_aggregation_verify import freeze, read, report

HERE = Path(__file__).resolve().parent


def verify_source(source):
    previous = read(source / 'freeze.json')
    verify_bindings(previous['source_and_artifact_bindings'])
    seal = read(source / 'completion_manifest.json')
    verify_bindings({str(source / k): v for k, v in seal['artifact_sha256'].items()})
    return read(source / 'artifacts/prepared.json')


def transport(args):
    artifacts = verify_source(args.source)
    before = {str(p): sha(p) for p in (HERE / 'c3_tps_aggregation_transport_v2.py',
                                      HERE / 'c3_tps_aggregation_board.py', Path(__file__).resolve())}
    result = smoke(args, artifacts, read(args.source / 'call_manifest.json'), args.output_dir)
    verify_bindings(before)
    write(args.output_dir / 'source_bindings.json', before)
    return result


def prepare(args):
    artifacts = verify_source(args.source)
    smoke_result = read(args.transport_check / 'summary.json')
    if smoke_result.get('status') != 'THIRTY_TRANSPORT_SESSIONS_PASSED' or smoke_result.get('target_calls') != 0:
        raise ValueError('TRANSPORT_NOT_QUALIFIED')
    verify_bindings(read(args.transport_check / 'source_bindings.json'))
    out = args.output_dir
    out.mkdir(exist_ok=False)
    shutil.copytree(args.source / 'artifacts', out / 'artifacts')
    # The copied descriptor deliberately retains exact old absolute binary paths.
    # No rebuild, target, FSim, method or shared-runtime change is introduced.
    preview = manifest(out.name, read(args.source / 'call_manifest.json')['programs'])
    write(out / 'transport_contract.json', transport_contract(preview))
    write(out / 'resume_contract.json', {
        'previous_attempt': str(args.source), 'previous_status': read(args.source / 'board/summary.json')['status'],
        'previous_completion_manifest_sha256': sha(args.source / 'completion_manifest.json'),
        'previous_FPGA_calls_reused': 0, 'new_FSim_calls': 0, 'new_ARM_builds': 0,
        'new_target_calls_planned': 180, 'transport_check': str(args.transport_check),
        'transport_check_sha256': {str(p): sha(p) for p in args.transport_check.rglob('*') if p.is_file()},
        'only_change': 'unique preassigned RPC ports, owned-listener checks and explicit connection shutdown',
        'qualified_programs_runtime_helper_unchanged': True})
    result = freeze(out)
    if read(out / 'call_manifest.json') != preview:
        raise ValueError('RESUMED_CALL_PROTOCOL_CHANGED')
    return result


def board(args):
    from c3_tps_aggregation_board_v2 import execute
    out = args.output_dir
    frozen = read(out / 'freeze.json')
    verify_bindings(frozen['source_and_artifact_bindings'])
    resume = read(out / 'resume_contract.json')
    verify_bindings(resume['transport_check_sha256'])
    ledger = read(out / 'call_manifest.json')
    _check_manifest(ledger)
    args.transport_contract = read(out / 'transport_contract.json')
    if args.transport_contract != transport_contract(ledger):
        raise ValueError('TRANSPORT_PLAN_CHANGED')
    bindings = {**frozen['source_and_artifact_bindings'], str(out / 'freeze.json'): sha(out / 'freeze.json')}
    return execute(args, read(out / 'artifacts/prepared.json'), ledger, out / 'board', bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('transport','prepare','board','report'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--transport-check', type=Path)
    parser.add_argument('--host', default='192.168.1.247')
    parser.add_argument('--known-hosts', default='/home/orange/.ssh/known_hosts')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.source:
        args.source = args.source.resolve()
    if args.transport_check:
        args.transport_check = args.transport_check.resolve()
    if args.phase == 'report':
        result = report(args.output_dir)
    else:
        result = globals()[args.phase](args)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
