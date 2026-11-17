#!/usr/bin/env python3
"""ResNet50 frozen-method migration; board work is always explicitly gated."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def main(argv=None):
    from resnet50_validation.common import DEFAULT_OUTPUT, DEFAULT_RUN, LOCAL_CONFIG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'prepare', 'validate', 'rank', 'board', 'status', 'analyze'))
    p.add_argument('action', nargs='?', choices=('run', 'resume'))
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    p.add_argument('--local-config', type=Path, default=LOCAL_CONFIG)
    p.add_argument('--stage', choices=('graph', 'modules', 'references', 'profile', 'pipeline', 'all'), default='all')
    p.add_argument('--execute-board', action='store_true')
    p.add_argument('--resume', action='store_true', help='resume an identity-matched offline DP checkpoint')
    p.add_argument('--verify-search', action='store_true',
                   help='execute offline small-domain C++ and fresh ResNet18 DP acceptance')
    p.add_argument('--seal', action='store_true',
                   help='seal the validated controller identity before the first board operation')
    p.add_argument('--archived', action='store_true',
                   help='read-only audit of the stopped preparation batch and its frozen source snapshot')
    a = p.parse_args(argv)
    if a.seal and (a.command != 'validate' or a.verify_search):
        p.error('--seal applies to validate after search acceptance and profile preparation')
    if a.archived and a.command != 'audit':
        p.error('--archived applies only to the read-only audit command')
    if a.command == 'audit':
        if a.archived:
            from resnet50_validation.archive_audit import audit
            from resnet50_validation.common import ARCHIVED_OUTPUT, ARCHIVED_RUN
            value = audit(ARCHIVED_OUTPUT if a.output == DEFAULT_OUTPUT else a.output,
                          ARCHIVED_RUN if a.run_dir == DEFAULT_RUN else a.run_dir)
        else:
            from resnet50_validation.prepare import audit
            value = audit(a.output, a.local_config)
    elif a.command == 'prepare':
        if a.stage in ('graph', 'all'):
            from cpu_vta_partitioning.common import read
            from resnet50_validation.assets import source_path
            repository = Path(__file__).resolve().parent.parents[4]
            for folder in (repository / 'python', repository / 'vta/python'):
                sys.path.insert(0, str(folder))
            os.environ['TVM_LIBRARY_PATH'] = str(source_path(read(a.local_config)['host_tvm_library']).parent)
            os.environ.setdefault('OMP_NUM_THREADS', '1')
        from resnet50_validation.workflow import prepare
        value = prepare(a.output, a.local_config, stage=a.stage)
    elif a.command == 'validate':
        if a.verify_search:
            from resnet50_validation.search_validation import acceptance
            value = acceptance(a.output, resume=a.resume)
        else:
            from resnet50_validation.validation import validate
            value = validate(a.output)
            if a.seal:
                from resnet50_validation.ranking import verify_search_acceptance
                from resnet50_validation.common import freeze_controllers
                if not value['checks'].get('profile_capsule'):
                    raise ValueError('complete validated profile capsule is required before controller sealing')
                verify_search_acceptance(a.output)
                freeze_controllers(a.output)
                value = {**value, 'controller_identity_sealed': True}
    elif a.command == 'rank':
        from resnet50_validation.ranking import rank
        value = rank(a.output, a.run_dir, resume=a.resume)
    elif a.command == 'board':
        if not a.execute_board:
            p.error('board operations require --execute-board')
        if not a.action:
            p.error('board run or board resume is required')
        from resnet50_validation.campaign import run
        value = run(a.output, a.run_dir, a.local_config, execute_board=True)
    elif a.command == 'status':
        from resnet50_validation.campaign import status
        value = status(a.output, a.run_dir)
    else:
        from resnet50_validation.analysis import analyze
        value = analyze(a.output, a.run_dir)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


if __name__ == '__main__':
    main()
