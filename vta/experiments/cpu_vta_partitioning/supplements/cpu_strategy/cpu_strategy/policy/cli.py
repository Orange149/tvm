"""One explicit entry point; all local commands are board-free."""
import argparse
import json
from pathlib import Path
import re

from cpu_vta_partitioning.common import read
from .domain import ROOT,PROJECT,protocol


def main(argv=None):
    p=argparse.ArgumentParser(description='ResNet18 成熟资源规则与完整DP Top20')
    commands=p.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--base-bundle',type=Path);prep.add_argument('--group-bundle',type=Path)
    prep.add_argument('--calibration-source',type=Path);prep.add_argument('--local-config',type=Path,default=PROJECT/'configs/local.json')
    prep.add_argument('--config',type=Path,default=ROOT/'configs/resnet18.json')
    v=commands.add_parser('validate');v.add_argument('--bundle',type=Path,required=True)
    for name in ('status','analyze'):
        c=commands.add_parser(name);c.add_argument('--run-dir',type=Path,required=True)
    boards=commands.add_parser('board').add_subparsers(dest='action',required=True)
    start=boards.add_parser('run');start.add_argument('--bundle',type=Path,required=True);start.add_argument('--run-id',default='resnet18_resource_policy_dp_01')
    start.add_argument('--local-config',type=Path,default=PROJECT/'configs/local.json')
    resume=boards.add_parser('resume');resume.add_argument('--run-dir',type=Path,required=True)
    for c in (start,resume):c.add_argument('--execute-board',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':
        from .prepare import prepare,DEFAULT_PARENT,DEFAULT_GROUP,DEFAULT_SOURCE
        print(prepare(a.base_bundle or DEFAULT_PARENT,a.group_bundle or DEFAULT_GROUP,a.calibration_source or DEFAULT_SOURCE,a.local_config,a.config));return 0
    if a.command=='validate':
        from .prepare import validate
        print(json.dumps({'status':'VALIDATED_OFFLINE','bundle_id':validate(a.bundle)['bundle_id']}));return 0
    if a.command=='status':
        from .campaign import validate_state
        s=validate_state(a.run_dir);print(json.dumps({'status':s['status'],'completed_boots':s['boot_index'],'total_boots':6,
            'phase':protocol()['boot_phases'][s['boot_index']] if s['boot_index']<6 else 'complete','active_boot':s['active_boot']},ensure_ascii=False,indent=2));return 0
    if a.command=='analyze':
        from .analysis import analyze
        r=analyze(a.run_dir);print(json.dumps({'status':r['status'],'report':str(a.run_dir/'analysis/REPORT_zh.txt')},ensure_ascii=False));return 0
    if not a.execute_board:p.error('board access requires --execute-board')
    from .transport import SSH
    from .campaign import preflight,create,run_current,validate_state
    if a.action=='run':
        if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):p.error('invalid run identity')
        root=ROOT/'runs'/a.run_id
        if root.exists():p.error('run exists; resume it')
        connection=read(a.local_config)['board'];t=SSH(connection,execute_board=True)
        q=preflight(a.bundle,t);s=create(a.bundle,root,connection,q['excluded_boot_id'])
    else:
        root=a.run_dir.resolve();s=validate_state(root);s=run_current(root,SSH(s['connection'],execute_board=True))
    print(json.dumps({'status':s['status'],'completed_boots':s['boot_index'],'run_dir':str(root)},ensure_ascii=False,indent=2))
    if s['status']=='WAITING_NEW_BOOT':print('数据已回收并同步；按约定等待手动重启。')
    return 0
