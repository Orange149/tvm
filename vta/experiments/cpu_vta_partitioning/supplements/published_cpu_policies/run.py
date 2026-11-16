#!/usr/bin/env python3
"""Source-audited CPU methods and the explicit single-boot board pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from cpu_policies.packrat import solve_packrat
from cpu_policies.pipeit import CoreGroup,find_split,merge_stage,work_flow
from cpu_policies.runtime_contract import build_runtime_contract,validate_runtime_contract

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def packrat_input(data):
    return {int(t):{int(b):float(v) for b,v in batches.items()} for t,batches in data['latencies_ms'].items()}


def pipeit_input(data):
    if data.get('time_unit')!='ms':raise ValueError('CLI requires an explicit millisecond time matrix')
    columns={}
    for row in data['time_columns']:
        key=(row['core_type'],row['core_count'])
        if key in columns:raise ValueError('duplicate time configuration')
        columns[key]=row['times']
    if data.get('operation','merge_stage')=='work_flow':
        result=work_flow(columns,[CoreGroup(g['core_type'],tuple(g['cores'])) for g in data['groups']])
        result['reproduction_level']='paper_algorithms_1_and_2'
    elif data.get('operation','merge_stage')=='merge_stage':
        result=merge_stage(columns,data['core_inventory'],variant=data['merge_variant'])
        result['reproduction_level']='alg1_2_plus_unverified_alg3_traversal_interpretation'
        result['eligible_as_official_author_dse']=False
    else:raise ValueError('unknown Pipe-it operation')
    return {**result,'time_unit':'ms','predicted_fps':1000/result['interval'] if result['interval'] else None,
            'fixture_role':data.get('fixture_role','user_supplied_costs')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    common=argparse.ArgumentParser(add_help=False);common.add_argument('--output',type=Path)
    verify=commands.add_parser('verify',parents=[common]);verify.add_argument('--author-optimizer',type=Path)
    for name in ('pipeit','packrat','runtime'):
        p=commands.add_parser(name,parents=[common]);p.add_argument('--input',type=Path,required=True)
    existing=commands.add_parser('inspect-existing',parents=[common]);existing.add_argument('--run-dir',type=Path,required=True)
    adapt=commands.add_parser('adapt-existing',parents=[common]);adapt.add_argument('--run-dir',type=Path,required=True);adapt.add_argument('--candidate-id',required=True)
    requirements=commands.add_parser('requirements',parents=[common]);requirements.add_argument('--run-dir',type=Path,required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--base-bundle',type=Path)
    valid=commands.add_parser('validate');valid.add_argument('--bundle',type=Path,required=True)
    for name in ('status','analyze'):
        p=commands.add_parser(name);p.add_argument('--run-dir',type=Path,required=True)
    boards=commands.add_parser('board').add_subparsers(dest='action',required=True)
    start=boards.add_parser('run');start.add_argument('--bundle',type=Path,required=True)
    start.add_argument('--run-id',default='resnet18_published_policy_dp_pilot_01')
    start.add_argument('--local-config',type=Path)
    resume=boards.add_parser('resume');resume.add_argument('--run-dir',type=Path,required=True)
    for p in (start,resume):p.add_argument('--execute-board',action='store_true')
    args=parser.parse_args()
    if args.command in ('prepare','validate','board','status','analyze'):
        from cpu_policies.pilot_common import DEFAULT_LOCAL_CONFIG,load_legacy
        if args.command=='prepare':
            from cpu_policies.pilot_prepare import prepare
            result={'status':'PREPARED_OFFLINE','bundle':str(prepare(args.base_bundle))}
        elif args.command=='validate':
            from cpu_policies.pilot_prepare import validate_bundle
            result={'status':'VALIDATED_OFFLINE','bundle_id':validate_bundle(args.bundle)['bundle_id']}
        elif args.command=='status':
            from cpu_policies.pilot_campaign import status
            result=status(args.run_dir)
        elif args.command=='analyze':
            from cpu_policies.pilot_analysis import analyze
            result=analyze(args.run_dir)
        else:
            if not args.execute_board:parser.error('board access requires --execute-board')
            from cpu_policies.pilot_campaign import create,run_current,validate_state
            from cpu_policies.pilot_runtime import SSH
            import re
            if args.action=='run':
                if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):parser.error('invalid run identity')
                connection=read(args.local_config or DEFAULT_LOCAL_CONFIG)['board']
                root=ROOT/'runs'/args.run_id
                state=create(args.bundle,root,connection)
            else:
                root=args.run_dir.resolve();state=validate_state(root);connection=state['connection']
            result=run_current(root,SSH(connection,execute_board=True,bundle=state['bundle']))
            result={k:result.get(k) for k in ('status','phase','boot_id','completed_components','completed_pipeline_configurations')}
            result['run_dir']=str(root)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False));return
    if args.command=='packrat':
        data=read(args.input);result=solve_packrat(packrat_input(data),data['total_threads'],data['total_batch'])
    elif args.command=='pipeit':result=pipeit_input(read(args.input))
    elif args.command=='runtime':result=build_runtime_contract(**read(args.input))
    elif args.command=='adapt-existing':
        from cpu_policies.integration import adapt_existing_candidate
        result=adapt_existing_candidate(args.run_dir,args.candidate_id)
    elif args.command=='requirements':
        from cpu_policies.integration import profile_requirements
        result=profile_requirements(args.run_dir)
    elif args.command=='inspect-existing':
        from cpu_policies.existing_profiles import inspect_existing
        result=inspect_existing(args.run_dir)
        result['batch1_local_results']={}
        for program,curve in result['service_curves_ms'].items():
            result['batch1_local_results'][program]=solve_packrat({int(t):{1:v} for t,v in curve.items()},4,1)
    else:
        data=read(ROOT/'examples/packrat.json');profile=packrat_input(data)
        pack=solve_packrat(profile,4,4)
        if pack['predicted_latency_ms']!=8 or pack['num_instances']!=2:raise AssertionError('Packrat reference mismatch')
        pipe=pipeit_input(read(ROOT/'examples/pipeit.json'))
        if abs(pipe['interval']-6.6)>1e-10 or len(pipe['groups'])!=1:raise AssertionError('Pipe-it reference mismatch')
        if find_split([0,1],[2,2],[3,3])!=([0,1],[]):raise AssertionError('Pipe-it strict heuristic was replaced')
        runtime=build_runtime_contract(**read(ROOT/'examples/runtime.json'));validate_runtime_contract(runtime)
        result={'scope':'offline method checks; synthetic costs, not measured CPU-FPGA performance',
                'packrat':pack,'pipeit':pipe,'runtime':runtime,'board_jobs_started':0,
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [ROOT/'run.py',*sorted((ROOT/'cpu_policies').glob('*.py'))]}}
        if args.author_optimizer:
            from cpu_policies.external_packrat import run_original_methods
            result['author_packrat_checks']={}
            for label,curve,batch in [('ordinary_batch',profile,4),('batch1',profile,1),
                 ('nonmonotonic',{1:{1:8.},2:{1:4.},3:{1:2.},4:{1:3.}},1),
                 ('tied',{1:{1:6.,2:10.,4:16.},2:{1:4.,2:6.,4:10.},3:{1:3.8,2:5.,4:8.},4:{1:3.7,2:4.7,4:7.5}},4)]:
                result['author_packrat_checks'][label]={
                    'author':run_original_methods(args.author_optimizer,curve,4,batch),
                    'paper_formula':solve_packrat(curve,4,batch)}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
