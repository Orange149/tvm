#!/usr/bin/env python3
"""Local preparation for a true-profile transfer test; no implicit board actions."""
import argparse
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare-graph');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--cfg',type=Path,required=True);a.add_argument('--weights',type=Path,required=True)
    a.add_argument('--darknet-lib',type=Path,required=True);a.add_argument('--image',type=Path,required=True)
    a=sub.add_parser('prepare-units');a.add_argument('--graph',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    for name in ('build-probes','build-domain'):
        a=sub.add_parser(name);a.add_argument('--graph',type=Path,required=True);a.add_argument('--workspace',type=Path,required=True)
        a.add_argument('--local-config',type=Path,required=True);a.add_argument('--program',action='append')
    a=sub.add_parser('prepare-reference');a.add_argument('--build',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--parent',type=Path,required=True)
    a=sub.add_parser('board-reference');a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--local-config',type=Path,required=True)
    a.add_argument('--execute-board',action='store_true')
    a=sub.add_parser('prepare-profile');a.add_argument('--build',type=Path,required=True)
    a.add_argument('--references',type=Path,required=True);a.add_argument('--parent',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--repair-build',type=Path)
    a=sub.add_parser('prepare-pipeline');a.add_argument('--profile-bundle',type=Path,required=True)
    a.add_argument('--ranked',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('board-pipeline');a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--local-config',type=Path,required=True)
    a.add_argument('--execute-board',action='store_true')
    a=sub.add_parser('analyze');a.add_argument('--run-dir',type=Path,required=True)
    for name in ('board-profile','board-traffic'):
        a=sub.add_parser(name);a.add_argument('--bundle',type=Path,required=True)
        a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--local-config',type=Path,required=True)
        a.add_argument('--execute-board',action='store_true');a.add_argument('--limit',type=int)
    a=sub.add_parser('rank');a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--profile-run',type=Path,required=True);a.add_argument('--traffic-run',type=Path,required=True)
    a.add_argument('--hardware-source',type=Path,required=True);a.add_argument('--protocol',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare-graph':
        from yolo_transfer.prepare.common_graph import prepare
        result={'graph_directory':str(prepare(args.output,cfg=args.cfg,weights=args.weights,darknet_lib=args.darknet_lib,image=args.image))}
    elif args.command=='prepare-units':
        from yolo_transfer.prepare.units import prepare
        result=prepare(args.graph,args.output)
    elif args.command in ('build-probes','build-domain'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from yolo_transfer.prepare.build import build
        result=build(args.graph,args.workspace,args.local_config,names=args.program,full_domain=args.command=='build-domain')
    elif args.command in ('prepare-reference','board-reference'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from yolo_transfer import references
        if args.command=='prepare-reference':
            result=references.prepare(args.build,args.output,args.parent)
        else:
            result=references.run(args.bundle,args.run_dir,args.local_config,execute_board=args.execute_board)
    elif args.command in ('prepare-profile','board-profile','board-traffic'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from yolo_transfer import profile
        if args.command=='prepare-profile':
            result=profile.prepare(args.build,args.references,args.parent,args.output,repair_build=args.repair_build)
        elif args.command=='board-profile':
            result=profile.run(args.bundle,args.run_dir,args.local_config,execute_board=args.execute_board,limit=args.limit)
        else:
            from yolo_transfer import traffic
            result=traffic.run(args.bundle,args.run_dir,args.local_config,execute_board=args.execute_board,limit=args.limit)
    elif args.command in ('prepare-pipeline','board-pipeline'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from yolo_transfer import pipeline
        if args.command=='prepare-pipeline':
            result=pipeline.prepare(args.profile_bundle,args.ranked,args.output)
        else:
            result=pipeline.run(args.bundle,args.run_dir,args.local_config,execute_board=args.execute_board)
    elif args.command=='analyze':
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from yolo_transfer.analysis import analyze
        result=analyze(args.run_dir)
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
        from yolo_transfer.ranking import run
        result=run(args.bundle,args.profile_run,args.traffic_run,args.hardware_source,args.protocol,args.output)
    print(json.dumps(result))

if __name__=='__main__':main()
