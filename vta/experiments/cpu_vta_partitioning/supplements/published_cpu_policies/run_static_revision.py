#!/usr/bin/env python3
"""Evidence-led shared CPU–VTA static-model revision."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':
    from static_revision.common import OUTPUT
    from static_revision import workflow
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('audit','replay','analyze','run','status'))
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    if a.command=='status':
        from cpu_vta_partitioning.common import read
        result=read(a.output/'GOAL_PROGRESS.json')
    elif a.command=='run':
        result={}
        for action in ('audit','replay','analyze'):result[action]=getattr(workflow,action)(a.output)
    else:result=getattr(workflow,a.command)(a.output)
    print(json.dumps(result,ensure_ascii=False))
