from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from dispatch_service import workflow

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Offline CPU dispatch-elapsed semantics check, no board commands')
    parser.add_argument('command',choices=('prepare','platform','replay','analyze'))
    parser.add_argument('--output',type=Path,default=workflow.OUTPUT)
    args=parser.parse_args()
    print(json.dumps(getattr(workflow,args.command)(args.output),ensure_ascii=False,indent=2))
