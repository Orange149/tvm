from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from recent_load import workflow
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Offline recent-load host placement, no board commands')
    p.add_argument('command',choices=('prepare','replay','analyze'))
    p.add_argument('--output',type=Path,default=workflow.OUTPUT);a=p.parse_args()
    print(json.dumps(getattr(workflow,a.command)(a.output),ensure_ascii=False,indent=2))
