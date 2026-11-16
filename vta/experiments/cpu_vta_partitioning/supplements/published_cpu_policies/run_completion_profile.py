from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from completion_profile import workflow as w
from cpu_vta_partitioning.common import read
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Isolated VTA completion brackets; no pipeline FPS input')
    p.add_argument('command',choices=('prepare','validate','board','status'))
    p.add_argument('--output',type=Path,default=w.OUTPUT);p.add_argument('--run-dir',type=Path,default=w.RUN)
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='prepare':v=w.prepare(a.output)
    elif a.command=='validate':v={n:w.validate(a.output/n/'capsule') for n in read(a.output/'PLAN.json')['bundle_ids']}
    elif a.command=='board':v=w.board(a.output,a.run_dir,execute_board=a.execute_board)
    else:v=read(a.run_dir/'state.json') if (a.run_dir/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(v,ensure_ascii=False,indent=2))
