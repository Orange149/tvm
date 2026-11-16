from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from strong_baseline import bridge as w
from cpu_vta_partitioning.common import read
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Exact 13FPS layout: legacy/current executor and classification/raw accounting')
    p.add_argument('command',choices=('prepare','qemu','validate','board','status'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board':v=w.board(execute_board=a.execute_board)
    elif a.command=='validate':v=w.validate(w.DEST/'capsule')
    elif a.command=='status':v=read(w.RUN/'state.json') if (w.RUN/'state.json').exists() else dict(status='NOT_STARTED')
    else:v=getattr(w,a.command)()
    print(json.dumps(v,ensure_ascii=False,indent=2))
