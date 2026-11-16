from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from polling_profile import workflow
from cpu_vta_partitioning.common import read
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Isolated VTA polling counters, zero complete DNN inferences')
    p.add_argument('command',choices=('prepare','validate','board','status'))
    p.add_argument('--output',type=Path,default=workflow.OUTPUT)
    p.add_argument('--run-dir',type=Path,default=workflow.RUN)
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='prepare':value=workflow.prepare(a.output)
    elif a.command=='validate':value={n:workflow.validate(a.output/n/'capsule') for n in workflow.protocol()['components']}
    elif a.command=='board':value=workflow.board(a.output,a.run_dir,execute_board=a.execute_board)
    else:value=read(a.run_dir/'state.json') if (a.run_dir/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(value,ensure_ascii=False,indent=2))
