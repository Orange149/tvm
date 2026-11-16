#!/usr/bin/env python3
"""One finite zero-DNN SWITCH-record capability test, never DNN execution."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))

if __name__=='__main__':
    from scheduler_event_audit.workflow import OUTPUT,prepare,run,analyze
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','board','analyze','status'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='prepare':
        d=prepare(a.output);result=dict(prepared=True,QEMU_load_passed=d['QEMU_load_passed'],binary_sha256=d['binary_sha256'],DNN_inferences=0)
    elif a.command=='board':result=run(a.output,execute_board=a.execute_board)
    elif a.command=='analyze':result=analyze(a.output)
    else:
        from cpu_vta_partitioning.common import read
        d=read(a.output/'BOARD_SELF_TEST.json');result=dict(passed=d['validation']['passed'],boot_id=d['boot_id'],DNN_inferences=0)
    print(json.dumps(result,ensure_ascii=False))
