"""Convenience entry for the frozen scoped-clock controller and its analysis.

Creates required local transfer directories before first deployment without
changing the measured controller, task identities or accepted old receipts.
"""
import argparse
import json
import run_scoped_thread_profile as frozen
import analyze_scoped_thread_effect as effect


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board':
        if not a.execute_board:raise ValueError('--execute-board required')
        for network in frozen.PARENTS:(frozen.RUN/network).mkdir(parents=True,exist_ok=True)
        value=frozen.board(execute_board=True)
    elif a.command=='status':value=frozen.read(frozen.RUN/'state.json')
    elif a.command=='analyze':value=effect.run()
    else:value=getattr(frozen,a.command)()
    if a.command!='analyze':print(json.dumps(value,ensure_ascii=False,indent=2))
