#!/usr/bin/env python3
"""Finite CPU service: zero DNN, no FPGA or system parameter changes."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':
    from finite_service.common import OUTPUT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','board','calibrate','validate'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command in ('prepare','board'):
        from finite_service import native
        result=getattr(native,a.command)(a.output,**({'execute_board':a.execute_board} if a.command=='board' else {}))
        if a.command=='prepare':result={k:result[k] for k in ('binary_sha256','QEMU_load_passed','protocol')}
    else:
        from finite_service import calibration
        result=getattr(calibration,'validate_mixed' if a.command=='validate' else 'calibrate')(a.output)
    print(json.dumps(result,ensure_ascii=False))
