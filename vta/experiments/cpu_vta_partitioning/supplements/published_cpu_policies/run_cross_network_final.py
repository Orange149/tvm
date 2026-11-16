#!/usr/bin/env python3
"""Read frozen ranking evidence, freeze long winners, and finish common reporting."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
if __name__=='__main__':
    import cross_network_final as final
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('features','short','long','long-results','report','audit'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='features':result=final.intrinsic_features()
    elif a.command=='short':result=final.short()
    elif a.command=='long':result=final.long(execute_board=a.execute_board)
    elif a.command=='long-results':result=final.long_results()
    elif a.command=='report':result=final.report()
    else:result=final.audit_delivery()
    print(json.dumps(result,ensure_ascii=False))
