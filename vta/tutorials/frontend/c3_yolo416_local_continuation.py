"""Continue once from a running operator attempt through local graph gates.

Never retries or opens another attempt after a failure. No ARM/FPGA entry.
All subordinate output directories must be absent before this monitor starts.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
from c3_tps_sync_release_local import HERE,PYTHON,read,write,sha,require,env_for_worker
from c3_yolo416_local_closeout import seal_operators
from c3_yolo416_graph_closeout import seal as seal_graph


def continue_local(root,operators,source,prepared,qualified):
    root.mkdir(exist_ok=False)
    try:
        require(not prepared.exists() and not qualified.exists(),'SUBORDINATE_DIRECTORY_ALREADY_EXISTS')
        require(not (operators/'completion_manifest.json').exists(),'OPERATOR_ATTEMPT_ALREADY_SEALED')
        write(root/'contract.json',{'operators':str(operators),'source':str(source),
              'graph_prepared':str(prepared),'graph_qualified':str(qualified),
              'new_operator_attempts':0,'graph_attempts_per_stage':1,
              'graph_target_calls_max':24,'FPGA_calls':0,'retry_allowed':False,
              'source_sha256':sha(__file__)})
        start=time.monotonic()
        while not (operators/'status.json').exists():
            require(time.monotonic()-start<21600,'OPERATOR_CONTROLLER_STATUS_NOT_OBSERVED')
            time.sleep(5)
        ok=seal_operators(operators)
        require(ok,'OPERATOR_ATTEMPT_STOPPED_NO_GRAPH_EXECUTION')
        stages=[('graph_prepare','c3_yolo416_graph_prepare.py',
                 ['prepare',str(prepared),'--source',str(source),'--qualified',str(operators)]),
                ('graph_qualify','c3_yolo416_graph_local_v2.py',
                 ['run',str(qualified),'--prepared',str(prepared)])]
        for name,script,args in stages:
            with (root/(name+'.log')).open('x') as log:
                code=subprocess.call([str(PYTHON),str(HERE/script),*args],env=env_for_worker(),
                                     stdout=log,stderr=subprocess.STDOUT)
            folder=prepared if name=='graph_prepare' else qualified
            ok=seal_graph(folder,'prepare' if name=='graph_prepare' else 'qualify')
            write(root/(name+'_result.json'),{'returncode':code,'sealed_success':ok,
                  'output':str(folder),'FPGA_calls':0})
            require(code==0 and ok,name.upper()+'_STOPPED_NO_RETRY')
        write(root/'result.json',{'status':'OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED',
              'FPGA_calls':0,'performance_samples':0})
        print('OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED',flush=True)
        return 0
    except Exception as e:
        write(root/'failure.json',{'type':type(e).__name__,'message':str(e),'FPGA_calls':0})
        traceback.print_exc();return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--operators',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--qualified',type=Path,required=True);a=p.parse_args()
    sys.exit(continue_local(*(p.resolve() for p in (a.output,a.operators,a.source,a.prepared,a.qualified))))
