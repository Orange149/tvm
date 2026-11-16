"""Adopt a sealed unchanged build; start a fresh 24-call graph attempt."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback
from c3_tps_sync_release_local import HERE,PYTHON,read,write,sha,verify,require,env_for_worker
from c3_yolo416_graph_identity import compare
from c3_yolo416_graph_closeout import seal


def run(root,prepared,old,qualified,identity_fix,lifetime_fix):
    root.mkdir(exist_ok=False)
    try:
        require(not qualified.exists(),'QUALIFICATION_DIRECTORY_EXISTS')
        for folder in (prepared,old,identity_fix,lifetime_fix):
            m=read(folder/'completion_manifest.json')
            verify({str(folder/p):d for p,d in m['artifact_sha256'].items()})
        require(read(prepared/'status.json')['status']=='WHOLE_TH_P_GRAPHS_AND_HELPER_BUILT_LOCAL_QUALIFICATION_PENDING',
                'BUILD_NOT_SUCCESSFUL')
        require(read(identity_fix/'completion_manifest.json')['status']=='SEALED_GRAPH_IDENTITY_CONTRACT_FIX',
                'IDENTITY_FIX_NOT_QUALIFIED')
        require(read(lifetime_fix/'completion_manifest.json')['status']=='SEALED_CPU_ONLY_PLUGIN_LIFETIME_FIX',
                'LIFETIME_FIX_NOT_QUALIFIED')
        verify(read(prepared/'freeze.json')['bindings'])
        write(root/'target_unchanged.json',compare(old,prepared))
        write(root/'contract.json',{'prepared':str(prepared),'old_prepared':str(old),'qualified':str(qualified),
            'identity_fix':str(identity_fix),'lifetime_fix':str(lifetime_fix),
            'target_IR_and_LLVM_unchanged':True,'adopted_old_graph_calls':0,'new_graph_calls_limit':24,'FPGA_calls':0})
        with (root/'qualify.log').open('x') as log:
            code=subprocess.call([str(PYTHON),str(HERE/'c3_yolo416_graph_local_v2.py'),'run',str(qualified),
                 '--prepared',str(prepared)],env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT)
        ok=seal(qualified,'qualify')
        write(root/'qualify_result.json',{'returncode':code,'sealed_success':ok})
        require(code==0 and ok,'GRAPH_QUALIFICATION_FAILED_NO_RETRY')
        write(root/'result.json',{'status':'OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED',
              'new_graph_calls':24,'adopted_old_graph_calls':0,'FPGA_calls':0,'performance_samples':0})
        print('OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED',flush=True);return 0
    except Exception as e:
        write(root/'failure.json',{'type':type(e).__name__,'message':str(e),'FPGA_calls':0})
        traceback.print_exc();return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    for n in ('prepared','old','qualified','identity-fix','lifetime-fix'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();sys.exit(run(*(x.resolve() for x in
        (a.output,a.prepared,a.old,a.qualified,a.identity_fix,a.lifetime_fix))))
