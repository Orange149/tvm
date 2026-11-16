"""Bounded serial archived-program census; never execute a compiled target."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil
from c3_uop_identity_applicability import HERE,REPO,HOLD,inventory,read,write,sha,require

PYTHON=Path('/home/orange/miniconda3/envs/vta-resnet/bin/python')


def main(root):
    root.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (root/'tests.log').open('x') as log:
        p=subprocess.run([str(PYTHON),'-m','unittest','-v','test_c3_uop_identity_applicability'],env=env,
                         stdout=log,stderr=subprocess.STDOUT,timeout=120)
    require(p.returncode==0,'READONLY_TOOL_PREFLIGHT_FAILED')
    objects=inventory();write(root/'inventory.json',objects)
    paths=[Path(__file__),HERE/'c3_uop_identity_applicability.py',HERE/'test_c3_uop_identity_applicability.py',
           HERE/'c3_official_window_method.py',HERE/'c3_valid_window.py',REPO/'build/libtvm.so',
           REPO/'src/target/llvm/codegen_cpu.cc',REPO/'vta/runtime/runtime.cc',
           HOLD/'20261014_c3_official_valid_window_run01/closure_final02/local_call_index.json',
           HOLD/'20261012_c3_banerjee_double_buffer_local_run01/author_sources/runtime.cc',
           HOLD/'20261017_c3_node34_uop_identity_board_run01/completion_manifest.json']
    paths += [Path(o['path']) for o in objects]
    bindings={str(p.resolve()):sha(p) for p in paths}
    write(root/'freeze.json',{'bindings':bindings,'target_calls':0,'compilations':0,'board_contact':False,
                              'scope':'archived programs only; no candidate construction or configuration search'})
    (root/'objects').mkdir();results=[]
    for index,row in enumerate(objects,1):
        name='%02d_%s'%(index,row['id']);source=root/'objects'/(name+'.input.json');write(source,row)
        out=root/'objects'/(name+'.json')
        argv=[str(PYTHON),str(HERE/'c3_uop_identity_applicability.py'),'--row',str(source),'--output',str(out)]
        start=time.monotonic();peak=0;limit=None
        with (root/'objects'/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                try:
                    proc=psutil.Process(child.pid)
                    peak=max(peak,sum(p.memory_info().rss for p in [proc,*proc.children(recursive=True)]))
                except psutil.Error:pass
                if peak>8*1024**3 or time.monotonic()-start>600:
                    limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(.1)
            rc=child.wait()
        write(root/'objects'/(name+'.worker.json'),{'returncode':rc,'limit':limit,'peak_tree_RSS':peak,
                                                   'resource_seconds':time.monotonic()-start,'target_calls':0})
        if limit:result={**row,'status':'ANALYSIS_RESOURCE_LIMIT','limit':limit}
        else:
            result=read(out)
            require(rc==0,'READONLY_WORKER_FAILED:'+name+':'+str(result))
        results.append(result)
        print(row['id'],result['status'],result.get('MAC_identities'),result.get('MAC_content_classes'),flush=True)
    for p,v in bindings.items():require(sha(p)==v,'SOURCE_CHANGED_DURING_CENSUS')
    write(root/'results.json',results)
    write(root/'status.json',{'status':'ARCHIVED_CENSUS_COMPLETE','objects':len(results),'target_calls':0,
                              'compilations':0,'board_contacted':False,'paused':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path)
    main(p.parse_args().output_dir.resolve())
