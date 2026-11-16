"""Reuse the sealed host runner; verify three-stage masks in serial and pipeline."""
import argparse,json,os,re,sys
from pathlib import Path
from run_cpu_vta_core_partition_validation import load,read
from run_cpu_vta_residual_validation import verify_affinity
from thesis_supplement_run import sha256,write_new_json
from cpu_vta_residual_protocol import ROOT


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args();s=load(a.run_dir)
    d=s.directory/'raw/host_validation';d.mkdir();binary=s.directory/'raw/golden_all_stage_host_runner'
    fixture=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_residual_validation/20260916T135816Z_residual_pilot_v2/raw/host_validation/historical_fixture'
    cpus=sorted(os.sched_getaffinity(0))[:4]
    if len(cpus)<4:raise ValueError('Four host cores required')
    commands=[]
    for condition in ('S','D'):
        masks=[cpus[:3],cpus[:3] if condition=='S' else cpus[3:],cpus[:3]]
        # CPU fixture parser requires mask size == logical threads; VTA is exempt.
        counts_expected=[3,3 if condition=='S' else 1,3]
        for mode in ('serial','pipeline'):
            name=condition+'_'+mode;output=d/(name+'.jsonl');aff=d/(name+'.affinity.json')
            argv=[str(binary),'--input',str(fixture/'input.bin'),'--runs','80','--queue-depth','2','--output-jsonl',str(output),'--thread-affinity-report',str(aff),'--defer-result-write']
            for i in range(3):
                for k,v in {'graph':fixture/'graph.json','params':fixture/'params.bin','lib':fixture/'module.so','device':'cpu','name':f'cpu{i}',
                            'input-names':'data0','runtime-num-threads':str(counts_expected[i]),'cpu-affinity':','.join(map(str,masks[i]))}.items():argv+=[f'--stage{i}-'+k,str(v)]
                if i:argv+=[f'--stage{i}-input-sources','input:0']
            if mode=='serial':argv+=['--serial']
            else:argv+=['--cpu-stage-gate','exclusive','--stage-gate-scope','all','--vta-host-nice','0']
            r=s.execute(['env','TVM_NUM_THREADS=3','TVM_THREAD_POOL_SPIN_COUNT=0']+argv,'host_'+name,'host_validation',30);commands.append(r)
            if r['status']!='ok':raise ValueError((s.directory/r['stderr']).read_text())
            rows=[json.loads(l) for l in output.read_text().splitlines()]
            if [r['frame_id'] for r in rows]!=list(range(80)) or any(r['top1']!=4095 for r in rows):raise ValueError('CPU fixture correctness')
            verify_affinity(aff,{'affinity':masks},range(3),mode)
            counts={int(i):int(n) for i,n in re.findall(r'\[THREADPOOL\] '+mode+r'\.stage(\d+) runtime\.NumThreads=(\d+)',(s.directory/r['stdout']).read_text())}
            if counts!=dict(enumerate(counts_expected)):raise ValueError('Host actual active thread counts')
            if mode=='pipeline':
                if any(r[f'stage{i}_process_cpu_ms'] is not None for r in rows for i in range(3)):raise ValueError('Concurrent process CPU attribution')
                if any(t['nice']!=0 for t in read(str(aff)+'.priority.json')['threads']):raise ValueError('Host priority changed')
            if any('driver_snapshot' in k or 'monotonic_origin' in k for k in rows[0]):raise ValueError('Off schema drift')
    r=s.execute([sys.executable,'-m','pytest','-q',str(Path(__file__).with_name('test_cpu_vta_core_partition_validation.py'))],'partition_pytest','host_validation',60)
    if r['status']!='ok':raise ValueError('Partition tests failed')
    write_new_json(s.directory/'analysis/host_validation.json',{'passed':True,'CPU_frames':320,'CPU_fixture_thread_counts':{'S':[3,3,3],'D':[3,1,3]},'board_required_thread_counts':[3,1,3],
                   'serial_and_pipeline_actual_masks_verified':True,'host_runner_sha256':sha256(binary),'runner_compiled':False,'board_calls':0,'commands':commands,'pytest':r})
    print('Sealed runner: 320 CPU fixture frames, actual 3/1/3 and masks PASS',flush=True)


if __name__=='__main__':main()
