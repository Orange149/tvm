"""Compile frozen derived source; verify off schema, CPU compatibility and probes."""
import argparse
import json
import os
import sys
from pathlib import Path
from run_cpu_vta_driver_snapshot_validation import load, build, ROOT
from thesis_supplement_run import sha256, write_new_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args()
    s=load(a.run_dir);d=s.directory/'raw/host_validation';d.mkdir()
    src=s.directory/'raw/runner_sources';binary=d/'driver_snapshot_host_runner'
    inc=[src]+[ROOT/p for p in ('include','3rdparty/dlpack/include','3rdparty/dmlc-core/include','3rdparty/vta-hw/include','vta/include')]
    command=['g++','-std=c++17','-O2',str(src/'vta_stage_pipeline_runner.cc'),'-o',str(binary)]
    for path in inc:command+=['-I',str(path)]
    command+=['-L',str(ROOT/'build'),'-Wl,-rpath,'+str(ROOT/'build'),'-ltvm_runtime','-lvta_fsim','-ldl','-pthread']
    def checked(argv,name,timeout=120):
        r=s.execute(argv,name,'host_validation',timeout)
        if r['status']!='ok':raise RuntimeError(name+' failed: '+(s.directory/r['stderr']).read_text()[-3000:])
        return r
    checked(command,'compile_snapshot_host')
    fixture=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_residual_validation/20260916T135816Z_residual_pilot_v2/raw/host_validation/historical_fixture'
    cpus=sorted(os.sched_getaffinity(0))[:4]
    def run(name,runner,flags):
        output=d/(name+'.jsonl')
        argv=[str(runner),'--input',str(fixture/'input.bin'),'--runs','80','--queue-depth','2','--output-jsonl',str(output)]
        for i in range(2):
            prefix=f'--stage{i}-'
            for key,value in {'graph':fixture/'graph.json','params':fixture/'params.bin','lib':fixture/'module.so',
                              'device':'cpu','name':f'cpu{i}','input-names':'data0','runtime-num-threads':'2',
                              'cpu-affinity':','.join(map(str,cpus[:2]))}.items():argv+=[prefix+key,str(value)]
        argv+=['--stage1-input-sources','input:0']+flags
        checked(['env','TVM_NUM_THREADS=2','TVM_THREAD_POOL_SPIN_COUNT=0']+argv,'host_'+name,30)
        rows=[json.loads(l) for l in output.read_text().splitlines()]
        assert [r['frame_id'] for r in rows]==list(range(80)) and all(r['top1']==4095 for r in rows)
        assert all(r[f'stage{i}_process_cpu_ms'] is None for r in rows for i in range(2))
        return rows
    default=run('default',binary,[])
    golden=run('golden_default',s.directory/'raw/golden_all_stage_host_runner',[])
    assert set(default[0])==set(golden[0])
    assert not any('driver_snapshot' in k or 'monotonic_origin' in k for k in default[0])
    enabled=run('cpu_enabled',binary,['--vta-driver-snapshots','--cpu-stage-gate','exclusive','--stage-gate-scope','all','--defer-result-write'])
    assert all(r[f'stage{i}_driver_snapshot'] is None for r in enabled for i in range(2))
    assert all('benchmark_monotonic_origin_ms' in r for r in enabled)
    test=d/'driver_snapshot_counter_check'
    checked(['g++','-std=c++17','-O2','-pthread','-I',str(src),str(s.directory/'raw/snapshot_controller_sources/test_vta_driver_snapshot.cc'),'-o',str(test)],'compile_counter_test',60)
    checked([str(test)],'counter_delta_and_actual_thread_probes',30)
    tests=checked([sys.executable,'-m','pytest','-q',str(Path(__file__).with_name('test_cpu_vta_driver_snapshot_validation.py'))],'pytest_snapshot_analyzer',60)
    build(s)
    write_new_json(s.directory/'analysis/host_validation.json',{'passed':True,'CPU_frames':240,'default_schema_equal_golden':True,
                   'actual_thread_clocks_and_schedstat_test_passed':True,'decreasing_and_nonfinite_counters_rejected':True,
                   'pytest':tests,'host_runner_sha256':sha256(binary),'board_calls':0})
    print('Host compatibility, actual thread probes, analyzer and cross build PASS',flush=True)


if __name__=='__main__':main()
