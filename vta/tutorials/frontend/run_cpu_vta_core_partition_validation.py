"""Fixed 3/1/3 core partition pilot; reuse the sealed runner and runtime unchanged."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import shutil
import statistics as st
import uuid
import re

import run_cpu_vta_host_priority_validation as parent
import run_cpu_vta_residual_validation as native
from cpu_vta_residual_protocol import ROOT, verify_assets, matrix
from thesis_supplement_run import RunStore,sha256,utc,write_new_json
from analyze_cpu_vta_driver_snapshot_validation import analyze
from analyze_cpu_vta_host_priority_validation import verify_priority

SOURCE=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_host_priority_validation/20260917T140924Z_host_priority_v1'
VERSION='cpu_vta_core_partition_v1'
CONTROLS=(Path(__file__).name,'analyze_cpu_vta_core_partition_validation.py','test_cpu_vta_core_partition_validation.py',
          'validate_cpu_vta_core_partition_host.py','CPU_VTA_CORE_PARTITION_VALIDATION_RUNBOOK.md')+parent.CONTROLS


def read(path):return json.loads(Path(path).read_text())


def configuration(name,enabled=False):
    if name not in ('S','D','D_all'):raise ValueError('Unknown fixed condition')
    c=copy.deepcopy(matrix()[0])
    c.update(condition=name,threads=[3,1,3],affinity=[[0,1,2],[0,1,2] if name=='S' else [3],[0,1,2]],
             gate_mode='exclusive',gate_scope='all' if name=='D_all' else 'cpu',
             driver_snapshots=enabled,host_priority_enabled=True,vta_host_nice=0)
    return c


def freeze(directory):
    old=parent.load(SOURCE)
    if not read(SOURCE/'analysis/FINAL_DELIVERY_STATUS.json')['completed']:raise ValueError('Completed source required')
    seal=read(SOURCE/'analysis/FINAL_CONCLUSION_ZH.assets.json')
    for key,v in seal.items():
        if sha256(SOURCE/key)!=v['sha256']:raise ValueError('Parent sealed evidence drift '+key)
    c=copy.deepcopy(old.contract)
    for key,v in c['assets'].items():v['source_path']=str(SOURCE/key)
    prior=read(SOURCE/'analysis/runner_binding.json')
    additions={'raw/frozen_priority_runner':SOURCE/prior['path'],
               'raw/golden_all_stage_host_runner':SOURCE/'raw/host_validation/driver_snapshot_host_runner',
               'raw/parent_evidence/FINAL_DELIVERY_STATUS.json':SOURCE/'analysis/FINAL_DELIVERY_STATUS.json',
               'raw/parent_evidence/FINAL_CONCLUSION_ZH.md':SOURCE/'analysis/FINAL_CONCLUSION_ZH.md',
               'raw/parent_evidence/FINAL_PAIRED_STATISTICS.json':SOURCE/'analysis/FINAL_PAIRED_STATISTICS.json',
               'raw/parent_evidence/FINAL_CONCLUSION_ZH.assets.json':SOURCE/'analysis/FINAL_CONCLUSION_ZH.assets.json'}
    additions.update({'raw/partition_controller_sources/'+n:Path(__file__).with_name(n).resolve() for n in CONTROLS})
    for key,p in additions.items():c['assets'][key]={'source_path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size}
    c.update(protocol_version=VERSION,run_id=Path(directory).name,stage='cpu_vta_core_partition',
             campaign_id='core_partition_'+uuid.uuid4().hex[:12],created_at=utc(),parent_completed_run=str(SOURCE),
             inference='New 3/1/3 fixed affinity causal contrast; one boot exploratory, prior boots do not replace it',
             threads=[3,1,3],conditions={n:configuration(n) for n in ('S','D','D_all')},
             snapshot_orders=[[['S',False],['S',True],['D',False],['D',True]],
                              [['D',True],['D',False],['S',True],['S',False]]],
             treatment={'S':'VTA host shares CPU cores0-2','D':'VTA host dedicated core3'},
             all_stage_diagnostic={'trigger':'D CPU front or tail mean run exceeds matched before/after D serial reference by >5%',
                                   'scope':'same dedicated masks and 3/1/3, only gate scope cpu/all',
                                   'orders':[['D','D_all'],['D_all','D']],
                                   'performance_snapshots':False,'correctness_frames':10},
             matching='S and D serial full graph before and after; service reference follows each actual affinity',
             priority_scope='nice=0 in every condition; initialized process task audit and per-frame caller check',
             causal_threshold=.05,serial_drift_threshold=.05,perturbation_threshold=.02,
             performance_source='snapshot off; on diagnostic, calibrated within each affinity',
             process_affinity='inherit four-core host mask; do not pass global process mask that overwrites stage masks',
             hardware_compute_only='unknown',physical_DDR_internal_DMA='unknown',formal_static_score_changed=False,
             runner_source_modified=False,max_kernel_boots=3)
    s=RunStore.create(directory,c)
    for key,v in c['assets'].items():
        p=s.directory/key;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(v['source_path'],p)
        if sha256(p)!=v['sha256']:raise ValueError('Asset copy drift '+key)
    runtime={name:{'host_path':str(s.directory/'raw/runtime'/name),'sha256':c['assets']['raw/runtime/'+name]['sha256'],
                   'bytes':c['assets']['raw/runtime/'+name]['bytes']} for name in ('libtvm_runtime.so','libvta.so','tvm_rpc','u-dma-buf.ko','vta_hpc.bit')}
    write_new_json(s.directory/'analysis/board_artifact_bindings.json',runtime)
    write_new_json(s.directory/'analysis/runner_binding.json',{'path':'raw/frozen_priority_runner','sha256':prior['sha256'],
                  'runtime_bindings':runtime,'derived_source_sha256':prior['derived_source_sha256'],
                  'runner_source_modified':False,'copied_from':str(SOURCE/prior['path'])})
    s.event('core_partition_protocol_frozen',verified_parent_seal_files=len(seal),runner_reused_sha256=prior['sha256'])
    return s


def load(directory):
    s=RunStore(directory)
    if s.contract['protocol_version']!=VERSION:raise ValueError('Wrong protocol')
    verify_assets(s)
    for name in CONTROLS:
        v=s.contract['assets']['raw/partition_controller_sources/'+name]
        if sha256(v['source_path'])!=v['sha256']:raise ValueError('Live controller drift '+name)
    return s


def build(s):
    b=read(s.directory/'analysis/runner_binding.json')
    if sha256(s.directory/b['path'])!=b['sha256']:raise ValueError('Runner drift')
    return b


@contextmanager
def adapters():
    with parent.adapters():
        before=native.build_runner;native.build_runner=build
        try:yield
        finally:native.build_runner=before


def session(s,boot,name,mode,block,enabled=False):
    load(s.directory);c=configuration(name,enabled)
    if mode in ('serial','correctness_serial'):c.update(host_priority_enabled=False,gate_mode='off',driver_snapshots=False)
    r=native.session(s,boot,c,mode,block,profile=False)
    p=s.directory/'raw/sessions'/r['session_id'];text=(p/'native.stdout').read_text()
    if '[VTA_QUEUE]' in (p/'native.stderr').read_text() or '[C3_COMPLETION]' in (p/'native.stderr').read_text():raise ValueError('Submit logging forbidden')
    if '--process-cpu-affinity' in r['argv']:raise ValueError('Global process mask could overwrite phase affinity')
    kind='serial' if mode in ('serial','correctness_serial') else 'pipeline'
    configured={int(i):int(n) for i,n in re.findall(r'\[THREADPOOL\] '+kind+r'\.stage(\d+) runtime\.NumThreads=(\d+)',text)}
    if configured!={0:3,1:1,2:3}:raise ValueError('Actual runtime thread count drift')
    if kind=='pipeline':
        rows=[json.loads(l) for l in (p/'result.jsonl').read_text().splitlines()]
        r['snapshot_metrics']=analyze(rows,enabled,c['gate_scope'],7 if mode=='pipeline' else 0)
        r['gate_metrics']=r['snapshot_metrics']['gate_metrics']
        r['priority_audit']=verify_priority(rows,0,read(p/'affinity.json.priority.json'),read(p/'affinity.json'))
    r.update(driver_snapshots_enabled=enabled,gate_scope=c['gate_scope'],actual_runtime_NumThreads=configured,
             allocated_pool_threads=[len(x['thread_ids']) for x in r['affinity_report']['stages']])
    write_new_json(p/'partition_summary.json',r);return r


def execute(s,replicate=False):
    from analyze_cpu_vta_core_partition_validation import dedicated_inflation, serial_drift
    existing=sorted((s.directory/'analysis').glob('boot_[123].json'))
    if (s.directory/'invalid_run.json').exists() or (existing and not replicate) or len(existing)>=3:raise ValueError('No retry or extra boot')
    if replicate and not existing:raise ValueError('Pilot required before replicate')
    if not read(s.directory/'analysis/host_validation.json')['passed']:raise ValueError('Host qualification required')
    boot=native.preflight(s);boot['boot_index']=len(existing)+1
    if any(read(p)['boot_id']==boot['boot_id'] for p in existing):raise ValueError('Independent kernel boot required')
    with adapters():
        native.deploy(s,boot,build(s))
        try:
            for name in ('S','D'):
                session(s,boot,name,'correctness_serial','qualification')
                session(s,boot,name,'correctness_pipeline','qualification',True)
            before={n:session(s,boot,n,'serial','matched_before') for n in ('S','D')}
            perf=[]
            for i,order in enumerate(s.contract['snapshot_orders']):
                for n,on in order:perf.append(session(s,boot,n,'pipeline',f'pair{i}_'+('on' if on else 'off'),on))
            checkpoint={n:session(s,boot,n,'serial','matched_checkpoint') for n in ('S','D')}
            check=serial_drift(before,checkpoint)
            write_new_json(s.directory/f'analysis/serial_checkpoint_boot{boot["boot_index"]}.json',check)
            if not check['passed']:raise ValueError('Serial checkpoint drift; diagnostic stopped')
            inflation=dedicated_inflation(before['D'],checkpoint['D'],[r for r in perf if r['condition']=='D' and not r['driver_snapshots_enabled']])
            write_new_json(s.directory/f'analysis/diagnostic_trigger_boot{boot["boot_index"]}.json',inflation)
            if inflation['trigger']:
                session(s,boot,'D_all','correctness_pipeline','diagnostic_qualification',False)
                for i,order in enumerate(s.contract['all_stage_diagnostic']['orders']):
                    for n in order:session(s,boot,n,'pipeline',f'coactivity{i}_off',False)
                after={n:session(s,boot,n,'serial','matched_after') for n in ('S','D')}
            else:after=checkpoint
            drift={n:[after[n]['metrics']['stages'][i]['run_ms']['mean']/before[n]['metrics']['stages'][i]['run_ms']['mean']-1 for i in range(3)] for n in ('S','D')}
            passed=max(abs(v) for values in drift.values() for v in values)<=.05
            write_new_json(s.directory/f'analysis/serial_drift_boot{boot["boot_index"]}.json',{'fractions':drift,'passed':passed,'before':before,'after':after})
            if not passed:raise ValueError('Serial service drift')
        except Exception as e:
            write_new_json(s.directory/'invalid_run.json',{'error':str(e),'raw_preserved':True,'silent_retry_allowed':False});raise
        finally:
            native.cleanup_rpc(s,boot)
            r=native.script_action(s,'cd '+shlex.quote(boot['board_directory'])+'''
p=$(cat rpc.pid)
test ! -e /proc/$p/exe
test ! -d sessions || test -z "$(ls -A sessions)"
printf 'RPC_STOPPED %s\\n' "$p"
printf 'FIRMWARE %s\\n' "$(readlink /lib/firmware/vta_hpc.bit)"
cat rpc.stdout rpc.stderr
''','verify_owned_cleanup',30)
            write_new_json(s.directory/f'analysis/owned_cleanup_boot{boot["boot_index"]}.json',{'passed':True,'action':r})


def report(s):
    from analyze_cpu_vta_core_partition_validation import summarize
    d=s.directory/'analysis'/('report_'+uuid.uuid4().hex[:10]);d.mkdir()
    xs=[read(p) for p in sorted((s.directory/'raw/sessions').glob('*/partition_summary.json'))]
    result=summarize(xs);result.update(no_invalid_run=not (s.directory/'invalid_run.json').exists(),historical_34ms_closed=False,hardware_compute_only='unknown',physical_DDR_internal_DMA='unknown')
    write_new_json(d/'analysis.json',result)
    print(d,flush=True);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=('freeze','pilot','replicate','analyze'));p.add_argument('--run-dir',required=True);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.phase in ('pilot','replicate') and not a.execute_board:p.error('--execute-board required')
    s=freeze(a.run_dir) if a.phase=='freeze' else load(a.run_dir)
    if a.phase in ('pilot','replicate'):execute(s,a.phase=='replicate');report(s)
    elif a.phase=='analyze':report(s)


if __name__=='__main__':main()
