"""Independent, opt-in runner-only driver/schedstat snapshot experiment."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import statistics as st
import uuid

import run_cpu_vta_all_stage_gate_validation as previous
import run_cpu_vta_residual_validation as native
from analyze_cpu_vta_driver_snapshot_validation import analyze
from cpu_vta_residual_protocol import ROOT, matrix, verify_assets
from thesis_supplement_run import RunStore, sha256, utc, write_new_json

SOURCE=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_all_stage_gate_validation/20260917T121038Z_all_stage_v1'
PROBE=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_driver_snapshot_validation/20260917T134200Z_scheduler_readonly_probe'
VERSION='cpu_vta_driver_snapshot_v1'
CONTROLS=(Path(__file__).name,'analyze_cpu_vta_driver_snapshot_validation.py',
          'test_cpu_vta_driver_snapshot_validation.py','validate_cpu_vta_driver_snapshot_host.py',
          'test_vta_driver_snapshot.cc','test_cpu_vta_all_stage_gate_validation.py',
          'run_cpu_vta_all_stage_gate_validation.py','analyze_cpu_vta_all_stage_gate_validation.py',
          'analyze_cpu_stage_overlap_validation.py')+native.CONTROL_FILES


def read(p):return json.loads(Path(p).read_text())


def derive_runner(source):
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('Runner anchor drift: '+old[:80])
        source=source.replace(old,new)
    replace('#include "vta_cpu_stage_gate.h"','#include "vta_cpu_stage_gate.h"\n#include "vta_driver_snapshot.h"')
    replace('namespace {\n','namespace {\nbool low_driver_snapshots = false;\n')
    replace('  bool defer_result_write = false;','  bool defer_result_write = false;\n  bool driver_snapshots = false;')
    replace('    } else if (key == "--defer-result-write") {','    } else if (key == "--vta-driver-snapshots") {\n      args.driver_snapshots = true;\n    } else if (key == "--defer-result-write") {')
    replace('  bool cpu_gate_timing_valid{false};','  bool cpu_gate_timing_valid{false};\n  vta_deploy::DriverRunSnapshot driver_snapshot;')
    replace('''  timing->run_start_ms = timing->set_end_ms;
  if (hooks != nullptr) {''','''  const bool snapshot = low_driver_snapshots && stage->device == "vta";
  if (snapshot) {
    timing->driver_snapshot.Begin();
    timing->set_end_ms = NowMillis();
  }
  timing->run_start_ms = timing->set_end_ms;
  if (hooks != nullptr) {''')
    replace('''  if (hooks != nullptr && hooks->finished) hooks->finished(timing->get_start_ms);
  CheckStableStageOutputs(stage,''','''  if (hooks != nullptr && hooks->finished) hooks->finished(timing->get_start_ms);
  if (snapshot) timing->driver_snapshot.End(timing->get_start_ms);
  CheckStableStageOutputs(stage,''')
    replace('    Args args = ParseArgs(argc, argv);','    Args args = ParseArgs(argc, argv);\n    low_driver_snapshots = args.driver_snapshots;')
    replace('  if (args.output_mode == "classification") {\n    os << ",\\"top1\\":" << frame.top1;',
            '  if (args.driver_snapshots) os << ",\\"benchmark_monotonic_origin_ms\\":" << benchmark_start_ms;\n'
            '  if (args.output_mode == "classification") {\n    os << ",\\"top1\\":" << frame.top1;')
    replace('''    const StageTiming& timing = frame.stage_timings[i];
    os <<''','''    const StageTiming& timing = frame.stage_timings[i];
    if (args.driver_snapshots) {
      os << ",\\"stage" << i << "_driver_snapshot\\":";
      timing.driver_snapshot.WriteJSON(os);
    }
    os <<''')
    return source


def freeze(directory):
    old=previous.load(SOURCE)
    if read(SOURCE/'analysis/FINAL_DELIVERY_STATUS.json')['independent_boots']!=3:raise ValueError('Completed source required')
    c=copy.deepcopy(old.contract)
    for key,v in c['assets'].items():v['source_path']=str(SOURCE/key)
    derived=derive_runner((SOURCE/'raw/runner_sources/vta_stage_pipeline_runner.cc').read_text())
    v=c['assets']['raw/runner_sources/vta_stage_pipeline_runner.cc']
    v.update(derived_from_sha256=v['sha256'],sha256=hashlib.sha256(derived.encode()).hexdigest(),bytes=len(derived.encode()))
    additions={
        'raw/runner_sources/vta_driver_snapshot.h':ROOT/'vta/apps/native_deploy/vta_driver_snapshot.h',
        'raw/runner_sources/vta/driver.h':ROOT/'3rdparty/vta-hw/include/vta/driver.h',
        'raw/golden_all_stage_host_runner':SOURCE/'raw/host_validation/cpu_stage_gate_host_runner'}
    additions.update({'raw/snapshot_controller_sources/'+name:Path(__file__).with_name(name).resolve() for name in CONTROLS})
    additions.update({'raw/scheduler_probe/'+str(p.relative_to(PROBE)):p for p in PROBE.rglob('*') if p.is_file()})
    for key,p in additions.items():c['assets'][key]={'source_path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size}
    c.update(protocol_version=VERSION,run_id=Path(directory).name,stage='cpu_vta_driver_snapshot',
             campaign_id='driver_snapshot_'+uuid.uuid4().hex[:12],created_at=utc(),parent_completed_run=str(SOURCE),
             inference='One boot exploratory counter mechanism; prior concurrency mechanism has three independent boots',
             snapshot_orders=[[['B',False],['B',True],['C',False],['C',True]],
                              [['C',True],['C',False],['B',True],['B',False]]],
             perturbation_threshold=.02,scheduler_trace_available=False,
             scheduler_capability='CONFIG_FTRACE=n CONFIG_SCHEDSTATS=n CONFIG_SCHED_INFO=y CONFIG_TASK_DELAY_ACCT=y; schedstat data capability checked in actual runs',
             driver_snapshot_api='Existing VTADriverProfilerStatus, no runtime/driver changes',
             cpu_counter='CLOCK_THREAD_CPUTIME_ID of single VTA host worker; not process or all stage workers',
             snapshot_scope='Probe bracket includes before/after overhead. Driver partition and thread CPU/runqueue counters overlap, cannot sum them.',
             software_timing='same steady-clock absolute benchmark origin and snapshot brackets; no fitted clock offset',
             formal_static_score_changed=False,actual_fpga_frequency='unknown')
    s=RunStore.create(directory,c)
    for key,v in c['assets'].items():
        p=s.directory/key;p.parent.mkdir(parents=True,exist_ok=True)
        if key=='raw/runner_sources/vta_stage_pipeline_runner.cc':
            with p.open('x') as f:f.write(derived)
        else:shutil.copy2(v['source_path'],p)
        if sha256(p)!=v['sha256']:raise ValueError('Copy drift '+key)
    runtime={name:{'host_path':str(s.directory/'raw/runtime'/name),
                   'sha256':c['assets']['raw/runtime/'+name]['sha256'],'bytes':c['assets']['raw/runtime/'+name]['bytes']}
             for name in ('libtvm_runtime.so','libvta.so','tvm_rpc','u-dma-buf.ko','vta_hpc.bit')}
    write_new_json(s.directory/'analysis/board_artifact_bindings.json',runtime)
    s.event('driver_snapshot_protocol_frozen',parent=str(SOURCE))
    return s


def load(directory):
    s=RunStore(directory)
    if s.contract['protocol_version']!=VERSION:raise ValueError('Wrong protocol')
    verify_assets(s)
    for name in CONTROLS:
        v=s.contract['assets']['raw/snapshot_controller_sources/'+name]
        if sha256(v['source_path'])!=v['sha256']:raise ValueError('Live controller drift '+name)
    for key in ('raw/runner_sources/vta_driver_snapshot.h','raw/runner_sources/vta/driver.h'):
        v=s.contract['assets'][key]
        if sha256(v['source_path'])!=v['sha256']:raise ValueError('Live header drift '+key)
    return s


def build(s):
    p=s.directory/'analysis/runner_binding.json'
    if p.exists():
        b=read(p)
        if sha256(s.directory/b['path'])!=b['sha256']:raise ValueError('Runner drift')
        return b
    d=s.directory/'raw/build';d.mkdir();output=d/'vta_stage_pipeline_runner_driver_snapshots'
    src=s.directory/'raw/runner_sources'
    inc=[src]+[ROOT/p for p in ('include','3rdparty/dlpack/include','3rdparty/dmlc-core/include','3rdparty/vta-hw/include','vta/include')]
    script=d/'build.sh'
    with script.open('x') as f:f.write('unset LD_LIBRARY_PATH\nsource '+shlex.quote(native.SDK)+'\n'+
        'aarch64-xilinx-linux-g++ -std=c++17 -O2 --sysroot="$SDKTARGETSYSROOT" '+
        ' '.join('-I '+shlex.quote(str(p)) for p in inc)+' '+shlex.quote(str(src/'vta_stage_pipeline_runner.cc'))+
        ' -o '+shlex.quote(str(output))+' -L '+shlex.quote(str(s.directory/'raw/runtime'))+
        ' -Wl,-rpath-link,"$SDKTARGETSYSROOT/lib" -Wl,-rpath-link,"$SDKTARGETSYSROOT/usr/lib" -ltvm_runtime -lvta -ldl -pthread\n')
    native.checked(s,['bash',str(script)],'compile_frozen_snapshot_runner',300)
    b={'path':str(output.relative_to(s.directory)),'sha256':sha256(output),
       'runtime_bindings':read(s.directory/'analysis/board_artifact_bindings.json'),
       'derived_source_sha256':sha256(src/'vta_stage_pipeline_runner.cc'),'driver_snapshots_default':False,
       'runtime_driver_modified':False}
    write_new_json(p,b);return b


@contextmanager
def adapters():
    with previous.adapters():
        old_build,old_argv=native.build_runner,native.runner_argv
        def argv(board,output,stages,c,mode,runs,telemetry=False,profile=False):
            if profile:raise ValueError('Submit diagnostics forbidden in low snapshot experiment')
            result=old_argv(board,output,stages,c,mode,runs,telemetry,False)
            if c.get('driver_snapshots'):result+=['--vta-driver-snapshots']
            return result
        native.build_runner,native.runner_argv=build,argv
        try:yield
        finally:native.build_runner,native.runner_argv=old_build,old_argv


def session(s,boot,name,mode,block,enabled=False):
    load(s.directory)
    c=copy.deepcopy(matrix()[0]);c.update(condition=name,gate_mode='exclusive' if name in ('B','C') else 'off',
                                       gate_scope='all' if name=='C' else 'cpu',driver_snapshots=enabled)
    r=native.session(s,boot,c,mode,block,profile=False)
    p=s.directory/'raw/sessions'/r['session_id']
    if '[VTA_QUEUE]' in (p/'native.stderr').read_text() or '[C3_COMPLETION]' in (p/'native.stderr').read_text():
        raise ValueError('Unexpected submission logging')
    if mode in ('pipeline','correctness_pipeline'):
        rows=[json.loads(l) for l in (p/'result.jsonl').read_text().splitlines()]
        r['snapshot_metrics']=analyze(rows,enabled,c['gate_scope'],7 if mode=='pipeline' else 0)
        r['gate_metrics']=r['snapshot_metrics']['gate_metrics']
    r['driver_snapshots_enabled']=enabled
    write_new_json(p/'snapshot_summary.json',r)
    return r


def execute(s):
    if (s.directory/'invalid_run.json').exists() or (s.directory/'analysis/boot_1.json').exists():raise ValueError('No resume/retry in same protocol')
    if not read(s.directory/'analysis/host_validation.json')['passed']:raise ValueError('Host qualification required')
    boot=native.preflight(s);boot['boot_index']=1
    with adapters():
        native.deploy(s,boot,build(s))
        try:
            session(s,boot,'serial','correctness_serial','qualification')
            for name in ('B','C'):session(s,boot,name,'correctness_pipeline','qualification',True)
            before=session(s,boot,'serial','serial','matched_before')
            for i,order in enumerate(s.contract['snapshot_orders']):
                for name,enabled in order:session(s,boot,name,'pipeline',f'pair{i}_'+('on' if enabled else 'off'),enabled)
            after=session(s,boot,'serial','serial','matched_after')
            drift=[after['metrics']['stages'][i]['run_ms']['mean']/before['metrics']['stages'][i]['run_ms']['mean']-1 for i in range(3)]
            write_new_json(s.directory/'analysis/serial_drift.json',{'before':before,'after':after,'fractions':drift,'passed':max(map(abs,drift))<=.05})
            if max(map(abs,drift))>.05:raise ValueError('Serial service drift')
        except Exception as e:
            write_new_json(s.directory/'invalid_run.json',{'error':str(e),'raw_preserved':True,'silent_retry_allowed':False});raise
        finally:
            native.cleanup_rpc(s,boot)
            action=native.script_action(s,'cd '+shlex.quote(boot['board_directory'])+'''
residual_pid=$(cat rpc.pid)
test ! -e /proc/$residual_pid/exe
test ! -d sessions || test -z "$(ls -A sessions)"
printf 'RPC_STOPPED %s\\n' "$residual_pid"
printf 'FIRMWARE %s\\n' "$(readlink /lib/firmware/vta_hpc.bit)"
cat rpc.stdout rpc.stderr
''','verify_owned_cleanup',30)
            write_new_json(s.directory/'analysis/owned_cleanup.json',{'passed':True,'action':action})


def report(s):
    out=s.directory/'analysis'/('report_'+uuid.uuid4().hex[:10]);out.mkdir()
    sessions=[read(p) for p in sorted((s.directory/'raw/sessions').glob('*/snapshot_summary.json'))]
    perf=[r for r in sessions if r['mode']=='pipeline'];pairs=[]
    for i in range(2):
        for n in ('B','C'):
            arms={r['driver_snapshots_enabled']:r for r in perf if r['condition']==n and r['block'].startswith(f'pair{i}_')}
            if len(arms)!=2:continue
            pairs.append({'block':i,'condition':n,
                          'ii_change_fraction':arms[True]['metrics']['ii_ms']/arms[False]['metrics']['ii_ms']-1,
                          'vta_run_change_fraction':arms[True]['gate_metrics']['stages'][1]['run_mean_ms']/arms[False]['gate_metrics']['stages'][1]['run_mean_ms']-1})
    qualified=len(pairs)==4 and max(max(abs(p['ii_change_fraction']),abs(p['vta_run_change_fraction'])) for p in pairs)<=.02
    means={}
    for n in ('B','C'):
        xs=[r['snapshot_metrics']['means'] for r in perf if r['condition']==n and r['driver_snapshots_enabled']]
        if xs:means[n]={k:st.mean(x[k] for x in xs) for k in xs[0]}
    delta={k:means['B'][k]-means['C'][k] for k in means.get('B',{})} if len(means)==2 else {}
    observable=all(r['snapshot_metrics']['sched_counter_observable'] for r in perf if r['driver_snapshots_enabled']) and len(perf)==8
    result={'sessions':sessions,'calibration_pairs':pairs,'II_and_run_perturbation_within_two_percent':qualified,
            'on_means':means,'B_minus_C_ms_or_counts':delta,'schedstat_observable':observable,
            'performance_source':'snapshot off sessions','scheduler_timeline_available':False,
            'complete':len(sessions)==13 and not (s.directory/'invalid_run.json').exists(),
            'inference':'one boot exploratory; counter windows include measured probes',
            'historical_34ms_closed':False,'hardware_compute_only':'unknown','physical_DDR_DMA':'unknown'}
    write_new_json(out/'analysis.json',result)
    lines=['# 低扰动driver与VTA host线程累计计数诊断','',
           f'完整会话：{len(sessions)}；II与run扰动≤2%：{qualified}；schedstat实际可观测：{observable}。',
           'CPU/VTA线程4/1/4、同图03:17、同运行库与完成协议，关闭逐提交日志。B为CPU互斥、C为全阶段互斥；一个boot补充诊断。',
           '', '| 均值（ms或次数/帧） | B | C | B−C |','|---|---:|---:|---:|']
    for k in delta:lines.append(f'| {k} | {means["B"][k]:.6f} | {means["C"][k]:.6f} | {delta[k]:.6f} |')
    lines+=['','driver_total=MMIO+post-start sleep+poll wait+other。线程CPU与runqueue描述同一host线程探针窗口，和driver计时重叠，不能相加。',
            'schedstat等待仅为已就绪后在运行队列中的累计等待；非CPU剩余时间还包含poll睡眠、timer唤醒延迟、硬件等待等，不能直接叫作硬件计算。',
            '板端CONFIG_FTRACE=n，未改内核或启用全局追踪。CONFIG_SCHED_INFO=y；若实际计数无增长，标记unknown。CPU时间仅为VTA host线程，不归给整个CPU阶段。',
            '校准未通过时，开启快照数据仅为诊断，关闭快照性能为主。历史34ms不拟合，正式评分不改。']
    with (out/'CONCLUSION_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(out,flush=True);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=('freeze','pilot','analyze'))
    p.add_argument('--run-dir',required=True);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.phase=='pilot' and not a.execute_board:p.error('--execute-board required')
    s=freeze(a.run_dir) if a.phase=='freeze' else load(a.run_dir)
    if a.phase=='pilot':execute(s);report(s)
    elif a.phase=='analyze':report(s)


if __name__=='__main__':main()
