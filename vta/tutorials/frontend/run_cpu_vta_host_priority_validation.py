"""Fixed-overlap owned VTA host priority causal experiment; defaults remain off."""
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
from analyze_cpu_vta_host_priority_validation import analyze, verify_priority
import run_cpu_vta_driver_snapshot_validation as low
from cpu_vta_residual_protocol import ROOT, matrix, verify_assets
from thesis_supplement_run import RunStore, sha256, utc, write_new_json

SOURCE=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_driver_snapshot_validation/20260917T134930Z_driver_snapshot_v3'
PROBE=ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_driver_snapshot_validation/20260917T134200Z_scheduler_readonly_probe'
VERSION='cpu_vta_host_priority_v1'
CONTROLS=(Path(__file__).name,'analyze_cpu_vta_host_priority_validation.py',
          'test_cpu_vta_host_priority_validation.py','validate_cpu_vta_host_priority_host.py','run_cpu_vta_driver_snapshot_validation.py','analyze_cpu_vta_driver_snapshot_validation.py',
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
    replace('#include "vta_driver_snapshot.h"','#include "vta_driver_snapshot.h"\n#include "vta_host_priority.h"')
    replace('bool low_driver_snapshots = false;', 'bool low_driver_snapshots = false;\nbool host_priority_enabled = false;\nint requested_vta_host_nice = 0;')
    replace('  bool driver_snapshots = false;', '  bool driver_snapshots = false;\n  bool host_priority_enabled = false;\n  int vta_host_nice = 0;')
    replace('    } else if (key == "--vta-driver-snapshots") {', '    } else if (key == "--vta-host-nice") {\n      args.host_priority_enabled = true;\n      args.vta_host_nice = std::stoi(need_value(key));\n      if (args.vta_host_nice != 0 && args.vta_host_nice != -10)\n        throw std::runtime_error("VTA host nice must be 0 or -10");\n    } else if (key == "--vta-driver-snapshots") {')
    replace('  vta_deploy::DriverRunSnapshot driver_snapshot;', '  vta_deploy::DriverRunSnapshot driver_snapshot;\n  long priority_tid{0};\n  int actual_host_nice{0};')
    replace('  timing->set_start_ms = NowMillis();', '  if (host_priority_enabled) {\n    timing->priority_tid = vta_deploy::HostPriorityTid();\n    timing->actual_host_nice = vta_deploy::ReadHostNice(timing->priority_tid);\n    const int expected = stage->device == "vta" ? requested_vta_host_nice : 0;\n    if (timing->actual_host_nice != expected) throw std::runtime_error("Per-frame host nice drift");\n  }\n  timing->set_start_ms = NowMillis();')
    replace('    low_driver_snapshots = args.driver_snapshots;', '    low_driver_snapshots = args.driver_snapshots;\n    host_priority_enabled = args.host_priority_enabled;\n    requested_vta_host_nice = args.vta_host_nice;\n    if (args.serial && args.host_priority_enabled) throw std::runtime_error("Priority intervention requires distinct pipeline stage workers");')
    replace('            publish_initialization(stage_index);\n            std::shared_ptr<Frame> frame;', '            if (args.host_priority_enabled && stages[stage_index].device == "vta")\n              vta_deploy::ApplyOwnedHostNice(args.vta_host_nice);\n            publish_initialization(stage_index);\n            std::shared_ptr<Frame> frame;')
    replace('    const StageTiming& timing = frame.stage_timings[i];\n    if (args.driver_snapshots)', '    const StageTiming& timing = frame.stage_timings[i];\n    if (args.host_priority_enabled) {\n      os << ",\\\"stage" << i << "_host_priority\\\":{\\\"tid\\\":" << timing.priority_tid\n         << ",\\\"nice\\\":" << timing.actual_host_nice << "}";\n    }\n    if (args.driver_snapshots)')
    replace('  WriteFile(args.thread_affinity_report, os.str());', '  WriteFile(args.thread_affinity_report, os.str());\n  if (args.host_priority_enabled && !args.serial) {\n    long vta_tid = 0;\n    for (size_t i = 0; i < records.size(); ++i) {\n      if (args.stages[i].device == "vta") {\n        if (vta_tid) throw std::runtime_error("Priority intervention supports a single VTA worker");\n        vta_tid = records[i].caller_tid;\n      }\n    }\n    std::ostringstream audit;\n    audit << "{\\\"vta_tid\\\":" << vta_tid << ",\\\"requested_nice\\\":" << args.vta_host_nice << ",\\\"threads\\\":[";\n    bool first = true;\n    for (long tid : ProcessThreadIds()) {\n      const int actual = vta_deploy::ReadHostNice(tid);\n      const int expected = tid == vta_tid ? args.vta_host_nice : 0;\n      if (actual != expected) throw std::runtime_error("Initialization thread priority drift");\n      if (!first) audit << ",";\n      first = false;\n      audit << "{\\\"tid\\\":" << tid << ",\\\"nice\\\":" << actual << "}";\n    }\n    audit << "]}";\n    WriteFile(args.thread_affinity_report + ".priority.json", audit.str());\n  }')
    return source


def freeze(directory):
    old=low.load(SOURCE)
    if not read(SOURCE/'analysis/FINAL_DELIVERY_STATUS.json')['snapshot_calibration_passed']:raise ValueError('Qualified low overhead source required')
    c=copy.deepcopy(old.contract)
    for key,v in c['assets'].items():v['source_path']=str(SOURCE/key)
    derived=derive_runner((SOURCE/'raw/runner_sources/vta_stage_pipeline_runner.cc').read_text())
    v=c['assets']['raw/runner_sources/vta_stage_pipeline_runner.cc']
    v.update(derived_from_sha256=v['sha256'],sha256=hashlib.sha256(derived.encode()).hexdigest(),bytes=len(derived.encode()))
    additions={
        'raw/runner_sources/vta_driver_snapshot.h':ROOT/'vta/apps/native_deploy/vta_driver_snapshot.h',
        'raw/runner_sources/vta/driver.h':ROOT/'3rdparty/vta-hw/include/vta/driver.h',
        'raw/golden_all_stage_host_runner':SOURCE/'raw/host_validation/driver_snapshot_host_runner',
        'raw/runner_sources/vta_host_priority.h':ROOT/'vta/apps/native_deploy/vta_host_priority.h'}
    additions.update({'raw/snapshot_controller_sources/'+name:Path(__file__).with_name(name).resolve() for name in CONTROLS})
    additions.update({'raw/scheduler_probe/'+str(p.relative_to(PROBE)):p for p in PROBE.rglob('*') if p.is_file()})
    for key,p in additions.items():c['assets'][key]={'source_path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size}
    c.update(protocol_version=VERSION,run_id=Path(directory).name,stage='cpu_vta_host_priority',
             campaign_id='host_priority_'+uuid.uuid4().hex[:12],created_at=utc(),parent_completed_run=str(SOURCE),
             inference='New fixed-overlap priority contrast: pilot exploratory; independent boot means require three kernel boots',
             causal_threshold=.05,serial_drift_threshold=.05,
             endpoints='runnable queue reduction, VTA service and CPU stage trajectories; II improvement not required',
             priority_scope='only initialized VTA stage caller TID; all CPU pool workers nice=0; per-frame caller recheck',
             next_boot_rule='replicate only qualified pilot; real user kernel reboot; no retry after failure',
             snapshot_orders=[[['N0',False],['N0',True],['N1',False],['N1',True]],
                              [['N1',True],['N1',False],['N0',True],['N0',False]]],
             treatment={'N0':0,'N1':-10},gate_scope_fixed='cpu',max_kernel_boots=3,
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
    s.event('host_priority_protocol_frozen',parent=str(SOURCE))
    return s


def load(directory):
    s=RunStore(directory)
    if s.contract['protocol_version']!=VERSION:raise ValueError('Wrong protocol')
    verify_assets(s)
    for name in CONTROLS:
        v=s.contract['assets']['raw/snapshot_controller_sources/'+name]
        if sha256(v['source_path'])!=v['sha256']:raise ValueError('Live controller drift '+name)
    for key in ('raw/runner_sources/vta_driver_snapshot.h','raw/runner_sources/vta/driver.h','raw/runner_sources/vta_host_priority.h'):
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
    with low.adapters():
        old_build,old_argv=native.build_runner,native.runner_argv
        def argv(board,output,stages,c,mode,runs,telemetry=False,profile=False):
            if profile:raise ValueError('Submit diagnostics forbidden in low snapshot experiment')
            result=old_argv(board,output,stages,c,mode,runs,telemetry,False)
            if c.get('host_priority_enabled'):result+=['--vta-host-nice',str(c['vta_host_nice'])]
            return result
        native.build_runner,native.runner_argv=build,argv
        try:yield
        finally:native.build_runner,native.runner_argv=old_build,old_argv


def session(s,boot,name,mode,block,enabled=False):
    load(s.directory)
    c=copy.deepcopy(matrix()[0]);c.update(condition=name,gate_mode='exclusive' if name in ('N0','N1') else 'off',
                                       gate_scope='cpu',driver_snapshots=enabled,
                                       host_priority_enabled=name in ('N0','N1'),vta_host_nice=s.contract['treatment'].get(name,0))
    r=native.session(s,boot,c,mode,block,profile=False)
    p=s.directory/'raw/sessions'/r['session_id']
    if '[VTA_QUEUE]' in (p/'native.stderr').read_text() or '[C3_COMPLETION]' in (p/'native.stderr').read_text():
        raise ValueError('Unexpected submission logging')
    if mode in ('pipeline','correctness_pipeline'):
        rows=[json.loads(l) for l in (p/'result.jsonl').read_text().splitlines()]
        r['snapshot_metrics']=analyze(rows,enabled,c['gate_scope'],7 if mode=='pipeline' else 0)
        r['gate_metrics']=r['snapshot_metrics']['gate_metrics']
        r['priority_audit']=verify_priority(rows,c['vta_host_nice'],read(p/'affinity.json.priority.json'),read(p/'affinity.json'))
    r['driver_snapshots_enabled']=enabled
    write_new_json(p/'snapshot_summary.json',r)
    return r


def execute(s,replicate=False):
    existing=sorted((s.directory/'analysis').glob('boot_[123].json'))
    if (s.directory/'invalid_run.json').exists() or (existing and not replicate) or len(existing)>=3:raise ValueError('No resume/retry or extra boot')
    if replicate and not existing:raise ValueError('Pilot required')
    if not read(s.directory/'analysis/host_validation.json')['passed']:raise ValueError('Host qualification required')
    boot=native.preflight(s);boot['boot_index']=len(existing)+1
    if any(read(p)['boot_id']==boot['boot_id'] for p in existing):raise ValueError('Real independent kernel reboot required')
    with adapters():
        native.deploy(s,boot,build(s))
        try:
            session(s,boot,'serial','correctness_serial','qualification')
            for name in ('N0','N1'):session(s,boot,name,'correctness_pipeline','qualification',True)
            before=session(s,boot,'serial','serial','matched_before')
            for i,order in enumerate(s.contract['snapshot_orders']):
                for name,enabled in order:session(s,boot,name,'pipeline',f'pair{i}_'+('on' if enabled else 'off'),enabled)
            after=session(s,boot,'serial','serial','matched_after')
            drift=[after['metrics']['stages'][i]['run_ms']['mean']/before['metrics']['stages'][i]['run_ms']['mean']-1 for i in range(3)]
            write_new_json(s.directory/f'analysis/serial_drift_boot{boot["boot_index"]}.json',{'before':before,'after':after,'fractions':drift,'passed':max(map(abs,drift))<=.05})
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
            write_new_json(s.directory/f'analysis/owned_cleanup_boot{boot["boot_index"]}.json',{'passed':True,'action':action})


def report(s):
    from analyze_cpu_vta_host_priority_validation import summarize
    out=s.directory/'analysis'/('report_'+uuid.uuid4().hex[:10]);out.mkdir()
    sessions=[read(p) for p in sorted((s.directory/'raw/sessions').glob('*/snapshot_summary.json'))]
    result=summarize(sessions)
    result.update(inference='Boot level paired means; one boot exploratory, three boots t(df=2)',
                  historical_34ms_closed=False,hardware_compute_only='unknown',physical_DDR_DMA='unknown',
                  no_invalid_run=not (s.directory/'invalid_run.json').exists())
    write_new_json(out/'analysis.json',result)
    lines=['# 固定CPU/VTA并发的VTA host nice因果验证','',
           'CPU/VTA线程4/1/4、CPU阶段互斥、同图03:17、同库、同输入与完成协议。N0 nice=0，N1仅VTA host worker nice=−10。',
           f'有效会话{len(sessions)}；独立boot {result["boots"]}；快照II/run扰动≤2%：{result["calibration_passed"]}。',
           '', '| boot | 快照 | VTA run降低ms | run降低比例 | II降低ms | runnable等待降低ms |',
           '|---|---|---:|---:|---:|---:|']
    for b in result['boot_pairs']:
        lines.append(f'| {b["boot"]} | {b["snapshots"]} | {b["vta_run_reduction_ms"]:.4f} | {b["vta_run_reduction_fraction"]:.2%} | {b["ii_reduction_ms"]:.4f} | {str(b.get("runqueue_reduction_ms","unknown"))} |')
    lines+=['','线程运行队列等待和driver计时相互重叠，不能求和，不能把局部run下降直接加到II或历史34ms。',
            '优先级可能改变CPU服务与内存访问时序；本干预不是纯DDR对照。单boot不作跨boot因果结论；硬件compute-only及物理DDR/DMA仍unknown。',
            '阶段CPU时间不使用进程CPU时间代替。正式静态评分及候选排名保持冻结。']
    with (out/'CONCLUSION_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(out,flush=True);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=('freeze','pilot','replicate','analyze'))
    p.add_argument('--run-dir',required=True);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.phase in ('pilot','replicate') and not a.execute_board:p.error('--execute-board required')
    s=freeze(a.run_dir) if a.phase=='freeze' else load(a.run_dir)
    if a.phase in ('pilot','replicate'):execute(s,a.phase=='replicate');report(s)
    elif a.phase=='analyze':report(s)


if __name__=='__main__':main()
