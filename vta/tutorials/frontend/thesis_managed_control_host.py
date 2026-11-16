#!/usr/bin/env python3
"""Freeze and execute production lifecycle gates with native manager/queue cores."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

from thesis_c2_host_lifecycle import extract
from thesis_supplement_run import RunStore, sha256, write_new_json
from thesis_supplement_audit import reconcile, seal

FRONTEND=Path(__file__).resolve().parent
ROOT=FRONTEND.parents[2]
BASE=FRONTEND/'report_out/stage_tile_cotuning/thesis_supplement_experiments'


def execute(core_run):
    runner=ROOT/'vta/apps/native_deploy/vta_stage_pipeline_runner.cc'
    header=runner.with_name('vta_managed_lifecycle_control.h')
    source=runner.read_text()
    current_core=extract(source,'class BoundarySlotManager {')
    previous=json.loads((core_run/'analysis/extraction.json').read_text())
    if hashlib.sha256(current_core.encode()).hexdigest()!=previous['production_class_sha256']:
        raise ValueError('Core regression source differs from current production manager')
    if json.loads((core_run/'summary.json').read_text())['status']!='HOST_PREQUALIFICATION_PASS':
        raise ValueError('Matching core regression did not pass')
    queue=extract(source,'class BoundedQueue {')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_managed_control_host_'+uuid.uuid4().hex[:8]
    directory=BASE/'results/03'/run_id
    store=RunStore.create(directory,{
        'protocol_version':'thesis_supplement_v1_20260916_managed_lifecycle_host',
        'campaign_id':json.loads((BASE/'results/ACTIVE_CAMPAIGN.json').read_text())['campaign_id'],
        'stage':'03','run_id':run_id,'purpose':'真实生产生命周期control与manager/queue核心本机故障预资格',
        'scope':'host_functional_prequalification_only','formal_phase03_completed':False,
        'core_regression_run':str(core_run),'core_regression_manifest_sha256':sha256(core_run/'artifact_hashes.json'),
        'current_runner_sha256':sha256(runner),'current_control_header_sha256':sha256(header),
        'production_core_sha256':previous['production_class_sha256'],
        'production_queue_sha256':hashlib.sha256(queue.encode()).hexdigest(),
        'controller_sha256':sha256(Path(__file__)),
        'test_source_sha256':sha256(FRONTEND/'thesis_managed_control_host.cc'),
        'manager_adapter':'exact current production manager with constructor observer callback; CPU shared vectors only',
        'queue_adapter':'verbatim production BoundedQueue template, no alternate queue logic',
        'control_adapter':'none: production header, actual CV/poll/FIFO/exclusive files',
        'seed':None,'repeats':10,'cases_per_repeat':16,'total_control_cases':160,
        'case_order':['direct_release','direct_cancel','fifo_release','fifo_abort','fifo_bad',
                      'premature_release','duplicate_release','unknown_release','timeout','trace_exclusive',
                      'duplicate_selector','trace_escape','queue_producer_close','queue_consumer_close',
                      'fifo_queue_abort','fifo_full_manager_abort'],
        'gate_timeout_seconds':1,'timeout_amendment':'本机确定性timeout负对照1s，正式板端仍00的60s上限',
        'compile_timeout_seconds':60,'probe_timeout_seconds':90,
        'sanitizers':{'address':True,'undefined_behavior':True,'leak':False},
        'leak_disabled_reason':'当前ptrace环境已证实LSan致命失败，不主张无泄漏',
        'expected_faults':'Reject/timeout/cancel/abort must satisfy assertions; unexpected case failure invalidates run',
        'FIFO_cleanup':'own kernel IPC FIFO unlinked after watcher joins; trace/commands/costs retained',
        'hardware_calls':0,'model_calls':0,'phase01_certificate':None,'phase02_real_data':None,
        'timing':'native state-event clocks descriptive only; action costs not inference/online wall',
        'remaining_gates':['actual native managed VTA stage execution','real bundle','two boots','current FINISH peaks/Q1'],
    })
    print(json.dumps({'run_started':str(directory)}),flush=True)
    for p,name in [(header,'vta_managed_lifecycle_control.h'),(runner,'production_runner.cc'),
                   (FRONTEND/'thesis_managed_control_host.cc','control_test.cc'),
                   (core_run/'raw/harness.cc','thesis_c2_host_lifecycle.cc'),
                   (core_run/'raw/manager_under_test.inc','manager_under_test.inc'),
                   (Path(__file__),'controller.py')]:
        shutil.copyfile(p,directory/'raw'/name)
    (directory/'raw/queue_under_test.inc').write_text('template <typename T>\n'+queue+'\n')
    traces=directory/'raw/traces';traces.mkdir()
    compiler=store.execute(['g++','-std=c++17','-O1','-g','-pthread','-fsanitize=address,undefined',
                            '-fno-omit-frame-pointer',str(directory/'raw/control_test.cc'),
                            '-o',str(directory/'raw/control_test')],
                           'compile_production_control_test','functional_qualification',60)
    probe=None
    if compiler['status']=='ok':
        probe=store.execute(['env','ASAN_OPTIONS=detect_leaks=0',str(directory/'raw/control_test'),str(traces)],
                            'exercise_production_control','functional_qualification',90)
    events=[]
    if probe:
        events=[json.loads(line) for line in (directory/probe['stdout']).read_text().splitlines()]
    passed=[e['case'] for e in events if e['event']=='control_case_pass']
    failed=[e for e in events if e['event']=='control_case_fail']
    success=probe is not None and probe['status']=='ok' and len(passed)==160 and not failed
    trace_audit=[]
    for path in sorted(traces.glob('*.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if [r['seq'] for r in rows]!=list(range(1,len(rows)+1)):
            raise ValueError('Native trace event sequence changed')
        if any(a['monotonic_ns']>b['monotonic_ns'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Native trace monotonic order changed')
        if path.stem.startswith('trace_escape'):
            if rows[0]['detail']!='quote"slash\\newline\n':raise ValueError('Trace JSON escaping changed')
        trace_audit.append({'path':str(path.relative_to(directory)),'events':len(rows),
                            'pid':rows[0]['pid'] if rows else None})
    if list(traces.glob('*.fifo')):raise ValueError('Watcher/FIFO cleanup unfinished')
    write_new_json(directory/'analysis/control_case_results.json',{'passed':passed,'failed':failed})
    write_new_json(directory/'analysis/native_trace_audit.json',trace_audit)
    summary={'status':'PRODUCTION_CONTROL_HOST_PREQUALIFICATION_PASS' if success else 'FAILED_NO_CERTIFICATE',
             'control_cases_passed':len(passed),'control_cases_failed':len(failed),
             'native_trace_files':len(trace_audit),'native_trace_events':sum(r['events'] for r in trace_audit),
             'formal_phase03_completed':False,'actual_hardware_calls':0,'actual_model_calls':0,
             'remaining':'native managed VTA execution/real bundle/two boots/FINISH/Q1 pending'}
    write_new_json(directory/'summary.json',summary)
    if not success:write_new_json(directory/'invalid_session.json',{'reason':'unexpected compile/probe/assertion failure',
                                                                  'all_costs_preserved':True})
    store.event('production_control_host_audited',**summary)
    cost=reconcile(directory);write_new_json(directory/'analysis/cost_audit.json',cost)
    (directory/'REPORT.md').write_text(
        '# 生产生命周期控制与队列/slot核心本机预资格\n\n'
        f'{len(passed)}/160 case通过，{len(failed)}个失败，实际VTA/模型调用0。\n\n'
        '直接编译生产control header，使用真实CV/FIFO/poll和排他trace文件；queue逐字提取，'
        'manager复用匹配current class hash的160/160核心回归fixture，只有CPU vector分配adapter。'
        '固定16类×10重复，覆盖释放/取消、FIFO release/abort/非法指令、早/重复/未知release、'
        'timeout、重复selector、排他文件与JSON转义，以及生产queue满/空关闭、FIFO→queue abort、'
        'FIFO→满slot manager abort。等待证据由锁内实际wait事件与promise/CV获得，无sleep证明阻塞。\n\n'
        'abort后的新acquire/publish被拒绝，合法held consumer可release；另外READY slot仍busy，'
        '没有强制FREE。FIFO watcher结束后仅清理自己创建的IPC对象，原始trace/命令及失败成本保留。'
        '地址与未定义行为检查开启；ptrace环境关闭leak扫描，不主张无泄漏。\n\n'
        '生产native main已接managed StageRunHooks以及slot/queue观察器，control CLI只允许'
        'two-slot managed pipeline且warmup=0；默认关闭。等待门在VTA全stage互斥锁外，'
        '运行开始/完成只记录事件。此host测试未执行native managed VTA stage，'
        '不得认领真实DMA/bundle/boot/health/FINISH峰值/Q1认证。\n\n'
        f'已测action成本{cost["measured_action_total_seconds"]:.6f}s，开发/人工等待unknown/null；'
        'trace为同一个native进程时钟，仅是host状态证据，不是板端吞吐或在线预算。'
        '正式03仍准备中，必须先恢复01并完成02。\n')
    seal(directory)
    print(json.dumps({'run_dir':str(directory),'summary':summary},ensure_ascii=False))
    return 0 if success else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-run',type=Path,required=True)
    args=parser.parse_args();sys.exit(execute(args.core_run.resolve()))
