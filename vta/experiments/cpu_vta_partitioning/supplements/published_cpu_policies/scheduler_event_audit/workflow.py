from pathlib import Path
import json,os,shlex,shutil,subprocess
from cpu_vta_partitioning.common import digest,inventory,now,read,sha256,write,sync_tree
from workload_model.common import ROOT
from .decode import check

OUTPUT=ROOT/'artifacts/scheduler_event_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_scheduler_event_audit'
SDK_SOURCE=ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def sources():
    files=[ROOT/'run_scheduler_event_audit.py',*sorted(Path(__file__).parent.glob('*.py')),
           *sorted(Path(__file__).parent.glob('*.cc')),ROOT/'tests/test_scheduler_event_audit.py']
    return {str(p):sha256(p) for p in files}


def qualify(document):
    try:return check(document)
    except (ValueError,KeyError,TypeError) as e:return dict(passed=False,reason=str(e),invalid_records_not_zero_wait=True)


def prepare(output):
    output=Path(output).resolve();build=output/'build';build.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('switch_probe.cc');tc=read(SDK_SOURCE)['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup changed')
    for t in list(tc['tools'].values())+list(tc.get('setup_files',{}).values()):
        if sha256(t['path'])!=t['sha256']:raise ValueError('SDK dependency changed')
    manifest=build/'PREPARED.json'
    if manifest.exists():
        proof=read(manifest)
        if proof['sources']!=sources() or proof['binary_sha256']!=sha256(build/'switch_probe'):raise ValueError('prepared event test identity changed')
        return proof
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    argv=['bash','-c','source "$1" || exit $?; shift; exec "$@"','switch-sdk',tc['sdk_setup'],
        'bash','-c','exec $CXX -std=c++17 -O2 "$1" -o "$2" -pthread','switch-compile',str(source),str(build/'switch_probe')]
    p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=120)
    (build/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('switch-event test compilation failed')
    runtime=ROOT/'artifacts/scheduler_gap_audit_01/resnet18/capsule/runtime'
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise ValueError('QEMU required for loading check')
    q=subprocess.run([qemu,str(runtime/'ld-linux-aarch64.so.1'),'--library-path',str(runtime),str(build/'switch_probe'),'--describe'],capture_output=True,text=True,timeout=20)
    if q.returncode:raise ValueError('AArch64 event test loading failed')
    desc=json.loads(q.stdout)
    if desc['DNN_inferences']!=0 or desc['periodic_sampling'] or desc['system_parameter_changes']:raise ValueError('unexpected self-test scope')
    proof=dict(sources=sources(),source_sha256=sha256(source),binary_sha256=sha256(build/'switch_probe'),toolchain=tc,
        description=desc,QEMU_load_passed=True,QEMU_not_native_event_validation=True,
        max_native_sessions=1,DNN_inferences=0,rebooted=False,FPGA_access=False,recorded_at=now())
    write(manifest,proof);return proof


def run(output,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    from cpu_vta_partitioning.transport import SSH
    output=Path(output).resolve();proof=read(output/'build/PREPARED.json')
    if proof['sources']!=sources() or sha256(output/'build/switch_probe')!=proof['binary_sha256']:raise ValueError('self-test source or executable changed')
    accepted=output/'BOARD_SELF_TEST.json'
    if accepted.exists():
        old=read(accepted)
        if sha256(old['raw'])!=old['raw_sha256'] or old['binary_sha256']!=proof['binary_sha256']:raise ValueError('accepted event evidence changed')
        calculated=qualify(read(old['raw']))
        if calculated!=old['validation']:raise ValueError('native event evidence does not replay')
        return dict(status='CACHED',passed=calculated['passed'],new_native_sessions=0,DNN_inferences=0)
    ssh=SSH(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    marker=output/'ATTEMPT.json';identity=dict(boot_id=boot,binary_sha256=proof['binary_sha256'],sources=sources())
    if marker.exists() and read(marker)!=identity:raise ValueError('event self-test identity/boot changed; preserve previous attempt')
    if not marker.exists():write(marker,identity)
    remote=ssh.config['remote_base'].rstrip('/')+'/diagnostics/scheduler_event_01/'+boot+'/'+proof['binary_sha256'][:24]
    ssh.idle_command('mkdir -p '+shlex.quote(remote),boot=boot);ssh.upload(output/'build/switch_probe',remote+'/probe')
    q=shlex.quote
    script='\n'.join(['cd '+q(remote),'test "$(sha256sum probe | cut -d " " -f 1)" = '+q(proof['binary_sha256']),
        'chmod 755 probe',
        'if test -f exit_code; then :; elif test -d .attempt; then echo "event attempt incomplete; no automatic rerun" >&2; exit 86; else',
        '  mkdir .attempt',
        '  if ./probe > events.json.tmp 2> native.log; then code=0; else code=$?; fi',
        '  mv events.json.tmp events.json',
        '  printf "%s\\n" "$code" > exit_code',
        'fi',
        'cat events.json'])
    raw=ssh.idle_command(script,boot=boot,timeout=20)
    path=output/'native'/boot/'events.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(raw)
    document=json.loads(raw)
    validation=qualify(document)
    after=ssh.boot_id()
    if after!=boot:raise ValueError('boot changed during event capability test')
    value=dict(boot_id=boot,binary_sha256=proof['binary_sha256'],sources=sources(),raw=str(path),raw_sha256=sha256(path),
        validation=validation,new_native_sessions=1,DNN_inferences=0,rebooted=False,FPGA_access=False,system_parameters_changed=False,
        same_boot_after_test=True,recorded_at=now())
    # The raw file and receipt remain available even when capability is absent.
    write(accepted,value);sync_tree(output)
    return dict(status='PASSED' if validation['passed'] else 'NOT_VALIDATED',passed=validation['passed'],
        reason=validation.get('reason'),new_native_sessions=1,DNN_inferences=0,
        records=validation.get('total_switch_records'),boot_id=boot)


def analyze(output):
    output=Path(output);receipt=read(output/'BOARD_SELF_TEST.json')
    if sha256(receipt['raw'])!=receipt['raw_sha256']:raise ValueError('raw switch-event output changed')
    document=read(receipt['raw']);validation=qualify(document)
    if validation!=receipt['validation']:raise ValueError('capability acceptance does not replay')
    REPORT.mkdir(parents=True,exist_ok=True)
    if not validation['passed']:
        text='# 调度事件接口自测未通过\n\n原始记录保留；未运行DNN、未访问FPGA、未重启或修改系统参数。\n\n'+json.dumps(validation,ensure_ascii=False,indent=2)+'\n'
    else:
        text='\n'.join(['# 调度事件接口：零DNN推理资格自测','',
            '本轮仅执行一次原生能力测试，含单线程忙运行、双线程同核竞争、睡眠、显式换核四种受控场景。没有执行DNN、访问FPGA、重启、修改调度参数或运行周期采样线程。','',
            '| 场景 | 线程 | 切换记录 | 抢占离开记录 | 非抢占离开记录 | 线程CPU时间ms | 时间线重建运行ms |',
            '|---|---:|---:|---:|---:|---:|---:|',
            *[f"| {r['case']} | {r['index']} | {r['switches']} | {r['preempt_out']} | {r['nonpreempt_out']} | {sum(s['CPU_clock_ns'] for s in r['segments'])/1e6:.6f} | {sum(s['reconstructed_CPU_ns'] for s in r['segments'])/1e6:.6f} |" for r in validation['streams']],
            '',f"全部{validation['total_switch_records']}条事件解码通过，无环形缓冲溢出或丢失记录。两竞争线程的重建运行片段在同一核心上的最大重叠为{validation['max_same_core_TID_overlap_ns']/1000:.3f}微秒。",'',
            '通过本次受控正负对照，确认此板可提供逐线程进入／离开CPU的时间、核心及抢占标记。无需ftrace或schedstats，本次也未开启它们。换核由测试程序显式改变自身允许集合完成，不是自然迁移策略的复现。','',
            '## 能说明什么，仍不能说明什么','',
            '可据此准备实际DNN线程的运行片段诊断，核对调用线程、worker和主机何时占用CPU，以及这些片段如何影响完成顺序。本轮仅证明接口可用，没有新的DNN时间线、FPS提升或模型修复。',
            'PERF_RECORD_SWITCH本身没有精确唤醒时刻。离开到再进入CPU之间的时间包含多种非运行原因，不能全部当作CPU排队；非抢占离开也可能是主动让出CPU等情况，不能全部称为睡眠。只有本次已知nanosleep场景能按受控操作验证睡眠行为。',
            '首次／末次不完整非运行区间保持未知；进入和离开记录并不等于全系统所有任务的调度轨迹。实际DNN应用仍需验证缓冲区容量、计时范围与探针扰动，成功前不宣称已能复原完整调度。',
            '', '## 后续范围','',
            '下一步可针对既有YOLO四线程重叠配置，设计一次关闭／开启事件记录的匹配诊断，关联阶段与逻辑worker的TID；不改评分、DP或Top-20。该DNN诊断尚未执行，本轮到接口自测结束。',
            '', '接口依据：[Linux perf_event_open](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)的context_switch、sample_id_all、use_clockid及PERF_RECORD_SWITCH；本板能力由本次原生数据验证，QEMU只验证此前的AArch64加载。',''])
    (REPORT/'FINDINGS_zh.md').write_text(text)
    write(REPORT/'DECODED.json',validation)
    shutil.copy2(receipt['raw'],REPORT/'RAW.json');shutil.copy2(output/'BOARD_SELF_TEST.json',REPORT/'BOARD_SELF_TEST.json')
    delivery=dict(status='CAPABILITY_PASSED' if validation['passed'] else 'CAPABILITY_NOT_VALIDATED',sources=sources(),
        new_native_sessions=1,DNN_inferences=0,rebooted=False,FPGA_access=False,system_parameters_changed=False,
        files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},raw_sha256=receipt['raw_sha256'],recorded_at=now())
    write(output/'DELIVERY.json',delivery)
    return dict(report=str(REPORT/'FINDINGS_zh.md'),status=delivery['status'],records=validation.get('total_switch_records'),DNN_inferences=0)
