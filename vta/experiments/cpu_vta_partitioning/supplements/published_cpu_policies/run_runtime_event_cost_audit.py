"""Authenticate old counters and measure only frozen event appends, zero DNN."""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from cpu_vta_partitioning.transport import SSH
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/runtime_event_cost_02'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_runtime_event_cost'
CAPSULE=ROOT/'artifacts/cpu_exclusion_gap_01/yolov3_tiny416/capsule'
BOOT='6018b3b7-bb29-4f80-92c1-0bfd8c81bb99'
SYMBOL='_ZN3vta15RuntimeProfiler15PushEventLockedENS0_5EventE'


def summarize_counter(d):
    fields=('load_buffer_2d_calls','store_buffer_2d_calls','push_gemm_op_calls','push_alu_op_calls',
        'synchronize_calls','mem_copy_from_host_calls','mem_copy_to_host_calls','flush_cache_calls',
        'invalidate_cache_calls','template_capture_calls','template_replay_hits','template_replay_fallbacks')
    if any(type(d.get(k)) is not int or d[k]<0 for k in fields):
        raise ValueError('runtime counters missing or negative')
    # Both the buffer-copy wrapper and its inner memcopy emit an event. These
    # are event counts, never additive copy durations or independent DMA work.
    emitted=sum(d[k] for k in fields)+d['mem_copy_from_host_calls']+d['mem_copy_to_host_calls']
    if d['event_count']!=min(4096,emitted):
        raise ValueError('retained event count differs from known emitted kinds')
    return dict(emitted_events_per_component=emitted,retained_events=d['event_count'],
        command_API_calls=sum(d[k] for k in fields[:4]),
        nonadditive_API_wall_us={k:d[k] for k in
            ('load_buffer_2d_enqueue_us','store_buffer_2d_enqueue_us','push_gemm_op_us','push_alu_op_us')},
        explicit_flush_or_invalidate_calls=d['flush_cache_calls']+d['invalidate_cache_calls'])


def audit():
    OUT.mkdir(parents=True,exist_ok=True);sources={};rows=[]
    yindex=ROOT/'artifacts/yolo_validation_01/ranking/inputs/traffic.json'
    index=read(yindex);sources[str(yindex)]=sha256(yindex)
    resrun=ROOT/'runs/resnet18_open_cpu_resource_02'
    for network,names in [('yolov3_tiny416',('vta_03_06','vta_09_12')),
                          ('resnet18',('vta_03_04','vta_08_08','vta_10_19'))]:
        for name in names:
            if network=='yolov3_tiny416':
                source=index['records'][name]['source'];p=Path(source['receipt'])
                if sha256(p)!=source['receipt_sha256']:raise ValueError('YOLO traffic receipt changed')
                receipt=read(p);verify_inventory(receipt['raw'],receipt['files']);raw=Path(receipt['raw'])
                status=raw/'traffic_profile/single_run_status.json';events=raw/'traffic_profile/single_run_events.json'
                bundle=ROOT/'artifacts/yolo_transfer_01/profile_bundle_bridge02'
            else:
                origin=resrun/'traffic'/(name+'.json');v=read(origin);sources[str(origin)]=sha256(origin)
                meta=v['measurement']['summary']['_receipt'];p=resrun/meta['path']
                if sha256(p)!=meta['sha256']:raise ValueError('ResNet traffic receipt changed')
                receipt=read(p);verify_inventory(resrun,receipt['files'])
                status=resrun/next(k for k in receipt['files'] if k.endswith('/traffic_profile/single_run_status.json'))
                events=resrun/next(k for k in receipt['files'] if k.endswith('/traffic_profile/single_run_events.json'))
                bundle=ROOT/'artifacts/bundles/8a942b2047c031339168f9d2'
            manifest=read(bundle/'manifest.json');programs=read(bundle/'plans/programs.json')
            if receipt['task']['bundle_id']!=manifest['bundle_id'] or receipt['task']['scored']!=1:
                raise ValueError('traffic task identity or single-run scope differs')
            runtime=bundle/'runtime/libvta.so'
            if sha256(runtime)!=sha256(CAPSULE/'runtime/libvta.so'):
                raise ValueError('old runtime differs from audited frozen library')
            for leaf in ('module.so','graph.json','params.bin'):
                info=programs[name]['files'][leaf]
                if sha256(bundle/'programs'/name/leaf)!=info['sha256']:
                    raise ValueError('traffic program changed')
            dumped=read(events);d=read(status);count=summarize_counter(d)
            if not dumped or dumped[-1]['seq']!=count['emitted_events_per_component']:
                raise ValueError('event sequence total differs from counters')
            sources.update({str(q):sha256(q) for q in (p,status,events,runtime,bundle/'manifest.json')})
            rows.append(dict(network=network,program=name,source_boot=receipt['boot_id'],
                source_task=receipt['task']['task_id'],dumped_events=len(dumped),
                timestamps_not_sufficient_for_precise_alignment=True,**count))
    library=CAPSULE/'runtime/libvta.so'
    asm=subprocess.run(['aarch64-linux-gnu-objdump','-d','--disassemble=VTALoadBuffer2D',str(library)],
        capture_output=True,text=True,check=True).stdout
    body=subprocess.run(['aarch64-linux-gnu-objdump','-d','--disassemble='+SYMBOL,str(library)],
        capture_output=True,text=True,check=True).stdout
    if not all(anchor in asm for anchor in ('#0x98','pthread_mutex_lock@plt',SYMBOL+'@plt')):
        raise ValueError('frozen runtime append call ABI changed')
    if not all(anchor in body for anchor in ('#928','#0x98','steady_clock3nowEv@plt')):
        raise ValueError('frozen runtime append implementation changed')
    (OUT/'VTALoadBuffer2D.asm').write_text(asm);(OUT/'PushEventLocked.asm').write_text(body)
    result=dict(rows=rows,sources=sources,runtime_sha256=sha256(library),new_DNN_inferences=0,
        old_event_records_not_complete_timelines=True,API_duration_excludes_subsequent_profiler_bookkeeping=True,
        no_CPU_cost_causality_established=True,new_static_model=False,
        profiler_global_mutex_and_signature_maps_not_isolated=True)
    write(OUT/'AUDIT.json',result,replace=True);csv_write(OUT/'counters.csv',rows)
    return result


def build():
    a=audit();source=ROOT/'runtime_event_cost_probe.cc'
    tc=read(ROOT/'artifacts/completion_breakpoint_capability_01/BUILD.json')['toolchain']
    for p,h in [(tc['sdk_setup'],tc['sdk_setup_sha256'])]+[(t['path'],t['sha256']) for t in tc['tools'].values()]:
        if sha256(p)!=h:raise ValueError('SDK identity changed')
    binary=OUT/'probe';env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    recipe='exec $CXX -std=c++14 -O2 -fno-ipa-cp -Wall -Wextra -Werror "$1" -ldl -o "$2"'
    r=subprocess.run(['bash','-c','source "$1" || exit $?; shift; exec "$@"','event-sdk',tc['sdk_setup'],
        'bash','-c',recipe,'event-compile',str(source),str(binary)],capture_output=True,text=True,env=env,timeout=120)
    (OUT/'compile.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('event-cost build failed; compile.log preserved')
    proof=dict(source_sha256=sha256(source),controller_sha256=sha256(__file__),binary_sha256=sha256(binary),
        toolchain=tc,recipe=recipe,runtime_sha256=a['runtime_sha256'],
        tvm_runtime_sha256=sha256(CAPSULE/'runtime/libtvm_runtime.so'),expected_boot=BOOT,
        scope='one fresh process, zero DNN, no FPGA; event append only, not the entire profiler overhead',
        counts=dict(warmup=5000,small=5000,large=20000,empty=20000),automatic_repeat=False)
    write(OUT/'BUILD.json',proof)
    return proof


def validate():
    proof=read(OUT/'BUILD.json')
    if (sha256(ROOT/'runtime_event_cost_probe.cc')!=proof['source_sha256'] or
        sha256(__file__)!=proof['controller_sha256'] or sha256(OUT/'probe')!=proof['binary_sha256'] or
        sha256(CAPSULE/'runtime/libvta.so')!=proof['runtime_sha256'] or
        sha256(CAPSULE/'runtime/libtvm_runtime.so')!=proof['tvm_runtime_sha256']):
        raise ValueError('event-cost source, program or runtime changed')
    return proof


def check_result(value):
    if (value['status']!='VALIDATED' or value['DNN_inferences']!=0 or value['FPGA_access'] or
        value['affinity']!=[3] or value['event_size']!=152 or value['total_events']!=30000 or
        value['retained_events']!=4096 or [r['calls'] for r in value['blocks']]!=[20000,5000,20000] or
        any(type(r['CPU_ns']) is not int or r['CPU_ns']<=0 for r in value['blocks'])):
        raise ValueError('native event-cost accounting invalid')
    return value


def qemu():
    validate();runtime=CAPSULE/'runtime'
    r=subprocess.run(['qemu-aarch64-static',str(runtime/'ld-linux-aarch64.so.1'),'--library-path',str(runtime),
        str(OUT/'probe'),str(runtime/'libvta.so')],capture_output=True,text=True,timeout=20)
    write(OUT/'QEMU.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
        scope='ABI, output and CPU program execution only; timing is not board performance'),replace=True)
    if r.returncode:raise ValueError('QEMU ABI/program validation failed')
    check_result(json.loads(r.stdout));return dict(valid=True,real_kernel_or_performance_proven=False)


def board(*,execute_board=False):
    if not execute_board:raise ValueError('explicit execute-board required')
    proof=validate();result_path=OUT/'BOARD_RECEIPT.json'
    if result_path.exists():return read(result_path)
    ssh=SSH(read(ROOT/'artifacts/cross_network_model_01/board.local.json')['board'],execute_board=True)
    if ssh.boot_id()!=BOOT:raise ValueError('boot changed; no native test executed')
    q=shlex.quote;manifest=read(CAPSULE/'manifest.json')
    remote_bundle=ssh.config['remote_base'].rstrip('/')+'/bundles/'+manifest['bundle_id']
    remote=ssh.config['remote_base'].rstrip('/')+'/capability/runtime_event_cost_02'
    started=OUT/'NATIVE_STARTED.json'
    if not started.exists():
        ssh.idle_command('mkdir -p '+q(remote)+'; test ! -e '+q(remote+'/started'),boot=BOOT)
        ssh.command('cat > '+q(remote+'/probe.upload'),input_path=OUT/'probe')
        script='\n'.join(['set -eu',
            'test "$(sha256sum '+q(remote+'/probe.upload')+' | cut -d \' \' -f 1)" = '+q(proof['binary_sha256']),
            'test "$(sha256sum '+q(remote_bundle+'/runtime/libvta.so')+' | cut -d \' \' -f 1)" = '+q(proof['runtime_sha256']),
            'test "$(sha256sum '+q(remote_bundle+'/runtime/libtvm_runtime.so')+' | cut -d \' \' -f 1)" = '+q(proof['tvm_runtime_sha256']),
            'mv '+q(remote+'/probe.upload')+' '+q(remote+'/probe'),'chmod 700 '+q(remote+'/probe'),
            'set -C','printf started > '+q(remote+'/started'),'set +C','set +e',
            q(remote_bundle+'/runtime/ld-linux-aarch64.so.1')+' --library-path '+q(remote_bundle+'/runtime')+
            ' '+q(remote+'/probe')+' '+q(remote_bundle+'/runtime/libvta.so')+
            ' > '+q(remote+'/stdout.json')+' 2> '+q(remote+'/stderr.txt'),
            'code=$?','printf "%s\\n" "$code" > '+q(remote+'/exit_code')])
        write(started,dict(boot=BOOT,remote=remote,binary_sha256=proof['binary_sha256'],DNN_inferences=0))
        ssh.idle_command(script,boot=BOOT,timeout=15)
    # A disconnected observation is recovered read-only; never launch again.
    code=ssh.command('cat '+q(remote+'/exit_code')).strip()
    stdout=ssh.command('cat '+q(remote+'/stdout.json'));stderr=ssh.command('cat '+q(remote+'/stderr.txt'))
    write(OUT/'RAW_NATIVE.json',dict(stdout=stdout,stderr=stderr,exit_code=code),replace=True)
    if code!='0' or ssh.boot_id()!=BOOT:raise ValueError('native event-cost test failed or boot changed')
    value=check_result(json.loads(stdout))
    result=dict(boot_id=BOOT,result=value,binary_sha256=proof['binary_sha256'],
        runtime_sha256=proof['runtime_sha256'],raw_sha256=sha256(OUT/'RAW_NATIVE.json'),
        new_native_processes=1,new_DNN_inferences=0,old_runtime_unchanged=True,
        no_system_parameter_or_FPGA_access=True)
    write(result_path,result);return result


def analyze():
    validate();native=read(OUT/'BOARD_RECEIPT.json');a=read(OUT/'AUDIT.json')
    if sha256(OUT/'RAW_NATIVE.json')!=native['raw_sha256']:raise ValueError('native raw changed')
    value=check_result(native['result']);rows=[]
    baseline=value['blocks'][0]['CPU_ns']/20000
    for item in value['blocks'][1:]:
        per_call=item['CPU_ns']/item['calls']
        rows.append(dict(block=item['name'],append_CPU_ns_per_call=per_call,
            empty_call_CPU_ns_per_call=baseline,increment_above_empty_ns=per_call-baseline))
    estimates=[]
    for r in a['rows']:
        estimates.append(dict(network=r['network'],program=r['program'],events=r['emitted_events_per_component'],
            synthetic_append_only_CPU_ms=[r['emitted_events_per_component']*x['increment_above_empty_ns']/1e6 for x in rows],
            synthetic_estimate_not_measured_component_CPU=True))
    result=dict(blocks=rows,component_event_work_estimates=estimates,
        mutex_signature_maps_and_full_Add_cost_unmeasured=True,
        causal_pipeline_slowdown_not_proven=True,new_DNN=0,new_native_processes=1,
        no_static_or_runtime_change=True)
    write(OUT/'RESULTS.json',result,replace=True);REPORT.mkdir(parents=True,exist_ok=True)
    text=['运行时事件记录：既有统计与零推理成本核查','',
        '读取已验收DMA统计，核验原始文件与模块/运行库身份。新增自测只调用冻结运行库的事件追加函数，不调用DNN、DMA、设备或控制接口。',
        '原运行库每次VTA API执行后仍会更新统计、记录事件；API duration不包含随后统计和记录的全部开销。',
        '旧记录只输出最后200条，内部最多保留4096条，不能据此恢复全帧精确时间线。',
        '下列数量是实际单帧计数；缓存flush/invalidate为0只对应当前相干运行配置，不代表没有内存流量。','']
    for r in a['rows']:text.append(f"{r['network']} {r['program']}：事件{r['emitted_events_per_component']}条，命令API {r['command_API_calls']}次。")
    text+=['','一次进程，5000条预热，之后分别5000和20000条；另测20000次同ABI空调用。只测事件追加，不包含外层锁、签名表及所有统计操作。']
    for r in rows:text.append(f"{r['block']}：每次追加CPU {r['append_CPU_ns_per_call']:.1f} ns；空调用{baseline:.1f} ns；差值{r['increment_above_empty_ns']:.1f} ns。")
    for e in estimates:text.append(f"{e['network']} {e['program']}：按上述微基准折算的追加成本范围{min(e['synthetic_append_only_CPU_ms']):.3f}—{max(e['synthetic_append_only_CPU_ms']):.3f}核ms；不是该子图或流水的实测归因。")
    text+=['','不能将折算值直接加到旧主机核时里：旧隔离成本已经包含原运行时开销。',
        '不能把API内部duration相加充当CPU核时；嵌套和墙钟含义不同。',
        '这项自测不证明日志造成38ms差距，也不证明关闭记录必然改善流水。',
        '依据测得量级决定是否值得进行执行机制匹配验证；本入口不自动修改运行库、不补测DNN、不调整评分或Top-20。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in
        [OUT/'AUDIT.json',OUT/'BUILD.json',OUT/'BOARD_RECEIPT.json',OUT/'RESULTS.json',REPORT/'FINDINGS_zh.txt']}),replace=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','build','validate','qemu','board','analyze'))
    parser.add_argument('--execute-board',action='store_true');args=parser.parse_args()
    value=board(execute_board=args.execute_board) if args.command=='board' else globals()[args.command]()
    if args.command in ('audit','build','validate'):print(json.dumps(dict(command=args.command,valid=True),ensure_ascii=False))
    else:print(json.dumps(value,ensure_ascii=False,indent=2))
