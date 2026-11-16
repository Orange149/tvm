"""Locate wait mismatch by compiled kernel; no model edits or native calls."""
from pathlib import Path
import json
import re
import shutil
import statistics as st
import subprocess

from run_three_thread_worker_gap import ROOT,POOL,selection
from run_three_thread_worker_gap_compat import verified_receipts
import run_three_thread_host_gap as clean
from cpu_vta_partitioning.common import read,write,sha256
from scheduler_gap_audit.analysis import call_details
from scheduler_gap_audit.curves import csv_write
from phase_model.simulator import rational


def audit():
    bundle=POOL/'yolov3_tiny416/capsule';module=bundle/'programs/cpu_00_02/module.so'
    receipts=[r for _,r in verified_receipts() if r['task']['spec']['phase']=='observed']
    iso=next(r for r in receipts if r['task']['spec']['scope']=='component')
    _,ik,_=call_details(iso['raw'],iso['task']);ik=[k for k in ik if k['stage']=='cpu_00_02']
    offsets=[next(k['function_offset'] for k in ik if k['kernel']==i) for i in range(12)]
    names=subprocess.run(['addr2line','-f','-e',str(module),*map(hex,offsets)],capture_output=True,text=True,check=True).stdout.splitlines()
    if len(names)!=24 or any(names[2*i]=='??' for i in range(12)):raise ValueError('kernel debug mapping unavailable')
    rows=[]
    for r in receipts:
        t=r['task'];s=t['spec']
        if s['scope']!='pipeline':continue
        _,ks,_=call_details(r['raw'],t);ks=[k for k in ks if k['stage']=='cpu_00_02']
        model_path=clean.POOL/'existing_model_diagnostic'/(s['case']+'.json');model=read(model_path)['result']
        tasks={(v['frame'],v['kernel'],v['slot']):v for v in model['tasks'] if v['stage']==0 and v['operation']=='worker_compute' and v['generation']==0}
        for index in range(12):
            group=[k for k in ks if k['kernel']==index];old=[k for k in ik if k['kernel']==index]
            actual=[st.mean(next(w for w in k['worker_events'] if w['task']==slot)['cpu_end_ns']-next(w for w in k['worker_events'] if w['task']==slot)['cpu_start_ns'] for k in group)/1e6 for slot in range(3)]
            original=[st.mean(next(w for w in k['worker_events'] if w['task']==slot)['cpu_end_ns']-next(w for w in k['worker_events'] if w['task']==slot)['cpu_start_ns'] for k in old)/1e6 for slot in range(3)]
            rows.append(dict(role=selection()['roles'][s['case']],kernel=index,function_offset=offsets[index],
                debug_function_label=names[2*index],module_sha256=sha256(module),
                native_CPU_ms_by_slot=actual,isolated_CPU_ms_by_slot=original,
                native_callback_queue_lower_ms=st.mean(k['last_worker_body_queue_lower_ms'] for k in group),
                native_callback_queue_upper_ms=st.mean(k['last_worker_body_queue_upper_ms'] for k in group),
                native_entry_counter_ms=st.mean(k['last_worker_pre_entry_queue_counter_ms'] for k in group),
                model_same_slot_ready_wait_ms=st.mean(float(rational(tasks[k['frame'],index,k['last_worker']]['ready_not_running']))/1e6 for k in group),
                scopes_differ_no_fitted_penalty=True,model_source_sha256=sha256(model_path)))
    source=ROOT/'artifacts/finite_service_01/development/INPUTS.json';x=read(source)
    if sha256(source)!=read(source.parent/'PLAN.json')['inputs_sha256']:raise ValueError('isolated profile input changed')
    networks=[]
    for key in ('resnet18/static_first','yolov3_tiny416/split'):
        p=x['isolated']['cases'][key]['stages'][0]['profile']
        work=[float(sum((rational(v) for k in p['kernels'] for v in k['workers'][slot]['compute_segments']),0)/1e6) for slot in range(3)]
        networks.append(dict(network=key.split('/')[0],program='cpu_00_02',threads=3,
            kernel_launches=len(p['kernels']),callback_CPU_ms_by_slot=work,
            located_serial_caller_CPU_ms=float(sum((rational(v) for v in p['serial_cpu_segments']),0)/1e6),
            caller_CPU_outside_compute_callbacks_ms=float(rational(p['caller_CPU'])/1e6)-work[0],
            already_present_in_replay=True,not_same_batch_FPS_comparison=True))
    # Inspect computation bodies, not their argument-checking wrappers.
    objdump=Path('/home/orange/alinx_plsdk/install/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-objdump')
    relocation_text=subprocess.run(['readelf','-Wr',str(module)],capture_output=True,text=True,check=True).stdout
    relocations={int(p[0],16):p[4] for line in relocation_text.splitlines()
                 if len(p:=line.split())>=5 and p[2]=='R_AARCH64_GLOB_DAT'}
    pooling=[]
    for suffix in ('','_1','_2'):
        name='tvmgen_default_fused_nn_max_pool2d_multiply_round_clip_cast'+suffix+'_compute_'
        output=subprocess.run([str(objdump),'-d','--disassemble='+name,str(module)],capture_output=True,text=True,check=True).stdout
        if '<'+name+'>:' not in output:raise ValueError('pooling body symbol absent')
        calls=[line for line in output.splitlines() if '\tbl\t' in line or '\tblr\t' in line]
        base=None;target=None;resolved=[]
        for line in output.splitlines():
            m=re.search(r'\tadrp\tx8,\s+([0-9a-f]+)',line)
            if m:base=int(m[1],16);target=None
            m=re.search(r'\tldr\tx8, \[x8, #([0-9a-fx]+)\]',line)
            if m and base is not None:target=relocations.get(base+int(m[1],0))
            if '\tblr\tx8' in line:resolved.append(dict(instruction=line.strip(),GOT_symbol=target))
        if (len(calls)!=2 or len(resolved)!=2 or
                {r['GOT_symbol'] for r in resolved}!={'__TVMBackendAllocWorkspace','__TVMBackendFreeWorkspace'}):
            raise ValueError('pooling call targets not fully resolved; do not claim serial body')
        pooling.append(dict(symbol=name,branch_with_link_calls=calls,resolved_calls=resolved,
            only_workspace_allocation_and_release_subcalls=True,no_parallel_launch_in_compute_body=True))
    out=dict(rows=rows,network_work_structure=networks,pooling_body_audit=pooling,
        isolated_source=str(source),isolated_source_sha256=sha256(source),
        source_sha256=sha256(__file__),native_sessions_added=0,
        input_resolution_and_compiler_schedule_are_confounders=True,
        serial_work_already_modeled_not_an_extra_penalty=True,
        debug_parent_labels_do_not_identify_exact_tensor_loop=True,
        no_new_dispatch_model=True)
    dest=POOL/'kernel_audit';write(dest/'RESULTS.json',out,replace=True)
    csv_write(dest/'kernel_wait_comparison.csv',rows);csv_write(dest/'network_work_structure.csv',networks)
    lines=['逐内核等待差异与两个网络的执行负载','',
        '这是只读离线分析，没有新推理或模型改动。逐内核数字受主报告中探针质量限制；原生排队计数与模型ready等待的标记不同，不能直接相减当修正量。','',
        '主机固定重worker核心时，差异主要集中在哪些内核：',
        '内核 | 编译函数的调试标签 | 实测回调CPU排队上侧估计ms | 实测入口前计数增长ms | 模型同槽位ready等待ms']
    for r in rows:
        if r['role']=='host_on_heavier_worker':
            lines.append(f"{r['kernel']} | {r['debug_function_label']} | {r['native_callback_queue_upper_ms']:.3f} | {r['native_entry_counter_ms']:.3f} | {r['model_same_slot_ready_wait_ms']:.3f}")
    lines += ['', '观察：前两个卷积计算回调（内核2、6）实测排队约20.8／18.5ms，模型同槽位等待约7.5／6.4ms；第三个卷积回调（内核10）则较接近。模型也把若干毫秒等待分配到实际几乎未排队的短回调上。',
        '因此误差不只是等待总量偏少，还包含等待发生位置与完成顺序不符。继续修改线程份额或给所有算子统一加慢化系数，没有得到本批证据支持。','',
        '冻结隔离记录中的两个网络前段结构（各自CPU0–2三线程）：',
        '网络 | 并行启动数 | 可定位串行调用线程CPU ms | 全部回调CPU ms | 调用线程回调外CPU ms']
    for r in networks:
        lines.append(f"{r['network']} | {r['kernel_launches']} | {r['located_serial_caller_CPU_ms']:.3f} | {sum(r['callback_CPU_ms_by_slot']):.3f} | {r['caller_CPU_outside_compute_callbacks_ms']:.3f}")
    lines += ['YOLO在本编译模块中具有约24.6ms可定位串行段，ResNet约5.9ms。YOLO三个池化计算体各有两处间接调用，已通过GOT重定位核实为工作区分配和释放；它们没有并行启动，完整回调序列也未出现池化并行启动。',
        '串行段和并行段交替，会改变主机工作能否利用空闲核心的时间窗口；相同整帧CPU核时不决定每次完成汇合的耗时。这个差异有助于解释为何容量近似对两个负载的适用程度不同。',
        '这些串行段已经进入现有时序回放，不是新发现的漏计成本，不能再次相加。该比较还包含224与416输入尺寸、网络形状及编译调度差异，不能概括为所有ResNet或YOLO都遵循同一规律。','',
        '下一步需要解决的是可运行任务与CPU服务时序的预测，而不是补造几十毫秒计算量。本轮没有把新的流水排队或CPU时间回填为可部署profile。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    report=ROOT.parents[4]/'paper/my_paper/analysis/20261111_three_thread_worker_gap/kernel_audit'
    report.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():
        if p.is_file():shutil.copy2(p,report/p.name)
    return dict(report=str(report/'FINDINGS_zh.txt'),network_structure=networks,pooling=pooling)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
