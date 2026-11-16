"""Locate worker entry delay in existing traces, without fitting a predictor.

The real runtime posts all worker queues before invoking logical task zero.
Caller callback entry is therefore a conservative 'posted by' landmark,
not a measured sched_wakeup event or a guarantee of immediate execution.
"""
from collections import defaultdict
from pathlib import Path
from statistics import median
import json
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from yolo_scheduler_trace.events import clip,intersection
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/worker_dispatch_gap_01'


def entry_gap(worker,caller,stream,raw):
    if raw['tid']!=worker['tid'] or raw['task']!=worker['slot'] or raw['launch']!=worker['launch']:
        raise ValueError('callback identity mismatch')
    if raw['wall_start_ns']!=worker['callback_start_ns']:
        raise ValueError('callback entry projection differs')
    posted_by=caller['callback_start_ns'];entry=worker['callback_start_ns']
    a,b=min(posted_by,entry),entry
    body=clip(stream,a,b)
    if not body['fully_covered']:raise ValueError('entry interval not fully captured')
    q0,q1=raw['queue_read_start0_ns'],raw['queue_read_end0_ns']
    if not 0<q0<=q1<=entry:raise ValueError('entry probe envelope invalid')
    lo,hi=max(a,q0),min(b,q1)
    probe=clip(stream,lo,hi) if hi>lo else dict(on_CPU_ns=0,preempt_off_ns=0,nonpreempt_off_ns=0,unknown_ns=0)
    fields=('on_CPU_ns','preempt_off_ns','nonpreempt_off_ns','unknown_ns')
    if sum(body[k] for k in fields)!=b-a:raise ValueError('entry time partition does not conserve wall')
    if sum(probe[k] for k in fields)!=intersection(a,b,q0,q1):raise ValueError('entry probe counted twice')
    return dict(frame=worker['frame'],kernel=worker['kernel'],slot=worker['slot'],tid=worker['tid'],
        last_worker=worker['last'],posted_by_ns=posted_by,entry_ns=entry,
        began_before_caller=entry<posted_by,post_caller_entry_gap_ns=b-a,
        **{k:body[k] for k in fields},entry_counter_probe_overlap_ns=max(0,hi-lo),
        probe_on_CPU_ns=probe['on_CPU_ns'],probe_preempt_off_ns=probe['preempt_off_ns'],
        probe_nonpreempt_off_ns=probe['nonpreempt_off_ns'],
        off_after_nonpreempt_out_is_not_all_sleep=True,exact_wakeup_time_known=False)


def run():
    base=ROOT/'artifacts/yolo_scheduler_trace_repair_01';path=base/'analysis/RESULTS.json'
    if sha256(path)!=read(base/'DELIVERY.json')['results_sha256']:
        raise ValueError('accepted trace changed')
    trace=read(path)
    if read(base/'FINAL_VALIDATION.json')['status']!='PASSED' or not trace['probe_quality_passed']:
        raise ValueError('selected native trace validity unproven')
    for rp,h in trace['source_receipts'].items():
        if sha256(rp)!=h:raise ValueError('native source receipt changed')
        item=read(rp)
        if item['task']['spec']['phase']=='observed':receipt=item
    verify_inventory(receipt['raw'],receipt['files'])
    raw_path=Path(receipt['raw'])/'wait_trace.jsonl'
    raw=[json.loads(s) for s in raw_path.read_text().splitlines()]
    index={}
    launches={}
    for r in raw:
        if r['kind']==1:
            key=r['launch'],r['task']
            if key in index:raise ValueError('duplicate raw worker callback')
            index[key]=r
        elif r['kind']==0:
            if r['launch'] in launches:raise ValueError('duplicate raw launch')
            launches[r['launch']]=r
    groups=defaultdict(list)
    for w in trace['worker_rows']:groups[w['frame'],w['kernel']].append(w)
    if set(groups)!={(f,k) for f in range(12,24) for k in range(12)}:
        raise ValueError('native launch population differs')
    rows=[]
    for (frame,kernel),workers in sorted(groups.items()):
        if sorted(w['slot'] for w in workers)!=[0,1,2,3]:raise ValueError('logical slots missing or duplicate')
        caller=next(w for w in workers if w['slot']==0)
        for w in workers:
            row=entry_gap(w,caller,trace['observed']['scheduler']['streams'][str(w['tid'])],index[w['launch'],w['slot']])
            # This is an independently observed cumulative increment, not an
            # exact within-landmark queue time: pending delay may be accounted
            # only at dispatch, and the remote baseline has a read envelope.
            baseline=launches[w['launch']]['worker_queue_baselines'][w['slot']]
            r=index[w['launch'],w['slot']]
            if baseline['tid']!=w['tid'] or r['queue_before_ns']<baseline['queue_ns']:
                raise ValueError('worker queue baseline TID/counter differs')
            row.update(pre_callback_queue_increment_ns=r['queue_before_ns']-baseline['queue_ns'],
                queue_baseline_read_start_ns=baseline['read_start_ns'],
                queue_baseline_read_end_ns=baseline['read_end_ns'],
                queue_increment_not_exact_post_caller_interval=True)
            rows.append(row)
    critical=[r for r in rows if r['last_worker']]
    if len(critical)!=144:raise ValueError('critical worker population differs')
    totals={k:sum(r[k] for r in critical) for k in (
        'post_caller_entry_gap_ns','on_CPU_ns','preempt_off_ns','nonpreempt_off_ns','unknown_ns',
        'entry_counter_probe_overlap_ns','probe_on_CPU_ns','probe_preempt_off_ns','probe_nonpreempt_off_ns')}
    # Caller task zero defines the landmark, so its pre-callback baseline
    # increment is a separate scope and excluded from the worker cross-check.
    totals['noncaller_queue_increment_ns']=sum(r['pre_callback_queue_increment_ns'] for r in critical if r['slot']!=0)
    summary={k+'_ms_per_front_frame':v/12/1e6 for k,v in totals.items()}

    # Existing model already has queueing. Match the same logical slot as
    # well as the model's own last-finisher, rather than adding native gaps.
    dev=ROOT/'artifacts/finite_service_01/development'
    seal=read(dev/'REPLAY_FREEZE.json');model_path=None
    for p,h in seal['sources'].items():
        if sha256(p)!=h:raise ValueError('frozen finite model source changed')
    for p,h in seal['files'].items():
        d=read(dev/p)
        if d['spec']==dict(case='yolov3_tiny416/overlap',mode='finite',placement='wake_boundary'):
            if sha256(dev/p)!=h:raise ValueError('frozen finite model replay changed')
            model_path,model=dev/p,d['result']
    if model_path is None:raise ValueError('matched frozen model absent')
    task_index={(r['frame'],r['kernel'],r['slot']):r for r in model['tasks']
                if r['stage']==0 and r['operation']=='worker_compute' and r['generation']==0}
    joins={(r['frame'],r['kernel']):r for r in model['kernel_joins'] if r['stage']==0}
    matched=[]
    for native in critical:
        key=native['frame'],native['kernel'];caller=task_index[key+(0,)]
        worker=task_index[key+(native['slot'],)];join=joins[key]
        modeled=max(0,worker['first_service_ns']-caller['first_service_ns'])
        own_last=max(max(0,task_index[key+(s,)]['first_service_ns']-caller['first_service_ns']) for s in join['last_slots'])
        actual_ends=sorted(w['callback_end_ns'] for w in groups[key])
        matched.append(dict(frame=key[0],kernel=key[1],native_last_slot=native['slot'],
            model_last_slots=join['last_slots'],last_worker_identity_agrees=native['slot'] in join['last_slots'],
            native_post_caller_callback_entry_ns=native['post_caller_entry_gap_ns'],
            model_post_caller_first_service_same_slot_ns=modeled,
            model_post_caller_first_service_own_last_slot_ns=own_last,
            native_last_to_second_last_callback_end_margin_ns=actual_ends[-1]-actual_ends[-2],
            callback_entry_and_first_service_are_distinct_envelopes=True))
    comparison=dict(matched_launches=len(matched),
        native_post_caller_entry_ms_per_frame=sum(r['native_post_caller_callback_entry_ns'] for r in matched)/12/1e6,
        model_same_slot_post_caller_first_service_ms_per_frame=sum(r['model_post_caller_first_service_same_slot_ns'] for r in matched)/12/1e6,
        model_own_last_post_caller_first_service_ms_per_frame=sum(r['model_post_caller_first_service_own_last_slot_ns'] for r in matched)/12/1e6,
        last_worker_identity_agreement_fraction=sum(r['last_worker_identity_agrees'] for r in matched)/len(matched),
        median_native_last_to_second_margin_ms=median(r['native_last_to_second_last_callback_end_margin_ns'] for r in matched)/1e6,
        native_last_margin_over_1ms=sum(r['native_last_to_second_last_callback_end_margin_ns']>1_000_000 for r in matched),
        no_difference_added_to_prediction=True,existing_model_is_not_latest_independent_device_variant=True)

    runtime=base/'capsule/runtime/libtvm_runtime.so'
    if sha256(runtime)!='724d15640bc5f75548725048f9b6adb1750bce768aa6ea39fe3fbaeb6726af98':
        raise ValueError('actual runtime differs from the native diagnostic')
    objdump=Path(read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['toolchain']['tools']['CXX']['path']).with_name('aarch64-xilinx-linux-objdump')
    text=subprocess.check_output([str(objdump),'-d','--start-address=0xe8fe0','--stop-address=0xe998c',str(runtime)],text=True)
    if not all(s in text for s in ('e95c8:', 'condition_variable10notify_one','e9240:', 'e95f4:')):
        raise ValueError('frozen runtime dispatch landmarks not found')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'actual_runtime_launch.asm').write_text(text)
    kernel_rows=[]
    for k in range(12):
        selected=[r for r in critical if r['kernel']==k]
        kernel_rows.append(dict(kernel=k,calls=len(selected),
            mean_post_caller_entry_gap_ms=sum(r['post_caller_entry_gap_ns'] for r in selected)/12/1e6,
            mean_preempt_off_ms=sum(r['preempt_off_ns'] for r in selected)/12/1e6,
            mean_off_after_nonpreempt_out_ms=sum(r['nonpreempt_off_ns'] for r in selected)/12/1e6,
            mean_counter_read_overlap_ms=sum(r['entry_counter_probe_overlap_ns'] for r in selected)/12/1e6))
    result=dict(summary=summary,workers=rows,kernels=kernel_rows,matched_model=comparison,
        matched_model_path=str(model_path),matched_model_sha256=sha256(model_path),matched_kernel_rows=matched,
        native_trace_sha256=sha256(path),raw_path=str(raw_path),raw_sha256=sha256(raw_path),
        actual_runtime_sha256=sha256(runtime),objdump_sha256=sha256(objdump),
        runtime_assembly_sha256=sha256(OUT/'actual_runtime_launch.asm'),source_sha256=sha256(__file__),
        native_boot=trace['boot_id'],selected_frames=list(range(12,24)),
        task_posting_landmark_confirmed_in_actual_binary=True,
        no_new_native_sessions=True,no_new_DNN=True,no_model_change=True,
        no_sum_of_all_worker_delays_added_to_stage=True,
        exact_wakeup_time_unknown=True,probe_interference_not_eliminated=True,
        queue_accounting_source='https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/delayacct.c',
        pending_run_delay_accounting_source='https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/stats.h')
    write(OUT/'RESULTS.json',result,replace=True)
    csv_write(OUT/'worker_entry_intervals.csv',rows);csv_write(OUT/'kernel_entry_gaps.csv',kernel_rows)
    csv_write(OUT/'matched_model_entry_gaps.csv',matched)
    lines=['任务已投递，但worker尚未开始计算：已有轨迹核查','',
        '本轮离线，没有新增DNN会话、模型或DP修改；使用无丢失的帧12–23，共144次启动、576个回调。',
        '实际运行库反汇编已核对：全部worker队列投递及必要notify_one在逻辑线程0回调之前；以caller回调开始作为“已经投递”的保守时刻。',
        '该时刻不是sched_wakeup事件，不保证各worker已经被调度，也不将投递后的全部时间叫作CPU排队。','',
        f"每次启动只选最后完成的一个worker，其在caller开始后、自己进入计算前的累计间隔为{summary['post_caller_entry_gap_ns_ms_per_front_frame']:.3f}ms/前段帧。",
        f"其中实际运行{summary['on_CPU_ns_ms_per_front_frame']:.3f}ms，抢占标记离开CPU后未恢复{summary['preempt_off_ns_ms_per_front_frame']:.3f}ms，非抢占离开后未恢复{summary['nonpreempt_off_ns_ms_per_front_frame']:.3f}ms，未覆盖0。",
        f"已记录的入口计数读取包络与上述间隔重合{summary['entry_counter_probe_overlap_ns_ms_per_front_frame']:.3f}ms；其中非抢占off重合{summary['probe_nonpreempt_off_ns_ms_per_front_frame']:.3f}ms。不能用已定位的这次入口读数直接解释其余长间隔。",
        f"另一个累计CPU排队计数在前置baseline到关键非caller回调入口之间增长{summary['noncaller_queue_increment_ns_ms_per_front_frame']:.3f}ms/帧，支持启动前CPU排队确实存在；其起止窗口略宽、可含先前未结算等待，不能当作严格的同窗口17.85ms分解。",'',
        '重要解释：worker此前因队列空而睡眠，之后即使被唤醒、已能运行，直到再次上CPU之前不会新增一次switch-in记录。因此“上次是非抢占离开”并不能证明这整段都在睡眠。',
        '此处是有探针流水的局部时序，不能从关闭探针的220.26ms前段直接扣除17.85ms，也不能把它加到隔离成本上。模型本身已经有部分启动等待，二者需要匹配比较。',
        f"匹配的旧有限服务回放：按其自己预测的最后worker统计，已有{comparison['model_own_last_post_caller_first_service_ms_per_frame']:.3f}ms/帧的caller首次服务后启动延迟；若跟踪实测中真正最后完成的逻辑槽位，该模型仅给{comparison['model_same_slot_post_caller_first_service_ms_per_frame']:.3f}ms/帧。最后worker身份一致率{100*comparison['last_worker_identity_agreement_fraction']:.2f}%。",
        '这表明缺口涉及“谁先获得CPU、谁最后完成”的预测，不能归纳成完全漏掉了一笔17.85ms开销。原生callback入口与模型first-service边界也不同，不把两者相减当作已证明的阶段因果差额。',
        f"实测最后与倒数第二worker完成时间差的中位数为{comparison['median_native_last_to_second_margin_ms']:.3f}ms，144次中{comparison['native_last_margin_over_1ms']}次超过1ms；最后worker身份的差异并非全部来自极近的并列。该阈值只描述轨迹，不用于选点或统计显著性判断。",
        '当前可确认缺口：隔离工作量不足以唯一决定任务投递后何时获得CPU；现有理想/有限共享近似对该时序仍有误差。尚未证明通用修正或FPS收益。',
        '下一步优先检查现有模型是否重现同类投递后启动拖尾，并用独立的固定唤醒负载验证相应调度假设；不直接扩大DNN试验或全域搜索。']
    (OUT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    report=ROOT.parents[4]/'paper/my_paper/analysis/20261111_worker_dispatch_gap';report.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,report/p.name)
    return dict(summary=summary,report=str(report/'FINDINGS_zh.txt'))


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
