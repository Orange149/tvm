"""Project isolated polling counts without reading any pipeline timing files."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
import json,shutil,sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import digest,now,read,sha256,verify_inventory,write
from phase_model.extract import average
from device_phase.extract import split_run
from scheduler_gap_audit.curves import csv_write
import polling_profile_bridge as board

ROOT=board.ROOT
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_driver_polling'


def project_calls(spans,device_rows,warmup,scored):
    calls=[]
    for frame in range(warmup,warmup+scored):
        selected=[s for s in spans if s['operation']=='run' and s['call']==frame]
        if len(selected)!=1:raise ValueError('driver run phase incomplete')
        projected=split_run(selected[0],device_rows)
        events=sorted([r for r in device_rows if r['call']==frame],key=lambda r:r['wall_start_ns'])
        if len(projected['devices'])!=len(events):raise ValueError('driver event pairing incomplete')
        for row,event in zip(projected['devices'],events):
            row['poll_checks']=event['driver_poll_iters']
            for key in ('run_total','submit_mmio','post_start_sleep','poll_wait'):
                row[key+'_ns']=Fraction(str(event['driver_'+key+'_us']))*1000
        calls.append(projected)
    signature=lambda c:[d['instructions'] for d in c['devices']]
    if any(signature(c)!=signature(calls[0]) for c in calls):
        raise ValueError('device instruction sequence changed across scored frames')
    return calls


def analyze():
    state=read(board.RUN/'state.json');plan=read(board.OUTPUT/'PLAN.json')
    if state['status']!='COMPLETE' or state['boot_id']!=plan['expected_boot_id']:
        raise ValueError('polling component cohort not complete or mixed boot')
    profiles={};sources={};table=[];devices=[]
    for spec in plan['specs']:
        if spec['mode']!='observed':continue
        bundle=board.OUTPUT/spec['network']/'capsule';board.validate(bundle)
        docs={};items={}
        for mode in ('qualify','clean','observed'):
            doc=board.task(bundle,{**spec,'mode':mode});docs[mode]=doc
            path=board.RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json');item=read(path)
            if item['task']!=doc or item['boot_id']!=state['boot_id']:
                raise ValueError('polling accepted identity differs')
            verify_inventory(item['raw'],item['files'])
            if board.summarize(item['raw'],doc,bundle)!=item['summary']:
                raise ValueError('polling accepted result no longer replays')
            if Path(item['raw'],'task_hash').read_text().strip()!=digest(doc):
                raise ValueError('polling raw task hash differs')
            items[mode]=item;sources[str(path)]=sha256(path)
        fingerprints=[x['summary']['output_fingerprints'] for x in items.values()]
        if any(x!=fingerprints[0] for x in fingerprints):raise ValueError('polling output differs from qualification')
        doc=docs['observed'];observed=items['observed']['summary'];clean=items['clean']['summary']
        raw=Path(items['observed']['raw']);spans=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        calls=project_calls(spans,observed['device']['mapped'],doc['warmup'],doc['scored'])
        stages=dict(host_serial_CPU=[average([c['host_serial_CPU_ns'][i] for c in calls])
                                    for i in range(len(calls[0]['host_serial_CPU_ns']))],devices=[])
        for index,example in enumerate(calls[0]['devices']):
            group=[c['devices'][index] for c in calls]
            entry=dict(wall=average([r['wall_ns'] for r in group]),CPU=average([r['CPU_ns'] for r in group]),
                instructions=example['instructions'],poll_checks=average([r['poll_checks'] for r in group]))
            for key in ('run_total_ns','submit_mmio_ns','post_start_sleep_ns','poll_wait_ns'):
                entry[key]=average([r[key] for r in group])
            stages['devices'].append(entry)
            devices.append(dict(network=spec['network'],program=spec['case'],device_index=index,
                instructions=example['instructions'],mean_poll_checks=mean(r['poll_checks'] for r in group),
                minimum_poll_checks=min(r['poll_checks'] for r in group),maximum_poll_checks=max(r['poll_checks'] for r in group),
                device_CPU_ms=mean(r['CPU_ns'] for r in group)/1e6,
                device_wall_ms=mean(r['wall_ns'] for r in group)/1e6,
                submit_mmio_ms=float(sum(r['submit_mmio_ns'] for r in group)/len(group)/1_000_000),
                post_start_sleep_ms=float(sum(r['post_start_sleep_ns'] for r in group)/len(group)/1_000_000),
                poll_wait_ms=float(sum(r['poll_wait_ns'] for r in group)/len(group)/1_000_000),
                poll_cycle_proxy_us=float(sum(r['poll_wait_ns']/max(1,r['poll_checks']-1) for r in group)/len(group)/1000),
                poll_cycle_is_not_directly_observed_sleep_or_wakeup=True))
        cpu_change=100*(observed['cost']['run_core_ms']/clean['cost']['run_core_ms']-1)
        wall_change=100*(observed['cost']['run_ms']/clean['cost']['run_ms']-1)
        quality=abs(cpu_change)<=20 and abs(wall_change)<=20
        key=spec['network']+'/'+spec['case']
        profiles[key]=dict(network=spec['network'],program=spec['case'],stages=stages,
            clean_cost=clean['cost'],observed_cost=observed['cost'],source_boot_id=state['boot_id'],
            probe_run_CPU_change_percent=cpu_change,probe_run_wall_change_percent=wall_change,
            probe_quality_passed=quality,pipeline_FPS_read=False,
            queue_fields_are_not_observations=True,
            module_sha256=read(bundle/'plans/programs.json')[spec['case']]['files']['module.so']['sha256'])
        table.append(dict(network=spec['network'],program=spec['case'],
            device_calls_per_stage=len(stages['devices']),poll_checks_per_stage=mean(sum(d['poll_checks'] for d in c['devices']) for c in calls),
            inferred_nanosleep_calls_per_stage=mean(sum(d['poll_checks'] for d in c['devices']) for c in calls),
            inferred_syscall_count_not_measured_wakeup_count=True,
            clean_run_ms=clean['cost']['run_ms'],clean_run_CPU_ms=clean['cost']['run_core_ms'],
            observed_run_ms=observed['cost']['run_ms'],observed_run_CPU_ms=observed['cost']['run_core_ms'],
            observed_CPU_inside_driver_ms=mean(sum(d['CPU_ns'] for d in c['devices']) for c in calls)/1e6,
            observed_CPU_outside_driver_ms=mean(sum(c['host_serial_CPU_ns']) for c in calls)/1e6,
            probe_run_CPU_change_percent=cpu_change,probe_run_wall_change_percent=wall_change,probe_quality_passed=quality))
    if len(profiles)!=5:raise ValueError('all five isolated VTA signatures required')
    result=dict(profiles=profiles,component_table=table,device_table=devices,source_receipts=sources,
        native_state=state,failed_startup_batch=read(ROOT/'artifacts/polling_profile_01/FAILED_STARTUP.json'),
        no_pipeline_FPS_read=True,no_static_model_or_DP_changed=True,
        no_poll_count_as_pure_hardware_latency=True,
        no_actual_scheduler_wakeup_instants_measured=True,
        original_runtime_sha256=read(board.OUTPUT/'build/BUILD.json')['runtime_sha256'] if (board.OUTPUT/'build/BUILD.json').exists()
            else read(ROOT/'artifacts/polling_profile_01/build/BUILD.json')['runtime_sha256'],
        analysis_source_sha256=sha256(__file__),recorded_at=now())
    dest=board.OUTPUT/'analysis';dest.mkdir(parents=True,exist_ok=True)
    write(dest/'RESULTS.json',result,replace=True);csv_write(dest/'components.csv',table);csv_write(dest/'device_calls.csv',devices)
    REPORT.mkdir(parents=True,exist_ok=True)
    for path in (dest/'RESULTS.json',dest/'components.csv',dest/'device_calls.csv'):shutil.copy2(path,REPORT/path.name)
    lines=['VTA轮询组件核查','',
        '本批只运行5个真实VTA子图，15会话、105次组件调用，0次完整DNN。输入0，资格1帧，关闭/开启观察各2预热+8计入。',
        '运行库、执行器、模块均复用。仅每次真实VTADeviceRun前后读取已有驱动计数，不逐次拦截轮询。',
        '首次批次在首个阶段计时前缺少辅助接口而退出，原始失败保留；新批次补齐原有适配库，原等待、worker探针关闭、排队计数明确关闭。零排队字段不作为观测。','']
    for row in table:
        lines.append(f"{row['network']} {row['program']}：每次子图{row['device_calls_per_stage']}次设备调用、平均{row['poll_checks_per_stage']:.1f}次状态查询；干净运行{row['clean_run_ms']:.3f}ms、{row['clean_run_CPU_ms']:.3f}核ms；观察扰动CPU{row['probe_run_CPU_change_percent']:+.2f}%、墙钟{row['probe_run_wall_change_percent']:+.2f}%；质量条件通过={row['probe_quality_passed']}。")
    lines += ['',
        '成功设备调用中，驱动先休眠一次，随后每次未完成查询再休眠；由源码可推导休眠调用数等于成功查询计数。它不是调度唤醒次数的直接记录。',
        '这些计数可以检验“把驱动CPU工作集中成一块、只保留一个非运行区间”的模型简化。它们尚不能恢复每次轮询CPU片段、CPU排队和设备真实完成时刻。',
        '本次不输出吞吐优化结论，不把VTADeviceRun墙钟或poll_wait直接当硬件纯计算时间。后续预测必须保留这个区分。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n');shutil.copy2(dest/'FINDINGS_zh.txt',REPORT/'FINDINGS_zh.txt')
    write(board.OUTPUT/'DELIVERY.json',dict(files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},
        results_sha256=sha256(dest/'RESULTS.json'),goal_complete=False,recorded_at=now()),replace=True)
    return dict(components=table,native_sessions=15,VTA_component_calls=105,DNN_inferences=0,
                report=str(REPORT/'FINDINGS_zh.txt'))


if __name__=='__main__':print(json.dumps(analyze(),ensure_ascii=False,indent=2))
