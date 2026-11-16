"""Independent, offline report for the sealed current-YOLO VTA diagnosis.

Never change the native controller or promote pipeline observations into costs.
Graph-frame averages and the network-completion window have distinct scopes.
"""
from pathlib import Path
import json
import os
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, digest, verify_inventory
from scheduler_gap_audit.curves import csv_write
from matched_device_profile.records import ns
from audit_formal_window_resources import complement

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/matched_device_profile_20261112_01/analysis'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_matched_device_profile'


def frame_parts(document, rows, projection):
    """Nonoverlapping wall-clock pieces, with integer-ns conservation."""
    if not projection['per_frame_projection_supported']:
        raise ValueError('a frame association cannot be invented for decomposition')
    answer = {}
    for index, stage in enumerate(document['stages']):
        if stage['device'] != 'vta':
            continue
        per_frame = []
        for event in projection['stages'][stage['program']]['frames']:
            row = rows[event['frame']]
            boundary = ns(row[f'stage{index}_set_ms']) + ns(row[f'stage{index}_get_ms'])
            run = ns(row[f'stage{index}_run_ms'])
            service = ns(row[f'stage{index}_ms'])
            if abs(run - event['run_wall_ns']) > 2 or abs(boundary + run - service) > 2:
                raise ValueError('phase clocks do not conserve native stage service')
            driver = event['driver_wall_ns']
            outside = event['outside_driver_run_wall_ns']
            if driver + outside != event['run_wall_ns'] or min(boundary, driver, outside) < 0:
                raise ValueError('device pieces overlap or lose time')
            per_frame.append(dict(frame=event['frame'], boundary_ns=boundary,
                driver_wall_ns=driver, outside_driver_ns=outside,
                service_ns=boundary + driver + outside,
                driver_CPU_ns=event['driver_CPU_ns'],
                completion_lower_ns=event['completion_lower_sum_ns'],
                completion_upper_ns=event['completion_upper_sum_ns'],
                read_clock_envelope_ns=event['read_clock_envelope_ns']))
        fields = set(per_frame[0]) - {'frame'}
        answer[stage['program']] = dict(frames=per_frame,
            means_ms={k.removesuffix('_ns') + '_ms': statistics.mean(v[k] for v in per_frame)/1e6
                      for k in sorted(fields)},
            frame_scope='graph frame IDs in the scored set; not clipped to network completion window')
    return answer


def compare_parts(isolated, pipeline):
    if set(isolated) != set(pipeline):
        raise ValueError('different programs cannot be compared as a matched profile')
    rows = []
    for program in sorted(isolated):
        a, b = isolated[program]['means_ms'], pipeline[program]['means_ms']
        row = dict(program=program, isolated=a, pipeline=b,
            changes_ms={k: b[k]-a[k] for k in a},
            completion_difference_interval_ms=[b['completion_lower_ms']-a['completion_upper_ms'],
                                               b['completion_upper_ms']-a['completion_lower_ms']],
            completion_intervals_are_not_statistical_CI=True,
            outside_driver_not_identified_as_all_CPU_or_scheduler_wait=True)
        rows.append(row)
    totals = {side: {k: sum(row[side][k] for row in rows) for k in rows[0][side]}
              for side in ('isolated', 'pipeline')}
    totals['changes_ms'] = {k: totals['pipeline'][k]-totals['isolated'][k]
                            for k in totals['isolated']}
    d = totals['changes_ms']
    if abs(d['service_ms'] - d['boundary_ms'] - d['driver_wall_ms'] - d['outside_driver_ms']) > 1e-8:
        raise ValueError('matched differences fail conservation')
    totals['sums_of_VTA_services_not_a_universal_pipeline_II_formula'] = True
    return rows, totals


def read_native_rows(receipt):
    return [json.loads(line) for line in (Path(receipt['raw'])/'result.jsonl').read_text().splitlines()]


def run():
    import run_matched_device_profile as native
    if OUT.exists() or REPORT.exists():
        raise FileExistsError('preserve the accepted offline report')
    native.validate()
    plan = read(native.OUT/'PLAN.json')
    state = read(native.RUN/'state.json')
    if state['status'] != 'COMPLETE' or state['accepted_sessions'] != 11:
        raise ValueError('all eleven frozen native sessions must be accepted first')
    tasks = read(native.OUT/'TASKS.json')
    if tasks['plan_sha256'] != sha256(native.OUT/'PLAN.json'):
        raise ValueError('native task list is not the frozen plan')
    sources = {str(native.OUT/name): sha256(native.OUT/name)
               for name in ('PLAN.json', 'TASKS.json', 'COMPONENT_FREEZE.json', 'QEMU.json')}
    sources[str(native.RUN/'state.json')] = sha256(native.RUN/'state.json')
    sources[str(Path(__file__))] = sha256(__file__)
    sources[str(ROOT/'matched_device_profile/records.py')] = sha256(ROOT/'matched_device_profile/records.py')
    items = {}
    native_calls = full_calls = 0
    for doc in tasks['tasks']:
        path = native.RUN/'accepted'/state['boot_id']/(doc['task_id']+'.json')
        receipt = read(path)
        if receipt['task'] != doc or receipt['boot_id'] != plan['expected_boot']:
            raise ValueError('receipt boot/task changed')
        verify_inventory(receipt['raw'], receipt['files'])
        raw = Path(receipt['raw'])
        if (raw/'boot_id').read_text().strip() != state['boot_id'] or (raw/'task_hash').read_text().strip() != digest(doc):
            raise ValueError('native raw identity changed')
        # Closed interval tuples are stored as JSON arrays in accepted receipts.
        # Compare their canonical serialized content, retaining every integer.
        if digest(native.summarize(raw, doc, native.BUNDLE)) != digest(receipt['summary']):
            raise ValueError('accepted measurements cannot be independently reconstructed')
        spec = doc['spec']; key = spec['scope'], spec['case'], spec['mode']
        if key in items:
            raise ValueError('duplicate accepted session')
        items[key] = receipt
        count = len(read_native_rows(receipt))
        if spec['scope'] == 'component': native_calls += count
        else: full_calls += count
        sources[str(path)] = sha256(path)
    if (len(items), native_calls, full_calls) != (11, 63, 49):
        raise ValueError('native scope differs from the frozen small study')
    component_freeze = read(native.OUT/'COMPONENT_FREEZE.json')
    if not component_freeze['pipeline_allowed'] or not all(r['passed'] for r in component_freeze['quality'].values()):
        raise ValueError('component probe quality failed')
    first_pipeline_start = min(r['measurement_started_at'] for k, r in items.items() if k[0] == 'pipeline')
    if any(r['accepted_at'] >= first_pipeline_start for k, r in items.items() if k[0] == 'component'):
        raise ValueError('component profiles were not accepted before pipeline observation')
    for path, expected in component_freeze['source_receipts'].items():
        if sha256(path) != expected:
            raise ValueError('component projection source changed')
    for scope, case in {(k[0], k[1]) for k in items}:
        fingerprints = items[scope, case, 'qualify']['summary']['output_fingerprints']
        if any(v['summary']['output_fingerprints'] != fingerprints for k, v in items.items() if k[:2] == (scope, case)):
            raise ValueError('probe or measurement changed complete outputs')
    clean_ref = plan['clean_reference']
    if sha256(clean_ref['receipt']) != clean_ref['receipt_sha256']:
        raise ValueError('previously accepted clean pipeline changed')
    clean = read(clean_ref['receipt']); verify_inventory(clean['raw'], clean['files'])
    if clean['boot_id'] != state['boot_id'] or clean['task']['configuration'] != items['pipeline', 'static_first', 'observed']['task']['configuration']:
        raise ValueError('clean pipeline is not matched to the observer layout and boot')
    from common_top20.runtime import summary as old_summary
    clean_bundle = native.PARENT/'yolov3_tiny416/capsule'
    if digest(old_summary(clean['raw'], clean['task'], clean_bundle)) != digest(clean['summary']):
        raise ValueError('clean pipeline summary changed')
    sources[clean_ref['receipt']] = clean_ref['receipt_sha256']
    isolated = {}
    refresh = []
    for name in ('vta_01_01', 'vta_05_06', 'vta_09_11'):
        observed = items['component', name, 'observed']
        isolated.update(frame_parts(observed['task'], read_native_rows(observed), observed['summary']['projection']))
        plain = items['component', name, 'clean']['summary']['cost']
        refresh.append(dict(program=name, clean_isolated_service_ms=plain['service_ms'],
            observed_isolated_service_ms=observed['summary']['cost']['service_ms'],
            isolated_total_CPU_ms=plain['run_core_ms']+plain['boundary_core_ms'],
            probe_changes_percent=component_freeze['quality'][name]['changes_percent']))
    observed = items['pipeline', 'static_first', 'observed']
    raw = read_native_rows(observed)
    pipeline = frame_parts(observed['task'], raw, observed['summary']['projection'])
    rows, totals = compare_parts(isolated, pipeline)
    probe = dict(clean_pipeline_FPS=1000/clean['summary']['II_ms'],
        observed_pipeline_FPS_not_formal=1000/observed['summary']['II_ms'],
        clean_II_ms=clean['summary']['II_ms'], observed_II_ms=observed['summary']['II_ms'],
        II_change_percent=100*(observed['summary']['II_ms']/clean['summary']['II_ms']-1),
        VTA_stage_service_change_percent={s['program']: 100*(observed['summary']['stage_service_ms'][i]/clean['summary']['stage_service_ms'][i]-1)
            for i,s in enumerate(observed['task']['stages']) if s['device']=='vta'},
        separate_processes_and_time_blocks=True, not_a_statistical_replicate=True)
    # Resource occupancy is separately clipped to the actual completion window.
    begin, end = [ns(raw[i]['completion_ms']) for i in (8, 23)]
    holds = [(ns(r[f'stage{i}_vta_mutex_acquire_ms']), ns(r[f'stage{i}_vta_mutex_release_ms']))
             for i,s in enumerate(observed['task']['stages']) if s['device']=='vta' for r in raw]
    busy, gaps = complement(begin, end, holds)
    occupancy = dict(VTA_software_lock_occupancy_percent=100*busy/(end-begin),
        held_ms_per_completion=busy/15/1e6, unheld_ms_per_completion=sum(b-a for a,b in gaps)/15/1e6,
        not_physical_device_compute_utilization=True)
    # TID counters are whole-session, coarse-grained evidence, never scored CPU time.
    host_ledger = [r for r in observed['summary']['thread_ledger'] if r['role']=='caller' and r.get('stage') in isolated]
    if len(host_ledger) != 3 or any(r['scope'] != 'whole_session_including_warmup_and_tail' for r in host_ledger):
        raise ValueError('host CPU counter scope/identity differs')
    all_driver_cpu = sum(e['cpu_end_ns']-e['cpu_start_ns'] for e in observed['summary']['device']['whole_session_events'])
    whole = dict(stage_caller_CPU_ms=sum(r['cpu_time_delta_ms'] for r in host_ledger),
        driver_CPU_ms=all_driver_cpu/1e6, graph_calls=48*3,
        all_three_hosts_CPU_ms_per_DNN_proxy=sum(r['cpu_time_delta_ms'] for r in host_ledger)/48,
        includes_initialization_warmup_tail_and_coarse_clock=True,
        not_subtractable_from_scored_wall_for_exact_off_CPU_attribution=True)
    result = dict(stage_rows=rows, totals=totals, isolated=isolated, pipeline=pipeline,
        refreshed_costs=refresh, probe_effect=probe, occupancy=occupancy, whole_session_CPU=whole,
        frame_alignment_unique_across_all_admissible_clock_origins=True,
        every_native_receipt_reconstructed=True, sources=sources,
        native_sessions=11, VTA_component_calls=63, full_DNN_inferences=49,
        previous_clean_pipeline_not_remeasured=True, no_reboot=True, no_reprogram=True,
        original_executor_and_runtime_unchanged=True, new_DP_or_Top20=False,
        no_new_score_or_pipeline_fitted_cost=True, whole_goal_complete=False)
    OUT.mkdir(); REPORT.mkdir(parents=True)
    write(OUT/'RESULTS.json', result)
    csv_write(REPORT/'stage_comparison.csv', [dict(program=r['program'],
        **{side+'_'+k: v for side in ('isolated', 'pipeline', 'changes_ms') for k,v in r[side].items()}) for r in rows])
    csv_write(REPORT/'probe_effect.csv', refresh)
    report(result)
    draw(result)
    files = [OUT/'RESULTS.json', *REPORT.iterdir()]
    write(OUT/'MANIFEST.json', dict(files={str(p):sha256(p) for p in files if p.is_file()}, sources=sources,
        new_board_connections_for_analysis=0, native_scope_reverified=[11,63,49],
        analyzer_source_sha256=sha256(__file__)))
    print(json.dumps(dict(totals=totals, probe=probe, report=str(REPORT/'FINDINGS_zh.txt')), ensure_ascii=False))


def report(result):
    t, p = result['totals'], result['probe_effect']
    delta = t['changes_ms']
    lines = [
        '当前YOLO全域静态第一名：三个VTA子图的匹配隔离／流水诊断',
        '对象为cfg_2787a77188dd5f9363d00b15；不是早期CPU0–2四线程的固定切图。',
        '当前切图：CPU0 → VTA1 → CPU2–4 → VTA5–6 → CPU7–8 → VTA9–11 → CPU12–13。',
        'VTA主机均固定核心3，控制共享四核；普通复制、逐线程固定、门控关闭、队列2、输入0。',
        '本批11原生会话、63次隔离VTA调用、49次完整推理；原先已验收的无探针48帧流水不重测。',
        '未重启、未重载FPGA、未修改执行器／运行库／模块／评分／DP。全部输出、亲和性、任务、boot、原始文件哈希复核通过。',
        '', '先看测量可信度：',
        '三个隔离子图的探针run墙钟变化分别为%+.3f%%、%+.3f%%、%+.3f%%；run核时变化分别为%+.3f%%、%+.3f%%、%+.3f%%。' %
        tuple([r['probe_changes_percent'][k] for k in ('run_ms','run_core_ms') for r in result['refreshed_costs']]),
        '探针只包裹原有调用及MMIO读写，不增加MMIO读取；完整流水288次设备调用均被记录，无溢出。',
        '相对时钟原点虽未知，但在全部可行原点下事件属于同一帧，故逐帧对应成立；不选择最接近FPS的原点。',
        '无探针流水%.6f FPS、II %.3f ms；探针流水%.6f FPS、II %.3f ms（II变化%+.3f%%）。探针FPS仅用于扰动核查。' %
        (p['clean_pipeline_FPS'],p['clean_II_ms'],p['observed_pipeline_FPS_not_formal'],p['observed_II_ms'],p['II_change_percent']),
        '三个VTA阶段服务时间探针变化为：'+', '.join('%s %+.3f%%'%v for v in p['VTA_stage_service_change_percent'].items())+'。',
        '', '每项为对应阶段帧8–23的均值；不是按同一网络完成窗口裁剪的时长：',
        '子图 / 隔离服务ms / 流水服务ms / 服务增量ms / 驱动调用外run增量ms / 驱动内墙钟增量ms',
    ]
    for r in result['stage_rows']:
        lines.append('%s / %.3f / %.3f / %+.3f / %+.3f / %+.3f' % (r['program'],r['isolated']['service_ms'],
            r['pipeline']['service_ms'],r['changes_ms']['service_ms'],r['changes_ms']['outside_driver_ms'],r['changes_ms']['driver_wall_ms']))
    lines += [
        '', '本轮实际定位到什么：',
        '三个VTA隔离服务合计%.3f ms，流水内合计%.3f ms，差%.3f ms。' % (t['isolated']['service_ms'], t['pipeline']['service_ms'], delta['service_ms']),
        '增长分解：驱动调用外run %+.3f ms；驱动调用内 %+.3f ms；输入／输出交接 %+.3f ms，三者守恒。' %
        (delta['outside_driver_ms'],delta['driver_wall_ms'],delta['boundary_ms']),
        '设备启动至完成的观测区间总和：隔离[%.3f, %.3f] ms，流水[%.3f, %.3f] ms。二者区间重叠，未观察到约24ms的纯设备完成时间增长。' %
        (t['isolated']['completion_lower_ms'],t['isolated']['completion_upper_ms'], t['pipeline']['completion_lower_ms'],t['pipeline']['completion_upper_ms']),
        '上述区间是最后未完成／首次完成读取给出的时间包络，不是置信区间，也不证明设备绝对不受访存竞争影响。',
        '同一完成窗口内VTA软件锁占用%.3f%%；这把当前瓶颈指向被串行持有的完整VTA服务，不能称为纯FPGA计算利用率。' % result['occupancy']['VTA_software_lock_occupancy_percent'],
        '', '为什么已经计入主机CPU需求仍会乐观：',
        '现有静态评分把主机核时放入CPU容量，同时把隔离VTA墙钟累计成单VTA需求。前者计量CPU工作量，后者默认隔离服务仍适用。',
        '真实执行在持有同一VTA软件锁时，还执行驱动调用外的图运行、布局／复制及主机工作。它们与CPU线程争用核心时，锁内墙钟可能延长；两个总量取最大值并不能表达这一时序耦合。',
        '因此，本轮支持“误差主要落在驱动之外的图运行墙钟”这一定位，不支持“还没把主机核时计进去”或“FPGA算力突然变差”的说法。',
        '驱动之外的时间仍混有CPU执行、运行时、访存、缺CPU和其他非运行时间；没有逐段TID时钟，不能把它全叫调度等待。',
        '', '成本刷新与模型改进分开：',
        '新无探针隔离VTA服务之和%.3f ms，与旧冻结表约170.127 ms接近，成本陈旧不足以解释本次约193ms的II。' % sum(r['clean_isolated_service_ms'] for r in result['refreshed_costs']),
        '这些新记录没有回填原DP；流水内增量不能作为所有切图通用的慢化系数。',
        '已有时序回放曾区分主机CPU与独立设备完成，但旧细轨迹属于另一切图。本轮补齐的是当前三个VTA子图的匹配证据，不宣称首次提出这一区分或已经改善Top20选点。',
        '', '下一步最小范围：',
        '优先检查现有隔离资料是否足以为当前四个CPU子图及三个VTA子图构建同身份的主机／设备依赖输入；有缺项先列清输入和成本。',
        '在不使用流水FPS拟合的前提下，复用既有时序模型检查共享CPU如何延长VTA锁内主机段；不能直接把这24ms加到原评分。',
        '如果输入不足或回放不能改善未测选点，就保留当前静态方法及其适用边界，不扩展成新的无依据调度模拟器。',
        '本报告是已完成的误差定位，尚不是新的可部署预测公式或性能提升。',
        '', '文献对应：GCAPS ECRTS2024区分加速器请求中的CPU段和设备段：',
        'https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf',
        '这里复用资源区分，未复现其优先级调度或宣称其公式直接适用TVM/Linux流水。',
    ]
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')


def draw(result):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/cpu_vta_matched_device_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(13.4,6.4))
    colors = {'boundary_ms':'#A5AFBC','driver_wall_ms':'#437FB1','outside_driver_ms':'#D8923B'}
    labels = {'boundary_ms':'输入／输出交接','driver_wall_ms':'设备驱动调用内','outside_driver_ms':'图运行中设备调用外'}
    ax=axes[0]
    for j,r in enumerate(result['stage_rows']):
        for side,shift in [('isolated',-.19),('pipeline',.19)]:
            bottom=0
            for key in colors:
                value=r[side][key];ax.bar(j+shift,value,bottom=bottom,width=.34,color=colors[key]);bottom+=value
            ax.text(j+shift,bottom+1,'%.1f'%bottom,ha='center',fontsize=10)
    ax.set_xticks(range(3),['VTA单元1','VTA单元5–6','VTA单元9–11'])
    ax.set_ylabel('每次子图服务时间（ms）');ax.set_ylim(0,86)
    ax.set_title('每对子图：左为隔离，右为流水',pad=17)
    ax=axes[1]
    totals=result['totals']
    for j,side in enumerate(('isolated','pipeline')):
        bottom=0
        for key in colors:
            value=totals[side][key];ax.bar(j,value,bottom=bottom,width=.5,color=colors[key]);bottom+=value
        ax.text(j,bottom+2,'%.2f ms'%bottom,ha='center',fontsize=12,fontweight='bold')
        ax.text(j,totals[side]['boundary_ms']+totals[side]['driver_wall_ms']+totals[side]['outside_driver_ms']/2,
                '调用外 %.2f'%totals[side]['outside_driver_ms'],ha='center',va='center',fontsize=11)
    ax.set_xticks([0,1],['隔离三个子图合计','流水三个子图合计']);ax.set_ylim(0,225)
    ax.set_ylabel('三个VTA阶段服务时间之和（ms）');ax.set_title('主要增量出现在设备调用外',pad=17)
    for ax in axes:
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for s in ('top','right'):ax.spines[s].set_visible(False)
    fig.suptitle('当前YOLO静态第一名：隔离profile为何仍偏乐观',fontsize=18,y=.98)
    fig.legend(handles=[Patch(facecolor=colors[k],label=labels[k]) for k in colors],
        loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.76,wspace=.28)
    fig.text(.5,.12,'设备完成包络合计：隔离104.40–104.78 ms，流水103.28–105.97 ms；区间重叠，未定位到同等规模的设备慢化。',ha='center',fontsize=10)
    fig.text(.5,.055,'数据来自探针会话，仅作归因；无探针正式结果5.176 FPS。不同阶段帧均值不能直接相加当作通用流水II。',ha='center',fontsize=10,color='#526070')
    for ext in ('png','svg','pdf'):fig.savefig(REPORT/('matched_stage_decomposition.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    run()
