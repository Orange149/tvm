"""Read-only component decomposition of the accepted CPU exclusion experiment.

No model, ranking, board connection, or new native sessions. Historical isolated
references remain explicitly distinct from the same-boot execution intervention.
"""
from pathlib import Path
import csv
import json
import math
import statistics as stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write

ROOT = Path(__file__).resolve().parent
POOL = ROOT / 'artifacts/cpu_exclusion_gap_01'
DEST = ROOT / 'artifacts/profile_pipeline_gap_focus_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_profile_pipeline_gap_focus'


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=0.000003):
        raise ValueError(label + ': inconsistent timing')


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def extract():
    plan = read(POOL / 'PLAN.json'); result = read(POOL / 'RESULTS.json')
    state = read(ROOT / 'runs/cpu_exclusion_gap_01/state.json')
    if state['status'] != 'COMPLETE' or state['accepted_sessions'] != 8:
        raise ValueError('source cohort is incomplete')
    stages = []; pipelines = []; source_hashes = {}; qualifications = {}
    # Verify qualification receipts before using their output fingerprints.
    receipts = []
    for path, expected in result['receipts'].items():
        if sha256(path) != expected:
            raise ValueError('source receipt changed')
        receipt = read(path); verify_inventory(receipt['raw'], receipt['files'])
        if receipt['boot_id'] != state['boot_id']:
            raise ValueError('source batch mixes boots')
        source_hashes[path] = expected; receipts.append(receipt)
        task = receipt['task']
        if task['spec']['phase'] == 'qualify':
            qualifications[(task['spec']['network'], task['cpu_admission'])] = receipt['summary']['output_fingerprints']
    for receipt in receipts:
        task = receipt['task']; summary = receipt['summary']; spec = task['spec']
        if spec['phase'] != 'clean':
            continue
        network = spec['network']; mode = task['cpu_admission']
        if summary['output_fingerprints'] != qualifications[(network, mode)]:
            raise ValueError('qualification and performance outputs differ')
        rows = [json.loads(line) for line in (Path(receipt['raw']) / 'result.jsonl').read_text().splitlines()]
        count = task['warmup'] + task['scored'] + task['tail']
        if [r['frame_id'] for r in rows] != list(range(count)):
            raise ValueError('missing, duplicate or unordered frame')
        kept = rows[task['warmup']:task['warmup'] + task['scored']]
        close((kept[-1]['completion_ms'] - kept[0]['completion_ms']) / (len(kept) - 1), summary['ii_ms'], 'completion interval')
        hosts = [r for r in summary['thread_ledger'] if r.get('stage', '').startswith('vta_') and r['role'] == 'caller']
        if any(r['cpu_time_delta_ms'] is None for r in hosts):
            raise ValueError('VTA host CPU time unavailable')
        first_end = rows[0]['stage0_end_ms']
        downstream_starts = [r[f'stage{i}_start_ms'] for r in rows for i in range(1, len(task['stages']))]
        first_has_no_downstream_stage = min(downstream_starts) >= first_end
        pipelines.append(dict(network=network, mode=mode, configuration=task['configuration']['id'],
            boot_id=receipt['boot_id'], executor_sha256=task['executor_sha256'],
            FPS=summary['fps'], II_ms=summary['ii_ms'],
            whole_process_core_ms_per_completion=summary['process_core_ms_per_completion'],
            host_CPU_ms_whole_session_per_frame=sum(r['cpu_time_delta_ms'] for r in hosts) / count,
            host_CPU_scope='whole session incl warmup/tail and startup probe; proc stat coarse counters; not a formal-window component cost',
            first_front_service_ms=rows[0]['stage0_ms'],
            first_front_has_no_downstream_stage=first_has_no_downstream_stage,
            first_frame_is_controlled_isolation=False,
            first_frame_caveat='startup/cache state differs; same-process contextual evidence only',
            task_id=task['task_id']))
        for index, stage in enumerate(task['stages']):
            prefix = 'stage' + str(index)
            parts = {k: stats.mean(r[prefix + '_' + k + '_ms'] for r in kept) for k in ('set', 'run', 'get')}
            service = stats.mean(r[prefix + '_ms'] for r in kept)
            close(service, sum(parts.values()), 'pipeline service components')
            close(service, summary['stage_service_ms'][index], 'accepted stage service')
            if stage['device'] != 'cpu':
                continue
            ref = next(r for r in plan['isolated_reference'][network] if r['program'] == stage['program'])
            if sha256(ref['receipt']) != ref['receipt_sha256']:
                raise ValueError('isolated reference changed')
            isolated = read(ref['receipt']); verify_inventory(isolated['raw'], isolated['files'])
            source_hashes[ref['receipt']] = ref['receipt_sha256']
            cost = isolated['summary']['cost']
            close(cost['service_ms'], sum(cost[k + '_ms'] for k in ('set', 'run', 'get')), 'isolated components')
            if task['runtime_sha256'] != isolated['task']['runtime_sha256']:
                raise ValueError('runtime differs from isolated reference')
            gate_wait = stats.mean(r.get(prefix + '_cpu_gate_wait_ms', 0) for r in kept)
            stages.append(dict(network=network, mode=mode, program=stage['program'], threads=stage['threads'],
                requested_mask=','.join(map(str, stage['mask'])),
                isolated_set_ms=cost['set_ms'], isolated_run_ms=cost['run_ms'], isolated_get_ms=cost['get_ms'],
                isolated_service_ms=cost['service_ms'],
                pipeline_set_ms=parts['set'], pipeline_run_ms=parts['run'], pipeline_get_ms=parts['get'],
                pipeline_service_ms=service, service_gap_ms=service-cost['service_ms'],
                run_gap_ms=parts['run']-cost['run_ms'],
                copy_gap_ms=parts['set']+parts['get']-cost['set_ms']-cost['get_ms'],
                CPU_gate_wait_outside_service_ms=gate_wait,
                isolated_scope=ref['scope'], isolated_boot=ref['boot_id'], pipeline_boot=receipt['boot_id'],
                isolated_executor_sha256=ref['executor_sha256'], pipeline_executor_sha256=task['executor_sha256'],
                isolated_reference_is_current_batch=False,
                homogeneous_core_permutation=ref['homogeneous_core_permutation'],
                isolated_receipt=ref['receipt'], pipeline_task=task['task_id']))
    return dict(stages=stages, pipelines=pipelines, sources=source_hashes,
        source_plan_sha256=sha256(POOL / 'PLAN.json'), source_results_sha256=sha256(POOL / 'RESULTS.json'),
        script_sha256=sha256(__file__), new_native_sessions=0, new_DNN_inferences=0,
        earlier_native_experiment=dict(sessions=8, DNN_inferences=116),
        static_model_DP_Top20_unchanged=True, full_domain_rank_deferred=True,
        limits=['Historical isolation is from a different boot and executor, not a new matched isolated measurement.',
                'Run wall time includes computation and runtime waits; no attribution to scheduling alone.',
                'Gate changes execution schedule; service reduction is not an additive causal contribution.',
                'VTA/controls and shared memory remain concurrent under global CPU exclusion.',
                'Whole-session thread CPU accounting differs from completion-window process CPU accounting.'])


def report(result):
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = result['stages']; pipelines = result['pipelines']
    names = {'resnet18': 'ResNet18', 'yolov3_tiny416': 'YOLOv3-tiny'}
    lines = ['隔离profile与流水差距：CPU阶段并发排除对照', '',
        '工作重点：解释阶段耗时差距。本次分析不运行新推理，不修改静态模型、DP或Top-20；新的全域排名暂不推进。',
        '本报告整理已完成的同批8个会话、116次完整推理。没有重启、没有新组件标定，正式流水没有详细等待探针。', '',
        '一、已经定位到哪一段',
        '阶段服务＝输入set_input墙钟＋run墙钟＋输出get_output墙钟。门控等待发生在服务计时之前，单独记录；它仍影响整体吞吐。',
        'run墙钟包含计算、启动、同步及完成等待，不等于纯计算CPU时间。', '',
        '网络 / CPU阶段 / 方式 / 隔离参照ms / 流水服务ms / run差额ms / 输入输出差额ms / 门控等待ms']
    for r in rows:
        lines.append(f"{names[r['network']]} / {r['program']} / {'开放并发' if r['mode']=='off' else '全部CPU阶段互斥'} / "
            f"{r['isolated_service_ms']:.3f} / {r['pipeline_service_ms']:.3f} / {r['run_gap_ms']:.3f} / "
            f"{r['copy_gap_ms']:.3f} / {r['CPU_gate_wait_outside_service_ms']:.3f}")
    lines += ['', '二、这次干预支持什么',
        '同一批内只改变全CPU阶段互斥开关，切图、线程数、核心映射、VTA主机及控制线程位置、运行库与模块均不变。',
        'CPU阶段互斥覆盖输入、运行和输出快照，但不互斥VTA主机与控制工作。',
        'ResNet18 CPU5–7由45.87降到29.26 ms，接近隔离参照27.74 ms；CPU子图并发会影响这一阶段。',
        'YOLO CPU0–2由218.73降到184.51 ms，仍比隔离参照139.51 ms多约45 ms。CPU子图互斥不足以消除差距。',
        '这不是采用全CPU串行的理由：排队时间转移到门控等待后，整体吞吐没有提高。', '']
    for network in names:
        before = next(p for p in pipelines if p['network']==network and p['mode']=='off')
        after = next(p for p in pipelines if p['network']==network and p['mode']=='exclusive')
        lines.append(f"{names[network]}：开放{before['FPS']:.4f} FPS，互斥{after['FPS']:.4f} FPS；变化{100*(after['FPS']/before['FPS']-1):+.2f}%。")
    lines += ['', '三、同一进程内的启动线索',
        'YOLO开放方式第一帧前段146.68 ms，计入窗口均值218.73 ms；CPU互斥方式第一帧147.27 ms，计入窗口均值184.51 ms。',
        '原始时间戳已核对：第一帧CPU前段完成前，下游真实阶段尚未开始。',
        '这是同一程序、同一启动内的旁证，不是受控隔离对照；首帧缓存与初始化状态不同，不能将首帧直接替代稳态隔离成本。', '',
        '四、下一步只针对尚未解释的部分',
        '优先核查全CPU互斥后仍与前段重叠的VTA主机运行区间、控制工作，以及其对最后完成worker的影响。',
        '先使用已有隔离／流水轨迹和本批无详细探针时间戳；区分run内计算CPU量、运行时等待CPU量、未运行时间及未知项。',
        '已有VTA主机CPU需求不是漏填为零。YOLO主机整段计数约每帧64–67核毫秒，ResNet约17核毫秒；作用域包含预热、尾部与初始化探针，不能直接当正式窗口成本。',
        '核时数值已计入并不表示主机工作的具体时序、落核及其触发的完成拖尾已被准确表达。',
        '目前不能将剩余45 ms全部归因于主机、调度或DDR，也不把它拟合为统一慢化系数。', '',
        '真正缺少的观测边界：已有回调探针的结束时刻，不等于TVM运行库的实际完成通知时刻。',
        '实际运行库机器码已经核对：计算回调返回后才递减num_pending；探针包装器在返回前还会记账。',
        '因此下一步需区分“worker回调结束→真实完成通知→caller恢复”，结合已有线程运行轨迹核查；不能把整段尾部直接叫caller排队。',
        '现有记录无法直接观测真实完成计数递减，这个缺口明确保留。先评估最小观测办法及扰动，不直接扩大CFS模拟器或重跑全域排名。',
        '边界依据：artifacts/completion_notification_audit_01/及对应源码／实际运行库机器码；旧观测记录不改写。', '',
        '五、证据边界',
        '隔离参照来自旧启动和不同诊断执行器；有些参照带探针或做了同构核心置换，逐项标在CSV中。同批干预差值不依赖这些参照。',
        '本批ResNet配置是已知的局部诊断配置，不是历史约13 FPS的全域第一名重测。',
        '单输入、单启动、短窗口；不报告稳定收益、显著性或统计等效。',
        '阶段差距尚未解决，新静态修正和Top-20收益均未证明。', '',
        '复现：python analyze_profile_pipeline_gap_focus.py',
        '原始批次：artifacts/cpu_exclusion_gap_01/；runs/cpu_exclusion_gap_01/。',
        '来源索引及逐项计时：artifacts/profile_pipeline_gap_focus_01/RESULTS.json、stages.csv、pipelines.csv。']
    (REPORT / 'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    keys = [('resnet18', 'cpu_00_02', 'ResNet18 CPU 0-2'),
            ('resnet18', 'cpu_05_07', 'ResNet18 CPU 5-7'),
            ('yolov3_tiny416', 'cpu_00_02', 'YOLOv3-tiny CPU 0-2')]
    for ax, (network, program, title) in zip(axes, keys):
        off = next(r for r in result['stages'] if r['network']==network and r['program']==program and r['mode']=='off')
        excluded = next(r for r in result['stages'] if r['network']==network and r['program']==program and r['mode']=='exclusive')
        values = [off['isolated_service_ms'], off['pipeline_service_ms'], excluded['pipeline_service_ms']]
        ax.bar(range(3), values, color=['#999999', '#cf654b', '#34889c'])
        ax.set_xticks(range(3), ['Isolated\nreference', 'Pipeline\nopen', 'CPU stages\nserialized'])
        ax.set_title(title); ax.set_ylabel('Stage service (ms)')
        ax.set_ylim(0, max(values)*1.18)
        for i, v in enumerate(values): ax.text(i, v+max(values)*.025, f'{v:.2f}', ha='center')
    fig.suptitle('Service recovery is partial; CPU serialization did not improve throughput')
    fig.text(.5, .01, 'Historical isolated reference has different boot/executor. Admission wait excluded from stage service; included in end-to-end interval.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(REPORT / 'stage_service_gap.png', dpi=160)
    fig.savefig(REPORT / 'stage_service_gap.svg')
    plt.close(fig)


def main():
    result = extract(); DEST.mkdir(parents=True, exist_ok=True)
    write(DEST / 'RESULTS.json', result, replace=True)
    csv_write(DEST / 'stages.csv', result['stages']); csv_write(DEST / 'pipelines.csv', result['pipelines'])
    report(result); plot(result)
    print(json.dumps(dict(stages=len(result['stages']), pipelines=len(result['pipelines']),
        new_native_sessions=0, new_DNN_inferences=0, report=str(REPORT / 'FINDINGS_zh.txt')), ensure_ascii=False))


if __name__ == '__main__': main()
