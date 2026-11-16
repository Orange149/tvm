"""Validate a zero-new-calibration reconstruction against fresh thread clocks."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from scheduler_gap_audit.curves import csv_write
from run_budget_reconstruction import reconstruct
import run_scoped_thread_profile as experiment

ROOT=experiment.ROOT
OUT=ROOT/'artifacts/scoped_thread_effect_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_scoped_thread_profile'


def run():
    source=experiment.analyze();source_path=experiment.OUT/'RESULTS.json'
    rows=[];sources={str(source_path):sha256(source_path)}
    for path,expected in source['source_receipts'].items():
        if sha256(path)!=expected:raise ValueError('accepted CPU clock receipt changed')
        receipt=read(path);verify_inventory(receipt['raw'],receipt['files']);sources[path]=expected
        task=receipt['task']
        if task['spec']['phase']!='clean':continue
        if task['environment']['spin_count']!=0:raise ValueError('quiet-worker assumption not supported')
        raw=Path(receipt['raw']);frames=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
        budget=read(raw/'thread_budget.json');n=task['stages'][0]['threads']
        ledger={t['logical_worker']:t['cpu_time_delta_ms'] for t in receipt['summary']['thread_ledger']
            if t['role'] in ('caller','worker')}
        reconstruction=reconstruct(frames,[ledger[i] for i in range(1,n)])
        direct=[sum(r['threads'][i]['CPU_end_ns']-r['threads'][i]['CPU_begin_ns'] for r in budget['records']) for i in range(n)]
        direct=[v/sum(direct) for v in direct]
        whole=[ledger[i]/sum(ledger.values()) for i in range(n)]
        gaps=[sum(b['threads'][i]['CPU_begin_ns']-a['threads'][i]['CPU_end_ns']
            for a,b in zip(budget['records'],budget['records'][1:]))/1e6 for i in range(1,n)]
        phase_rows=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        caller=budget['slots'][0]['tid']
        phase_cpu=sum(p['cpu_end_ns']-p['cpu_start_ns'] for p in phase_rows if p['tid']==caller)/1e6
        rows.append(dict(network=task['spec']['network'],program=task['stages'][0]['program'],threads=n,
            same_process_whole_weights=whole,direct_all_run_thread_weights=direct,
            reconstructed_weights=reconstruction['weights'],
            whole_share_L1_error=sum(abs(a-b) for a,b in zip(whole,direct)),
            reconstructed_share_L1_error=sum(abs(a-b) for a,b in zip(reconstruction['weights'],direct)),
            worker_CPU_between_run_snapshots_ms=gaps,
            caller_whole_counter_ms=ledger[0],caller_all_recorded_phase_CPU_ms=phase_cpu,
            caller_outside_phase_residual_from_coarse_counter_ms=ledger[0]-phase_cpu,
            residual_includes_result_statistics_configuration_probe_and_counter_scope=True,
            process_run_budget_envelopes_valid=receipt['summary']['run_thread_budget']['all_process_budgets_inside_thread_envelope'],
            clock_read_widening_ms=receipt['summary']['run_thread_budget']['max_thread_read_window_widening_ms'],
            original_CPU_milliseconds_not_updated=True,raw_task=task['task_id']))
    result=dict(rows=rows,sources=sources,new_native_sessions=6,CPU_component_calls=33,DNN_inferences=0,
        no_global_profile_or_ranking=True,no_formal_model_or_costs_changed=True,
        all_noncaller_between_run_increments_zero=all(all(v==0 for v in r['worker_CPU_between_run_snapshots_ms']) for r in rows),
        fresh_measurement_is_scope_validation_not_independent_throughput_validation=True,
        counter_reconstruction_requires_full_original_receipt_audit=True,
        coarse_noise_must_be_reported_and_missing_mass_must_not_be_clipped=True,
        source_files={str(p):sha256(p) for p in (Path(__file__),ROOT/'run_budget_reconstruction.py')})
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'comparison.csv',rows)
    text=['run区间逐线程CPU核时：三签名匹配验证','',
        '同一boot、输入0，三个已有CPU子图：每项资格1次，再2帧预热+8帧计入。共6个会话、33次CPU子图调用、0次完整DNN。',
        '复用原模块、TVM/VTA运行库；只编译可选测量执行器。无等待探针、FPGA访问或重启。',
        '所有输出、角色/TID、单核亲和性、读取包络、文件哈希和boot身份通过验收。进程run CPU增量落在逐线程读取包络之内。','',
        '网络 / 子图 / 整会话caller份额 / 全部run直接时钟caller份额 / 旧计数重建caller份额']
    for r in rows:text.append(f"{r['network']} / {r['program']} / {100*r['same_process_whole_weights'][0]:.2f}% / {100*r['direct_all_run_thread_weights'][0]:.2f}% / {100*r['reconstructed_weights'][0]:.2f}%")
    text+=['','旧数据候选修正的定义（尚未修改正式成本）：',
        '把该进程全部记录帧的run进程CPU时间相加，保留非调用线程整会话CPU计数；调用线程run CPU取前者减去后者，再归一化。',
        '这利用串行隔离、线程池空转关闭以及非调用线程在run外基本不工作的条件，去掉调用线程输出统计等工作对run份额的污染。',
        '本批直接线程时钟记录的非调用线程相邻run间CPU增量均为0，支持该条件在这三项中的可行性。',
        '估计仍受旧线程计数的较粗粒度、初始空回调及未观测工作影响；不是把旧计数改写成真实逐线程新测量。',
        '使用全部run对照全部run，避免把含预热总计数与仅8帧的直接时钟混淆。',
        '输入、输出及结果统计发生在调用线程上；整会话计数不是只有初始化。没有单独测量各段时，不把残差全叫作JSON或哈希成本。','',
        '能支持：测量作用域偏差真实存在，且依子图而异；已有原始记录可能支持低成本的计数重建，应先做全记录离线核验。',
        '不能支持：该修正已解决流水慢化或改善Top-20。细粒度时序回放原先使用计算回调CPU而非整会话份额，此项作用域修正不能自动解释其剩余32.8ms误差。',
        '下一步只读取两个网络已有隔离记录，核验全部run过程CPU、非调用线程计数、签名与原始哈希是否齐备；拒绝负的调用线程残差或不可观测计数，保留既有回退及不确定性。',
        '核验通过后先比较同一完整候选域下的权重修正贡献与总成本刷新贡献，再决定完整DP/Top-20验证；不按三个签名缩放其他子图，也不立刻新增全域板测。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in
        [OUT/'RESULTS.json',OUT/'comparison.csv',REPORT/'FINDINGS_zh.txt']}),replace=True)
    print(json.dumps(dict(rows=rows,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
