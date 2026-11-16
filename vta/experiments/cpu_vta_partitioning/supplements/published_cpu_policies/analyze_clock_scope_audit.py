"""Interpret the one executed clock-scope self-test, never calibrate a model."""
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from run_clock_scope_audit import OUT,ROOT,checked_plan,analyze_rows
from scheduler_gap_audit.curves import csv_write


def run():
    plan=checked_plan();r=read(OUT/'RECEIPT.json')
    if sha256(r['raw_path'])!=r['raw_sha256'] or r['binary_sha256']!=plan['binary_sha256']:
        raise ValueError('clock-scope native identity differs')
    rows=analyze_rows([json.loads(s) for s in Path(r['raw_path']).read_text().splitlines()])
    if rows!=r['summaries']:raise ValueError('clock self-test analysis not repeatable')
    off,on=rows[2],rows[3]
    off_difference=(off['thread_CPU_ms']-off['task_clock_ms'])*1e6
    boundary=off['counter_external_wall_envelope_ns']
    conclusion=dict(
        short_sleep_CPU_clock_and_perf_clock_not_equal=off_difference>boundary,
        trace_CPU_clock_and_switch_span_not_equal=on['thread_clock_minus_switch_ms']*1e6>on['counter_external_wall_envelope_ns'],
        difference_exists_without_DNN_or_active_switch_recording=True,
        source_sequence_consistent_with_different_accounting_boundaries=True,
        vendor_kernel_schedule_charge_exactly_identified=False,
        no_universal_switch_cost_estimated=True,
        no_prediction_formula_or_cost_refit=True,
        same_process_single_trial_not_stable_tracing_overhead_estimate=True)
    result=dict(rows=rows,conclusion=conclusion,source_receipt_sha256=sha256(OUT/'RECEIPT.json'),
        source_sha256=sha256(__file__),native_boot=r['boot_id'],new_DNN=0,new_VTA=0,native_processes=1,
        primary_sources=plan['sources'])
    write(OUT/'analysis/RESULTS.json',result,replace=True);csv_write(OUT/'analysis/clock_comparison.csv',rows)
    lines=['短睡眠计时口径自测：结论与限制','',
        '一次进程、四个事前固定情形，DNN推理0、VTA调用0；没有重启、系统参数变更或改写原运行库。',
        '每个情形都打开同样的事件描述符；关闭记录表示切换事件保持禁用，开启表示启用。不是完全移除所有perf描述符的对照。',
        '短睡眠负载为600次约5us忙段+请求1us睡眠；睡眠实际时间未假定为1us。连续忙计算约12ms。',
        'guard睡眠只用于形成覆盖工作窗口的边界事件，不计入工作窗口。perf计数读取包络独立保留。','']
    for row in rows:
        lines.append(f"{row['case']}，记录{'开' if row['trace'] else '关'}：墙钟{row['wall_ms']:.6f}ms，线程CPU钟{row['thread_CPU_ms']:.6f}ms，perf task-clock {row['task_clock_ms']:.6f}ms。")
        if row['trace']:
            lines.append(f"  IN→OUT重建{row['switch_span_CPU_ms']:.6f}ms，切出{row['switch_outs']}次，线程CPU钟减重建{row['thread_clock_minus_switch_ms']:.6f}ms。")
    lines+=['','可支持的结论：短睡眠中几个计数的差异在零DNN程序也存在，且远大于读取窗口包络；不能将VTA run中同类差额直接命名为新增算子工作。',
        'Linux v5.4源代码中，线程CPU钟读取调度器累计运行时间；perf task-clock使用perf事件上下文计时；switch记录由切换前后的不同钩子产生。它们不是同一个精确边界。',
        '这与数据中的计时差异一致，但未读取供应商内核完整构建源码，不能据此声称已经逐行确认差额如何收费，也不将约5–6us/次写成通用开销。',
        '记录开关两次按顺序各执行一次，短忙段按墙钟预算而非固定指令数，因此不把CPU时间之差当作纯探针成本或稳定提升。',
        '原隔离profile继续保留，不自动替换为另一种时钟，不以新计数重拟合慢化。',
        '后续归因：将线程CPU需求、实际运行区间与off-CPU区间分别报告；关键worker受其他工作占用的证据仍使用实际时间重合，不再要求两个计时数完全相等。','',
        '原始记录：'+r['raw_path']]
    (OUT/'analysis/FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    report=ROOT.parents[4]/'paper/my_paper/analysis/20261111_clock_scope_audit';report.mkdir(parents=True,exist_ok=True)
    for p in (OUT/'analysis').iterdir():
        if p.is_file():shutil.copy2(p,report/p.name)
    return dict(report=str(report/'FINDINGS_zh.txt'),conclusion=conclusion)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
