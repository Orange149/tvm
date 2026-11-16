"""Frozen one-parameter development replay; no native runs or FPS fitting."""
from pathlib import Path
import json
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from policy_branch_eval.common import isolated_inputs_only
from compiled_wakeup_model import KernelWakeFinite,adapted_select

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/three_thread_host_gap_01'
OUT=ROOT/'artifacts/compiled_wakeup_01/three_thread'


def run():
    start=time.monotonic();source=PARENT/'existing_model_diagnostic/INPUTS.json'
    parameter=ROOT/'artifacts/three_thread_worker_gap_01/kernel_parameter_sources/COMPILED_WAKEUP.json'
    with isolated_inputs_only():
        x=read(source);p=read(parameter)
        for path,h in x['replay_sources'].items():
            if sha256(path)!=h:raise ValueError('original model code changed')
        if p['DNN_FPS_used'] or p['wakeup_granularity_ns']!=3000000:raise ValueError('parameter derivation differs')
        plan=dict(input_sha256=sha256(source),compiled_parameter_sha256=sha256(parameter),
            old_model_sources=x['replay_sources'],
            new_sources={str(f):sha256(f) for f in (Path(__file__),ROOT/'compiled_wakeup_model.py',ROOT/'tests/test_compiled_wakeup_model.py')},
            only_change=adapted_select()[1],CPU_work_and_service_quanta_unchanged=True,
            sleeper_credit_host_placement_and_device_model_unchanged=True,
            one_parameter_value_no_search=True,known_case_development_only=True,new_DNN=0,
            stage_errors_not_equal_to_selection_gains=True,
            evaluate_all_three_positions_not_only_the_preferred_case=True,
            objective='test a compiled-kernel wake-rule correction before any new prospective measurement')
        freeze(OUT/'PLAN.json',plan);paths={}
        for cid,stages in x['cases'].items():
            path=OUT/(cid+'.json')
            if not path.exists():
                result=KernelWakeFinite(stages,wakeup_granularity_ns=p['wakeup_granularity_ns'],
                    quanta=x['quanta_ns'],placement='wake_boundary',completion_model='independent',
                    completion_point='midpoint',runnable_completion_wait=True,controls=None,
                    gate='off',host_order='before',**x['window']).run()
                freeze(path,dict(result=result,plan_sha256=sha256(OUT/'PLAN.json'),no_native_performance_read=True))
            if read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('cached correction input differs')
            paths[cid]=dict(path=str(path),sha256=sha256(path))
        freeze(OUT/'REPLAY_FREEZE.json',dict(paths=paths,plan_sha256=sha256(OUT/'PLAN.json'),
            all_new_predictions_sealed_before_native_comparison=True))
    # Native references are deliberately downstream of the new replay freeze.
    native=read(PARENT/'analysis/GAP_COMPARISON.json');rows=[]
    for n in native['rows']:
        old_path=PARENT/'existing_model_diagnostic'/(n['configuration_id']+'.json')
        old=read(old_path)['result'];new=read(paths[n['configuration_id']]['path'])['result']
        if new['replay']['processed_frames']!=old['replay']['processed_frames']:
            raise ValueError('comparison windows differ')
        rows.append(dict(role=n['role'],configuration_id=n['configuration_id'],
            old_front_ms=old['replay']['stage_service_ns'][0]/1e6,
            new_front_ms=new['replay']['stage_service_ns'][0]/1e6,
            native_clean_front_ms=n['pipeline_front_ms'],
            old_front_error_ms=old['replay']['stage_service_ns'][0]/1e6-n['pipeline_front_ms'],
            new_front_error_ms=new['replay']['stage_service_ns'][0]/1e6-n['pipeline_front_ms'],
            old_II_ms=old['replay']['interval_ns']/1e6,new_II_ms=new['replay']['interval_ns']/1e6,
            native_clean_II_ms=n['pipeline_II_ms'],native_clean_FPS=n['pipeline_FPS'],
            old_replay_sha256=sha256(old_path),new_replay_sha256=paths[n['configuration_id']]['sha256']))
    ordering={}
    for key in ('old','new'):
        order=sorted(rows,key=lambda r:(r[key+'_II_ms'],r['configuration_id']))
        ordering[key]=dict(order=[r['role'] for r in order],selected_native_FPS=order[0]['native_clean_FPS'])
    out=dict(rows=rows,ordering=ordering,source_sha256=sha256(__file__),new_native_sessions=0,
        parameter_from_matched_kernel_not_FPS=True,known_case_diagnostic_only=True,
        no_full_CFS_claim=True,no_new_Top20_or_performance_gain_claim=True,
        CPU_model_and_control_approximations_remain=True,
        current_probe_does_not_support_exact_remaining_gap_attribution=True,
        native_source_sha256=sha256(PARENT/'analysis/GAP_COMPARISON.json'),
        plan_sha256=sha256(OUT/'PLAN.json'),compute_seconds=time.monotonic()-start)
    write(OUT/'RESULTS.json',out,replace=True)
    lines=['内核来源的单项唤醒规则修正：离线开发回放','',
        '只将原先q(n)/2的唤醒抢占阈值，替换为当前匹配内核Image及四核缩放推导的3ms。运行片、休眠补偿、主机放置、CPU工作及设备完成模型保持不变。',
        '内核镜像完整notes及build-id与当前运行内核匹配，参数没有从DNN FPS调优。这里是编译初始化规则推导值，未读取运行内核内存。',
        '所有新回放在读取本轮原生结果前冻结。配置与实测已知，因此属于开发诊断，不是盲测或独立收益验证。','',
        '主机角色 | 原回放前段ms | 修正回放前段ms | 实测前段ms | 原误差ms | 修正误差ms']
    for r in rows:
        lines.append(f"{r['role']} | {r['old_front_ms']:.3f} | {r['new_front_ms']:.3f} | {r['native_clean_front_ms']:.3f} | {r['old_front_error_ms']:+.3f} | {r['new_front_error_ms']:+.3f}")
    lines += ['', '本轮不改变任何原生FPS，也没有将这项修正接入完整DP。相同局部第一名只说明此池没有选点差异。完整搜索、未测配置验证及同预算1／5／20比较仍未完成。',
        '若效果不足，不能根据这些结果继续调整3ms；它来自独立内核证据。其他模型假设也没有因此得到验证。',
        '源码依据：https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c，wakeup_gran/wakeup_preempt_entity及update_sysctl；实际镜像反汇编和校验在three_thread_worker_gap_01/kernel_parameter_sources中。']
    (OUT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    report=ROOT.parents[4]/'paper/my_paper/analysis/20261111_compiled_wakeup'
    report.mkdir(parents=True,exist_ok=True)
    for name in ('RESULTS.json','FINDINGS_zh.txt','PLAN.json','REPLAY_FREEZE.json'):
        shutil.copy2(OUT/name,report/name)
    return out


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
