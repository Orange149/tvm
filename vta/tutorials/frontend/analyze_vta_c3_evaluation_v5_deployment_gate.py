#!/usr/bin/env python3
"""Seal the final C3 v5 deployment gate, including correctness rejection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from c3_evaluation_v5 import read_json,read_jsonl,seal_flat,verify_flat,write_json
HERE=Path(__file__).resolve().parent
GROUPED=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
DEFAULT_OPERATOR=GROUPED/'20260926_c3_yolo512_evaluation_v5_cross_boot_analysis_run01'
DEFAULT_COMPLETION=GROUPED/'20260926_c3_yolo512_evaluation_v5_completion_audit_run01'
DEFAULT_FAILED=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_boot1_run01'
DEFAULT_DIAG=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_correctness_diagnostic_board_run01'
DEFAULT_OUTPUT=GROUPED/'20260926_c3_yolo512_evaluation_v5_deployment_gate_run01'

def run(args):
 roots={k:Path(v).resolve() for k,v in {'operator':args.operator_analysis,'completion':args.completion_audit,'failed_graph':args.failed_graph_attempt,'diagnostic':args.correctness_diagnostic}.items()}
 out=Path(args.output_dir).resolve()
 if out.exists():raise FileExistsError(out)
 manifests={k:verify_flat(v) for k,v in roots.items()}
 op=read_json(roots['operator']/'analysis.json');comp=read_json(roots['completion']/'summary.json')
 failed=read_json(roots['failed_graph']/'invalid_session.json');diag=read_json(roots['diagnostic']/'summary.json')
 failed_rows=read_jsonl(roots['failed_graph']/'calls.jsonl');diag_rows=read_jsonl(roots['diagnostic']/'calls.jsonl')
 if not op['at_least_one_stable_mode11'] or len([x for x in op['deployment_decisions'].values() if x['stable_mode11']])!=2:raise ValueError('operator gate drift')
 if comp.get('status')!='COMPLETION_AUDIT_COMPLETE' or comp.get('completion_mode')!='run_only':raise ValueError('completion gate drift')
 if failed.get('status')!='INVALID_NON_SPLICEABLE_GRAPH_BOOT' or any(r.get('latency_ms') is not None for r in failed_rows):raise ValueError('failed graph attempt exposed formal latency')
 singleton=diag['singleton_mode11_correct']
 if diag.get('latency_values_collected') is not False or any(r.get('latency_ms') is not None for r in diag_rows):raise ValueError('correctness diagnostic contains latency')
 if diag.get('combined_selector_correct') or any(singleton.values()):raise ValueError('diagnostic no longer supports rejection')
 out.mkdir(parents=True)
 result={'schema':'c3_evaluation_v5_deployment_gate_v1','status':'GRAPH_DEPLOYMENT_REJECTED_CORRECTNESS',
  'manifests':manifests,'selector_method_primary_comparison_passed':op['primary_selector_comparison']['passed'],
  'stable_operator_mode11_tasks':[k for k,v in op['deployment_decisions'].items() if v['stable_mode11']],
  'completion_mode':comp['completion_mode'],'incumbent_graph_correct':True,
  'combined_selector_graph_correct':False,'singleton_mode11_graph_correct':singleton,
  'formal_graph_timing_calls':0,'graph_boot2_required':False,'additional_reboot_required':False,
  'deployment_program':'incumbent','post_failure_reselection':False,
  'evidence_interpretation':('both sealed mode11 programs are FPGA-correct and fast under their isolated operator contracts, but neither is correct when routed alone into the real YOLO-512 graph; local scoped-completion certificates are therefore insufficient production eligibility evidence'),
  'root_cause_status':'not established; do not attribute to debug transitions, fusion, or parameter reuse without a targeted trace',
  'next_gate':'repair and independently certify whole-graph mode11 correctness before any graph latency measurement'}
 write_json(out/'analysis.json',result);write_json(out/'summary.json',result)
 report = """# C3 v5 最终部署门禁

- selector-v4 对 bytes-only 的 B3 主比较未通过：两者在两个任务上选择相同 oracle。
- 两个封存 mode11 在两次独立启动中均 6/6 轮胜出，孤立算子改善约 95.8% 和 96.2%。
- `run-only` 是正确的整图完成边界；额外同步会新增提交。
- stock 与 incumbent 整图三个 seed 全等。
- 两个 mode11 无论组合部署还是分别单独部署，整图输出均与 stock 不一致。
- 正式整图 latency 调用为 0；部署门禁拒绝 selector-v4，保留 incumbent。

因此当前证据支持“孤立算子机制有效”，不支持“可安全整图部署”或“端到端加速”。下一步应先修复并重新认证 mode11 的真实图正确性，而不是继续测速度。
"""
 (out/'REPORT.md').write_text(report,encoding='utf-8')
 seal_flat(out,(Path(__file__).resolve(),));print(json.dumps(result,indent=2,sort_keys=True))

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--operator-analysis',default=str(DEFAULT_OPERATOR));p.add_argument('--completion-audit',default=str(DEFAULT_COMPLETION));p.add_argument('--failed-graph-attempt',default=str(DEFAULT_FAILED));p.add_argument('--correctness-diagnostic',default=str(DEFAULT_DIAG));p.add_argument('--output-dir',default=str(DEFAULT_OUTPUT));return p.parse_args()
if __name__=='__main__':run(parse_args())
