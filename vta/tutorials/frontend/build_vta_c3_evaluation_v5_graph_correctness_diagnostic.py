#!/usr/bin/env python3
"""Build two correctness-only singleton-mode11 YOLO-512 diagnostics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import tvm
from tvm import relay
import vta
from build_vta_c3_evaluation_v5_graphs import (DEFAULT_ANALYSIS, DEFAULT_FREEZE, DEFAULT_POOL,
    DEFAULT_PREPARE, build_routed, cross_compiler, make_relay_program, selected_routes, sha256, write_json)
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash

HERE=Path(__file__).resolve().parent
GROUPED=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
DEFAULT_REFERENCE=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_build_run01'
DEFAULT_OUTPUT=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_correctness_diagnostic_build_run01'

def run(args):
 out=Path(args.output_dir).resolve()
 if out.exists(): raise FileExistsError(out)
 routes=selected_routes(args.board_pool_dir,args.freeze_dir,args.analysis_dir)
 inc,sel=routes['incumbent'],routes['selector_v4']
 variants={
   'mode11_only_'+sel[0]['execution_task_id'][:8]: [sel[0],inc[1]],
   'mode11_only_'+sel[1]['execution_task_id'][:8]: [inc[0],sel[1]],
 }
 out.mkdir(parents=True)
 env=vta.get_env(); program,params,shape=make_relay_program(env,args); compiler=cross_compiler()
 builds=[]
 for name,route in variants.items(): builds.append(build_routed(name,route,program,params,out,env,compiler))
 ref=Path(args.reference_build)
 ref_graph=sha256(ref/'stock/graph.json')
 if any(row['graph_sha256']!=ref_graph for row in builds): raise RuntimeError('diagnostic graph JSON drift')
 reference_semantic=semantic_params_hash(ref/'stock/params.bin')
 if any(semantic_params_hash(out/row['deployment_variant']/'params.bin')!=reference_semantic for row in builds):
  raise RuntimeError('diagnostic parameter semantics drift')
 summary={'schema':'c3_evaluation_v5_graph_correctness_diagnostic_build_v1',
  'status':'TWO_SINGLETON_MODE11_DIAGNOSTIC_GRAPHS_BUILT','input_shape':list(shape),
  'variants':builds,'formal_latency_forbidden':True,'selection_changed':False,
  'purpose':'isolate graph correctness failure without observing performance'}
 write_json(out/'summary.json',summary)
 write_json(out/'artifact_hashes.json',{'artifacts':{str(x.relative_to(out)):sha256(x) for x in sorted(out.rglob('*')) if x.is_file() and x.name!='artifact_hashes.json'}})
 print(json.dumps({'status':summary['status'],'variants':list(variants)},indent=2))

def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--board-pool-dir',default=str(DEFAULT_POOL)); p.add_argument('--freeze-dir',default=str(DEFAULT_FREEZE))
 p.add_argument('--analysis-dir',default=str(DEFAULT_ANALYSIS)); p.add_argument('--reference-build',default=str(DEFAULT_REFERENCE))
 p.add_argument('--cfg',default=str(DEFAULT_PREPARE/'resolution_512/yolov3-tiny-512.cfg'))
 p.add_argument('--weights',default='/tmp/tvm_test_data/darknet/yolov3-tiny.weights')
 p.add_argument('--darknet-lib',default='/home/orange/.tvm_test_data/darknet/libdarknet2.0.so')
 p.add_argument('--output-dir',default=str(DEFAULT_OUTPUT)); return p.parse_args()
if __name__=='__main__': run(parse_args())
