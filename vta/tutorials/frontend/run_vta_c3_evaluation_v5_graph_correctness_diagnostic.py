#!/usr/bin/env python3
"""Run correctness-only YOLO-512 combined/singleton mode11 diagnostics."""
from __future__ import annotations
import argparse,gc,json,tempfile,time,traceback
from pathlib import Path
import numpy as np
from tvm import rpc
import run_vta_c3_reuse_board as reuse_board
from c3_evaluation_v5 import SEEDS,append_jsonl,read_json,seal_flat,sha256_file,verify_flat,write_json
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import load_executor,mismatch_counts,output_summary,random_input
from run_vta_p7r120_yolo_rpc_board import runtime_inventory
HERE=Path(__file__).resolve().parent
GROUPED=HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
DEFAULT_MAIN=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_build_run01'
DEFAULT_DIAG=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_correctness_diagnostic_build_run01'
DEFAULT_FREEZE=GROUPED/'20260926_c3_yolo512_evaluation_v5_freeze_run01'
DEFAULT_OUTPUT=GROUPED/'20260926_c3_yolo512_evaluation_v5_graph_correctness_diagnostic_board_run01'

def connect(host,port,timeout):
 for _ in range(30):
  try:return rpc.connect(host,port,session_timeout=timeout)
  except Exception:time.sleep(.25)
 raise RuntimeError('RPC unavailable')

def run(args):
 main,diag,freeze_dir=map(Path,(args.main_build,args.diagnostic_build,args.freeze_dir))
 output=Path(args.output_dir); 
 if output.exists():raise FileExistsError(output)
 manifests={'main':verify_flat(main),'diagnostic':verify_flat(diag),'freeze':verify_flat(freeze_dir)}
 freeze=read_json(freeze_dir/'contract.json'); runtime=read_json(freeze['runtime_contract'])
 for name,path in runtime['runtime_paths'].items():
  if sha256_file(path)!=runtime['runtime_sha256'][name]:raise ValueError('runtime changed '+name)
 variants=[('stock',main,'stock'),('incumbent',main,'incumbent'),('selector_v4',main,'selector_v4'),
           ('mode11_only_2fa33cab',diag,'mode11_only_2fa33cab'),
           ('mode11_only_eec24a99',diag,'mode11_only_eec24a99')]
 output.mkdir(parents=True); (output/'calls.jsonl').write_text('')
 write_json(output/'contract.json',{'schema':'c3_evaluation_v5_graph_correctness_diagnostic_board_v1',
  'status':'FROZEN_BEFORE_CORRECTNESS_DIAGNOSTIC','variants':[x[0] for x in variants],
  'seeds':list(SEEDS),'formal_latency_forbidden':True,'timer_calls':0,'manifests':manifests})
 board=ExperimentBoard(args,'/var/volatile/'+output.name,args.port); remote=None; rows=[]; refs={}
 try:
  initial=board.state(); reuse_board.validate_state(initial,exclusive=True); env=board.environment()
  reuse_board.RUNTIME=runtime['runtime_sha256']; write_json(output/'runtime_attestation.json',board.deploy(runtime['runtime_paths']))
  write_json(output/'bitstream_reload.json',board.reload())
  with tempfile.TemporaryDirectory(prefix='c3_v5_graph_correctness_') as temporary:
   for index,(label,root,artifact) in enumerate(variants):
    board.port=args.port+2*index; board.pid=None; board.start(); remote=connect(args.host,board.port,args.session_timeout)
    _,funcs=runtime_inventory(remote); executor=load_executor(remote,root,artifact,[remote.ext_dev(0),remote.cpu(0)],temporary)
    board.environment(require_affinity=True)
    for seed in SEEDS:
     executor.set_input('data',random_input(seed,resolution=512)); funcs['vta.runtime.profiler_clear'](); executor.run()
     outputs=[executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
     if label=='stock': refs[seed]=[np.array(x,copy=True) for x in outputs]
     mismatch=[0]*len(outputs) if label=='stock' else mismatch_counts(refs[seed],outputs)
     row={'variant':label,'seed':seed,'correct_vs_stock':not any(mismatch),
          'mismatch_count_by_output':mismatch,'outputs':output_summary(outputs),
          'runtime_profile_complete':json.loads(funcs['vta.runtime.profiler_status']()),
          'latency_ms':None,'invocation_index':len(rows)}
     rows.append(row);append_jsonl(output/'calls.jsonl',row)
    del executor;remote=None;gc.collect();board.stop();board.pid=None
    print(label,'complete',flush=True)
  status={label:all(r['correct_vs_stock'] for r in rows if r['variant']==label) for label,_,_ in variants}
  after=board.state();reuse_board.validate_state(after,exclusive=True)
  if after['boot_id']!=initial['boot_id']:raise RuntimeError('boot changed')
  summary={'schema':'c3_evaluation_v5_graph_correctness_diagnostic_board_v1',
   'status':'GRAPH_CORRECTNESS_DIAGNOSTIC_COMPLETE','boot_id':initial['boot_id'],
   'variant_correctness':status,'calls':len(rows),'latency_values_collected':False,
   'combined_selector_correct':status['selector_v4'],
   'singleton_mode11_correct':{k:v for k,v in status.items() if k.startswith('mode11_only_')}}
  write_json(output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
 except BaseException as e:
  write_json(output/'invalid_diagnostic.json',{'message':str(e),'traceback':traceback.format_exc()[-12000:],'latency_values_collected':False});raise
 finally:
  remote=None;gc.collect()
  try:board.stop()
  except:pass
  seal_flat(output,(Path(__file__).resolve(),))

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main-build',default=str(DEFAULT_MAIN));p.add_argument('--diagnostic-build',default=str(DEFAULT_DIAG));p.add_argument('--freeze-dir',default=str(DEFAULT_FREEZE));p.add_argument('--output-dir',default=str(DEFAULT_OUTPUT));p.add_argument('--host',default='192.168.1.247');p.add_argument('--port',type=int,default=9090);p.add_argument('--session-timeout',type=int,default=1800);p.add_argument('--known-hosts',default='/home/orange/.ssh/known_hosts');return p.parse_args()
if __name__=='__main__':run(parse_args())
