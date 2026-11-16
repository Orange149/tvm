"""Audit archived graph functions serially; never compile or invoke a target."""
import os,sys,time,signal,subprocess,json,shutil
from pathlib import Path
from collections import Counter
from c3_yolo224_sync_structure import SOURCE,HERE,HOLD,sha,read,write,require

def main():
 out=HOLD/'20261016_c3_yolo224_sync_structure_readonly03';out.mkdir(exist_ok=False)
 seal=read(SOURCE/'artifact_hashes.json')['artifacts']
 # Bind every archived artifact, including actual graphlib and full graph mapping.
 for name,digest in seal.items():require(sha(SOURCE/name)==digest,'SOURCE_SEAL_CHANGED:'+name)
 sources=[HERE/'c3_yolo224_sync_structure.py',Path(__file__).resolve(),HERE/'test_c3_yolo224_sync_structure.py',HERE/'c3_official_window_method.py',HERE/'c3_valid_window.py',HERE.parents[2]/'vta/runtime/runtime.cc',HERE.parents[2]/'vta/runtime/runtime.h']
 write(out/'freeze.json',{'source_archive':str(SOURCE),'archive_manifest_sha256':sha(SOURCE/'artifact_hashes.json'),
       'source_bindings':{str(p):sha(p) for p in sources},'variants':['G0','G1','G2'],'primary_existing_graph':'G2',
       'target_calls':0,'compilations':0,'board_operations':0,'full_native_safety_claim':False,
       'per_worker_seconds':600,'per_worker_memory_bytes':8*1024**3})
 for p in sources[:3]:shutil.copyfile(p,out/p.name)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 with (out/'tests.log').open('x') as log:
  p=subprocess.run([sys.executable,'-m','unittest','-v','test_c3_yolo224_sync_structure'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
 require(p.returncode==0,'PRECHECK_FAILED')
 import tvm
 import vta  # Register serialized TIR intrinsics before reading functions.
 from c3_official_window_baseline import is_vta_mac,buffer_abi
 from c3_official_window_workflow import _tree_rss
 unique={};mapping=[]
 for variant in ('G0','G1','G2'):
  graph=read(SOURCE/variant/'graph.json')
  for node_id,node in enumerate(graph['nodes']):
   if node['op']!='tvm_op' or 'conv2d' not in node['attrs']['func_name']:continue
   symbol=node['attrs']['func_name'];path=SOURCE/variant/(symbol+'.tir.json')
   text=path.with_suffix('').read_text()
   is_vta='T.tir.vta.uop_push(0, 0,' in text
   fn=tvm.ir.load_json(path.read_text()) if is_vta else None
   if is_vta:require(is_vta_mac(fn),'VTA_TEXT_JSON_CLASSIFICATION_DIFFERS')
   row={'variant':variant,'node':node_id,'symbol':symbol,'device':'VTA' if is_vta else 'CPU','tir':str(path),'sha256':sha(path),'abi':buffer_abi(fn) if is_vta else None}
   if is_vta:
    key=row['sha256']
    if key not in unique:unique[key]=dict(row,object_id='function_%02d'%(len(unique)+1))
    row['object_id']=unique[key]['object_id']
   mapping.append(row)
 write(out/'graph_mapping.json',mapping)
 write(out/'inventory.json',list(unique.values()))
 print(json.dumps({'unique_functions':len(unique),'graph_calls':len(mapping),'VTA_calls':sum(r['device']=='VTA' for r in mapping)}),flush=True)
 (out/'objects').mkdir()
 for row in unique.values():
  stem=out/'objects'/row['object_id'];write(stem.with_suffix('.input.json'),row)
  cmd=[sys.executable,str(HERE/'c3_yolo224_sync_structure.py'),'--row',str(stem.with_suffix('.input.json')),'--output',str(stem.with_suffix('.json'))]
  start=time.monotonic();peak=0;limit=None
  with stem.with_suffix('.log').open('x') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    peak=max(peak,_tree_rss(proc.pid))
    if peak>8*1024**3 or time.monotonic()-start>600:
     limit='MEMORY_OR_TIME_LIMIT';os.killpg(proc.pid,signal.SIGKILL);break
    time.sleep(.2)
   code=proc.wait()
  write(stem.with_suffix('.worker.json'),{'returncode':code,'resource_limit':limit,'peak_bytes':peak,'seconds':time.monotonic()-start})
  if not stem.with_suffix('.json').exists():write(stem.with_suffix('.json'),{'status':'ANALYSIS_UNRESOLVED','error':limit or 'WORKER_FAILED'})
  result=read(stem.with_suffix('.json'))
  print(json.dumps({'object':row['object_id'],'node':row['node'],'variant':row['variant'],'status':result['status'],'MAC':result.get('MAC_categories'),'RESET':result.get('RESET_categories'),'force_serial':result.get('force_serial_requested')}),flush=True)
 summary={}
 for variant in ('G0','G1','G2'):
  entries=[];totals=Counter();mac=Counter();reset=Counter()
  for row in mapping:
   if row['variant']!=variant or row['device']!='VTA':continue
   d=read(out/'objects'/(row['object_id']+'.json'))
   brief={**row,**{k:d.get(k) for k in ['status','categories','MAC_categories','RESET_categories','force_serial_requested','g2l_count','counts','error']}}
   entries.append(brief);totals.update(d.get('categories',{}));mac.update(d.get('MAC_categories',{}));reset.update(d.get('RESET_categories',{}))
  summary[variant]={'VTA_call_sites':len(entries),'unique_functions':len({r['object_id'] for r in entries}),'analyzed_call_sites':sum(x['status']=='TIR_STRUCTURE_AND_DIRECT_OVERWRITE_SCREEN_ONLY' for x in entries),'force_serial_call_sites':sum(x['force_serial_requested'] is True for x in entries),'categories':dict(totals),'MAC_categories':dict(mac),'RESET_categories':dict(reset),'entries':entries}
 write(out/'summary.json',summary)
 for path,digest in read(out/'freeze.json')['source_bindings'].items():require(sha(path)==digest,'TOOL_CHANGED')
 write(out/'completion_manifest.json',{'status':'READONLY_NETWORK_STRUCTURE_AUDIT_FINISHED','target_calls':0,'board_operations':0,'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
 print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
