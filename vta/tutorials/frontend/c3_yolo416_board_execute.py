"""Serial frozen pool executor; first failed unit closes the entire attempt."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from c3_tps_sync_release_local import HERE,PYTHON,read,write,sha,verify,require,env_for_worker
from c3_yolo416_board_transport import ComparisonBoard,transport_contract
from c3_yolo416_comparison_protocol import manifest,canonical,validate
from c3_tps_aggregation_board import submit_signature


def warm_contract(session,prior):
    rows=[r for r in prior if r['object_id']==session['object_id'] and r['variant']==session['variant']
          and r['rotation']==session['rotation'] and r['phase'] in ('diagnostic_prefix','diagnostic_warm')]
    require(len(rows)==6 and [r['position'] for r in rows]==list(range(1,7)),'OWN_WARM_DIAGNOSTIC_NOT_UNIQUE')
    return {'object_id':session['object_id'],'variant':session['variant'],'rotation':session['rotation'],
            'calls':[{'position':r['position'],'seed':r['seed'],
                'signature':submit_signature(r['helper']['submits'],'detailed')} for r in rows]}


def accounting(folder):
    rows=[];started=completed=0;unknown=[];invalid=[]
    def safe(path):
        if not path.exists():return {}
        try:
            value=read(path)
            if not isinstance(value,dict):raise ValueError('record not object')
            return value
        except (ValueError,OSError):invalid.append(str(path));return {}
    for unit in sorted(folder.glob('session_*')):
        if not unit.is_dir():continue
        s=c=0;uncertain=False
        for p in sorted(unit.glob('call_*_begin.json')):
            row=safe(p);position=int(p.stem.split('_')[1]);ep=unit/('call_%02d_execution.json'%position)
            end=unit/('call_%02d_end.json'%position);e=safe(ep);done=safe(end)
            m=e.get('metadata',{})
            if not isinstance(m,dict):invalid.append(str(ep));m={}
            if type(m.get('started_calls')) is int:s=max(s,m['started_calls'])
            if type(m.get('completed_calls')) is int:c=max(c,m['completed_calls'])
            uncertain|=e.get('helper_returned') is not True
            rows.append({'call_index':row.get('call_index'),'session':unit.name,'variant':row.get('variant'),
                         'object_id':row.get('object_id'),'certified':done.get('status')=='PASS',
                         'execution_evidence':str(ep) if ep.exists() else None})
        started+=s;completed+=c
        if uncertain:unknown.append(unit.name)
    return {'registered_attempts':len(rows),'helper_started_observed_lower_bound':started,
            'helper_completed_observed_lower_bound':completed,'certified':sum(r['certified'] for r in rows),
            'execution_unresolved_units':unknown,'invalid_records':invalid,'rows':rows,'begin_is_not_execution':True}


def execute(root):
    import psutil
    require(not (root/'completion_manifest.json').exists() and not (root/'prepare_failure.json').exists(),
            'ATTEMPT_ALREADY_CLOSED_OR_PREPARATION_FAILED')
    freeze=read(root/'freeze.json');verify(freeze['bindings'])
    require(freeze.get('status')=='TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING','BOARD_POOL_NOT_READY')
    ledger=read(root/'call_manifest.json');require(canonical(ledger)==canonical(manifest(ledger['attempt'],ledger['pool'])),
                                                  'CALL_MANIFEST_CHANGED')
    require(read(root/'transport_contract.json')==transport_contract(ledger),'TRANSPORT_CONTRACT_CHANGED')
    out=root/'board';out.mkdir(exist_ok=False);status='STOPPED_BOARD_ENVIRONMENT';all_rows=[]
    freeze_sha=sha(root/'freeze.json')
    write(out/'execution_contract.json',{'freeze_sha256':freeze_sha,'host':freeze['host'],
          'planned_calls':ledger['total_target_calls'],'runtime_and_program_bindings':freeze['bindings']})
    artifacts=read(root/'artifacts.json')
    args=SimpleNamespace(host=freeze['host'],known_hosts=freeze['known_hosts'],
                         transport_contract=read(root/'transport_contract.json'))
    board=ComparisonBoard(args,'/var/volatile/c3_yolo416_'+root.name,artifacts['bitstream']['sha256'])
    try:
        write(out/'preflight.json',board.preflight())
        write(out/'deployment.json',board.deploy(artifacts));status='STOPPED_BOARD_UNIT'
        for i,session in enumerate(ledger['sessions'],1):
            require(sha(root/'freeze.json')==freeze_sha,'EXECUTION_FREEZE_CHANGED')
            args=[str(PYTHON),str(HERE/'c3_yolo416_board_session.py'),str(root),'--ordinal',str(i),
                  '--freeze-sha256',freeze_sha]
            if session['phase']=='formal':
                path=out/('expected_warm_%03d.json'%i);write(path,warm_contract(session,all_rows))
                args+=['--warm-contract',str(path),'--warm-sha256',sha(path)]
            start=time.monotonic();peak=0;limit=None
            with (out/('session_%03d_worker.log'%i)).open('x') as log:
                child=subprocess.Popen(args,env=env_for_worker(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    try:
                        proc=psutil.Process(child.pid)
                        peak=max(peak,sum(p.memory_info().rss for p in [proc,*proc.children(recursive=True)]))
                    except psutil.Error:pass
                    if peak>8*1024**3 or time.monotonic()-start>600:
                        limit='MEMORY' if peak>8*1024**3 else 'TIME';os.killpg(child.pid,signal.SIGKILL);break
                    time.sleep(.1)
                code=child.wait()
            write(out/('session_%03d_worker.json'%i),{'returncode':code,'resource_limit':limit,
                'sampled_peak_tree_RSS':peak,'resource_control_seconds':time.monotonic()-start,'session':session})
            if code or limit:
                # The watchdog may prevent the child's finally block. Only
                # attempt-owned processes are eligible for this cleanup.
                board.session=board.root+'/session_%02d'%i
                try:
                    board.stop()
                    unit=out/('session_%03d'%i)
                    if unit.exists() and not (unit/'rpc.log').exists():
                        (unit/'rpc.log').write_text(board.ssh('cat '+shlex.quote(board.session+'/rpc.log')))
                except Exception as e:write(out/'cleanup_warning.json',{'type':type(e).__name__,'message':str(e)})
                finally:board.session=None
                write(out/'stop.json',{'ordinal':i,'session':session,'returncode':code,'resource_limit':limit})
                raise RuntimeError('FIRST_BOARD_UNIT_FAILED_NO_RETRY')
            unit=out/('session_%03d'%i);summary=read(unit/'summary.json')
            require(summary['status']=='COMPLETE_FRESH_UNIT_SUCCESS' and len(summary['results'])==6,
                    'UNIT_NOT_COMPLETE')
            all_rows.extend(summary['results'])
            print(json.dumps({'unit':i,'units_total':len(ledger['sessions']),
                'certified_target_calls':len(all_rows),'object_id':session['object_id'],
                'variant':session['variant'],'phase':session['phase']}),flush=True)
        validate(ledger,all_rows);verify(freeze['bindings'])
        write(out/'results.json',all_rows);write(out/'final_state.json',board.preflight(read(out/'preflight.json')['boot_id']))
        status='COMPLETE_TH_P_OPERATORS_AND_GRAPH_SINGLE_BOOT_ATTEMPT'
    except Exception as e:
        write(out/'failure.json',{'type':type(e).__name__,'message':str(e),'fully_certified_rows':len(all_rows),
                                'performance_conclusion_allowed':False,'retry_allowed':False})
    account=accounting(out);write(out/'call_accounting.json',account)
    write(out/'summary.json',{'status':status,'accounting':account,'planned_calls':ledger['total_target_calls'],
          'formal_samples':sum(r['timed'] for r in all_rows),'failed_sample_splicing_allowed':False})
    print(json.dumps({'status':status,**{k:v for k,v in account.items() if k!='rows'}}),flush=True)
    return 0 if status.startswith('COMPLETE_TH_P') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();sys.exit(execute(a.output.resolve()))
