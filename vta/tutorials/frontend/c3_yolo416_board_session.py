"""Execute exactly six frozen calls in one owned fresh RPC unit."""
import argparse
import gc
import json
import math
from pathlib import Path
import shlex
import time
from types import SimpleNamespace
from c3_tps_sync_release_local import read,write,sha,verify,require,capture_helper_call
from c3_yolo416_board_transport import ComparisonBoard
from c3_tps_aggregation_board import submit_signature


def metadata_gate(execution,position,kind,diagnostic):
    m=execution.get('metadata',{});count=execution.get('counts')
    require(type(count) is int and type(m.get('started_calls')) is int and
            type(m.get('completed_calls')) is int and
            count==m['started_calls']==m['completed_calls']==position and
            execution.get('helper_returned') is True and not execution.get('observation_errors'),
            'SINGLE_TARGET_CALL_NOT_COMPLETED')
    require(type(m.get('packed_calls_per_invocation')) is int and type(m.get('parameter_count')) is int and
            m['packed_calls_per_invocation']==1 and m['adds_synchronize'] is False and
            m['last_kind']==('diagnostic' if diagnostic else 'timer'),'HELPER_INVOCATION_CONTRACT_CHANGED')
    if kind=='operator':require(m['schema']=='c3_official_window_once_helper_v1' and m['parameter_count']==4,
                                'FUSED_HELPER_ABI_CHANGED')
    else:require(m['schema']=='c3_yolo416_graph_once_v1' and m['parameter_count']==0,'GRAPH_HELPER_ABI_CHANGED')


def run(root,ordinal,freeze_sha,warm_path=None,warm_sha=None):
    require(sha(root/'freeze.json')==freeze_sha,'EXECUTION_FREEZE_CHANGED')
    freeze=read(root/'freeze.json');verify(freeze['bindings'])
    require(freeze.get('status')=='TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING','BOARD_POOL_NOT_READY')
    require(not (root/'prepare_failure.json').exists(),'BOARD_PREPARATION_FAILED')
    artifacts=read(root/'artifacts.json');manifest=read(root/'call_manifest.json')
    session=manifest['sessions'][ordinal-1];rows=[r for r in manifest['calls'] if r['session_id']==session['session_id']]
    require(len(rows)==6 and [r['position'] for r in rows]==list(range(1,7)),'FROZEN_SIX_CALL_UNIT_CHANGED')
    warm=None
    if session['phase']=='formal':
        require(warm_path is not None and sha(warm_path)==warm_sha,'WARM_DIAGNOSTIC_CONTRACT_CHANGED')
        warm=read(warm_path)
        require(all(warm[k]==session[k] for k in ('object_id','variant','rotation')) and len(warm['calls'])==6,
                'WARM_DIAGNOSTIC_PROGRAM_OR_ROTATION_CHANGED')
    out=root/'board'/('session_%03d'%ordinal);out.mkdir(exist_ok=False)
    args=SimpleNamespace(host=freeze['host'],known_hosts=freeze['known_hosts'],
                         transport_contract=read(root/'transport_contract.json'))
    board=ComparisonBoard(args,'/var/volatile/c3_yolo416_'+root.name,artifacts['bitstream']['sha256'])
    remote=adapter=None;results=[]
    try:
        baseline=read(root/'board/preflight.json');boot=baseline['boot_id']
        write(out/'preflight.json',board.preflight(boot))
        write(out/'start.json',board.start(ordinal,session['collection']))
        remote=board.connect(out);get=remote.get_function
        clear=get('vta.runtime.official_submit_ledger_clear');submit=get('vta.runtime.official_submit_ledger')
        collection=get('vta.runtime.official_collection_status')
        capacity=get('vta.runtime.queue_capacity_status');replay=get('vta.runtime.replay_status')
        inventory={'queue':json.loads(capacity()),'replay':json.loads(replay()),'collection':json.loads(collection())}
        require(all(inventory['queue'][k]==33554432 for k in
                    ('insn_capacity_bytes','uop_capacity_bytes','insn_submit_threshold_bytes')) and
                inventory['queue']['submissions']==0 and inventory['queue']['replay_policy']=='disabled' and
                inventory['replay']['mode']=='disabled' and inventory['replay']['template_count']==0,
                'ACTUAL_RUNTIME_QUEUE_OR_REPLAY_CHANGED')
        require(inventory['collection']['mode']==session['collection'] and inventory['collection']['debug_flag']==0,
                'INITIAL_COLLECTION_MODE_CHANGED')
        write(out/'inventory.json',inventory)
        descriptor=artifacts['programs'][rows[0]['program_key']]
        diagnostic=session['phase']=='diagnostic'
        if descriptor['kind']=='operator':
            from c3_yolo416_operator_board_adapter import OperatorAdapter
            adapter=OperatorAdapter(remote,descriptor,artifacts,out,diagnostic)
        else:
            from c3_yolo416_graph_board_adapter import GraphAdapter
            adapter=GraphAdapter(remote,descriptor,artifacts,out,diagnostic)
        require(json.loads(submit())['attempted_submits']==0,'REFERENCE_PATH_ENQUEUED_VTA')
        expected_files={'libvta.so':artifacts['runtime']['sha256'],
                        'libtvm_runtime.so':artifacts['runtime_tvm']['sha256'],**adapter.expected_files}
        write(out/'actual_environment.json',board.environment(boot,expected_files))
        h=read(artifacts['hardware_archive']['path'])['fifo_depths']
        commands={i:h[k] for i,k in enumerate(('load_queue','gemm_queue','store_queue'),1)}
        from c3_tps_window_progress import LANES
        tokens=dict(zip(LANES,[h[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
        for row in rows:
            require(sha(root/'freeze.json')==freeze_sha,'EXECUTION_FREEZE_CHANGED')
            identity=adapter.prepare_call(row);clear()
            write(out/('call_%02d_begin.json'%row['position']),{**row,'identity':identity,'start_unix':time.time()})
            value,execution=capture_helper_call(row['position'],lambda:adapter.invoke(diagnostic),
                adapter.counts,adapter.metadata,submit,
                lambda v:write(out/('call_%02d_execution.json'%row['position']),v))
            metadata_gate(execution,row['position'],descriptor['kind'],diagnostic)
            elapsed=None
            if diagnostic:
                require(isinstance(value,str),'DIAGNOSTIC_DUMP_NOT_TEXT')
                (out/('native_%02d.txt'%row['position'])).write_text(value)
            else:
                require(type(value) in (int,float) and math.isfinite(value) and value>0,'INVALID_HELPER_TIME')
                elapsed=float(value)
            state=json.loads(collection())
            require(state['mode']==session['collection'] and state['debug_flag']==0 and
                    (session['collection']!='minimal' or state['detailed_clock_reads']==0),
                    'COLLECTION_OR_DETAIL_CLOCKS_CHANGED')
            checked=adapter.check(row)
            signature=submit_signature(execution['submits'],session['collection'])
            require(len(signature)==(1 if descriptor['kind']=='operator' else 12),'SUBMISSION_PATH_COUNT_CHANGED')
            if diagnostic:
                certificate=adapter.certify(row,value,execution,commands,tokens)
                for key,obj in certificate.items():write(out/('%s_%02d.json'%(key,row['position'])),obj)
                write(out/('signature_%02d.json'%row['position']),signature)
            elif session['phase']=='formal':
                require(sha(warm_path)==warm_sha,'WARM_DIAGNOSTIC_CONTRACT_CHANGED')
                own=warm['calls'][row['position']-1]
                require(own['position']==row['position'] and own['seed']==row['seed'] and signature==own['signature'],
                        'FORMAL_PATH_DIFFERS_FROM_OWN_DIAGNOSTIC')
            else:
                require(signature==submit_signature(adapter.local_signature(row['position']),'detailed'),
                        'FPGA_QUALIFICATION_PATH_DIFFERS_FROM_LOCAL')
            require(json.loads(replay())['mode']=='disabled' and json.loads(capacity())['replay_policy']=='disabled',
                    'REPLAY_CHANGED_DURING_UNIT')
            result={**row,'status':'PASS','target_calls':1,'elapsed_ms':elapsed,'helper':execution,
                    'collection_status':state,'output_checks':checked,'reference_identity':identity,'end_unix':time.time()}
            write(out/('call_%02d_end.json'%row['position']),result);results.append(result)
        write(out/'final_environment.json',board.environment(boot,expected_files))
        verify(freeze['bindings']);require(len(results)==6,'INCOMPLETE_FRESH_UNIT')
        adapter.release();adapter=None;get=clear=submit=collection=capacity=replay=None
        gc.collect();board.close_connection(remote,out);remote=None;gc.collect();board.stop()
        write(out/'stopped_sockets.json',board.sockets())
        (out/'rpc.log').write_text(board.ssh('cat '+shlex.quote(board.session+'/rpc.log')));board.session=None
        write(out/'summary.json',{'status':'COMPLETE_FRESH_UNIT_SUCCESS','session':session,'target_calls':6,'results':results})
        return 0
    except BaseException as e:
        if not (out/'failure.json').exists():write(out/'failure.json',{'type':type(e).__name__,'message':str(e),
                'passed_calls':len(results),'no_retry':True})
        raise
    finally:
        if adapter is not None:adapter.release()
        if board.session is not None:
            try:
                board.stop()
                if not (out/'rpc.log').exists():
                    (out/'rpc.log').write_text(board.ssh('cat '+shlex.quote(board.session+'/rpc.log')))
            except Exception as e:write(out/'cleanup_warning.json',{'type':type(e).__name__,'message':str(e)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--ordinal',type=int,required=True)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--warm-contract',type=Path);p.add_argument('--warm-sha256')
    a=p.parse_args();run(a.output.resolve(),a.ordinal,a.freeze_sha256,a.warm_contract,a.warm_sha256)
