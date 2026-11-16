from contextlib import contextmanager
from pathlib import Path
import builtins
import io
import json
import os
import socket
import subprocess
from cpu_vta_partitioning.common import digest,inventory,now,read,sha256,write
from workload_model.common import ROOT,OUTPUT as STUDY

PARENT=ROOT/'artifacts/scheduler_gap_audit_01'
OUTPUT=ROOT/'artifacts/model_semantics_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261110_model_semantics_audit'
RUN=ROOT/'runs/cross_network_scheduler_gap_01'


def protocol():
    return dict(experiment='model_semantics_audit_01',revision=1,
        window=dict(frames=104,warmup=8,scored=64,tail=32,queue_depth=2),
        control_modes=['zero','diagnostic_observed'],host_modes=['greedy','all_fixed_assignments'],
        native_sessions=0,new_inferences=0,max_replays=198,
        host_order='before',completion_at='last_stage_finish_before_output_queue_push',
        old_scores_and_rankings_unchanged=True,thread_domain_pruned=False,
        results_are_known_data_diagnostic_replay=True,no_CFS_reproduction_claim=True)


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('offline freeze changed: '+str(path))
    else:write(path,value)
    return value


def sources():
    files=[ROOT/'run_model_semantics_audit.py',*sorted(Path(__file__).parent.glob('*.py')),
           ROOT/'tests/test_model_semantics_audit.py',ROOT/'pool_temporal/simulator.py',
           ROOT/'phase_model/simulator.py',ROOT/'pool_temporal/ranking.py']
    return {str(p):sha256(p) for p in files}


@contextmanager
def offline_guard():
    ledger=dict(network_attempts=0,forbidden_process_attempts=0,local_commands=[])
    saved=(socket.socket.connect,socket.socket.connect_ex,socket.create_connection,
           socket.getaddrinfo,subprocess.Popen,os.system)
    def denied_network(*a,**kw):
        ledger['network_attempts']+=1;raise RuntimeError('offline audit forbids network/board access')
    def denied_process(*a,**kw):
        ledger['forbidden_process_attempts']+=1;raise RuntimeError('offline audit forbids this process')
    def local_popen(args,*a,**kw):
        if isinstance(args,(tuple,list)) and args and not kw.get('shell'):
            name=Path(str(args[0])).name
            if name in ('nm','readelf','fc-list','fc-match'):
                ledger['local_commands'].append(list(map(str,args)));return saved[4](args,*a,**kw)
        return denied_process()
    socket.socket.connect=socket.socket.connect_ex=denied_network
    socket.create_connection=socket.getaddrinfo=denied_network
    subprocess.Popen=local_popen;os.system=denied_process
    try:yield ledger
    finally:
        socket.socket.connect,socket.socket.connect_ex=saved[:2]
        socket.create_connection,socket.getaddrinfo=saved[2:4]
        subprocess.Popen,os.system=saved[4:]


def _skip_value(text,start):
    """Skip a JSON value without decoding performance-bearing subtrees."""
    i=start
    if text[i] not in '[{':
        if text[i]=='"':return json.JSONDecoder().raw_decode(text,i)[1]
        while i<len(text) and text[i] not in ',}]':i+=1
        return i
    depth=0;string=False;escaped=False
    while i<len(text):
        c=text[i]
        if string:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':string=False
        elif c=='"':string=True
        elif c in '[{':depth+=1
        elif c in ']}':
            depth-=1
            if depth==0:return i+1
        i+=1
    raise ValueError('unterminated metadata source')


def metadata_projection(path,allowed):
    """Only decode named top-level metadata. summary/FPS values never exposed.

    Bytes are read/hashed for provenance; excluded subtrees are lexically
    skipped, so this is not a model input containing performance results.
    """
    text=Path(path).read_text();decoder=json.JSONDecoder();i=0;result={};seen=set()
    def whitespace(pos):
        while pos<len(text) and text[pos].isspace():pos+=1
        return pos
    i=whitespace(i)
    if text[i]!='{':raise ValueError('metadata root must be object')
    i+=1
    while True:
        i=whitespace(i)
        if text[i]=='}':break
        key,i=decoder.raw_decode(text,i)
        if key in seen:raise ValueError('duplicate metadata key')
        seen.add(key);i=whitespace(i)
        if text[i]!=':':raise ValueError('metadata separator missing')
        i=whitespace(i+1)
        if key in allowed:result[key],i=decoder.raw_decode(text,i)
        else:i=_skip_value(text,i)
        i=whitespace(i)
        if text[i]==',':i+=1
        elif text[i]!='}':raise ValueError('metadata object delimiter missing')
    if not set(allowed).issubset(result):raise ValueError('required metadata missing')
    return result


@contextmanager
def replay_input_guard(output):
    """A replay may read its projections, never raw receipts or performance."""
    output=Path(output).resolve();saved=(builtins.open,io.open,Path.open)
    def wrap(original):
        def guarded(path,mode='r',*a,**kw):
            if isinstance(path,(str,os.PathLike)) and ('r' in mode or '+' in mode):
                p=Path(path).resolve()
                if 'runs' in p.parts or ('paper' in p.parts and 'analysis' in p.parts) or (
                    p.name in ('RESULTS.json','FINAL_SHORT_RESULTS.json','FINAL_LONG_RESULTS.json','cases.csv','stage_errors.csv')):
                    raise RuntimeError('replay cannot read pipeline performance or traces: '+str(p))
            return original(path,mode,*a,**kw)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,saved)
    try:yield
    finally:builtins.open,io.open,Path.open=saved
