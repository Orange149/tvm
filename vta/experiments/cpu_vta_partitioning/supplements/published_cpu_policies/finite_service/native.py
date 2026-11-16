"""Derive a finite CPU-only calibration from the frozen validated event probe."""
from pathlib import Path
import os,shlex,subprocess,shutil
from cpu_vta_partitioning.common import read,write,sha256,now
from cpu_vta_partitioning.transport import SSH
from yolo_scheduler_trace_repair.header_patch import once
from scheduler_event_audit.decode import validate_stream,intersection
from .common import ROOT,OUTPUT,CURRENT,CONFIG,freeze,protocol,sources


def source_text():
    original=ROOT/'scheduler_event_audit/switch_probe.cc'
    receipt=read(CURRENT/'platform'/read(CURRENT/'PLATFORM_BOOT_CHANGE.json')['observed_boot']/'events/build/PREPARED.json')
    if sha256(original)!=receipt['source_sha256']:raise ValueError('source probe differs')
    text=original.read_text()
    text=once(text,' if(argc!=1)return 3;',' if(argc!=1)return 3;\n pin(1);')
    text=once(text,'{{"alone",1},{"contended",2},{"sleep",1},{"migrate",1}}',
        '{{"busy2",2},{"busy3",3},{"busy4",4},{"busy5",5},{"busy6",6},{"mixed",3}}')
    text=once(text,'const bool ok=rows.size()==5','const bool ok=rows.size()==23')
    text=once(text,'   if(!strcmp(name,"sleep"))work(r,"sleep",0,150000000);',r'''
   if(!strcmp(name,"mixed")) {
     if(i==0)work(r,"busy",0,80000000);
     else for(int repeat=0;repeat<16;++repeat){work(r,"sleep",0,1000000);work(r,"busy",0,i==1?1000000:30000);}
   }
   else if(!strcmp(name,"sleep"))work(r,"sleep",0,150000000);''')
    text=once(text,'   else work(r,"busy",0,150000000);','   else work(r,"busy",0,80000000);')
    old='  puts("{\\\"interface\\\":\\\"PERF_RECORD_SWITCH\\\",'
    # Keep the original ABI description, but replace its old finite case list.
    text=once(text,r'\"workload_threads\":5',r'\"workload_threads\":23')
    text=once(text,r'\"cases\":[\"alone\",\"contended\",\"sleep\",\"migrate\"]',
        r'\"cases\":[\"busy2\",\"busy3\",\"busy4\",\"busy5\",\"busy6\",\"mixed\"]')
    return text


def prepare(output=OUTPUT):
    output=Path(output);build=output/'build';build.mkdir(parents=True,exist_ok=True)
    proof_path=build/'PREPARED.json'
    if proof_path.exists():
        proof=read(proof_path)
        if proof['sources']!=sources() or sha256(build/'probe')!=proof['binary_sha256']:
            raise ValueError('finite calibration preparation differs')
        return proof
    code=source_text();src=build/'calibration.cc';src.write_text(code)
    boot=read(CURRENT/'PLATFORM_BOOT_CHANGE.json')['observed_boot']
    cap=CURRENT/'platform'/boot/'events/BOARD_SELF_TEST.json'
    if not read(cap)['validation']['passed']:raise ValueError('current boot switch capability missing')
    freeze(output/'PLAN.json',dict(protocol=protocol(),sources=sources(),expected_boot_id=boot,
        capability_sha256=sha256(cap),source_sha256=sha256(src),
        predictions_and_test_criteria_frozen_before_native_calibration=True))
    tc=read(ROOT/'artifacts/cross_network_model_01/phase_profile/build/BUILD.json')['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:raise ValueError('SDK setup differs')
    for t in list(tc['tools'].values())+list(tc.get('setup_files',{}).values()):
        if sha256(t['path'])!=t['sha256']:raise ValueError('SDK tool differs')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    argv=['bash','-c','source "$1" || exit $?; shift; exec "$@"','finite-sdk',tc['sdk_setup'],
        'bash','-c','exec $CXX -std=c++17 -O2 "$1" -o "$2" -pthread','finite-compile',str(src.resolve()),str((build/'probe').resolve())]
    p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=120)
    (build/'compile.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('finite calibration build failed')
    runtime=ROOT/'artifacts/scheduler_gap_audit_01/resnet18/capsule/runtime'
    q=shutil.which('qemu-aarch64-static')
    if not q:raise ValueError('QEMU required')
    p=subprocess.run([q,str(runtime/'ld-linux-aarch64.so.1'),'--library-path',str(runtime),str((build/'probe').resolve()),'--describe'],capture_output=True,text=True,timeout=30)
    import json
    if p.returncode:raise ValueError('AArch64 load failed')
    desc=json.loads(p.stdout)
    if desc['DNN_inferences']!=0 or desc['workload_threads']!=23:raise ValueError('calibration scope differs')
    proof=dict(sources=sources(),binary_sha256=sha256(build/'probe'),source_sha256=sha256(src),
        toolchain=tc,QEMU_load_passed=True,QEMU_not_native_validation=True,protocol=protocol(),description=desc)
    write(proof_path,proof);return proof


def check(document):
    if document.get('status')!='MEASURED' or document.get('DNN_inferences')!=0 or document.get('periodic_sampling') is not False:
        raise ValueError('finite platform event capture unavailable')
    expected=[('busy'+str(n),i) for n in range(2,7) for i in range(n)]+[('mixed',i) for i in range(3)]
    streams=document['streams']
    if [(s['case'],s['index']) for s in streams]!=expected or len({s['tid'] for s in streams})!=23:
        raise ValueError('finite platform task list differs')
    result=[]
    for s in streams:
        if s['data_size']-(s['head']-s['tail'])<=64:raise ValueError('event ring near full')
        if s['nice']!=0:raise ValueError('calibration requires equal nice-0 tasks')
        row=validate_stream(s)
        want=(['sleep','busy']*16 if s['case']=='mixed' and s['index'] else ['busy'])
        if [x['kind'] for x in row['segments']]!=want:raise ValueError('finite work segments incomplete')
        for seg in row['segments']:
            if seg['requested_core']!=0:raise ValueError('calibration escaped workload core')
        if s['case'].startswith('busy') or s['index']==0:
            if not 80_000_000<=row['segments'][0]['CPU_clock_ns']<=85_000_000:
                raise ValueError('CPU work budget differs from finite calibration')
        else:
            target=1_000_000 if s['index']==1 else 30_000
            if any(not target<=seg['CPU_clock_ns']<=target+1_000_000 for seg in row['segments'] if seg['kind']=='busy'):
                raise ValueError('short held-out CPU work budget differs')
        result.append(row)
    # Target task timelines must not claim simultaneous execution on one core.
    for case in sorted({s['case'] for s in streams}):
        group=[r for r in result if r['case']==case];spans=[]
        for row in group:
            for span in row['on_CPU']:spans.append({**span,'tid':row['tid']})
        ordered=sorted(spans,key=lambda s:(s['start_ns'],s['end_ns']));previous=None
        for span in ordered:
            if previous is not None and span['tid']!=previous['tid'] and intersection(span['start_ns'],span['end_ns'],previous['start_ns'],previous['end_ns'])>100_000:
                raise ValueError('two target TIDs overlap on one core')
            if previous is None or span['end_ns']>previous['end_ns']:previous=span
    return dict(passed=True,streams=result,records=sum(len(r['events']) for r in result),
                no_DNN=True,no_FPGA=True,heldout_data_not_used_for_calibration=True)


def board(output=OUTPUT,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output=Path(output);plan=read(output/'PLAN.json');proof=read(output/'build/PREPARED.json')
    if sources()!=plan['sources'] or sources()!=proof['sources'] or sha256(output/'build/probe')!=proof['binary_sha256']:
        raise ValueError('platform calibration identity changed')
    receipt=output/'PLATFORM_RECEIPT.json'
    if receipt.exists():
        old=read(receipt)
        if sha256(old['raw'])!=old['raw_sha256'] or check(read(old['raw']))!=old['validation']:
            raise ValueError('platform receipt differs')
        return dict(status='CACHED',new_native_sessions=0,DNN_inferences=0)
    ssh=SSH(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=plan['expected_boot_id']:raise ValueError('boot differs from frozen platform plan')
    remote=ssh.config['remote_base'].rstrip('/')+'/diagnostics/finite_service_01/'+boot+'/'+proof['binary_sha256'][:24]
    q=shlex.quote;ssh.idle_command('mkdir -p '+q(remote),boot=boot);ssh.upload(output/'build/probe',remote+'/probe')
    script='\n'.join(['cd '+q(remote),'test "$(sha256sum probe | cut -d " " -f 1)" = '+q(proof['binary_sha256']),
        'chmod 755 probe',
        'if test -f exit_code; then :; elif test -d .attempt; then exit 86; else',
        'mkdir .attempt',
        'if ./probe > raw.json.tmp 2> native.log; then code=0; else code=$?; fi',
        'mv raw.json.tmp raw.json','printf "%s\\n" "$code" > exit_code','fi','cat raw.json'])
    raw=ssh.idle_command(script,boot=boot,timeout=30);path=output/'raw.json';path.write_text(raw)
    document=read(path);validation=check(document)
    if ssh.boot_id()!=boot:raise ValueError('boot changed during platform calibration')
    write(receipt,dict(boot_id=boot,raw=str(path.resolve()),raw_sha256=sha256(path),
        binary_sha256=proof['binary_sha256'],sources=sources(),validation=validation,
        native_sessions=1,DNN_inferences=0,FPGA_access=False,system_changes=False,
        recorded_at=now()))
    return dict(passed=True,records=validation['records'],new_native_sessions=1,DNN_inferences=0)
