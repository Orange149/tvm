#!/usr/bin/env python3
"""Frozen historical YOLO416 C2 quick experiment; never rebuild model stages."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, os, re, shlex, shutil, sys
import statistics, subprocess, tarfile
from pathlib import Path
from types import SimpleNamespace
from c2_yolo224_prepare import ROOT, HERE, sha, write, capture_workspace
from c2_yolo224_board import Board, BIT, read
from c2_yolo224_analyze import reference, metrics, outputs
from c2_yolo224_campaign import Campaign
from c2_yolo224_command import reconcile as command_reconcile

HISTORY = HERE/'report_out/yolov3_tiny_pipeline_baseline/20260514_yolo_multisplit100/packages/yolo_multi_pool2_logits'
FROZEN = HERE/'report_out/stage_tile_cotuning/c2_yolov3_tiny/20261018_run03'
SDK = Path('/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux')


def sdk_tool(script,*arguments):
    # Keep cross-compilation environment changes out of network subprocesses.
    subprocess.run(['/bin/bash','-c','unset LD_LIBRARY_PATH; source "$1"; shift; exec "$@"',
        'c2-sdk',str(SDK),sys.executable,str(script),*map(str,arguments)],check=True,cwd=ROOT)


def once(path,value):
    if Path(path).exists():
        if read(path)!=json.loads(json.dumps(value)):raise ValueError('immutable artifact drift: '+str(path))
    else:write(path,value)


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('asset drift: '+str(target))
    else:shutil.copy2(source,target)


def schema(graph):
    return [{'shape':graph['attrs']['shape'][1][graph['node_row_ptr'][h[0]]+h[1]],
             'dtype':graph['attrs']['dltype'][1][graph['node_row_ptr'][h[0]]+h[1]]} for h in graph['heads']]


def prepare(a):
    out=a.out
    if not out.exists():capture_workspace(out)
    if (out/'prepared.json').exists():return
    package=out/'package';package.mkdir(exist_ok=True)
    old=read(a.package/'manifest.json');manifest={**old};stages=[];identities={}
    if old['input_shape']!=[1,3,416,416]:raise ValueError('frozen 416 package required')
    for i,s in enumerate(old['stages']):
        r={**s,'runtime_num_threads':(3,1,4)[i],'artifact_sha256':{}}
        for k,n in [('graph','graph.json'),('lib','graphlib.so'),('params','params.params')]:
            src=a.package/s[k];relative=f'stages/stage{i}/{n}'
            copy_file(src,package/relative);r[k]=relative
            identities[s[k]]=sha(src);r['artifact_sha256'][k+'_sha256']=sha(src)
            copy_file(src,out/'legacy'/s[k])
        actual=schema(read(package/r['graph']))
        if actual!=s['output_schema']:raise ValueError('historical schema differs from compiled graph')
        r['output_schema']={'slots':actual};stages.append(r)
    manifest.update(stages=stages,candidate_id='yolo416_historical_pool2_logits_c2',
        c2_quantization_scope='unchanged per-stage historical program',historical_package=str(a.package))
    once(package/'manifest.json',manifest)
    for name in ('manifest.json','run_native_pipeline.sh','input.bin','libvta.so','libtvm_runtime.so','vta_stage_pipeline_runner'):
        copy_file(a.package/name,out/'legacy'/name);identities[name]=sha(a.package/name)
    for name in ('libvta.so','libtvm_runtime.so'):
        copy_file(FROZEN/'package'/name,package/name)
    (out/'runtime').mkdir(exist_ok=True)
    copy_file(package/'libvta.so',out/'runtime/libvta.so')
    copy_file(FROZEN/'runtime/build_manifest.json',out/'runtime/reused_build_manifest.json')
    once(out/'runtime/build_manifest.json',{'artifacts':{'libvta.so':sha(package/'libvta.so')},
        'source':str(FROZEN),'scope':'reuse frozen instrumented runtime, no model compilation'})
    import numpy as np
    from tvm.relay.testing.darknet import load_image
    inputs=[{'path':'inputs/000.bin','sha256':sha(a.package/'input.bin'),
             'source':'historical input.bin','source_sha256':sha(a.package/'input.bin')}]
    copy_file(a.package/'input.bin',package/'inputs/000.bin')
    seen={inputs[0]['sha256']};source=read(a.inputs)
    for entry in source:
        image=Path(entry.get('image',entry.get('path','')))
        data=np.ascontiguousarray(load_image(str(image),416,416)[None],dtype='float32')
        if data.shape!=(1,3,416,416):raise ValueError('preprocess shape')
        raw=data.tobytes();digest=hashlib.sha256(raw).hexdigest()
        if digest in seen:continue
        seen.add(digest);relative=f'inputs/{len(inputs):03d}.bin'
        (package/relative).write_bytes(raw)
        inputs.append({'path':relative,'sha256':digest,'image':str(image),'image_sha256':sha(image)})
        if len(inputs)==4:break
    if len(inputs)!=4:raise ValueError('four distinct inputs required')
    once(package/'inputs.json',inputs)
    once(out/'historical_asset_identity.json',{'source':str(a.package),'source_hashes':identities,
        'quantization':old['split_quantization_mode'],'unchanged_graphs_libraries_parameters':True})
    from audit_shared_buffer_storage import tensor_bytes
    payloads=[sum(tensor_bytes(x['shape'],x['dtype']) for x in s['output_schema']['slots']) for s in stages[:2]]
    once(out/'prepared.json',{'input_count':4,'stage_outputs':[len(s['output_schema']['slots']) for s in stages],
        'input_slot_bytes':payloads[0],'expected_boundary_api_bytes':2*sum(payloads),
        'scope':'single boot quick paired observation; no C1/C3/Q2 or model rebuilding'})


def compile_runner(out):
    if (out/'compile_runner.json').exists():
        raise FileExistsError('runner already built; use the next phase or a fresh output directory')
    compiler=shutil.which('aarch64-xilinx-linux-g++')
    if not compiler or not os.environ.get('SDKTARGETSYSROOT'):
        sdk_tool(HERE/'c2_yolo416_quick.py','compile','--out',out)
        return
    package=out/'package'
    cmd=[compiler,'--sysroot='+os.environ['SDKTARGETSYSROOT'],'-std=c++17','-O2','-pthread',
         '-ffunction-sections','-fdata-sections','-DTVM_KALLOC_ALIGNMENT=256','-DTVM_INDEX_DEFAULT_I64=1',
         '-DDMLC_USE_FOPEN64=0','-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>']
    cmd+=['-I'+str(ROOT/p) for p in ('include','3rdparty/dlpack/include','3rdparty/dmlc-core/include',
                                   '3rdparty/vta-hw/include','3rdparty/picojson')]
    cmd += [str(HERE/'c2_yolo224_runner.cc'),'-L'+str(package),'-Wl,--gc-sections','-Wl,-rpath,$ORIGIN',
            '-ltvm_runtime','-lvta','-ldl','-lrt','-o',str(package/'c2_yolo224_runner')]
    r=subprocess.run(cmd,capture_output=True,text=True)
    write(out/'compile_runner.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    r.check_returncode()
    depcmd=[x for x in cmd[:-2] if not x.startswith(('-Wl,','-L','-l'))]
    depcmd+=['-MM','-MF',str(out/'runner_dependencies.d')]
    subprocess.run(depcmd,check=True,capture_output=True)
    dependencies={}
    raw=(out/'runner_dependencies.d').read_text().replace('\\\n',' ')
    for name in raw.split(':',1)[1].split():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        relative=path.relative_to(ROOT);copy_file(path,out/'native_build_dependencies'/relative)
        dependencies[str(relative)]=sha(path)
    once(out/'native_build_dependencies/manifest.json',{'files':dependencies,'preprocessor_command':depcmd})
    snap=out/'source_delivery';snap.mkdir()
    sources=[HERE/n for n in ('c2_yolo416_quick.py','c2_yolo224_runner.cc','c2_yolo224_analyze.py',
                             'c2_yolo224_board.py','c2_yolo224_command.py','c3_input_early_release_runner.cc')]
    for p in sources:copy_file(p,snap/p.name)
    once(snap/'manifest.json',{p.name:sha(p) for p in sources})


def storage(out):
    from audit_shared_buffer_storage import audit_package
    from make_shared_buffer_plan import make_plan
    package=out/'package';checks=out/'package_checks.sha256'
    files=sorted(p for p in package.rglob('*') if p.is_file())
    checks.write_text(''.join(sha(p)+'  '+str(p.relative_to(package))+'\n' for p in files))
    audit=audit_package(package,checks);audit['audit_source_sha256']=sha(HERE/'audit_shared_buffer_storage.py')
    once(out/'storage_audit.json',audit)
    once(out/'shared_buffer_plan.json',make_plan(audit,read(out/'runtime/build_manifest.json')))


def certify_reader(out):
    """ARM-only certificate for the unchanged ELF; missing proof disables E1."""
    from c3_input_early_release_audit import SDK_OBJDUMP, POOL_LAUNCH, direct_calls, certify_wait_loop, instructions
    package=out/'package';s=read(package/'manifest.json')['stages'][1]
    graph=read(package/s['graph']);nodes=graph['nodes'];rows=graph['node_row_ptr'];attrs={k:v[1] for k,v in graph['attrs'].items()}
    inp=next(i for i,n in enumerate(nodes) if n['name']==s['input_names'][0]);entry=rows[inp];sid=attrs['storage_id'][entry]
    uses=[(i,p) for i,n in enumerate(nodes) for p,r in enumerate(n['inputs']) if r[:2]==[inp,0]]
    if len(uses)!=1 or [i for i,v in enumerate(attrs['storage_id']) if v==sid]!=[entry]:raise ValueError('multiple readers/aliases')
    node,port=uses[0];reader=nodes[node]['attrs']['func_name']
    if port or nodes[node]['attrs']['num_inputs']!='1' or nodes[node]['attrs']['num_outputs']!='1' or any(h[0]==inp for h in graph['heads']):
        raise ValueError('unreviewed reader ABI')
    evidence=out/'reader_evidence';evidence.mkdir(exist_ok=True)
    def dump(label,path,*args):
        result=subprocess.run([str(SDK_OBJDUMP),*args,str(path)],check=True,capture_output=True,text=True)
        (evidence/(label+'.txt')).write_text(result.stdout);return result.stdout
    lib=package/s['lib'];runtime=package/'libtvm_runtime.so'
    wrapper=dump('wrapper_arm',lib,'-d','--disassemble='+reader)
    body=dump('compute_arm',lib,'-d','--disassemble='+reader+'_compute_')
    if direct_calls(wrapper)!=['VTATLSCommandHandle',reader+'_compute_']:raise ValueError('wrapper effects changed')
    if direct_calls(body)!=['VTABufferCPUPtr','VTABufferCPUPtr'] or sum(x['mnemonic']=='blr' for x in instructions(body))!=1:
        raise ValueError('compute effects changed')
    pair=re.search(r'adrp\s+x0,\s*([0-9a-f]+)[^\n]*\n[^\n]*add\s+x0, x0, #0x([0-9a-f]+)',body)
    if not pair:raise ValueError('CPU callback pointer unresolved')
    address=int(pair[1],16)+int(pair[2],16);symbols=dump('symbol_table',lib,'-t')
    following=[int(m[1],16) for m in re.finditer(r'^([0-9a-f]+)\s+.*?\bF\s+\.text\s+',symbols,re.M) if int(m[1],16)>address]
    if not following:raise ValueError('callback boundary unresolved')
    leaf=dump('callback_arm',lib,'-d','--start-address='+hex(address),'--stop-address='+hex(min(following)))
    code=[x for x in instructions(leaf) if x['mnemonic']!='nop']
    if not code or code[-1]['mnemonic']!='ret' or any(x['mnemonic'] in ('bl','blr','br','svc','hvc','smc') for x in code):
        raise ValueError('callback contains external effects')
    reloc=dump('relocations',lib,'-R')
    if '__TVMBackendParallelLaunch' not in reloc:raise ValueError('parallel launch relocation missing')
    launch=dump('parallel_launch_arm',runtime,'-d','--disassemble=TVMBackendParallelLaunch')
    pool=dump('thread_pool_arm',runtime,'-d','--disassemble='+POOL_LAUNCH)
    if not any(POOL_LAUNCH in x for x in direct_calls(launch)):raise ValueError('unexpected synchronization implementation')
    wait=certify_wait_loop(pool)
    resolver=dump('cpu_pointer_arm',package/'libvta.so','-d','--disassemble=VTABufferCPUPtr')
    if any(x in resolver for x in ('VTADeviceRun','VTASynchronize','VTALoadBuffer2D')):raise ValueError('asynchronous resolver')
    certificate={'schema':'c2-yolo416-last-reader-v1','status':'static_qualified_dynamic_pending',
        'input_node':inp,'input_entry':entry,'storage_id':sid,'node':node,'function':reader,
        'input_shape':attrs['shape'][entry],'input_dtype':attrs['dltype'][entry],
        'reader_output_entry':rows[node],'reader_output_shape':attrs['shape'][rows[node]],
        'reader_output_dtype':attrs['dltype'][rows[node]],'single_direct_reader':uses,
        'callback_ARM_range':[hex(address),hex(min(following))],'reader_has_no_hardware_commands':True,
        'synchronous_wait':wait,'no_llvm_reconstruction':True,
        'identities':{'graph':sha(package/s['graph']),'library':sha(lib),'runtime':sha(runtime),'driver':sha(package/'libvta.so')},
        'evidence':{p.name:sha(p) for p in evidence.iterdir()},
        'dynamic_requirements':['all later ABI ranges disjoint','real resolver input/output ranges disjoint',
                                'all eight outputs exact','before/after overwrite barriers']}
    once(out/'last_reader_certificate.json',certificate)


class QuickBoard(Board):
    def __init__(self,out,host):
        super().__init__(out,host);self.remote='/var/volatile/c2_yolo416_'+self.out.name

    def recover_completed(self,label):
        """Fetch a completed immutable block after a transport-only failure."""
        local=self.out/'runs'/label;dest=self.remote+'/'+label;q=shlex.quote
        state=read(self.out/'board_state.json');before=self.preflight(label+'_recover_before')
        if before['boot_id']!=state['boot_id']:raise RuntimeError('boot changed before recovery')
        for line in before['raw'].splitlines():
            if 'sh -c ' not in line and '/c2_yolo224_runner --' in line:
                raise RuntimeError('native process still running; cannot recover')
        self.ssh('test -s '+q(dest+'/exitcode')+' && test -s '+q(dest+'/result.json'),label+'_recover_completed_check')
        if not (local/'recovery_started.json').exists():
            self.ssh('if [ -e '+q(dest+'.tar')+' ]; then mv '+q(dest+'.tar')+' '+q(dest+'.tar.failed_collection')+'; fi',label+'_preserve_failed_collection')
        once(local/'recovery_started.json',{'native_not_rerun':True,'reason':'collection failed after completed target execution'})
        self.ssh('tar -C '+q(dest)+' -cf '+q(dest+'.tar')+' .',label+'_recover_pack')
        self.call(['scp',*self.options,self.host+':'+dest+'.tar',str(local/'raw.tar')],label+'_recover_fetch',120)
        with tarfile.open(local/'raw.tar') as tar:
            for member in tar.getmembers():
                if not member.isfile():continue
                name=member.name.removeprefix('./')
                if name not in ('stdout','stderr','exitcode','result.json','outputs.bin','reader_addresses.jsonl'):
                    raise ValueError('unexpected recovered artifact: '+name)
                (local/name).write_bytes(tar.extractfile(member).read())
        once(local/'artifact_recovery.json',{'native_not_rerun':True,'boot_id':state['boot_id'],
              'result_sha256':sha(local/'result.json'),'outputs_sha256':sha(local/'outputs.bin')})

    def deploy_quick(self):
        if not (self.out/'board_state.json').exists():self.restore()
        else:
            if self.preflight('resume_identity')['boot_id']!=read(self.out/'board_state.json')['boot_id']:raise RuntimeError('boot changed')
        if (self.out/'deployed.json').exists():return
        inventory={str(p.relative_to(self.out)):sha(p) for folder in ('package','legacy') for p in (self.out/folder).rglob('*') if p.is_file()}
        archive=self.out/'deployment.tar'
        with tarfile.open(archive,'x') as tar:
            for folder in ('package','legacy'):tar.add(self.out/folder,arcname=folder)
        once(self.out/'deployment_inventory.json',inventory)
        q=shlex.quote;self.ssh('test ! -e '+q(self.remote)+' && mkdir '+q(self.remote),'deploy_mkdir')
        self.call(['scp',*self.options,str(archive),self.host+':'+self.remote+'/deployment.tar'],'deploy_upload',180)
        self.ssh('cd '+q(self.remote)+' && task_digest=$(sha256sum deployment.tar) && test "${task_digest%% *}" = '+sha(archive)+' && tar -xf deployment.tar','deploy_extract')
        once(self.out/'deployed.json',{'boot_id':read(self.out/'board_state.json')['boot_id'],'remote':self.remote})

    def configure_quick(self):
        out=self.out;package=out/'package';state=read(out/'board_state.json');base=self.remote+'/package'
        stages=[]
        for s in read(package/'manifest.json')['stages']:
            r={**s,**s['artifact_sha256']}
            for k in ('graph','lib','params'):r[k]=base+'/'+s[k]
            stages.append(r)
        p={'schema':'c2-yolo416-program-v1','input_count':4,'stages':stages,
           'inputs':[{**x,'path':base+'/'+x['path']} for x in read(package/'inputs.json')],
           'boot_id':state['boot_id'],'bitstream':state['firmware'],'bitstream_sha256':state['bitstream_sha256'],
           'runner_sha256':sha(package/'c2_yolo224_runner'),'runtime_sha256':sha(package/'libtvm_runtime.so'),
           'driver_sha256':sha(package/'libvta.so'),'shared_buffer_plan':self.remote+'/shared_buffer_plan.json',
           'capacity_certificate':self.remote+'/q1/command_capacity_certificate.json'}
        if (out/'last_reader_certificate.json').exists():
            reader=read(out/'last_reader_certificate.json')
            p['last_reader']={'status':'qualified','qualification_scope':'ARM static proof; dynamic checks required',
                'certificate':self.remote+'/last_reader_certificate.json','certificate_sha256':sha(out/'last_reader_certificate.json'),
                **{k:reader[k] for k in ('input_node','node','function')}}
        once(out/'program.json',p)
        names=['program.json','shared_buffer_plan.json']
        if 'last_reader' in p:names.append('last_reader_certificate.json')
        for name in names:self.call(['scp',*self.options,str(out/name),self.host+':'+self.remote+'/'+name],'configure_'+name)
        self.ssh('chmod +x '+shlex.quote(base+'/c2_yolo224_runner'),'configure_executable')

    def legacy(self):
        out=self.out;local=out/'legacy_replay'
        if (local/'verified.json').exists():return read(local/'verified.json')
        if local.exists():
            if not (local/'exitcode').exists():raise RuntimeError('legacy attempt completion unknown; inspect before retry')
            attempt=len(list(out.glob('legacy_replay_failed_*')))+1
            local.rename(out/f'legacy_replay_failed_{attempt:02d}')
        attempt=len(list(out.glob('legacy_replay_failed_*')))+1
        local.mkdir();base=self.remote+'/legacy';dest=self.remote+f'/legacy_replay_attempt{attempt:02d}';q=shlex.quote
        old=read(out/'legacy/manifest.json');cmd=[base+'/vta_stage_pipeline_runner']
        for s in old['stages']:
            prefix='--stage'+str(s['index'])
            for k in ('graph','lib','params'):cmd += [prefix+'-'+k,base+'/'+s[k]]
            for k,v in [('name',s['name']),('device',s['device']),('input-names',','.join(s['input_names'])),
                        ('input-sources',','.join(s['input_sources'])),('runtime-num-threads',str((3,1,4)[s['index']]))]:cmd += [prefix+'-'+k,v]
        cmd+=['--input',base+'/input.bin','--runs','20','--queue-depth','2','--runtime-num-threads','4',
              '--output-mode','raw_all_stages','--output-dump-dir',dest+'/outputs','--output-jsonl',dest+'/frames.jsonl']
        env={'LD_LIBRARY_PATH':base,'LD_PRELOAD':base+'/libtvm_runtime.so:'+base+'/libvta.so','TVM_NUM_THREADS':'4',
             'TVM_THREAD_POOL_SPIN_COUNT':'0','AXU5EVB_DRIVER_POST_START_SLEEP_NS':'1000','AXU5EVB_DRIVER_POLL_SLEEP_NS':'1000'}
        once(local/'contract.json',{'argv':cmd,'environment':env,'boot_id':read(out/'board_state.json')['boot_id'],
                                  'historical_assets':read(out/'historical_asset_identity.json')})
        self.preflight('legacy_before')
        self.ssh('\n'.join(['set -eu','mkdir '+q(dest),'cd '+q(base),'set +e',
            'env '+shlex.join([k+'='+v for k,v in env.items()]+cmd)+' > '+q(dest+'/stdout')+' 2> '+q(dest+'/stderr'),
            'r=$?','printf "%s\\n" "$r" > '+q(dest+'/exitcode'),'exit 0']),'legacy_run',150)
        self.ssh('tar -C '+q(dest)+' -cf '+q(dest+'.tar')+' .','legacy_pack')
        self.call(['scp',*self.options,self.host+':'+dest+'.tar',str(local/'raw.tar')],'legacy_fetch',120)
        with tarfile.open(local/'raw.tar') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    relative=Path(member.name.removeprefix('./'))
                    if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe archived path')
                    target=local/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        if int((local/'exitcode').read_text()):raise RuntimeError('legacy replay failed: '+(local/'stderr').read_text()[-1000:])
        rows=[json.loads(s) for s in (local/'frames.jsonl').read_text().splitlines() if s.strip()]
        rows=[r for r in rows if 'frame_id' in r]
        if len(rows)!=20 or any(len(r['raw_output_files'])!=8 for r in rows):raise ValueError('incomplete legacy outputs')
        ref=[]
        for i,r in enumerate(rows):
            values=[(local/Path(x['path']).relative_to(dest)).read_bytes() for x in r['raw_output_files']]
            if i and values!=ref:raise ValueError('legacy input output is unstable')
            ref=values
        ii=(rows[-1]['stage2_end_ms']-rows[0]['stage2_end_ms'])/19
        value={'frames':20,'ii_ms':ii,'fps':1000/ii,'all_eight_outputs_repeatable':True,
               'stage_run_ms_mean':[statistics.mean(r[f'stage{i}_run_ms'] for r in rows) for i in range(3)],
               'output_sha256':[hashlib.sha256(x).hexdigest() for x in ref]}
        once(local/'verified.json',value);self.preflight('legacy_after');return value


class QuickCampaign(Campaign):
    def __init__(self,out,host):
        self.out=self.master=Path(out);self.board=QuickBoard(out,host);self.ref=None;self.small=None
        self.source_reference=self.out/'runs/reference_M0_01'
        cert=self.out/'q1/command_capacity_certificate.json'
        if cert.exists():
            d=read(cert)['derivation'];self.small=d['insn_capacity_bytes'],d['uop_capacity_bytes']
        if (self.source_reference/'exitcode').exists():self.ref=reference(self.source_reference,4)

    def execute(self,label,*args,**kwargs):
        directory=self.out/'runs'/label
        if directory.exists() and not (directory/'exitcode').exists():
            self.board.recover_completed(label)
        return super().execute(label,*args,**kwargs)

    def establish_reference(self):
        # execute() requires an existing reference, so run the first serial M0 directly.
        if self.ref is None:
            o=SimpleNamespace(label='reference_M0_01',mode='M0',frames=8,warmup=0,settle=0,serial=True,
                detailed=False,diagnostics=True,proxy_off=False,control='',insn=1<<25,uop=1<<25,
                threshold=0,expect_invalid=False,program='program.json')
            self.board.run(o);self.ref=reference(self.source_reference,4)
        legacy=read(self.out/'legacy_replay/verified.json')
        observed=[hashlib.sha256(x).hexdigest() for x in self.ref[0]]
        if observed!=legacy['output_sha256']:raise ValueError('runtime migration changes historical outputs; C2 qualification stopped')
        once(self.out/'migration_output_check.json',{'status':'passed','eight_outputs_byte_identical':True,
            'legacy_sha256':legacy['output_sha256'],'M0_sha256':observed})
        once(self.source_reference/'verified.json',metrics(self.source_reference,self.ref))

    def main_group(self):
        self.establish_reference()
        for mode in ('M0','M1','M2'):
            self.execute('qual_'+mode,mode,frames=8,diagnostics=True,allocation_log=True,loader_log=True)
        self.q1()
        for mode in ('M3','manual'):
            self.execute('qual_'+mode,mode,frames=8,small=True,diagnostics=True,allocation_log=True,loader_log=True)
        once(self.out/'command_reconciliation.json',command_reconcile(self.out))
        pilot=self.execute('pilot_M0','M0',frames=16,warmup=4,settle=8)
        legacy=read(self.out/'legacy_replay/verified.json')
        bridge={'legacy_fps':legacy['fps'],'C2_M0_fps':pilot['fps'],'ii_ratio':pilot['ii_ms']/legacy['ii_ms'],
                'source_scope':'same stages/inputs; historical runner and libraries versus unified C2 runner and frozen instrumented libraries'}
        if bridge['ii_ratio']>1.10:
            repeat=self.execute('migration_repeat_M0','M0',frames=16,warmup=4,settle=8)
            bridge['repeat_fps']=repeat['fps']
            r=read(self.out/'runs/migration_repeat_M0/result.json')
            bridge['stage_run_ms_mean']=[statistics.mean(f['stage_end_ms'][i]-f['stage_begin_ms'][i] for f in r['frames']) for i in range(3)]
            bridge['unexplained_regression']=repeat['ii_ms']/legacy['ii_ms']>1.10
        once(self.out/'migration_performance_check.json',bridge)
        if bridge.get('unexplained_regression'):raise RuntimeError('M0 migration slowdown >10%; inspect stages before timing shared configurations')
        n=max(64,math.ceil(10000/pilot['ii_ms'])+1)
        protocol={'statistical_unit':'one actual Linux boot; two blocks per mode, descriptive only',
                  'orders':[['M0','M1','M2','M3'],['M3','M2','M1','M0']],
                  'frames':n,'warmup':4,'settle':8,'minimum_window_ms':10000,
                  'timing':'all three stages and eight output snapshots; preprocessing/NMS/disk/checking excluded'}
        once(self.out/'quick_protocol.json',protocol)
        for round_index,order in enumerate(protocol['orders'],1):
            for mode in order:
                m=self.execute(f'main_{round_index}_{mode}',mode,frames=n,warmup=4,settle=8,small=mode=='M3')
                if m['completion_window_ms']<10000:raise ValueError('window shorter than frozen protocol')
        self.execute('manual_timing','manual',frames=n,warmup=4,settle=8,small=True)

    def controls(self,early):
        self.establish_reference()
        import copy
        shared=copy.deepcopy(read(self.out/'shared_buffer_plan.json'))
        shared['edges'][0]['tensors'][0]['entry']=999999
        program=copy.deepcopy(read(self.out/'program.json'))
        program['shared_buffer_plan']=self.board.remote+'/shared_buffer_plan_bad_entry.json'
        for name,value in [('shared_buffer_plan_bad_entry.json',shared),('program_bad_entry.json',program)]:
            path=self.out/name;once(path,value)
            self.board.call(['scp',*self.board.options,str(path),self.board.host+':'+self.board.remote+'/'+name],'fallback_upload_'+name)
        fallback=self.execute('fallback_bad_entry','M2',frames=4,small=True,program='program_bad_entry.json')
        if fallback['borrowed_pool_payload_bytes'] or fallback['slot_external_actual_bytes']!=read(self.out/'runs/qual_M1/verified.json')['slot_external_actual_bytes']:
            raise ValueError('unqualified pool entry did not safely fall back to external slots')
        self.execute('control_full','M1',frames=4,control='full',small=True)
        for control in ('stale','duplicate','incomplete','abort'):
            self.execute('control_'+control,'M1',frames=4,control=control,small=True,expect_invalid=True)
        self.execute('control_bad_capacity','manual',frames=2,insn=256,uop=4096,expect_invalid=True)
        self.execute('health_after_faults','M1',frames=3,small=True)
        if early:
            for control in ('before','after'):
                self.execute('control_'+control,'E1',frames=4,control=control,small=True,detailed=True)
        once(self.out/'controls_complete.json',{'normal_full_slow_consumer':True,'stale_duplicate_incomplete_abort':True,
              'undersized_command_rejected':True,'fresh_executor_health':True,'early_barriers':early,
              'bad_pool_entry_whole_edge_fallback':True})

    def early_group(self):
        self.establish_reference()
        eligible=(self.out/'last_reader_certificate.json').exists()
        if eligible:
            from c2_yolo224_address_observer import verify
            if not (self.out/'address_observer_config.json').exists():
                sdk_tool(HERE/'c2_yolo224_address_observer.py','build','--out',self.out)
            path=self.out/'c2_yolo224_address_observer.so'
            self.board.call(['scp',*self.board.options,str(path),self.board.host+':'+self.board.remote+'/'+path.name],'observer_upload')
            self.execute('address_observed_E1','E1',frames=8,small=True,detailed=True,observer=True)
            verify(self.out,'address_observed_E1')
        self.controls(eligible)
        modes=['L1','L2','CP']+(['E1'] if eligible else [])
        for mode in modes:self.execute('qual_'+mode,mode,frames=8,small=True,detailed=eligible,
                                      proxy_off=not eligible,diagnostics=True,allocation_log=True,loader_log=True)
        n=read(self.out/'quick_protocol.json')['frames']
        for index,order in enumerate((modes,list(reversed(modes))),1):
            for mode in order:
                m=self.execute(f'lease_{index}_{mode}',mode,frames=n,warmup=4,settle=8,
                               small=True,proxy_off=not eligible)
                if m['completion_window_ms']<10000:raise ValueError('short lease timing window')
        if eligible:
            from c2_yolo224_qualification import reconcile
            reconcile(self.out)
        once(self.out/'early_complete.json',{'early_reader_qualified':eligible,'modes':modes})


def report(out):
    from c2_yolo224_command import account
    from c2_yolo224_allocation import reconcile as allocation_reconcile
    if not (out/'early_complete.json').exists() or not (out/'controls_complete.json').exists():
        raise ValueError('campaign/control qualification has not completed')
    ref=reference(out/'runs/reference_M0_01',4);runs={}
    for p in sorted((out/'runs').iterdir()):
        if (p/'exitcode').exists() and (p/'exitcode').read_text().strip()=='0':runs[p.name]=metrics(p,ref)
    once(out/'analysis.json',runs)
    mode_names=[m for m in ('M0','M1','M2','M3','manual','L1','L2','CP','E1') if 'qual_'+m in runs]
    all_commands={m:account(out/'runs'/('qual_'+m)) for m in mode_names}
    baseline=all_commands['M2']
    for m,value in all_commands.items():
        if value!=baseline:raise ValueError('complete VTA command work differs in '+m)
    once(out/'all_mode_command_reconciliation.json',{'status':'passed','modes':all_commands})
    allocation_reconcile(out,prefix='qual_',selected_modes=mode_names)
    final_sources=out/'source_delivery_final';final_sources.mkdir(exist_ok=True)
    names=['c2_yolo416_quick.py','test_c2_yolo416_quick.py','c2_yolo224_runner.cc','c2_yolo224_analyze.py',
        'c2_yolo224_allocation.py','c2_yolo224_board.py','c2_yolo224_campaign.py','c2_yolo224_command.py',
        'c2_yolo224_qualification.py','c2_yolo224_address_observer.py','c2_yolo224_address_observer.cc',
        'c2_yolo224_host_test.py','c3_input_early_release_runner.cc','c3_input_early_release_audit.py',
        'audit_shared_buffer_storage.py','make_shared_buffer_plan.py',
        'aggregate_thesis_vta_queue_profile.py','build_thesis_command_capacity_certificate.py']
    for name in names:copy_file(HERE/name,final_sources/name)
    once(final_sources/'manifest.json',{name:sha(HERE/name) for name in names})
    frozen_sources=FROZEN/'runtime/source_snapshot'
    if frozen_sources.exists() and not (out/'runtime/source_snapshot').exists():
        shutil.copytree(frozen_sources,out/'runtime/source_snapshot')
    rows=[]
    for mode in ('M0','M1','M2','M3','manual','L1','L2','CP','E1'):
        timed=[r for label,r in runs.items() if label.startswith(('main_','lease_')) and label.endswith('_'+mode)]
        if mode=='manual' and 'manual_timing' in runs:timed=[runs['manual_timing']]
        if not timed:continue
        if len(timed)!=(1 if mode=='manual' else 2):raise ValueError('incomplete paired timing blocks: '+mode)
        if any(r['completion_window_ms']<10000 for r in timed):raise ValueError('short completion window: '+mode)
        if any(r['synchronize_calls']!=r['frames']*baseline['complete_submissions']/baseline['frames'] for r in timed):
            raise ValueError('timed submission count differs from qualification: '+mode)
        ii=statistics.mean(r['ii_ms'] for r in timed)
        qual=runs['qual_'+mode]
        ledger=account(out/'runs'/('qual_'+mode))
        rows.append({'mode':mode,'blocks':len(timed),'ii_ms':ii,'fps':1000/ii,
            'fps_blocks':[r['fps'] for r in timed],'copied_bytes':sum(qual['boundary_copied_bytes_per_frame']),
            'handoff_ms':statistics.mean(r['handoff_ms_median'] for r in timed),
            'pool_high_water_bytes':qual['pool_high_water_bytes'],'new_slot_bytes':qual['slot_external_actual_bytes'],
            'borrowed_payload_bytes':qual['borrowed_pool_payload_bytes'],'submits_per_frame':ledger['complete_submissions']/ledger['frames'],
            'instruction_peak':ledger['complete_tail_peaks'][0],'uop_peak':ledger['complete_tail_peaks'][1],
            'overlap':[r['overlap'] for r in timed]})
    once(out/'summary.json',{'rows':rows,'positive_runs':len(runs),'verified_frames':sum(r['frames'] for r in runs.values()),
        'verified_outputs':sum(r['raw_outputs_checked'] for r in runs.values()),
        'verified_bytes':sum(r['raw_bytes_checked'] for r in runs.values()),'all_byte_exact':True,
        'boot_id':read(out/'board_state.json')['boot_id'],'scope':'single boot, two blocks, no stable-performance claim'})
    environments=[]
    for label in runs:
        if not label.startswith(('main_','lease_')) and label!='manual_timing':continue
        pair=[]
        after_file=label+'_after.json'
        for suffix in ('before','after'):
            sample=out/'transport'/(label+'_'+suffix+'.json')
            if suffix=='after' and not sample.exists():
                sample=out/'transport'/(label+'_recover_before.json')
                after_file=sample.name
            raw=read(sample)['stdout']
            cpu=re.search(r'^(\d+)\n(userspace|performance|powersave|ondemand|schedutil)\s*$',raw,re.M)
            pl=re.search(r'^\s*pl0_ref\s+\d+\s+\d+\s+\d+\s+(\d+)',raw,re.M)
            if not cpu or not pl:raise ValueError('missing measured hardware clocks: '+label)
            value={'boot_id':raw.splitlines()[0],'cpu_khz':int(cpu[1]),'governor':cpu[2],'pl0_hz':int(pl[1])}
            pair.append(value)
        if pair[0]!=pair[1] or pair[0]['boot_id']!=read(out/'board_state.json')['boot_id']:
            raise ValueError('hardware environment drift: '+label)
        environments.append({'label':label,**pair[0],'post_sample_file':after_file})
    if len({(e['cpu_khz'],e['governor'],e['pl0_hz']) for e in environments})!=1:
        raise ValueError('hardware clocks changed between blocks')
    once(out/'hardware_environment_reconciliation.json',{'status':'passed','blocks':environments,
        'temperature_scope':'thermal sensor data unavailable; no measured temperature stability claim'})
    with (out/'measurements.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['label','mode','fps','ii_ms','completion_window_ms','pool_high_water_bytes',
            'slot_external_actual_bytes','borrowed_pool_payload_bytes','handoff_ms_median','synchronize_calls'])
        writer.writeheader()
        for label,r in runs.items():writer.writerow({'label':label,**{k:r[k] for k in writer.fieldnames if k!='label'}})
    legacy=read(out/'legacy_replay/verified.json');cert=read(out/'q1/command_capacity_certificate.json')['derivation']
    text=['# YOLOv3-tiny-416 C2快速验证','', '## 问题动机',
          '检验C2能否叠加到历史5.440 FPS对应的416部署；本轮为单启动两轮配对观察。','',
          '## 方案',
          '历史三阶段图、参数和动态库逐文件哈希不变，保留分阶段量化。线程3/1/4、队列深度2；历史输入及另外三张真实图片。',
          '', '## 受控实验',
          'M0/M1/M2/M3依次对应复制、外部双槽共享、合格输入池复用及Q1缩容；manual使用相同容量数字。归还组统一外部输入槽及Q1容量，L1/L2/CP/E1分别为单槽结束归还、双槽结束归还、复制后归还及单槽最后读后归还。',
          '主实验及归还组分别执行正逆序两轮；每块4帧预热、8帧稳定前缀、64个正式帧，完成窗口均不少于10秒。手工同容量额外运行一块。',
          '全部正例逐帧检查8项原始输出，容差为零。性能以完整三阶段及八输出快照的稳定完成窗口计算；预处理、NMS、写盘和核对不计时。',
          f"当前boot：`{read(out/'board_state.json')['boot_id']}`。不将本轮结果与224共同量化结果拼接。",'',
          '## 结果',f"原运行器20帧复测：{legacy['fps']:.6f} FPS，阶段run均值 {legacy['stage_run_ms_mean']} ms。",
          'M0与原运行器在历史输入上的全部8项输出逐字节一致。',
           f"全部计时块前后CPU频率均为 {environments[0]['cpu_khz']} kHz、governor={environments[0]['governor']}，PL0为 {environments[0]['pl0_hz']} Hz；未取得温度传感器数据。",
           'lease_1_L1因计时后收集超时，常规after读数缺失，改用恢复取回前的较晚环境读数；该差异在hardware_environment_reconciliation.json中明确保留，不视为紧邻计时结束的采样。',
          f"Q1实际推导：`{json.dumps(cert,ensure_ascii=False)}`。",'',
          '| 模式 | 块数 | II ms | FPS | 边界复制 B/帧 | 新增槽 B | 池内高水位 B | 提交/帧 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['mode']} | {r['blocks']} | {r['ii_ms']:.6f} | {r['fps']:.6f} | {r['copied_bytes']} | {r['new_slot_bytes']} | {r['pool_high_water_bytes']} | {r['submits_per_frame']} |")
    by_mode={r['mode']:r for r in rows}
    a,b=by_mode['M2'],by_mode['M3']
    text+=['',f"M0→M1：边界框架复制由 {by_mode['M0']['copied_bytes']:,} B/帧降为0；M1→M2少分配 {by_mode['M1']['new_slot_bytes']-a['new_slot_bytes']:,} B。",
           f"M2→M3：命令backing由67,108,864 B降至 {cert['insn_capacity_bytes']+cert['uop_capacity_bytes']:,} B；池高水位下降 {(a['pool_high_water_bytes']-b['pool_high_water_bytes'])/a['pool_high_water_bytes']*100:.4f}%。",
           f"M0→M3：两块平均II由 {by_mode['M0']['ii_ms']:.6f} ms变为 {b['ii_ms']:.6f} ms，描述性FPS变化 {(b['fps']/by_mode['M0']['fps']-1)*100:.3f}%；不作稳定性推断。"]
    if 'E1' in by_mode:
        l1,l2,e=by_mode['L1'],by_mode['L2'],by_mode['E1']
        text += [f"E1相对L1：II缩短 {(1-e['ii_ms']/l1['ii_ms'])*100:.3f}%；相对L2少分配 {l2['new_slot_bytes']-e['new_slot_bytes']:,} B，FPS分别为 {e['fps']:.6f} 与 {l2['fps']:.6f}。",
                 '自然重叠逐块统计（开始生产/完整发布/可比较对数）：']
        for mode in ('L1','L2','CP','E1'):
            text.append(mode+'：'+json.dumps([[x['next_production_starts_before_old_consumer_end'],x['next_production_finishes_before_old_consumer_end'],x['eligible_pairs']] for x in by_mode[mode]['overlap']]))
    text+=['','完整逐块值、边界交接耗时、输入等待和自然重叠分别见 measurements.csv、summary.json 和各 runs/*/verified.json。',
           '实际分配按槽物理地址去重；借用视图不重复计入新增分配。命令backing、片上UOP SRAM与内核固定预留分开解释：池高水位降低不等于内核192 MiB预留释放。',
           'M0另有3,108,248 B普通CPU复制staging有效载荷，不计入u-dma-buf池高水位；公共输入、输出暂存、CPU堆元数据及总RSS不作为已测的C2节省量。外部命令容量缩小也不改变32 KiB片上UOP SRAM。',
           '', '## 结论与限制',
           '本轮全部已核对正例输出逐字节一致。复制、分配及命令需求的归因以相邻配置对照为准；Q1不改变完整提交内容与次数，手工同数字具有相同空间需求。',
           '输入池复用和外部单槽提前归还属于两组独立对照，其节省不叠加。提前归还仅转移输入使用权，底层分配、中间结果及整个VTA阶段互斥锁均保留。',
           '单启动两轮仅提供描述性吞吐观察，不证明跨启动稳定提升或统计等效。没有标签集，不报告mAP；Q2、三启动复核和共同量化416均未开展。',
           '', '## 证据入口',
           '- historical_asset_identity.json：原阶段程序不变的哈希证据。',
           '- legacy_replay/、migration_output_check.json、migration_performance_check.json：复测与迁移桥接。',
           '- storage_audit.json、shared_buffer_plan.json、command_reconciliation.json：准入、分配与完整命令账本。',
           '- qualification_reconciliation.json、dynamic_address_certificate.json：资格与强制交错。',
           '- runs/、transport/、source_delivery_final/：原始测量、环境和最终实现。',
           '- native_build_dependencies/：运行器实际36项编译源依赖；runtime/source_snapshot/为复用运行库的冻结来源。',
           '', '准备与收集异常保留：旧包首次启动因板端缺少timeout命令而未执行模型；SDK环境下观察器上传曾在启动模型前失败。lease_1_L1在完成计时后压缩收集超时，直接取回同一个已完成块的原始输出，未重跑或替换该块。相关失败记录及恢复哈希均保留。',
           '', '复现实验：设置PYTHONPATH为仓库python:vta/python，使用vta-resnet环境的Python执行 c2_yolo416_quick.py all --package <历史416包> --inputs <真实图片清单JSON> --out <全新目录>。编译步骤自动在独立子shell加载AXU SDK，网络运行沿用普通主机环境。']
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'status':'complete','out':str(out),'rows':rows},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','compile','audit','deploy','legacy','main','early','report','all'])
    p.add_argument('--package',type=Path,default=HISTORY);p.add_argument('--inputs',type=Path,default=FROZEN/'package/inputs.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--host',default='root@192.168.1.247')
    a=p.parse_args();a.out=a.out.resolve();a.package=a.package.resolve();a.inputs=a.inputs.resolve()
    phases=['prepare','compile','audit','deploy','legacy','main','early','report'] if a.phase=='all' else [a.phase]
    for phase in phases:
        print(json.dumps({'phase':phase,'out':str(a.out)}),flush=True)
        if phase=='prepare':prepare(a)
        elif phase=='compile':compile_runner(a.out)
        elif phase=='audit':
            storage(a.out)
            try:certify_reader(a.out)
            except (ValueError,subprocess.CalledProcessError) as error:
                write(a.out/'early_reader_not_qualified.json',{'reason':str(error),'fallback':'stage-end release; do not recompile historical model'})
        elif phase=='deploy':
            b=QuickBoard(a.out,a.host);b.deploy_quick();b.configure_quick()
        elif phase=='legacy':QuickBoard(a.out,a.host).legacy()
        elif phase=='main':QuickCampaign(a.out,a.host).main_group()
        elif phase=='early':QuickCampaign(a.out,a.host).early_group()
        else:report(a.out)


if __name__=='__main__':main()

