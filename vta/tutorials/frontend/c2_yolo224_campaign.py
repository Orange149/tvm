#!/usr/bin/env python3
"""Resume-safe controlled C2 qualification, paired pilots and frozen formal blocks."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shlex
from types import SimpleNamespace
from c2_yolo224_prepare import ROOT,write,sha
from c2_yolo224_board import Board,SSH,BIT,read
from c2_yolo224_analyze import reference,metrics
from aggregate_thesis_vta_queue_profile import load_records,aggregate
from build_thesis_command_capacity_certificate import create_profile,create_certificate

ORDER=((0,1,2,3),(1,3,0,2),(2,0,3,1),(3,2,1,0))


class Campaign:
    def __init__(self,out,host,master=None):
        self.out=Path(out).resolve();self.master=Path(master).resolve() if master else self.out
        self.board=Board(self.out,host);self.ref=None;self.small=None
        self.source_reference=self.master/'runs/reference_M0_01'

    def execute(self,label,mode,frames=64,warmup=0,settle=0,small=False,**kwargs):
        if self.ref is None:self.ref=reference(self.source_reference)
        caps=self.small or (1<<25,1<<25)
        values=dict(label=label,mode=mode,frames=frames,warmup=warmup,settle=settle,serial=False,detailed=False,
            diagnostics=False,proxy_off=False,control='',insn=caps[0] if small else 1<<25,
            uop=caps[1] if small else 1<<25,threshold=0,expect_invalid=False,program='program.json')
        values.update(kwargs);options=SimpleNamespace(**values);directory=self.out/'runs'/label
        if not (directory/'exitcode').exists():
            if directory.exists():raise RuntimeError('incomplete/unknown previous attempt: '+str(directory))
            self.board.run(options)
        contract=read(directory/'contract.json')
        if contract['runner_sha256']!=sha(self.out/'package/c2_yolo224_runner'):
            raise ValueError('cannot reuse a result from a different runner')
        if options.expect_invalid:
            if int((directory/'exitcode').read_text())==0:raise ValueError('negative case unexpectedly succeeded')
            result=read(directory/'result.json')
            if result['status']!='INVALID':raise ValueError('negative target did not exit through safe joined path')
            check={'expected_invalid':True,'reason':result['error'],'worker_exit':'joined before native main returned',
                   'result_sha256':sha(directory/'result.json')}
        else:check=metrics(directory,self.ref)
        if not (directory/'verified.json').exists():write(directory/'verified.json',check)
        print(json.dumps({'verified':label,**{k:check[k] for k in ('mode','fps','pool_high_water_bytes','expected_invalid','reason') if k in check}},ensure_ascii=False),flush=True)
        # Preserve every byte locally, then reclaim only this run's RAM dumps.
        if not (directory/'remote_dump_reclaimed.json').exists():
            dest=self.board.remote+'/'+label;q=shlex.quote
            raw=directory/'outputs.bin'
            if raw.exists():
                self.board.ssh('set -eu\ntest "$(sha256sum '+q(dest+'/outputs.bin')+" | cut -d' ' -f1)\" = "+sha(raw)+
                    '\nrm '+q(dest+'/outputs.bin')+' '+q(dest+'.tar'),label+'_reclaim')
            write(directory/'remote_dump_reclaimed.json',{'all_raw_bytes_saved_locally':True,'outputs_sha256':sha(raw) if raw.exists() else None})
        return check

    def q1(self):
        directory=self.out/'runs/qual_M2'
        contract=read(directory/'contract.json')
        status=aggregate(load_records(directory/'stderr'),contract['environment'])
        if not (self.out/'worker_queue_profile.json').exists():write(self.out/'worker_queue_profile.json',status)
        binding=dict(package_dir=str(self.out/'package'),runner=str(self.out/'package/c2_yolo224_runner'),
            runtime=str(self.out/'package/libtvm_runtime.so'),driver=str(self.out/'package/libvta.so'),
            bitstream=str(BIT),shared_buffer_plan=str(self.out/'shared_buffer_plan.json'),vta_max_xfer_bytes=1<<25,
            queue_status=str(self.out/'worker_queue_profile.json'),evidence_scope='board_profiled_program',
            boot_id=contract['boot_id'],output_dir=str(self.out/'capacity_profile'))
        profile=self.out/'capacity_profile/default_capacity_profile.json'
        if not profile.exists():create_profile(SimpleNamespace(**binding))
        target=self.out/'q1/command_capacity_certificate.json'
        if not target.exists():
            create_certificate(SimpleNamespace(**{**binding,'output_dir':str(self.out/'q1'),
                'peak_profile':str(profile),'mode':'q1'}))
        cert=read(target);d=cert['derivation'];self.small=(d['insn_capacity_bytes'],d['uop_capacity_bytes'])
        if not (self.out/'q1/uploaded.json').exists():
            self.board.ssh('mkdir '+shlex.quote(self.board.remote+'/q1'),'q1_mkdir')
            for name in ('command_capacity_certificate.json','default_capacity_profile.json'):
                self.board.call(['scp',*self.board.options,str(self.out/'q1'/name),self.board.host+':'+self.board.remote+'/q1/'+name],'q1_upload_'+name)
            write(self.out/'q1/uploaded.json',{'certificate_sha256':sha(target),'capacities':self.small})
        print(json.dumps({'Q1_capacities':self.small,'peaks':[status['insn_peak_bytes'],status['uop_peak_bytes']],
                         'queue_instances':status['queue_instances'],'complete_submissions':status['submissions']}),flush=True)

    def qualify(self):
        for mode in ('M0','M1','M2'):self.execute('qual_'+mode,mode,diagnostics=True)
        self.q1()
        self.execute('qual_M3','M3',small=True,diagnostics=True)
        self.execute('qual_manual','manual',small=True,diagnostics=True)
        signature=lambda label:Counter(tuple(r[k] for k in ('reason','insn_bytes','uop_bytes','load_bytes','store_bytes'))
            for r in load_records(self.out/'runs'/label/'stderr'))
        if not(signature('qual_M2')==signature('qual_M3')==signature('qual_manual')):
            raise ValueError('Q1 changed command work or submission multiplicity')
        for mode in ('L1','L2','CP','E1'):self.execute('qual_'+mode,mode,small=True,detailed=True)
        self.execute('control_full','M1',frames=3,control='full')
        for control,mode in (('stale','M1'),('duplicate','M1'),('incomplete','M1'),('abort','M1')):
            self.execute('control_'+control,mode,frames=3,control=control,expect_invalid=True)
        self.execute('control_before','E1',frames=3,small=True,detailed=True,control='before')
        self.execute('control_after','E1',frames=3,small=True,detailed=True,control='after')
        self.execute('control_bad_capacity','M2',frames=3,insn=256,uop=4096,expect_invalid=True)
        # A fresh, healthy executor after the controlled failures.
        self.execute('health_after_faults','M2',frames=3,small=True)
        for mode in ('M0','M1','M2','M3','L1','L2','CP','E1'):
            self.execute('pilot_'+mode,mode,frames=64,warmup=5,settle=16,small=mode not in ('M0','M1','M2'))
        self.execute('proxy_off_L2','L2',frames=64,warmup=5,settle=16,small=True,proxy_off=True)
        groups={}
        for group,modes in (('main',('M0','M1','M2','M3')),('lease',('L1','L2','CP','E1'))):
            fastest=min(read(self.out/'runs'/('pilot_'+m)/'verified.json')['ii_ms'] for m in modes)
            count=max(128,math.ceil(20000/fastest)+1)
            groups[group]={'modes':modes,'frames':count,'warmup':5,'settle':16,
                'round_orders':[[modes[i] for i in order] for order in ORDER],
                'minimum_planned_window_ms':20000,'independent_boots':3}
        if not (self.out/'formal_protocol.json').exists():write(self.out/'formal_protocol.json',groups)
        if not (self.out/'qualification_complete.json').exists():write(self.out/'qualification_complete.json',{
            'all_outputs_byte_exact':True,'all_modes':True,'controlled_faults':True,'formal_protocol_frozen':True,
            'q1_capacity':self.small,'boot_id':read(self.out/'board_state.json')['boot_id']})

    def formal(self):
        self.q1()
        protocol=read(self.master/'formal_protocol.json')
        for group,value in protocol.items():
            for round_index,order in enumerate(value['round_orders']):
                for mode in order:
                    self.execute(f'formal_{group}_r{round_index+1}_{mode}',mode,frames=value['frames'],
                        warmup=value['warmup'],settle=value['settle'],small=group=='lease' or mode=='M3')
        self.execute('formal_manual','manual',frames=protocol['main']['frames'],warmup=5,settle=16,small=True)
        if not (self.out/'formal_complete.json').exists():write(self.out/'formal_complete.json',{
            'boot_id':read(self.out/'board_state.json')['boot_id'],'protocol_sha256':sha(self.master/'formal_protocol.json')})

    def additional(self):
        self.q1()
        import copy
        q=shlex.quote
        def upload(path,relative):
            self.board.call(['scp',*self.board.options,str(path),self.board.host+':'+self.board.remote+'/'+relative],
                            'extra_upload_'+relative.replace('/','_'))
        plan=read(self.out/'program.json');shared=read(self.out/'shared_buffer_plan.json')
        bad=copy.deepcopy(shared);bad['edges'][0]['tensors'][0]['entry']=999999
        path=self.out/'shared_buffer_plan_bad_entry.json'
        if not path.exists():write(path,bad);upload(path,path.name)
        program=copy.deepcopy(plan);program['shared_buffer_plan']=self.board.remote+'/'+path.name
        name='program_bad_entry.json';path=self.out/name
        if not path.exists():write(path,program);upload(path,name)
        result=self.execute('fallback_bad_entry','M2',program=name)
        if result['borrowed_pool_payload_bytes']!=0 or result['slot_external_actual_bytes']!=526848:
            raise ValueError('ineligible anchor did not fall back to whole-edge external K2')
        # Whitespace-only graph revision keeps semantics unchanged, while an old
        # Q1 certificate must reject the changed exact artifact. Both the ordinary
        # program identity and shared plan are updated, so they cannot mask Q1.
        old=self.out/'package/stages/stage0/graph.json';changed=self.out/'graph_whitespace.json'
        if not changed.exists():changed.write_text(json.dumps(read(old),separators=(',',':'))+'\n');upload(changed,changed.name)
        if sha(changed)==sha(old):raise ValueError('drift fixture did not change graph bytes')
        variant=copy.deepcopy(shared);variant['stages'][0]['graph_sha256']=sha(changed)
        path=self.out/'shared_buffer_plan_whitespace.json'
        if not path.exists():write(path,variant);upload(path,path.name)
        program=copy.deepcopy(plan);program['stages'][0]['graph']=self.board.remote+'/'+changed.name
        program['stages'][0]['graph_sha256']=sha(changed)
        program['shared_buffer_plan']=self.board.remote+'/'+path.name
        name='program_whitespace.json';path=self.out/name
        if not path.exists():write(path,program);upload(path,name)
        self.execute('manual_identity_drift','manual',frames=3,small=True,program=name)
        self.execute('q1_identity_drift','M3',frames=3,small=True,program=name,expect_invalid=True)
        failed=read(self.out/'runs/q1_identity_drift/result.json')
        if failed['allocation_snapshots'] or '[LOAD]' in (self.out/'runs/q1_identity_drift/stdout').read_text():
            raise ValueError('old Q1 identity was not rejected before graph allocation/load')
        if not (self.out/'additional_complete.json').exists():write(self.out/'additional_complete.json',{
            'anchor_fallback_byte_exact':True,'manual_same_capacity_byte_exact':True,
            'old_Q1_certificate_rejected_before_graph_load':True,
            'drift_scope':'whitespace-only graph artifact revision; no claim that workload/peak changed'})
        helper=self.out/'c2_yolo224_address_observer.so'
        if not (self.out/'address_observer_uploaded.json').exists():
            upload(helper,helper.name);write(self.out/'address_observer_uploaded.json',{'sha256':sha(helper)})
        self.execute('address_observed_E1','E1',frames=64,small=True,detailed=True,observer=True)
        from c2_yolo224_address_observer import verify
        if not (self.out/'dynamic_address_certificate.json').exists():verify(self.out,'address_observed_E1')

    def boot_check(self):
        self.execute('qual_M0','M0',diagnostics=True)
        self.execute('qual_M2','M2',diagnostics=True)
        self.q1()
        original=read(self.master/'q1/command_capacity_certificate.json')['derivation']
        if self.small!=(original['insn_capacity_bytes'],original['uop_capacity_bytes']):
            raise ValueError('fresh-boot command demand differs; do not mix capacity configurations')
        self.execute('qual_M3','M3',small=True,diagnostics=True)
        self.execute('qual_L2','L2',frames=16,small=True,detailed=True)
        self.execute('qual_E1','E1',frames=16,small=True,detailed=True)
        self.board.call(['scp',*self.board.options,str(self.out/'c2_yolo224_address_observer.so'),
            self.board.host+':'+self.board.remote+'/c2_yolo224_address_observer.so'],'boot_address_observer_upload')
        self.execute('address_observed_E1','E1',frames=16,small=True,detailed=True,observer=True)
        from c2_yolo224_address_observer import verify
        verify(self.out,'address_observed_E1')
        self.execute('control_full','M1',frames=3,control='full')
        for control in ('stale','duplicate','incomplete','abort'):
            self.execute('control_'+control,'M1',frames=3,control=control,expect_invalid=True)
        for control in ('before','after'):
            self.execute('control_'+control,'E1',frames=3,small=True,detailed=True,control=control)
        self.execute('health_after_faults','M2',frames=3,small=True)
        from c2_yolo224_qualification import reconcile
        reconcile(self.out)
        pointer=self.master/'allocation_trace_location.json'
        if not pointer.exists():
            self.allocations()
            write(pointer,{'directory':str(self.out),'boot_id':read(self.out/'board_state.json')['boot_id'],
                'trace_sha256':sha(self.out/'allocation_trace_reconciliation.json'),
                'scope':'separate driver-INFO qualification; no formal timing samples'})
        if not (self.out/'qualification_complete.json').exists():write(self.out/'qualification_complete.json',{
            'fresh_boot_requalification':True,'main_graph_outputs_byte_exact':True,'lease_controls_passed':True,
            'q1_capacity':self.small,'boot_id':read(self.out/'board_state.json')['boot_id']})

    def q2(self):
        self.q1();peak=read(self.out/'worker_queue_profile.json')['insn_peak_bytes']
        results={}
        for ratio in (75,50,25):
            evidence=self.out/f'Q2_{ratio}_candidate.json'
            if evidence.exists():
                results[ratio]=read(evidence);continue
            threshold=(peak*ratio//100)//16*16
            if threshold<7*16:
                results[ratio]={'not_applicable':'below runtime minimum threshold'}
                write(evidence,results[ratio]);continue
            label=f'Q2_{ratio}_diag';active=label
            try:
                self.execute(label,'Q2',frames=64,diagnostics=True,threshold=threshold)
                records=load_records(self.out/'runs'/label/'stderr')
                insn=((max(4096,2*max(r['insn_bytes'] for r in records))+255)//256)*256
                uop=((max(4096,2*max(r['uop_bytes'] for r in records))+255)//256)*256
                if max(insn,uop)>1<<25:raise ValueError('Q2 margin exceeds runtime limit; no clipping permitted')
                active=f'Q2_{ratio}_small'
                self.execute(active,'Q2',frames=64,insn=insn,uop=uop,threshold=threshold,diagnostics=True)
                reduced=load_records(self.out/'runs'/active/'stderr')
                signature=lambda rows:Counter(tuple(r[k] for k in
                    ('reason','insn_bytes','uop_bytes','load_bytes','store_bytes')) for r in rows)
                if signature(reduced)!=signature(records):
                    raise ValueError('Q2 capacity changed full-tail command demand')
                for pair_index,mode in enumerate(('M3','Q2','Q2','M3')):
                    active=f'Q2_{ratio}_paired_{pair_index+1}_'+mode
                    self.execute(active,mode,frames=128,warmup=5,settle=16,
                        small=mode=='M3',insn=self.small[0] if mode=='M3' else insn,
                        uop=self.small[1] if mode=='M3' else uop,threshold=0 if mode=='M3' else threshold)
                results[ratio]={'status':'passed','threshold':threshold,'insn_capacity':insn,'uop_capacity':uop,
                    'complete_tail_peaks':[max(r['insn_bytes'] for r in records),max(r['uop_bytes'] for r in records)],
                    'submissions':len(records),'saving_vs_Q1':sum(self.small)-insn-uop,
                    'peak_source_sha256':sha(self.out/'runs'/label/'stderr'),
                    'program_sha256':sha(self.out/'program.json'),'runner_sha256':sha(self.out/'package/c2_yolo224_runner'),
                    'capacity_rule':'align256(max(4096, 2 * complete submission peak)); no clipping',
                    'performance_scope':'single boot ABBA exploration; no independent-boot performance claim'}
            except Exception as error:
                # Preserve the failed target. An SSH timeout leaves completion
                # unknown and stops the campaign; it never authorizes recovery.
                failed=self.out/'runs'/active
                if not (failed/'exitcode').exists():raise
                failure={'status':'failed','threshold':threshold,'failed_label':active,
                         'error':str(error),'retried':False}
                failure_path=self.out/f'Q2_{ratio}_failure.json'
                if not failure_path.exists():write(failure_path,failure)
                self.board.recover_fpga(f'Q2_{ratio}_failed')
                self.execute(f'Q2_{ratio}_recovery_health','M2',frames=3,small=True)
                results[ratio]={**failure,'recovery_fresh_executor_byte_exact':True}
            write(evidence,results[ratio])
        if not (self.out/'q2_complete.json').exists():write(self.out/'q2_complete.json',results)

    def allocations(self):
        self.q1()
        for mode in ('M0','M1','M2','M3','manual','L1','L2','CP','E1'):
            self.execute('allocation_'+mode,mode,frames=3,small=mode not in ('M0','M1','M2'),
                allocation_log=True,loader_log=True)
        from c2_yolo224_allocation import reconcile
        traces=reconcile(self.out)
        objects=sorted(set(x for mode in traces['modes'].values() for x in mode['loaded_init_objects']))
        self.board.ssh('sha256sum '+' '.join(shlex.quote(x) for x in objects),
            'allocation_observed_loaded_DSO_hashes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('qualify','formal','q2','additional','boot_check','allocations'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--master',type=Path)
    parser.add_argument('--host',default='root@192.168.1.247')
    args=parser.parse_args();getattr(Campaign(args.out,args.host,args.master),args.phase)()


if __name__=='__main__':main()
