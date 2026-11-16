"""Existing detached SSH lifecycle with explicit per-configuration executors."""
from pathlib import Path
from copy import deepcopy
import shlex
import time

from cpu_vta_partitioning.common import read,write,digest
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.transport import TransportError
from cpu_vta_partitioning.transport.scripts import session_script
from cpu_vta_partitioning.prepare.payload import job_storage_bytes
from ..transport import SSH as LegacySSH
from ..evidence import identified
from .domain import ENV,protocol,validate_candidate


def executor(config):
    return 'policy' if config['affinity_mode']=='pinned' or config['control_mask']!=[0,1,2,3] else 'legacy'


def make_task(kind,config,bundle,*,warmup=0,scored=4,tail=0,indices=None,executor_override=None,**labels):
    if set(labels)&{'configuration','stages','environment','experiment','executor','executor_sha256','task_id','bundle_id','input_indices'}:
        raise ValueError('task labels override frozen execution fields')
    programs=read(Path(bundle)/'plans/programs.json');validate_candidate(config,programs,kind in ('calibration','qualify_component'))
    if any(type(v)is not int or v<0 for v in (warmup,scored,tail)) or scored<1:raise ValueError('invalid native window')
    indices=list(indices if indices is not None else protocol()['qualification_indices'])
    if not indices or len(set(indices))!=len(indices) or any(i not in (0,16,17,18) for i in indices):raise ValueError('invalid input identities')
    if executor_override is not None and (executor_override!='policy' or kind not in ('executor_bridge','qualify_executor_bridge') or config['affinity_mode']!='group'):
        raise ValueError('executor override is confined to matched group-mode bridge diagnostics')
    exe=executor_override or executor(config);spec=read(Path(bundle)/'plans/executors.json')[exe]
    doc={'kind':kind,'configuration':deepcopy(config),'stages':deepcopy(config['stages']),'environment':deepcopy(ENV),
         'warmup':warmup,'scored':scored,'tail':tail,'input_indices':indices,
         'experiment':protocol()['experiment'],'executor':exe,'executor_sha256':spec['sha256'],'executor_override':executor_override,**labels}
    return identified(doc,read(Path(bundle)/'manifest.json')['bundle_id'])


def native_command(document,bundle,remote_bundle,remote_job):
    c=document['configuration']
    override=document.get('executor_override')
    if override and (override!='policy' or document['kind'] not in ('executor_bridge','qualify_executor_bridge') or c['affinity_mode']!='group'):
        raise ValueError('invalid executor bridge override')
    if document['experiment']!=protocol()['experiment'] or document['executor']!=(override or executor(c)):raise ValueError('wrong resource task/executor')
    spec=read(Path(bundle)/'plans/executors.json')[document['executor']]
    if spec['sha256']!=document['executor_sha256']:raise ValueError('executor hash changed')
    argv,inputs=base_command(document,bundle,remote_bundle,remote_job)
    argv[argv.index(remote_bundle+'/runner/vta_stage_pipeline_runner')]=remote_bundle+'/'+spec['path']
    argv[argv.index('--cpu-stage-gate')+1]=c['execution_mode']
    loader=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv.append('--allow-cpu-affinity-superset')
    if document['executor']=='policy':
        argv+=['--control-cpu-affinity',','.join(map(str,c['control_mask']))]
        if c['affinity_mode']=='pinned':argv.append('--cpu-worker-pinning')
        if document['kind'].startswith('qualify'):argv.append('--resource-affinity-probe')
    return argv,inputs


class SSH(LegacySSH):
    def measure(self,document,bundle,remote_bundle,run_dir,boot):
        root=Path(run_dir)/'raw'/boot/document['task_id'];root.mkdir(parents=True,exist_ok=True)
        remote=self.job_path(document,run_dir,boot);expected=digest(document);q=shlex.quote
        taskfile=root/'task.json'
        if taskfile.exists() and read(taskfile)!=document:raise ValueError('remote task identity collision')
        if not taskfile.exists():write(taskfile,document)
        if self.boot_id()!=boot:raise TransportError('board boot changed')
        self.command('mkdir -p '+q(remote));argv,inputs=native_command(document,bundle,remote_bundle,remote)
        (root/'inputs.txt').write_text(inputs)
        (root/'execute.sh').write_text(session_script(self.config['remote_base'],remote,boot,expected,argv))
        existing=self.command('test -f '+q(remote+'/task_hash')+' && cat '+q(remote+'/task_hash')+' || true').strip()
        if existing and existing!=expected:raise TransportError('remote hash differs')
        try:
            if not existing:
                self.check_space(remote,job_storage_bytes(document,read(Path(bundle)/'plans/programs.json'))+256*1024**2)
                self.upload(root/'inputs.txt',remote+'/inputs.txt');self.upload(root/'execute.sh',remote+'/execute.sh')
                self.command('cd '+q(remote)+' && nohup sh execute.sh </dev/null > launcher.log 2>&1 &')
            deadline=time.monotonic()+660
            while True:
                result=self.command('test -f '+q(remote+'/exit_code')+' && cat '+q(remote+'/exit_code')+' || true').strip()
                if result:break
                if time.monotonic()>deadline:raise TransportError('native deadline exceeded')
                time.sleep(2)
        except TransportError as e:raise TransportError(str(e)+'; retained '+self.failure_snapshot(remote,root)) from e
        output=self.collect(remote,root)
        if ((output/'boot_id').read_text().strip()!=boot or (output/'task_hash').read_text().strip()!=expected or self.boot_id()!=boot):
            raise TransportError('collected boot/task differs')
        if result!='0':raise TransportError('native failed; raw results retained: '+str(root))
        if (output/'inputs.txt').read_text()!=inputs:raise TransportError('native inputs differ')
        return output

    def qualify_environment(self,remote,boot,root,bundle):
        super().qualify_environment(remote,boot,root,bundle)
        raw=self.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
        if raw!=['1066666']*4:raise TransportError('CPU frequency differs; do not measure or silently retune')
