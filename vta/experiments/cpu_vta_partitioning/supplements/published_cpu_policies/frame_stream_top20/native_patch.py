"""Patch only a private copy of the authenticated CPU benchmark sources."""
from pathlib import Path
import shutil
from .common import api

def header(source):
    def replace(before,after):
        nonlocal source
        if source.count(before)!=1:raise ValueError('native source anchor differs: '+before[:70])
        source=source.replace(before,after)
    replace('int64_t wait_start{0},dequeue{0},start{0},end{0};',
            'int64_t wait_start{0},dequeue{0},gate_request{0},gate_acquire{0},start{0},end{0},gate_release{0};')
    replace('(actors.size()==2||actors.size()==3)','(actors.size()>=2&&actors.size()<=4)')
    replace('args.cpu_stage_gate=="off",','(args.cpu_stage_gate=="off"||args.cpu_stage_gate=="resource_group"),')
    replace('"frame stream requires individual pinning and no CPU gate"',
            '"frame stream requires individual pinning and supported CPU gate"')
    replace('  Flow<BoundedQueue> flow(count,2,14);',
            '  Flow<BoundedQueue> flow(count,2,14);\n'
            '  std::vector<std::vector<unsigned int>> masks;\n'
            '  if(args.cpu_stage_gate=="resource_group")\n'
            '    for(const auto& stage:args.stages) masks.push_back(stage.cpu_affinity);\n'
            '  vta_deploy::CpuResourceGroups groups(masks);')
    replace('flow.Cancel();ready.notify_all();','groups.Cancel();flow.Cancel();ready.notify_all();')
    replace('record.wait_start=before;record.dequeue=dequeue;record.start=NanoNow();',
            'record.wait_start=before;record.dequeue=dequeue;record.gate_request=NanoNow();\n'
            '          std::unique_ptr<vta_deploy::CpuStageGate::Lease> lease;\n'
            '          if(args.cpu_stage_gate=="resource_group") lease=groups.Acquire(index);\n'
            '          record.gate_acquire=NanoNow();record.start=NanoNow();')
    replace('record.end=NanoNow();','record.end=NanoNow();lease.reset();record.gate_release=NanoNow();')
    replace('cpu_frame_stream_v1','cpu_frame_stream_top20_v1')
    replace('       <<",\\\"set_ms\\\":"<<r.stage.SetMs()',
            '       <<",\\\"gate_mode\\\":\\\""<<args.cpu_stage_gate<<"\\\""\n'
            '       <<",\\\"gate_request_ns\\\":"<<r.gate_request-origin\n'
            '       <<",\\\"gate_acquire_ns\\\":"<<r.gate_acquire-origin\n'
            '       <<",\\\"gate_release_ns\\\":"<<r.gate_release-origin\n'
            '       <<",\\\"set_ms\\\":"<<r.stage.SetMs()')
    return source

def prepare_source(parent,destination):
    parent,destination=Path(parent),Path(destination);source=parent/'sources/frame_stream_native'
    destination.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.suffix in ('.cc','.h'):shutil.copy2(p,destination/p.name)
    c=api(parent);old=source/'frame_stream_native.h';new=destination/old.name
    new.write_text(header(old.read_text()))
    return {'source_path':str(destination/'vta_stage_pipeline_runner_frame_stream.cc'),
            'header_preimage_sha256':c.sha256(old),'header_patched_sha256':c.sha256(new)}
