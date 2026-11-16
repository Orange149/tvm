from pathlib import Path
from hashlib import sha256
import shutil

def patch(source):
    def replace(before,after):
        nonlocal source
        if source.count(before)!=1:raise ValueError('frame stream anchor differs: '+before[:80])
        source=source.replace(before,after)
    replace('struct Args {','struct Args {\n  std::string cpu_frame_stream_spec;')
    replace('      << "  --serial\\n"','      << "  --cpu-frame-stream-spec PATH (CPU-only bounded common frames)\\n"\n      << "  --serial\\n"')
    replace('    } else if (key == "--serial") {',
            '    } else if (key == "--cpu-frame-stream-spec") {\n      args.cpu_frame_stream_spec=need_value(key);\n    } else if (key == "--serial") {')
    replace('#include "contention_native.h"','#include "contention_native.h"\n#include "frame_stream_native.h"')
    replace('    if (!args.contention_spec.empty()) return controlled_contention::Run(args, &stages);',
            '    if (!args.cpu_frame_stream_spec.empty()) return cpu_frame_stream::Run(args, &stages);\n'
            '    if (!args.contention_spec.empty()) return controlled_contention::Run(args, &stages);')
    return source

def prepare_source(parent,destination):
    parent,destination=Path(parent),Path(destination)
    source=parent/'sources/contention_native'
    destination.mkdir(parents=True,exist_ok=True)
    for p in source.glob('*.h'):shutil.copy2(p,destination/p.name)
    shutil.copy2(Path(__file__).with_name('native.h'),destination/'frame_stream_native.h')
    shutil.copy2(Path(__file__).with_name('flow.h'),destination/'frame_stream_flow.h')
    original=source/'vta_stage_pipeline_runner_contention.cc'
    target=destination/'vta_stage_pipeline_runner_frame_stream.cc';target.write_text(patch(original.read_text()))
    return {'source_path':str(target),'source_preimage_sha256':sha256(original.read_bytes()).hexdigest(),
            'source_patched_sha256':sha256(target.read_bytes()).hexdigest()}
