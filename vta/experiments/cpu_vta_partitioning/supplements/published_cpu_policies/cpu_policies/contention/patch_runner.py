"""A small patch on the already-frozen core-model runner, with opt-in replay."""
from pathlib import Path
from hashlib import sha256
import shutil

def patch(source):
    def replace(before,after,count=1):
        nonlocal source
        if source.count(before)!=count: raise ValueError('contention patch anchor differs: '+before[:90])
        source=source.replace(before,after)
    replace('struct Args {','struct Args {\n  std::string contention_spec;')
    replace('      << "  --serial\\n"','      << "  --contention-spec PATH (independent frontier actors)\\n"\n      << "  --serial\\n"')
    replace('    } else if (key == "--serial") {','    } else if (key == "--contention-spec") {\n      args.contention_spec = need_value(key);\n    } else if (key == "--serial") {')
    replace('#ifndef VTA_STAGE_PIPELINE_NO_MAIN','#include "contention_native.h"\n\n#ifndef VTA_STAGE_PIPELINE_NO_MAIN')
    replace('    std::vector<InputRecord> inputs = LoadInputs(args, &stages[0]);',
            '    if (!args.contention_spec.empty()) return controlled_contention::Run(args, &stages);\n'
            '    std::vector<InputRecord> inputs = LoadInputs(args, &stages[0]);')
    # Qualification output is saved only on the first warmup frame; no scored-frame I/O.
    replace('ResultJSON(args, frame, completed, mode, benchmark_start_ms)',
            'ResultJSON(ContentionOutputArgs(args, frame.frame_id), frame, completed, mode, benchmark_start_ms)',count=2)
    replace('#include "contention_native.h"',
            'Args ContentionOutputArgs(const Args& args, int frame) {\n'
            '  Args result=args; if(frame!=0) { result.output_dump_dir.clear(); result.output_mode="raw"; } return result;\n}\n'
            '#include "contention_native.h"')
    return source

def prepare_source(parent,dest):
    parent,dest=Path(parent),Path(dest)
    src=parent/'sources/core_model_native/vta_stage_pipeline_runner_core_model.cc'
    manifest=__import__('json').loads((parent/'manifest.json').read_text())
    name=str(src.relative_to(parent))
    if sha256(src.read_bytes()).hexdigest()!=manifest['files'][name]['sha256']: raise ValueError('runner source hash differs')
    dest.mkdir(parents=True,exist_ok=True)
    for header in src.parent.glob('*.h'): shutil.copy2(header,dest/header.name)
    header=dest/'native_probe.h'
    header.write_text(header.read_text().replace('program == "cpu_05_07";', 'program == "cpu_05_07" || program == "cpu_17_20";'))
    shutil.copy2(Path(__file__).with_name('native.h'),dest/'contention_native.h')
    shutil.copy2(Path(__file__).with_name('turn_gate.h'),dest/'turn_gate.h')
    target=dest/'vta_stage_pipeline_runner_contention.cc'
    target.write_text(patch(src.read_text()))
    return {'source_path':str(target),'source_patched_sha256':sha256(target.read_bytes()).hexdigest(),
            'source_preimage_sha256':sha256(src.read_bytes()).hexdigest()}
