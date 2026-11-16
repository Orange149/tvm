"""A narrow patch on the authenticated existing pinned executor source."""
from hashlib import sha256

PINNED_SOURCE_SHA256='672710ad702b187225834949d233bf8ae917dae0750c230c60728d4646c87ca8'


def replace(source,before,after):
    if source.count(before)!=1:raise ValueError('phase patch anchor differs: '+before[:90])
    return source.replace(before,after,1)


def patch(source):
    if sha256(source.encode()).hexdigest()!=PINNED_SOURCE_SHA256:
        raise ValueError('phase patch requires frozen pinned executor source')
    source=replace(source,'#include "vta_session_profile.h"',
        '#include "vta_session_profile.h"\n#include "phase_probe.h"')
    source=replace(source,'  bool resource_affinity_probe{false};',
        '  bool resource_affinity_probe{false};\n  std::string phase_event_json;')
    source=replace(source,'      << "  --resource-affinity-probe (qualification only)\\n"',
        '      << "  --phase-event-json PATH (optional same-clock observation)\\n"\n'
        '      << "  --resource-affinity-probe (qualification only)\\n"')
    source=replace(source,'    } else if (key == "--resource-affinity-probe") {',
        '    } else if (key == "--phase-event-json") {\n'
        '      args.phase_event_json = need_value(key);\n'
        '    } else if (key == "--resource-affinity-probe") {')
    source=replace(source,'    Args args = ParseArgs(argc, argv);',
        '    Args args = ParseArgs(argc, argv);\n    vta_workload_phase::Configure(args.phase_event_json);')
    start=source.index('std::vector<NDArray> RunStage(')
    end=source.index('\nbool SameDType',start)
    original=source[start:end];body=original
    body=replace(body,'  timing->set_start_ms = NowMillis();',
        '  const auto phase_call = vta_workload_phase::Next(stage->name);\n'
        '  vta_workload_phase::Span phase_set(stage->name, phase_call, "set");\n'
        '  timing->set_start_ms = NowMillis();')
    body=replace(body,'  timing->set_end_ms = NowMillis();',
        '  timing->set_end_ms = NowMillis();\n  phase_set.End();')
    body=replace(body,'  stage->run();',
        '  vta_workload_phase::Span phase_run(stage->name, phase_call, "run");\n'
        '  stage->run();\n  phase_run.End();')
    body=replace(body,'  CheckStableStageOutputs(stage,',
        '  vta_workload_phase::Span phase_get(stage->name, phase_call, "get");\n'
        '  CheckStableStageOutputs(stage,')
    body=replace(body,'  timing->end_ms = NowMillis();',
        '  timing->end_ms = NowMillis();\n  phase_get.End();')
    return source[:start]+body+source[end:]
