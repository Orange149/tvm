"""Ordered local preparation, without freezing a still-evolving controller or using SSH."""
from pathlib import Path

from .common import DEFAULT_OUTPUT, LOCAL_CONFIG


def prepare(output=DEFAULT_OUTPUT, local_config=LOCAL_CONFIG, *, stage='all', names=None):
    from . import prepare as local
    from .bundle import prepare as pipeline

    if stage not in ('graph', 'modules', 'references', 'profile', 'pipeline', 'all'):
        raise ValueError('unknown local preparation stage')
    output = Path(output).resolve()
    steps = ('graph', 'modules', 'references', 'profile') if stage == 'all' else (stage,)
    result = {}
    for item in steps:
        if item == 'graph':
            from .graph import prepare_graph
            from .units import prepare_units
            manifest = prepare_graph(output / 'graph')
            units = prepare_units(output / 'graph', output / 'units')
            result[item] = dict(canonical_sha256=manifest['canonical_sha256'],
                                unit_count=units['units'], interval_checks=units['interval_checks'])
        elif item == 'pipeline':
            result[item] = pipeline(output)
        else:
            result[item] = local.prepare(output, local_config, stage=item,
                                         **({'names': names} if item == 'modules' else {}))
    if stage == 'all':
        if (output / 'ranking/FREEZE.json').is_file():
            result['pipeline'] = pipeline(output)
            status = 'PREPARED_OFFLINE'
        else:
            status = 'PROFILE_PREPARED_PIPELINE_PENDING'
    else:
        status = 'LOCAL_STAGE_COMPLETE'
    return dict(status=status, stages=result, pipeline_pending=not (output / 'pipeline_bundle/manifest.json').is_file(),
                board_accessed=False)
