"""Single-boot calibration, exact host ranking, and one frozen Top-20 union."""
from pathlib import Path
import shlex
import time

from cpu_vta_partitioning.common import digest, inventory, locked, now, read, sha256, sync_tree, verify_inventory, write
from yolo_transfer import profile
from .common import assert_controllers, frozen, timed_deploy
from . import components, runtime


def status(output, run_dir):
    output, run_dir = Path(output), Path(run_dir)
    result = {'network': 'resnet50_v2', 'board_access': False}
    catalog = output / 'build/plans/programs.json'
    if catalog.exists():
        specs = read(catalog)
        present = [p for p in specs if (output / 'build/programs' / p['name'] / 'record.json').is_file()]
        result['local_build'] = dict(recorded=len(present), required=len(specs),
            pending=len(specs)-len(present),
            CPU_search=sum(p['kind'] == 'interval' and p['device'] == 'cpu' for p in present),
            VTA_search=sum(p['kind'] == 'interval' and p['device'] == 'vta' for p in present),
            references=sum(p['kind'] != 'interval' for p in present),
            scope='progress markers only; validate verifies module bytes and identity')
    result['profile_capsule_prepared'] = (output / 'profile_capsule/manifest.json').is_file()
    for key, path in [('campaign', run_dir / 'state.json'), ('components', run_dir / 'components/state.json'),
                      ('traffic', run_dir / 'traffic/traffic.json'), ('pipeline', run_dir / 'pipeline/state.json'),
                      ('search', output / 'ranking/SEARCH_STATUS.json'),
                      ('offline_search_acceptance', output / 'SEARCH_ACCEPTANCE.json')]:
        if path.exists():
            doc = read(path)
            result[key] = {k: v for k, v in doc.items()
                           if k not in ('records', 'costs', 'sources', 'files', 'implementation',
                                        'static_source_hashes', 'rankings', 'methods')}
    if (output / 'ranking/FREEZE.json').exists():
        result['ranking_frozen'] = read(output / 'ranking/FREEZE.json')['candidate_count']
    return result


def accept(ssh, document, bundle, remote, run_dir, boot, expected=None):
    path = Path(run_dir) / 'accepted' / boot / (document['task_id'] + '.json')
    if path.exists():
        item = read(path)
        if item['task'] != document or item['boot_id'] != boot:
            raise ValueError('pipeline task or boot identity differs')
        verify_inventory(item['raw'], item['files'])
        result = runtime.summary(item['raw'], document, bundle)
        if result != item['summary']:
            raise ValueError('pipeline receipt summary no longer replays')
    else:
        began = time.monotonic()
        began_at = now()
        raw = ssh.measure(document, bundle, remote, run_dir, boot)
        result = runtime.summary(raw, document, bundle)
        if expected is not None and result['output_fingerprints'] != expected:
            raise ValueError('performance outputs differ from the complete qualification')
        raw = Path(raw)
        if ((raw / 'boot_id').read_text().strip() != boot
                or (raw / 'task_hash').read_text().strip() != digest(document)):
            raise ValueError('pipeline raw identity differs before acceptance')
        item = dict(task=document, boot_id=boot, raw=str(raw), files=inventory(raw), summary=result,
                    measurement_started_at=began_at, accepted_at=now(),
                    measurement_and_collection_seconds=time.monotonic() - began)
        write(path, item)
        ssh.acknowledge(document, run_dir, boot)
    raw = Path(item['raw'])
    if ((raw / 'boot_id').read_text().strip() != boot
            or (raw / 'task_hash').read_text().strip() != digest(document)):
        raise ValueError('pipeline raw identity differs')
    if expected is not None and item['summary']['output_fingerprints'] != expected:
        raise ValueError('accepted performance and qualification output fingerprints differ')
    return item


def pipeline(output, run_dir, local_config, boot):
    from .validation import validate_bundle
    output, run_dir = Path(output), Path(run_dir)
    bundle = output / 'pipeline_bundle'
    manifest = validate_bundle(bundle)
    ssh = runtime.transport(read(local_config)['board'], execute_board=True)
    if ssh.boot_id() != boot:
        raise ValueError('boot changed before full pipelines; preserve the current cohort')
    frozen(run_dir / 'run.json', dict(boot_id=boot, bundle_id=manifest['bundle_id']))
    remote = timed_deploy(ssh, bundle, run_dir)
    write(run_dir / 'hardware.json', components.check_hardware(ssh, bundle, boot), replace=True)
    order = read(bundle / 'plans/execution_order.json')
    pool = read(bundle / 'plans/pool.json')
    if len(order) > 40 or set(order) != set(pool) or len(order) != len(set(order)):
        raise ValueError('pipeline union is not the approved unique selection')
    completed = 0
    for name in order:
        qualify = accept(ssh, runtime.task(bundle, pool[name], 'qualify'), bundle, remote, run_dir, boot)
        accept(ssh, runtime.task(bundle, pool[name], 'short'), bundle, remote, run_dir, boot,
               expected=qualify['summary']['output_fingerprints'])
        completed += 1
        write(run_dir / 'state.json', dict(status='MEASURING', boot_id=boot,
            accepted_sessions=2 * completed, candidate_count=len(order), done_candidates=completed,
            updated_at=now()), replace=True)
        print('ResNet50 pipelines', completed, '/', len(order), name, flush=True)
    ssh.idle_command('cd ' + shlex.quote(remote) + ' && sha256sum -c board.sha256 > post_run_verification.log && sync',
                     boot=boot, timeout=600)
    result = dict(status='COMPLETE', boot_id=boot, candidate_count=len(order), accepted_sessions=2 * len(order),
                  complete_DNN_inferences=49 * len(order), post_deployment_hashes_passed=True,
                  single_input_single_boot_short_window=True, finished_at=now())
    write(run_dir / 'state.json', result, replace=True)
    sync_tree(run_dir)
    return result


def run(output, run_dir, local_config, *, execute_board=False):
    if not execute_board:
        raise ValueError('--execute-board is required')
    from .prepare import validate_profile
    from .ranking import rank, verify_search_acceptance
    from .bundle import prepare as prepare_pipeline
    from .analysis import analyze
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    assert_controllers(output)
    verify_search_acceptance(output)
    capsule = components.profile_bundle(output)
    validate_profile(capsule)
    from .common import ARCHIVED_OUTPUT, ARCHIVED_RUN
    if (ARCHIVED_OUTPUT / 'COMPILER_REGRESSION_STOP.json').is_file():
        from .archive_audit import import_components
        import_components(output, run_dir, archived_output=ARCHIVED_OUTPUT, archived_run=ARCHIVED_RUN)
    ssh = profile.matched_ssh(read(local_config)['board'], execute_board=True)
    with locked(run_dir):
        boot = ssh.boot_id()
        frozen(run_dir / 'run.json', dict(boot_id=boot, output=str(output),
            profile_capsule_id=read(capsule / 'manifest.json')['bundle_id'],
            controller_sha256=sha256(output / 'CONTROLLERS.json')))
        def state(stage, **fields):
            write(run_dir / 'state.json', dict(status=stage, boot_id=boot, updated_at=now(), **fields), replace=True)
        try:
            state('COMPONENT_PILOT')
            components.complete(output, run_dir / 'components', local_config, execute_board=True,
                                limit=39, expected_boot=boot)
            components.measure_traffic(output, run_dir / 'traffic', local_config, execute_board=True,
                                       limit=3, expected_boot=boot)
            components.pilot_cost(output, run_dir)
            state('COMPONENTS')
            components.complete(output, run_dir / 'components', local_config, execute_board=True, expected_boot=boot)
            state('DMA')
            components.measure_traffic(output, run_dir / 'traffic', local_config, execute_board=True, expected_boot=boot)
            if ssh.boot_id() != boot:
                raise ValueError('boot changed before ranking')
            state('RANKING')
            result = rank(output, run_dir, resume=True)
            if result['status'] != 'COMPLETE':
                state('SEARCH_INCOMPLETE', search=result)
                return status(output, run_dir)
            state('PIPELINE_PREPARATION')
            prepare_pipeline(output)
            state('PIPELINE')
            measured = pipeline(output, run_dir / 'pipeline', local_config, boot)
            state('ANALYZING', pipeline=measured)
            report = analyze(output, run_dir)
            state('COMPLETE', pipeline=measured, report=report)
            return status(output, run_dir)
        except Exception as error:
            state('STOPPED_ON_ERROR', error_type=type(error).__name__, error=str(error))
            raise
