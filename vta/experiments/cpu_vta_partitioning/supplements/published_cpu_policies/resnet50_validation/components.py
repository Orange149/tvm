"""Fresh ResNet50 component evidence with durable, identity-bound acceptance."""
from pathlib import Path
import shlex
import time
from types import FunctionType

from cpu_vta_partitioning.common import inventory, locked, now, read, sha256, sync_tree, verify_inventory, write
from yolo_transfer import profile, traffic
from .common import EXPECTED, assert_controllers, frozen, timed_deploy


def profile_bundle(output):
    return Path(output) / 'profile_capsule'


def check_hardware(ssh, bundle, boot):
    # All values are checked, never set: this campaign does not reconfigure/reboot hardware.
    ssh.wait_idle(boot)
    raw = ssh.idle_command('sha256sum /lib/firmware/vta_hpc.bit\n'
        'cat /sys/class/fpga_manager/fpga0/state\ncat /sys/class/u-dma-buf/udmabuf0/size', boot=boot)
    lines = raw.splitlines()
    policy = read(Path(bundle) / 'plans/board_execution.json')
    if (len(lines) != 3 or lines[0].split()[0] != sha256(Path(bundle) / 'firmware/vta_hpc.bit')
            or lines[1] != 'operating' or int(lines[2]) != policy['udmabuf_bytes']):
        raise ValueError('hardware identity differs; no implicit FPGA reload or buffer change')
    clocks = ssh.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
    if clocks != ['1066666'] * 4:
        raise ValueError('CPU frequency differs; no automatic clock change')
    return dict(boot_id=boot, raw=raw, frequency_khz=list(map(int, clocks)))


def record(bundle, key, path):
    path, bundle = Path(path), Path(bundle)
    item = read(path)
    doc = profile.make_task(bundle, key)
    run = read(path.parents[2] / 'run.json')
    if (item['task'] != doc or item['boot_id'] != path.parent.name
            or run['boot_id'] != item['boot_id'] or run['bundle_id'] != doc['bundle_id']):
        raise ValueError('component task, boot, or capsule identity differs')
    verify_inventory(item['raw'], item['files'])
    raw = Path(item['raw'])
    from cpu_vta_partitioning.common import digest
    if ((raw / 'boot_id').read_text().strip() != item['boot_id']
            or (raw / 'task_hash').read_text().strip() != digest(doc)):
        raise ValueError('component raw identity differs')
    result = profile.summary(raw, doc, bundle)
    if result != item['summary']:
        raise ValueError('component summary no longer replays')
    name = doc['stages'][0]['program']
    return {**result, 'configuration': doc['configuration'], 'source': dict(
        receipt=str(path.resolve()), receipt_sha256=sha256(path), boot_id=item['boot_id'],
        task_id=doc['task_id'], bundle_id=doc['bundle_id'],
        program_files={n: sha256(bundle / 'programs' / name / n)
                       for n in ('module.so', 'graph.json', 'params.bin')})}


def replay(output, run_dir, *, require_complete=True):
    output, run_dir = Path(output), Path(run_dir)
    assert_controllers(output)
    bundle = profile_bundle(output)
    manifest = profile.validate(bundle)
    keys = read(bundle / 'plans/component_configurations.json')
    from .archive_audit import imported_records
    imported = imported_records(bundle, run_dir)
    if not set(imported) <= set(keys):
        raise ValueError('imported costs are outside the repaired component domain')
    marker = run_dir / 'run.json'
    if not marker.exists():
        if require_complete and set(imported) != set(keys):
            raise ValueError('no ResNet50 isolated component records')
        return imported
    identity = read(marker)
    if identity['bundle_id'] != manifest['bundle_id']:
        raise ValueError('component run belongs to a different capsule')
    records = dict(imported)
    for key in keys:
        doc = profile.make_task(bundle, key)
        path = run_dir / 'accepted' / identity['boot_id'] / (doc['task_id'] + '.json')
        if key in imported:
            if path.exists():
                raise ValueError('imported component was also measured; do not double count receipts')
        elif path.exists():
            records[key] = record(bundle, key, path)
    if require_complete and set(records) != set(keys):
        raise ValueError('all ResNet50 isolated components must be accepted before ranking')
    return records


def complete(output, run_dir, local_config, *, execute_board=False, limit=None, expected_boot=None):
    if not execute_board:
        raise ValueError('--execute-board is required')
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    if (output / 'COMPILER_REGRESSION_STOP.json').exists():
        raise ValueError('archived compiler-regression batch cannot execute; use the repaired batch')
    assert_controllers(output)
    bundle = profile_bundle(output)
    manifest = profile.validate(bundle)
    order = read(bundle / 'plans/profile_order.json')
    if len(order) > EXPECTED['components']:
        raise ValueError('component budget expanded; do not automatically start a larger campaign')
    from .archive_audit import imported_records
    imported = imported_records(bundle, run_dir)
    if not set(imported) <= set(order):
        raise ValueError('imported costs are outside the repaired component domain')
    ssh = profile.matched_ssh(read(local_config)['board'], execute_board=True)
    with locked(run_dir):
        boot = ssh.boot_id()
        if expected_boot is not None and boot != expected_boot:
            raise ValueError('boot changed; never merge cohorts')
        frozen(run_dir / 'run.json', dict(boot_id=boot, bundle_id=manifest['bundle_id']))
        remote = timed_deploy(ssh, bundle, run_dir)
        write(run_dir / 'hardware.json', check_hardware(ssh, bundle, boot), replace=True)
        started = time.monotonic()
        records, newly = {}, 0
        for key in order:
            doc = profile.make_task(bundle, key)
            path = run_dir / 'accepted' / boot / (doc['task_id'] + '.json')
            if key in imported:
                if path.exists():
                    raise ValueError('imported component was also measured; do not double count receipts')
                result = imported[key]
            elif path.exists():
                result = record(bundle, key, path)
            else:
                if limit is not None and len(records) >= limit:
                    continue
                begin = time.monotonic()
                raw = ssh.measure(doc, bundle, remote, run_dir, boot)
                result = profile.summary(raw, doc, bundle)
                write(path, dict(task=doc, boot_id=boot, raw=str(raw), files=inventory(raw), summary=result,
                                 accepted_at=now(), measurement_and_collection_seconds=time.monotonic() - begin))
                # Reconstruct and validate before acknowledging remote deletion.
                result = record(bundle, key, path)
                ssh.acknowledge(doc, run_dir, boot)
                newly += 1
            records[key] = result
            table = dict(status='COMPLETE' if len(records) == len(order) else 'PARTIAL',
                         accepted=len(records), required=len(order), costs=records,
                         pipeline_FPS_used=False,
                         imported_components=sum(key in imported for key in records),
                         source_boot_ids=sorted({value['source']['boot_id'] for value in records.values()}))
            write(output / 'isolated_costs.json', table, replace=True)
            write(run_dir / 'state.json', dict(status=table['status'], accepted=len(records), required=len(order),
                boot_id=boot, updated_at=now(), newly_measured=newly), replace=True)
            print('ResNet50 components', len(records), '/', len(order), doc['stages'][0]['signature'], flush=True)
        result = dict(status='COMPLETE' if len(records) == len(order) else 'PARTIAL', boot_id=boot,
                      accepted=len(records), required=len(order), newly_measured=newly,
                      imported_components=sum(key in imported for key in records),
                      elapsed_seconds=time.monotonic() - started)
        write(run_dir / 'state.json', result, replace=True)
        ssh.idle_command('cd ' + shlex.quote(remote) + ' && sha256sum -c board.sha256 > post_run_verification.log && sync',
                         boot=boot, timeout=600)
        sync_tree(run_dir)
        return result


def measure_traffic(output, run_dir, local_config, *, execute_board=False, limit=None, expected_boot=None):
    if not execute_board:
        raise ValueError('--execute-board is required')
    from cpu_vta_partitioning.transport import SSH
    output, run_dir = Path(output), Path(run_dir)
    if (output / 'COMPILER_REGRESSION_STOP.json').exists():
        raise ValueError('archived compiler-regression batch cannot execute DMA; use the repaired batch')
    assert_controllers(output)
    bundle = profile_bundle(output)
    manifest = profile.validate(bundle)
    measure = FunctionType(SSH.measure.__code__, {**SSH.measure.__globals__, 'native_command': traffic.native_command})
    ssh = type('ResNet50TrafficSSH', (SSH,), {'measure': measure})(read(local_config)['board'], execute_board=True)
    with locked(run_dir):
        boot = ssh.boot_id()
        if expected_boot is not None and boot != expected_boot:
            raise ValueError('boot changed before DMA collection')
        frozen(run_dir / 'run.json', dict(boot_id=boot, bundle_id=manifest['bundle_id'],
            controller_sha256=sha256(Path(traffic.__file__))))
        remote = timed_deploy(ssh, bundle, run_dir)
        check_hardware(ssh, bundle, boot)
        cfgs = read(bundle / 'plans/component_configurations.json')
        keys = [key for key, cfg in cfgs.items()
                if cfg['stages'][0]['device'] == 'vta' and cfg['context_id'] == 'shared4']
        pilot = read(bundle / 'plans/pilot.json')['vta']
        keys.sort(key=lambda key: (cfgs[key]['stages'][0]['program'] not in pilot,
                                  pilot.index(cfgs[key]['stages'][0]['program'])
                                  if cfgs[key]['stages'][0]['program'] in pilot else 0,
                                  cfgs[key]['stages'][0]['program']))
        if len(keys) > EXPECTED['dma']:
            raise ValueError('DMA signature budget expanded')
        records, newly = {}, 0
        for key in keys:
            doc = traffic.task(bundle, key)
            path = run_dir / 'accepted' / boot / (doc['task_id'] + '.json')
            if path.exists():
                item = read(path)
                if item['task'] != doc or item['boot_id'] != boot:
                    raise ValueError('DMA receipt task/boot differs')
                verify_inventory(item['raw'], item['files'])
                raw = Path(item['raw'])
                from cpu_vta_partitioning.common import digest
                if ((raw / 'boot_id').read_text().strip() != boot
                        or (raw / 'task_hash').read_text().strip() != digest(doc)):
                    raise ValueError('DMA raw task or boot differs')
                result = traffic.summary(item['raw'], doc, bundle)
                if result != item['summary']:
                    raise ValueError('DMA receipt does not replay')
            else:
                if limit is not None and len(records) >= limit:
                    continue
                began = time.monotonic()
                raw = ssh.measure(doc, bundle, remote, run_dir, boot)
                from cpu_vta_partitioning.common import digest
                if ((Path(raw) / 'boot_id').read_text().strip() != boot
                        or (Path(raw) / 'task_hash').read_text().strip() != digest(doc)):
                    raise ValueError('DMA raw task or boot differs before acceptance')
                result = traffic.summary(raw, doc, bundle)
                write(path, dict(task=doc, boot_id=boot, raw=str(raw), files=inventory(raw), summary=result,
                    measurement_and_collection_seconds=time.monotonic() - began))
                ssh.acknowledge(doc, run_dir, boot)
                newly += 1
            records[doc['stages'][0]['program']] = {**result, 'source': dict(
                boot_id=boot, task_id=doc['task_id'], receipt_sha256=sha256(path))}
            result = dict(status='COMPLETE' if len(records) == len(keys) else 'PARTIAL',
                          accepted=len(records), required=len(keys), boot_id=boot,
                          pipeline_FPS_used=False, records=records)
            write(run_dir / 'traffic.json', result, replace=True)
            print('ResNet50 DMA', len(records), '/', len(keys), doc['stages'][0]['program'], flush=True)
        sync_tree(run_dir)
        return {k: v for k, v in result.items() if k != 'records'}


def pilot_cost(output, run_dir):
    """Planning timing only; these projections never become component costs."""
    import math
    import statistics
    output, run_dir = Path(output), Path(run_dir)
    bundle = profile_bundle(output)
    records = replay(output, run_dir / 'components', require_complete=False)
    order = read(bundle / 'plans/profile_order.json')[:39]
    if len(order) != 39:
        raise ValueError('the structural pilot must retain exactly thirty-nine component signatures')
    groups = {'cpu': dict(values=[], sources=[]), 'vta': dict(values=[], sources=[])}
    for key in order:
        doc = profile.make_task(bundle, key)
        if key not in records:
            raise ValueError('pilot is incomplete; a new or imported signature is missing: ' + key)
        source = records[key]['source']
        path = Path(source['receipt'])
        if sha256(path) != source['receipt_sha256']:
            raise ValueError('pilot source receipt identity differs')
        item = read(path)
        if (item['task']['task_id'] != source['task_id'] or item['boot_id'] != source['boot_id']
                or item['task']['component_key'] != key):
            raise ValueError('pilot source task or boot differs')
        value = item.get('measurement_and_collection_seconds')
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError('invalid measured pilot duration')
        group = groups[doc['stages'][0]['device']]
        imported = bool(source.get('archived_output')) or item['task']['bundle_id'] != doc['bundle_id']
        group['sources'].append(dict(component_key=key, receipt=str(path), receipt_sha256=sha256(path),
            task_id=source['task_id'], boot_id=source['boot_id'], imported=imported,
            measurement_and_collection_seconds=value))
        if value is not None:
            group['values'].append(value)
    if len(groups['cpu']['sources']) != 24 or len(groups['vta']['sources']) != 15:
        raise ValueError('structural pilot must retain twenty-four CPU and fifteen VTA signatures')
    rows = {}
    counts = {'cpu': 1256, 'vta': 1495}
    for device, group in groups.items():
        values, sources = group['values'], group['sources']
        median, maximum = (statistics.median(values), max(values)) if values else (None, None)
        rows[device] = dict(observed_sessions=len(sources), timed_sessions=len(values),
            unknown_duration_sessions=len(sources) - len(values),
            imported_sessions=sum(source['imported'] for source in sources),
            new_sessions=sum(not source['imported'] for source in sources),
            measured_seconds=sum(values) if len(values) == len(sources) else None,
            known_timed_seconds=sum(values) if values else None,
            session_min_seconds=min(values) if values else None, session_max_seconds=maximum,
            session_median_seconds=median,
            total_median_projection_seconds=median * counts[device] if median is not None else None,
            total_max_projection_seconds=maximum * counts[device] if maximum is not None else None,
            sources=sources)
    result = dict(scope='pilot timing projection, not actual full calibration cost or performance prediction',
                  expected_components=2751, component_types=rows, pilot_is_part_of_full_table=True,
                  pilot_sessions=len(order), unknown_durations_excluded_from_projection=True,
                  extra_stage_calls=0, component_costs_modified=False)
    write(output / 'PILOT_COST.json', result, replace=True)
    return result
