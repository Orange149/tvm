#!/usr/bin/env python3
"""Read-only progress and resource monitoring, outside sealed experiment sources.

This process never launches measurements, changes parameters, deletes files,
restarts a controller or sends messages. Board probes are optional, lightweight,
and deferred while its session lock exists or formal pipeline timing is active.
Reports and alerts are local files, not automatic chat notifications.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
BATCH = 'resnet50_v2_validation_native_history_02'
POLICY = dict(progress_interval_seconds=60, resource_interval_seconds=600,
              busy_retry_seconds=60, stalled_warning_seconds=1200,
              available_memory_warning_bytes=256 * 1024**2,
              storage_warning_bytes=512 * 1024**2,
              changes_experiment=False, automatic_retries=False,
              automatic_reboot=False, chat_notifications=False)


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.monitor-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()


def stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


@contextmanager
def monitor_lock(path):
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('a health monitor already owns this batch') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_info(pid):
    path = Path('/proc') / str(pid)
    try:
        raw = (path / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        argv = [x.decode(errors='replace') for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
        return dict(pid=int(pid), state=fields[0], start_ticks=fields[19], argv=argv)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def actors(output, continuation_file):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        item = process_info(path.name)
        if not item or len(item['argv']) < 2:
            continue
        argv = item['argv']
        if not any(Path(x).name in ('run_resnet50_validation.py', 'run_resnet50_top20_board.py') for x in argv[:3]):
            continue
        if '--output' in argv and Path(argv[argv.index('--output') + 1]).resolve() != Path(output).resolve():
            continue
        found.append(item)
    marker = load(continuation_file)
    continuation = process_info(marker['pid']) if marker else None
    if continuation and continuation['start_ticks'] != str(marker.get('start_ticks', marker.get('process_start_ticks'))):
        continuation = None
    return dict(controllers=found, continuation=continuation)


def effective_phase(campaign, continuation):
    stage = campaign.get('status', 'UNKNOWN')
    handoff = (continuation or {}).get('phase')
    if stage == 'STOPPED_ON_ERROR' and campaign.get('error_type') == 'ValueError' and campaign.get('error') == 'no checkpoint to resume':
        return 'HANDOFF_' + (handoff or 'MISSING')
    return stage


def progress_key(docs, phase, checkpoint_mtime):
    return [phase, (docs['components'] or {}).get('accepted'),
            (docs['DMA'] or {}).get('accepted'), (docs['pipeline'] or {}).get('done_candidates'),
            (docs['pipeline'] or {}).get('accepted_sessions'), checkpoint_mtime]


def probe_script(config, boot):
    lock = config['remote_base'].rstrip('/') + '/locks/' + boot + '/session.lock'
    mount = config['storage_mount']
    q = shlex.quote
    # Do not acquire the native-session lock: a monitoring process must never
    # make a scheduled measurement lose its own lock. A race discards metrics.
    return '\n'.join([
        'boot=$(cat /proc/sys/kernel/random/boot_id)',
        'printf "BOOT %s\\n" "$boot"',
        'if test "$boot" != ' + q(boot) + '; then echo STATE BOOT_CHANGED; exit 0; fi',
        'if test -d ' + q(lock) + '; then echo STATE BUSY; exit 0; fi',
        'echo STATE IDLE_SNAPSHOT',
        "awk '/^(MemTotal|MemAvailable|CmaTotal|CmaFree):/ {gsub(\":\",\"\",$1); print \"MEM\",$1,$2}' /proc/meminfo",
        'df -Pk ' + q(mount) + " | awk 'NR > 1 {print \"DISK_FREE_KB\",$4}'",
        "awk '{print \"DMA_BYTES\",$1}' /sys/class/u-dma-buf/udmabuf0/size",
        'if test -d ' + q(lock) + '; then echo STATE RACED; fi',
    ])


def parse_probe(raw, expected_boot):
    result, memory = {}, {}
    for line in raw.splitlines():
        fields = line.split()
        if fields[:1] == ['BOOT'] and len(fields) == 2:
            result['boot_id'] = fields[1]
        elif fields[:1] == ['STATE'] and len(fields) == 2:
            result['state'] = fields[1]
        elif fields[:1] == ['MEM'] and len(fields) == 3:
            memory[fields[1]] = int(fields[2]) * 1024
        elif fields[:1] == ['DISK_FREE_KB'] and len(fields) == 2:
            result['storage_available_bytes'] = int(fields[1]) * 1024
        elif fields[:1] == ['DMA_BYTES'] and len(fields) == 2:
            result['DMA_buffer_bytes'] = int(fields[1])
    if result.get('boot_id') != expected_boot:
        return dict(state='BOOT_CHANGED', boot_id=result.get('boot_id'))
    if result.get('state') in ('BUSY', 'RACED'):
        return dict(state=result['state'], boot_id=expected_boot)
    if (result.get('state') != 'IDLE_SNAPSHOT' or 'MemAvailable' not in memory
            or 'MemTotal' not in memory or 'storage_available_bytes' not in result
            or 'DMA_buffer_bytes' not in result or any(x < 0 for x in memory.values())
            or result['storage_available_bytes'] < 0 or result['DMA_buffer_bytes'] <= 0):
        raise ValueError('incomplete or invalid resource snapshot; missing data are not zero')
    return dict(**result, memory_bytes=memory, measured_peak=False,
                scope='idle-lock-checked lightweight snapshot, not a peak-memory measurement')


def board_probe(config, boot):
    sys.path.insert(0, str(ROOT.parents[1] / 'src'))
    from cpu_vta_partitioning.transport import SSH
    return parse_probe(SSH(config, execute_board=True).command(probe_script(config, boot), timeout=15), boot)


def warnings(docs, phase, live, last_progress, now, board, expected_boot):
    issues = []
    add = lambda code, message: issues.append(dict(code=code, message=message))
    campaign = docs['campaign']
    if campaign.get('boot_id') != expected_boot:
        add('BOOT_CHANGED', '控制器的开发板启动身份已变化。')
    if phase == 'SEARCH_INCOMPLETE':
        add('SEARCH_INCOMPLETE', 'DP在约定预算内未完成；未授权扩大搜索预算。')
    if phase == 'STOPPED_ON_ERROR':
        add('CONTROLLER_ERROR', '实验控制器已停止：' + str(campaign.get('error')))
    handoff = docs['continuation'] or {}
    # The helper handles first-rank initialization. Once ranking is frozen, an
    # explicitly resumed board controller can finish without that helper. Its
    # old failure remains in history; current controller failures still alert.
    needs_first_rank_handoff = (docs.get('search') or {}).get('status') != 'COMPLETE'
    if handoff.get('phase') == 'STOPPED' and needs_first_rank_handoff:
        add('CONTINUATION_ERROR', '自动衔接已停止：' + str(handoff.get('error')))
    if phase != 'COMPLETE' and not live['controllers'] and now - last_progress > 120:
        add('NO_CONTROLLER', '未检测到实验或排名进程，且超过两分钟没有进展。')
    if (phase != 'COMPLETE' and needs_first_rank_handoff and not live['continuation']
            and handoff.get('phase') not in ('COMPLETE', 'NOT_REQUIRED')):
        add('NO_CONTINUATION', '自动衔接进程已退出。')
    if (phase in ('COMPONENT_PILOT', 'COMPONENTS', 'DMA', 'PIPELINE')
            and now - last_progress >= POLICY['stalled_warning_seconds']):
        add('NO_RECENT_PROGRESS', '二十分钟未见验收数量推进；可能正在文件校验，需结合日志判断。')
    if board:
        if board.get('state') == 'BOOT_CHANGED':
            add('BOARD_BOOT_CHANGED', '开发板实际启动身份已变化。')
        if board.get('state') == 'IDLE_SNAPSHOT':
            if board['memory_bytes']['MemAvailable'] < POLICY['available_memory_warning_bytes']:
                add('LOW_MEMORY', '开发板可用内存不足256 MiB。')
            if board['storage_available_bytes'] < POLICY['storage_warning_bytes']:
                add('LOW_STORAGE', '实验分区剩余空间不足512 MiB；实际部署另有容量预检。')
            if board['DMA_buffer_bytes'] != 192 * 1024**2:
                add('DMA_CHANGED', 'DMA缓冲池大小与冻结的192 MiB不一致。')
    return issues


class Monitor:
    def __init__(self, output, run, destination, *, execute_board=False, probe=board_probe, find_actors=actors):
        self.output, self.run, self.destination = map(Path, (output, run, destination))
        self.destination.mkdir(parents=True, exist_ok=True)
        self.identity = load(self.run / 'run.json')
        self.boot = self.identity['boot_id']
        self.execute_board, self.probe, self.find_actors = execute_board, probe, find_actors
        self.config = load(ROOT.parents[1] / 'configs/local.json')['board']

    def tick(self, now=None):
        now = time.time() if now is None else now
        documents = dict(campaign=load(self.run / 'state.json'),
            components=load(self.run / 'components/state.json'), DMA=load(self.run / 'traffic/traffic.json'),
            pipeline=load(self.run / 'pipeline/state.json'), search=load(self.output / 'ranking/SEARCH_STATUS.json'),
            continuation=load(self.output / 'operations/continuation_state.json'))
        if documents['campaign'] is None:
            raise ValueError('campaign state is missing')
        phase = effective_phase(documents['campaign'], documents['continuation'])
        live = self.find_actors(self.output, self.output / 'operations/observer_process.json')
        checkpoint = self.output / 'ranking/search/search_checkpoint.jsonl.gz'
        key = progress_key(documents, phase, checkpoint.stat().st_mtime_ns if checkpoint.exists() else None)
        state = load(self.destination / 'state.json') or {}
        if state and state.get('boot_id') != self.boot:
            raise ValueError('monitor directory belongs to another boot')
        changed = state.get('progress_key') != key
        last = now if changed else state['last_progress_unix']
        board = state.get('last_board_snapshot')
        next_probe = state.get('next_board_probe_unix', 0)
        probe_status = state.get('last_probe_status', 'NOT_REQUESTED')
        probe_error = state.get('last_probe_error')
        # All formal full-pipeline timing is deliberately free of new SSH probes.
        if self.execute_board and now >= next_probe and phase != 'PIPELINE':
            try:
                sample = self.probe(self.config, self.boot)
                probe_error = None
                probe_status = sample['state']
                if sample['state'] in ('IDLE_SNAPSHOT', 'BOOT_CHANGED'):
                    board = dict(**sample, observed_at=stamp(now))
                next_probe = now + (POLICY['busy_retry_seconds'] if sample['state'] in ('BUSY', 'RACED')
                                    else POLICY['resource_interval_seconds'])
            except Exception as error:
                probe_status, probe_error = 'READ_FAILED', str(error)
                next_probe = now + POLICY['resource_interval_seconds']
        elif self.execute_board and phase == 'PIPELINE':
            probe_status = 'DEFERRED_FORMAL_PIPELINE'
        issues = warnings(documents, phase, live, last, now, board, self.boot)
        if probe_error:
            issues.append(dict(code='RESOURCE_READ_FAILED', message='最近一次只读资源检查失败，尚未成功复核：' + probe_error))
        result = dict(checked_at=stamp(now), boot_id=self.boot, phase=phase,
            health='ATTENTION' if issues else ('COMPLETE' if phase == 'COMPLETE' else 'RUNNING'),
            progress={name: {k: v for k, v in (documents[name] or {}).items()
                if k in ('status', 'accepted', 'required', 'done_candidates', 'candidate_count',
                         'accepted_sessions', 'updated_at', 'elapsed_seconds')}
                for name in ('components', 'DMA', 'search', 'pipeline')},
            last_progress_at=stamp(last), seconds_since_progress=max(0, now - last),
            actors=live, resource_probe_status=probe_status,
            first_rank_handoff_required=(documents['search'] or {}).get('status') != 'COMPLETE',
            prior_handoff_phase=(documents['continuation'] or {}).get('phase'),
            last_board_snapshot=board, issues=issues, policy=POLICY,
            monitoring_scope='status only; no new inference, experiment changes or chat delivery')
        active = {issue['code']: issue['message'] for issue in issues}
        previous = state.get('active_alerts', {})
        for code, message in active.items():
            if previous.get(code) != message:
                append(self.destination / 'alerts.jsonl', dict(at=stamp(now), event='OPEN', code=code, message=message))
        for code in previous.keys() - active.keys():
            append(self.destination / 'alerts.jsonl', dict(at=stamp(now), event='RESOLVED', code=code))
        save(self.destination / 'state.json', dict(boot_id=self.boot, progress_key=key,
            last_progress_unix=last, next_board_probe_unix=next_probe,
            last_board_snapshot=board, last_probe_status=probe_status,
            last_probe_error=probe_error, active_alerts=active))
        save(self.destination / 'latest.json', result)
        append(self.destination / 'history.jsonl', result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--batch', default=BATCH, help='separate execution-cohort directory name')
    parser.add_argument('--execute-board', action='store_true', help='allow read-only resource checks; zero inference')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.batch) or args.batch in ('.', '..'):
        parser.error('--batch must be one safe directory name')
    output, run = ROOT / 'artifacts' / args.batch, ROOT / 'runs' / args.batch
    destination = output / 'operations/health'
    monitor = Monitor(output, run, destination, execute_board=args.execute_board)
    with monitor_lock(destination / 'monitor.lock'):
        save(destination / 'process.json', dict(pid=os.getpid(), start_ticks=process_info(os.getpid())['start_ticks'],
            started_at=stamp(time.time()), script=str(Path(__file__).resolve()),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            read_only_board_probes=args.execute_board, policy=POLICY))
        while True:
            began = time.monotonic()
            try:
                result = monitor.tick()
            except Exception as error:
                append(destination / 'alerts.jsonl', dict(at=stamp(time.time()), event='MONITOR_ERROR',
                    code=type(error).__name__, message=str(error)))
                print('status read failed: ' + str(error), flush=True)
                if args.once:
                    return 1
                time.sleep(max(1, POLICY['progress_interval_seconds'] - (time.monotonic() - began)))
                continue
            print(json.dumps({k: result[k] for k in ('checked_at', 'phase', 'health', 'progress', 'issues')},
                             ensure_ascii=False), flush=True)
            if args.once or result['phase'] == 'COMPLETE' or any(
                    x['code'] in ('BOOT_CHANGED', 'BOARD_BOOT_CHANGED') for x in result['issues']):
                return 0 if not result['issues'] else 2
            time.sleep(max(1, POLICY['progress_interval_seconds'] - (time.monotonic() - began)))


if __name__ == '__main__':
    raise SystemExit(main())
