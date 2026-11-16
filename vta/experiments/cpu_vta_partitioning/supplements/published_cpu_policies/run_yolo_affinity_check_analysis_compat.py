#!/usr/bin/env python3
"""Analyze frozen receipts collected under transfers/<generation>/collected.

The frozen analyzer compared the returned collected directory to the parent
task directory.  Correct that read-only path check, without changing any
measurement source, task, configuration, executable, or raw evidence.
"""
from hashlib import sha256
import inspect
from pathlib import Path


def raw_scope_valid(raw, expected):
    raw, expected = Path(raw).resolve(), Path(expected).resolve()
    if not raw.is_relative_to(expected):
        return False
    parts = raw.relative_to(expected).parts
    return len(parts) == 3 and parts[0] == 'transfers' and parts[1].isdigit() and parts[2] == 'collected'


def install():
    from yolo_affinity_check import analysis
    original = inspect.getsource(analysis.analyze)
    before = "if raw.resolve() != (run_dir/'raw'/identity['boot_id']/doc['task_id']).resolve():"
    after = "if not raw_scope_valid(raw, run_dir/'raw'/identity['boot_id']/doc['task_id']):"
    if original.count(before) != 1:
        raise ValueError('frozen analysis path anchor changed')
    thread_before = "threads.append(dict(task_id=doc['task_id'], **spec, **item))"
    thread_after = "threads.append(dict(task_id=doc['task_id'], **spec, **{('clock_scope' if k == 'scope' else k): v for k, v in item.items()}))"
    if original.count(thread_before) != 1:
        raise ValueError('frozen analysis clock-scope anchor changed')
    source = original.replace(before, after).replace(thread_before, thread_after)
    proof = dict(original_analyze_sha256=sha256(original.encode()).hexdigest(),
        adapted_analyze_sha256=sha256(source.encode()).hexdigest(),
        exact_change=dict(before=before, after=after),
        clock_scope_change=dict(before=thread_before, after=thread_after), revision=2,
        controller_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_scope='raw/<boot>/<task>/transfers/<numeric_generation>/collected',
        original_frozen_sources_unchanged=True, native_executions_added=0,
        scores_windows_configurations_programs_and_raw_unchanged=True)
    namespace = dict(analysis.analyze.__globals__, raw_scope_valid=raw_scope_valid)
    exec(compile(source, '<affinity collected-directory read-only compatibility>', 'exec'), namespace)
    adapted = namespace['analyze']
    def analyze(output, run_dir):
        from cpu_vta_partitioning.common import write, sha256 as file_sha256
        from yolo_affinity_check.common import frozen
        frozen(Path(run_dir) / 'ANALYSIS_COMPATIBILITY.json', proof)
        snapshot = Path(run_dir) / 'sources' / 'analysis_compatibility_02' / Path(__file__).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and file_sha256(snapshot) != file_sha256(Path(__file__)):
            raise ValueError('analysis compatibility snapshot changed')
        if not snapshot.exists():
            snapshot.write_bytes(Path(__file__).read_bytes())
        return adapted(output, run_dir)
    analysis.analyze = analyze
    return proof


if __name__ == '__main__':
    from run_yolo_affinity_check import main
    install()
    main()
