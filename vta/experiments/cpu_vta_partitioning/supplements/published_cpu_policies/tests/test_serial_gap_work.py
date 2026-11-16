from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_serial_gap_work import disjoint_queries, query_occupancy, serial_call_proof


def launch(a, b, c, d, baselines=None):
    return dict(queue_read_start0_ns=a, queue_read_end0_ns=b,
                queue_read_start1_ns=c, queue_read_end1_ns=d,
                worker_queue_baselines=baselines or [])


def test_queries_do_not_turn_sleep_into_cpu_or_double_count_baselines():
    q = disjoint_queries([launch(10, 20, 40, 60)])
    stream = dict(on=[dict(start_ns=0, end_ns=15), dict(start_ns=50, end_ns=100)],
                  off=[dict(start_ns=15, end_ns=50, preempt=False)])
    assert query_occupancy(stream, 0, 100, q) == dict(query_wall_ns=30, query_on_CPU_ns=15)
    with pytest.raises(ValueError, match='overlap'):
        disjoint_queries([launch(10, 30, 40, 60), launch(20, 35, 70, 80)])


def test_binary_absence_of_symbol_is_insufficient_to_prove_serial_execution():
    asm = '''
  1000: 90000148 adrp x8, 2e000
  1004: f947f108 ldr x8, [x8, #4064]
  1008: f9400108 ldr x8, [x8]
  100c: d63f0100 blr x8
  1010: b0000108 adrp x8, 2e000
  1014: f947dd08 ldr x8, [x8, #4024]
  1018: f9400108 ldr x8, [x8]
  101c: d63f0100 blr x8
'''
    reloc = '''000000000002efe0  a R_AARCH64_GLOB_DAT 0 __TVMBackendAllocWorkspace + 0
000000000002efb8  a R_AARCH64_GLOB_DAT 0 __TVMBackendFreeWorkspace + 0'''
    assert [r['target'] for r in serial_call_proof(asm, reloc)] == [
        '__TVMBackendAllocWorkspace', '__TVMBackendFreeWorkspace']
    with pytest.raises(ValueError, match='relocation'):
        serial_call_proof(asm, reloc.replace('__TVMBackendFreeWorkspace', '__TVMBackendParallelLaunch'))
    with pytest.raises(ValueError, match='call structure'):
        serial_call_proof(asm + '\n  1020: 94000000 bl 2000', reloc)
