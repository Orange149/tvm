from pathlib import Path
from cpu_vta_partitioning.common import read,write,sha256
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/finite_service_01'
CURRENT=ROOT/'artifacts/static_revision_01'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('finite-service frozen identity differs: '+str(path))
    else:write(path,value)
    return value


def protocol():
    return dict(experiment='finite_service_01',hypothesis='finite single-TID service and persistent virtual runtime can expose fork/join drag hidden by fluid service',
        source_basis='Linux v5.4 fair.c sched_slice, calc_delta_fair and place_entity; an approximation, not vendor CFS reproduction',
        calibration_cases=[dict(name='busy'+str(n),threads=n,CPU_budget_ns=80_000_000) for n in range(2,7)],
        heldout_case=dict(name='mixed',threads=3,work=[dict(busy_ns=80_000_000),
            dict(repeats=16,sleep_ns=1_000_000,busy_ns=1_000_000),
            dict(repeats=16,sleep_ns=1_000_000,busy_ns=30_000)]),
        max_platform_native_sessions=1,DNN_inferences=0,FPGA_access=False,
        main_core=1,workload_core=0,CPU_work_budget_ns=1_696_480_000,
        calibration_statistic='median of fully interior IN-to-preempt-OUT spans while all case threads are busy',
        calibration_sample_minimum_per_case=5,
        scheduler_approximation=dict(service='one actor per physical core; persistent equal-weight virtual runtime',
            sleeper_lag_cap='one half of n times calibrated granule',
            wake_preemption_threshold='one half calibrated granule',
            granule_kept_across_immediate_work_segments=True,
            unsupported_runnable_counts='reject, do not extrapolate',
            actual_vendor_CFS_reproduced=False),
        mixed_acceptance='all three per-thread wall-time errors <= max(20 percent observed wall, largest calibrated service granule)',
        no_kernel_granularity_claim=True,no_DNN_timing_fitted=True,no_DNN_profiles_modified=True,
        no_reboot=True,no_sysctl_change=True,no_automatic_repeat=True)


def sources():
    files=[ROOT/'run_finite_service.py',*sorted(Path(__file__).parent.glob('*.py')),
           ROOT/'tests/test_finite_service.py',ROOT/'scheduler_event_audit/switch_probe.cc',
           ROOT/'scheduler_event_audit/decode.py']
    return {str(p):sha256(p) for p in files}
