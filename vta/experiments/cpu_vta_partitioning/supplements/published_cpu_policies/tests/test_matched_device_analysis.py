from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_matched_device_profile import frame_parts, compare_parts


def sample():
    document = {'stages':[{'program':'vta_a','device':'vta'}]}
    rows = [dict(stage0_set_ms=2,stage0_get_ms=3,stage0_run_ms=10,stage0_ms=15)]
    projection = dict(per_frame_projection_supported=True,stages={'vta_a':{'frames':[
        dict(frame=0,run_wall_ns=10_000_000,driver_wall_ns=6_000_000,
             outside_driver_run_wall_ns=4_000_000,driver_CPU_ns=1_000_000,
             completion_lower_sum_ns=5_000_000,completion_upper_sum_ns=5_500_000,
             read_clock_envelope_ns=20_000)]}})
    return document, rows, projection


def test_decomposition_rejects_overlapping_or_ambiguous_attribution():
    document, rows, projection = sample()
    result = frame_parts(document, rows, projection)['vta_a']['means_ms']
    assert result['boundary_ms']+result['driver_wall_ms']+result['outside_driver_ms'] == result['service_ms']
    invalid = deepcopy(projection)
    invalid['stages']['vta_a']['frames'][0]['driver_wall_ns'] += 100
    with pytest.raises(ValueError,match='pieces overlap'): frame_parts(document, rows, invalid)
    with pytest.raises(ValueError,match='association'): frame_parts(document, rows, {'per_frame_projection_supported':False})
    rows[0]['stage0_ms'] = 14
    with pytest.raises(ValueError,match='conserve'): frame_parts(document, rows, projection)


def test_driver_and_host_changes_conserve_without_equating_CPU_and_wall():
    a = frame_parts(*sample())
    b = deepcopy(a)
    m = b['vta_a']['means_ms']
    m['outside_driver_ms'] += 7
    m['driver_wall_ms'] -= 1
    m['service_ms'] += 6
    # CPU time is independently observed; it is not the wall-time residual.
    m['driver_CPU_ms'] -= .5
    rows, total = compare_parts(a, b)
    assert total['changes_ms']['service_ms'] == 6
    assert total['changes_ms']['outside_driver_ms'] == 7
    assert total['changes_ms']['driver_CPU_ms'] == -.5
    assert rows[0]['outside_driver_not_identified_as_all_CPU_or_scheduler_wait']


def test_completion_brackets_not_fit_to_a_point_or_called_confidence_intervals():
    a = frame_parts(*sample())
    b = deepcopy(a)
    b['vta_a']['means_ms']['completion_lower_ms'] = 5.1
    b['vta_a']['means_ms']['completion_upper_ms'] = 5.7
    rows, _ = compare_parts(a,b)
    lo, hi = rows[0]['completion_difference_interval_ms']
    assert lo == pytest.approx(-.4) and hi == pytest.approx(.7)
    assert rows[0]['completion_intervals_are_not_statistical_CI']
    with pytest.raises(ValueError,match='Different|different'): compare_parts(a,{'other':b['vta_a']})


def test_accepted_native_records_reconstruct_with_lossless_JSON_arrays():
    import run_matched_device_profile as native
    from cpu_vta_partitioning.common import read, digest, verify_inventory
    paths = sorted((native.RUN/'accepted').glob('*/*.json'))
    if not paths:
        pytest.skip('native records are not part of synthetic CI fixtures')
    native.validate()
    assert len(paths) == 11
    for p in paths:
        receipt = read(p)
        verify_inventory(receipt['raw'], receipt['files'])
        rebuilt = native.summarize(receipt['raw'], receipt['task'], native.BUNDLE)
        assert digest(rebuilt) == digest(receipt['summary'])
