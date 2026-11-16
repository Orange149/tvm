from audit_isolated_trace_inventory import capabilities, match


def test_old_host_split_does_not_imply_completion_brackets():
    profile = {'stages': {'devices': [{'CPU': 1, 'wall': 2}], 'host_serial_CPU': [3, 4]}}
    result = capabilities(profile, 'vta')
    assert result['host_cpu_split'] and not result['device_completion_brackets']
    profile['stages']['devices'][0].update(completion_lower_ns=1, completion_upper_ns=2)
    assert capabilities(profile, 'vta')['device_completion_brackets']


def test_core_permutation_is_not_an_exact_measurement():
    row = dict(network='yolo', program='cpu_02_04', module_sha256='abc', threads=3,
               mask=[0, 1, 2], worker_core_map=[0, 1, 2])
    assert match(row, row) == 'matching_placement'
    changed = dict(row, mask=[1, 2, 3], worker_core_map=[1, 2, 3])
    assert match(row, changed) == 'structure_only_requires_declared_core_permutation'
    assert match(row, dict(changed, module_sha256='other')) == 'different_module_or_thread_signature'


def test_host_split_requires_both_ends():
    assert not capabilities({'stages': {'devices': [dict()], 'host_serial_CPU': [1]}}, 'vta')['host_cpu_split']
