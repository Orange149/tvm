from validate_vta_p7r_command_resource_plan import align_up, capacity_for_roles, peak_by_role


def _row(role, insn, uop):
    return {
        "candidate_role": role,
        "outcome": "passed_local_signature",
        "command_signature": {
            "structural": {"peaks": {"insn_bytes": insn, "uop_bytes": uop}}
        },
    }


def test_page_aligned_capacity_for_allowed_identity_set():
    peaks = peak_by_role([_row("tophub", 3744, 1480), _row("hybrid", 3808, 1424)])
    assert capacity_for_roles(peaks, ("tophub", "hybrid")) == {
        "insn_capacity_bytes": 4096,
        "uop_capacity_bytes": 4096,
    }


def test_alignment_moves_exact_page_only_when_needed():
    assert align_up(4096) == 4096
    assert align_up(4097) == 8192
