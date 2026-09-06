import re
import json

import numpy as np

from calibrate_vta_dma_native import analyze, cases, lower_kernel, protocol_hash, shuffled_cases


def test_dma_identification_cases_cover_access_patterns_without_duplicates():
    matrix = cases(batch=1, block_out=16)
    assert len(matrix) == 36
    assert len({case["case_id"] for case in matrix}) == len(matrix)
    assert {case["access_kind"] for case in matrix} == {"contiguous", "strided", "padded"}
    assert {case["measurement_kind"] for case in matrix} == {
        "dma_component_identification", "matched_access_pair", "vta_compute_slope"
    }
    assert all(case["includes"] and case["excludes"] for case in matrix)
    assert not any(case["eligible_for_direct_physical_model"] for case in matrix)


def test_contiguous_identification_design_is_full_rank():
    rows = []
    for case in cases(batch=1, block_out=16):
        if case["family"] != "contiguous_load_store":
            continue
        elements = case["elements"]
        rows.append(
            [
                4 * elements * case["input_count"],
                elements * case["output_count"],
                case["input_count"],
                case["output_count"],
                1,
            ]
        )
    design = np.asarray(rows, dtype="float64")
    scales = np.maximum(np.max(np.abs(design), axis=0), 1.0)
    assert np.linalg.matrix_rank(design / scales) == 5


def test_single_and_double_input_cases_use_the_same_add_opcode():
    base = [case for case in cases(1, 16) if case["family"] == "contiguous_load_store"]
    assert {case["alu_opcode"] for case in base} == {"add"}
    assert {case["alu_repeats"] for case in base} == {1}
    assert {case["input_count"] for case in base} == {1, 2}


def test_load_and_store_sweeps_change_only_the_target_count():
    base = [case for case in cases(1, 16) if case["family"] == "contiguous_load_store"]
    for y, x in {(case["output_y"], case["output_x"]) for case in base}:
        for output_count in (1, 2):
            load_pair = [
                case for case in base
                if (case["output_y"], case["output_x"]) == (y, x)
                and case["output_count"] == output_count
            ]
            assert {case["input_count"] for case in load_pair} == {1, 2}
            invariant_keys = ("alu_fixed_sequence", "alu_repeats", "output_y", "output_x", "output_count")
            assert len({tuple(str(case[key]) for key in invariant_keys) for case in load_pair}) == 1
        for input_count in (1, 2):
            store_pair = [
                case for case in base
                if (case["output_y"], case["output_x"]) == (y, x)
                and case["input_count"] == input_count
            ]
            assert {case["output_count"] for case in store_pair} == {1, 2}
            invariant_keys = ("input_count", "input_y", "input_x", "alu_fixed_sequence", "alu_repeats")
            assert len({tuple(str(case[key]) for key in invariant_keys) for case in store_pair}) == 1


def test_lowered_single_and_double_input_use_the_same_vta_alu_opcode_sequence():
    opcode_sequences = []
    for input_count in (1, 2):
        case = next(
            case for case in cases(1, 16)
            if case["family"] == "contiguous_load_store"
            and case["input_count"] == input_count
            and case["output_count"] == 1
            and case["output_y"] == 8
        )
        lowered = str(lower_kernel(case))
        pushes = re.findall(r"uop_push\(([^)]+)\)", lowered)
        opcode_sequences.append([int(push.split(",")[5].strip()) for push in pushes])
    assert opcode_sequences[0] == opcode_sequences[1]
    assert len(opcode_sequences[0]) == 1


def test_access_pairs_hold_alu_output_and_store_work_constant():
    paired = [case for case in cases(1, 16) if case["pair_id"]]
    groups = {}
    for case in paired:
        groups.setdefault(case["pair_id"], {})[case["pair_role"]] = case
    assert len(groups) == 8
    for roles in groups.values():
        assert set(roles) == {"control", "treatment"}
        control, treatment = roles["control"], roles["treatment"]
        for key in (
            "alu_opcode", "alu_repeats", "output_y", "output_x", "output_count",
            "output_bits", "batch", "block_out", "elements",
        ):
            assert control[key] == treatment[key]


def test_padded_control_is_a_pre_padded_contiguous_tensor():
    paired = [case for case in cases(1, 16) if case["family"] == "padded_matched"]
    for case in paired:
        if case["pair_role"] == "control":
            assert case["access_kind"] == "contiguous"
            assert case["input_fill"] == "pre_padded"
            assert (case["input_y"], case["input_x"]) == (
                case["output_y"], case["output_x"]
            )
        else:
            assert case["access_kind"] == "padded"
            assert case["input_fill"] == "pattern"


def test_strided_control_differs_only_in_source_stride_geometry():
    paired = [case for case in cases(1, 16) if case["family"] == "strided_matched"]
    for case in paired:
        if case["pair_role"] == "control":
            assert case["access_kind"] == "contiguous"
            assert (case["input_y"], case["input_x"]) == (
                case["output_y"], case["output_x"]
            )
        else:
            assert case["access_kind"] == "strided"
            assert case["input_x"] > case["output_x"]


def test_compute_sweep_keeps_dma_shape_fixed_and_changes_only_alu_repeats():
    sweep = [case for case in cases(1, 16) if case["family"] == "compute_sweep"]
    assert [case["alu_repeats"] for case in sweep] == [1, 2, 4, 8]
    ignored = {"case_id", "alu_repeats"}
    reference = {key: value for key, value in sweep[0].items() if key not in ignored}
    for case in sweep[1:]:
        assert {key: value for key, value in case.items() if key not in ignored} == reference


def test_lowered_compute_sweep_has_fixed_dma_and_increasing_alu_work():
    observed = []
    for case in (case for case in cases(1, 16) if case["family"] == "compute_sweep"):
        lowered = str(lower_kernel(case))
        observed.append((
            case["alu_repeats"], lowered.count("VTALoadBuffer2D"),
            lowered.count("VTAStoreBuffer2D"), lowered.count("uop_push("),
        ))
    assert {(load_count, store_count) for _, load_count, store_count, _ in observed} == {(1, 1)}
    assert [uop_count for _, _, _, uop_count in observed] == [1, 2, 4, 8]


def test_vta_geometry_is_not_fixed_to_one_by_sixteen():
    case = cases(batch=2, block_out=32)[0]
    assert case["batch"] == 2
    assert case["block_out"] == 32
    assert case["elements"] == case["output_y"] * case["output_x"] * 2 * 32


def test_protocol_hash_is_reproducible_and_order_sensitive():
    first = {"seed": 7, "case_order": ["a", "b"]}
    same = {"case_order": ["a", "b"], "seed": 7}
    changed = {"seed": 7, "case_order": ["b", "a"]}
    assert protocol_hash(first) == protocol_hash(same)
    assert protocol_hash(first) != protocol_hash(changed)


def test_case_randomization_is_seeded_and_reproducible():
    matrix = cases(1, 16)
    order_a = [case["case_id"] for case in shuffled_cases(matrix, 4101)]
    order_b = [case["case_id"] for case in shuffled_cases(matrix, 4101)]
    order_c = [case["case_id"] for case in shuffled_cases(matrix, 4102)]
    assert order_a == order_b
    assert order_a != order_c
    assert sorted(order_a) == sorted(case["case_id"] for case in matrix)


def test_analyze_preserves_partial_results_and_records_missing_case(tmp_path):
    selected = cases(1, 16)[:2]
    package = tmp_path / "package"
    results = tmp_path / "board_results"
    package.mkdir()
    results.mkdir()
    (package / "manifest.json").write_text(json.dumps({"cases": selected}))
    sample = {
        "inner_repeats": 1,
        "wall_us": 10.0,
        "correct": True,
        "profile": {
            "device_run_wait_us": 8.0,
            "driver_submit_mmio_us": 1.0,
            "load_buffer_2d_bytes": 2048,
            "store_buffer_2d_bytes": 512,
            "load_buffer_2d_calls": 1,
            "store_buffer_2d_calls": 1,
        },
    }
    (results / (selected[0]["case_id"] + ".jsonl")).write_text(json.dumps(sample) + "\n")
    analyze(tmp_path)
    summary = json.loads((tmp_path / "dma_identification_summary.json").read_text())
    assert summary["successful_case_count"] == 1
    assert summary["failed_case_count"] == 1
    assert summary["failures"][0]["failure_type"] == "profile_missing"
