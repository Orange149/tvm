"""Protocol and actual lowering regressions; never contacts an FPGA."""
import copy
import json
from pathlib import Path

import pytest
import c3_thread_sync_protocol as p
import c3_thread_sync_audit as a


def fake_times(accepted, scales=None):
    scales = scales or {}
    return [{**r, "latency_ms": scales.get(r["candidate_id"], 10), "correct": True,
             "complete_operator_calls": 1, "hidden_warmup_calls": 0} for r in p.schedule(accepted)]


def test_williams_not_position_rotation():
    result = p.balance(p.LETTERS)
    assert result["rows"][1] == list("BCADHEGF")
    assert len(result["predecessor_counts"]) == 56
    assert set(result["predecessor_counts"].values()) == {1}
    assert set(result["position_counts"].values()) == {1}
    assert not p.balance("ABCDEFG")["complete_eight_version_williams"]
    assert set(p.balance("ABCDEFG")["predecessor_counts"].values()) != {1}


def test_exact_coverage_and_seed_order():
    accepted = [t+"_"+v for t in ("T8","T7") for v in p.LETTERS]
    rows = fake_times(accepted)
    assert len(rows) == 384
    p.validate_timing(rows,accepted)
    for r in range(8):
        groups={}
        for row in rows:
            if row["round"] == r: groups.setdefault(row["candidate_id"],[]).append(row["seed"])
        assert len({tuple(x) for x in groups.values()}) == 1
    rows[0]["complete_operator_calls"] = 2
    with pytest.raises(ValueError,match="extra invocation"): p.validate_timing(rows,accepted)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "wrong"])
def test_invalid_timing_fails(mutation):
    rows=fake_times(["T8_A"])
    if mutation=="missing": rows.pop()
    elif mutation=="duplicate": rows[-1]=rows[0]
    elif mutation=="nan": rows[0]["latency_ms"]=float("nan")
    else: rows[0]["correct"]=False
    with pytest.raises(ValueError): p.validate_timing(rows,["T8_A"])


def test_direction_two_boot_threshold_and_unmatched():
    rows=fake_times(["T8_A","T8_B"],{"T8_B":11})
    boot=p.paired(rows,"T8_A","T8_B")
    assert boot["paired_median_gain_pct"] == pytest.approx(-10)
    assert p.classify([boot])=="awaiting_second_boot"
    assert p.classify([boot,boot])=="stable_regression"
    assert p.classify([boot,boot],False)=="control_unmatched"
    improve=p.paired(rows,"T8_B","T8_A")
    assert p.classify([improve,improve])=="stable_improvement"
    assert p.classify([improve,boot])=="no_stable_direction_threshold"


def test_median_of_ratios_not_ratio_of_medians():
    rows=fake_times(["T8_A","T8_B"])
    for row in rows:
        row["latency_ms"] = ([1,2,3,4,10,20,30,40] if row["candidate_id"]=="T8_A" else [2,1,4,3,8,16,24,32])[row["round"]]
    boot=p.paired(rows,"T8_A","T8_B")
    import statistics
    assert boot["paired_median_gain_pct"] == pytest.approx(statistics.median(boot["round_gains_pct"]))


def test_fsim_has_no_board_driver_counter():
    local=dict(driver_run_calls=0, driver_run_insns=0, synchronize_calls=2, synchronize_insns=123)
    board={**local,"driver_run_calls":2,"driver_run_insns":123}
    assert p.logical_signature(local)==p.logical_signature(board)
    assert p.logical_signature(local)!=p.logical_signature({**board,"synchronize_insns":124})


def test_board_requires_explicit_execution_flag(tmp_path):
    from types import SimpleNamespace
    from c3_thread_sync_board import run
    with pytest.raises(ValueError,match="execute-board"):
        run(SimpleNamespace(execute_board=False),tmp_path)


def test_first_call_profile_keeps_requests_but_defers_warmed_signature():
    from c3_thread_sync_board import validate_profile, requires_local_warmed_signature
    row = {
        "candidate_id": "T8_B",
        "local_qualification": {"semantics": {"totals": {"load_buffer_2d_bytes": 745472}}},
        "command_qualification": {
            "calls": [{"profile": {
                "load_buffer_2d_bytes": 745472, "synchronize_calls": 1,
                "synchronize_insns": 263,
            }}],
            "instruction_audits": {"0": {"submission_count": 1}},
        },
    }
    first = {"runtime_profile_complete": {
        "load_buffer_2d_bytes": 745472, "synchronize_calls": 1,
        "synchronize_insns": 262, "driver_run_calls": 1,
    }}
    validate_profile(row, first, require_warmed_signature=False)
    with pytest.raises(ValueError, match="warmed command signature"):
        validate_profile(row, first, require_warmed_signature=True)
    changed_request = copy.deepcopy(first)
    changed_request["runtime_profile_complete"]["load_buffer_2d_bytes"] = 1
    with pytest.raises(ValueError, match="data requests differ"):
        validate_profile(row, changed_request, require_warmed_signature=False)
    assert requires_local_warmed_signature("diagnostic")
    assert not requires_local_warmed_signature("correctness_warmup")
    assert not requires_local_warmed_signature("warmup")
    assert not requires_local_warmed_signature("timing")


@pytest.mark.parametrize("port", [9090, 9091])
def test_experiment_rpc_uses_distinct_fixed_session_port(monkeypatch, port):
    from types import SimpleNamespace
    from c3_thread_sync_board import ExperimentBoard
    board = ExperimentBoard(SimpleNamespace(known_hosts="unused"), "/var/volatile/test", port)
    commands = []

    def fake_ssh(command, timeout=25):
        commands.append(command)
        return "123\n" if "echo $!" in command else ""

    monkeypatch.setattr(board, "ssh", fake_ssh)
    assert board.start() == 123
    assert "--port=" + str(port) in commands[-1]
    assert "--port-end=" + str(port + 1) in commands[-1]


def test_sdk_adapter_clears_and_restores_late_linker_environment(monkeypatch):
    import os
    import run_vta_c3_thread_sync_ablation as r
    monkeypatch.setenv("LD_LIBRARY_PATH", "/test/late/import")
    with r.adapter():
        assert "LD_LIBRARY_PATH" not in os.environ
        assert os.environ["VTA_REPLAY_POLICY"]=="disabled"
    assert os.environ["LD_LIBRARY_PATH"]=="/test/late/import"


def dump():
    return """There are 5 instructions
INSTRUCTION 0: LOAD UOP
 DRAM: 0x00000000, SRAM:0x0000
 y: size=1, pad=[0, 0]
 x: size=8, stride=8, pad=[0, 0]
INSTRUCTION 1: GEMM
 reset_out: 1
 range (0, 4)
INSTRUCTION 2: GEMM
 reset_out: 0
 range (4, 8)
INSTRUCTION 3: NOP-COMPUTE-STAGE
INSTRUCTION 4: FINISH
"""


def test_opcode_and_semantic_accounting():
    result=a.instruction_dump(dump())
    block=result["submissions"][0]
    assert block["raw_opcode"]==dict(LOAD=2,GEMM=2,FINISH=1)
    assert block["semantic"]["RESET_GEMM"]==1
    assert block["semantic"]["MAC_GEMM"]==1
    assert sum(block["semantic"].values())==5
    assert result["uop_load_bytes"]==32
    assert result["effective_uop_occupancy_peak"] is None


@pytest.mark.parametrize("bad", [lambda x:x.replace("SRAM:0x0000","SRAM:0x1fff"),
                                  lambda x:x.replace("range (4, 8)","range (4, 9)"),
                                  lambda x:x.replace("5 instructions","6 instructions"),
                                  lambda x:x.replace("INSTRUCTION 4: FINISH", "INSTRUCTION 4: UNKNOWN")])
def test_dump_rejects_unknown_or_illegal_addresses(bad):
    with pytest.raises(ValueError): a.instruction_dump(bad(dump()))


@pytest.fixture(scope="module")
def lowered():
    import run_vta_c3_thread_sync_ablation as r
    from c3_reuse_semantics import capture_lower,extract_requests
    env, fp=r.s.hardware()
    rows=r.make_candidates(env,fp,"test")
    result={}
    for row in rows:
        sch,tensors=r.instantiate(row)
        mod,_,_=capture_lower(sch,tensors)
        info=a.final_tir_audit(mod,extract_requests(mod,env))
        result[row["candidate_id"]]=(row,mod,info)
    return result


def test_sixteen_complete_entities_and_actual_address_checks(lowered):
    assert len(lowered)==16
    for cid,(row,mod,info) in lowered.items():
        assert info["status"]=="passed"
        assert info["max_kernel_uops"]<=8192
        assert row["visible_features"]["h_nthread"]==1
        assert row["visible_features"]["oc_nthread"]==(2 if cid[-1]=="A" else 1)
        assert row["experimental_reload_control"]==(cid[-1] in "EG")


def test_sync_identity_and_sram_difference_are_not_normalized_away(lowered):
    _,mod,info=lowered["T8_D"]
    _,other,other_info=lowered["T8_E"]
    actual=a.control_match(info,other_info,mod,other,"sync")
    assert actual["checks"]["ordered_data_and_compute"]
    broken=copy.deepcopy(other_info)
    broken["event_sha256"]="changed_sram_offset"
    assert not a.control_match(info,broken,mod,other,"sync")["matched"]


def test_genuine_tophub_adapter_equal(lowered):
    import tvm
    import run_vta_c3_thread_sync_ablation as r
    from c3_reuse_semantics import capture_lower
    row,mod,_=lowered["T8_A"]
    alias={**row,"source_tophub":False,"complete_config_entity":row["adapter_config"]}
    sch,tensors=r.instantiate(alias)
    alias_mod,_,_=capture_lower(sch,tensors)
    assert tvm.ir.structural_equal(mod,alias_mod,map_free_vars=True)


def test_buffers_update_without_reallocation_and_poison(monkeypatch):
    import numpy as np
    import run_vta_p7r132_y00_search_confirmation as buffers
    class Array:
        def copyfrom(self,value): self.value=np.array(value)
    expected=np.array([0,5,127],dtype="int8")
    monkeypatch.setattr(buffers,"reference_data_from_workload",lambda workload,seed:(expected+seed,expected,expected))
    pool={key:Array() for key in ("data","weight","output")}
    pool.update(data_shape=(3,),weight_shape=(3,),output_shape=(3,))
    identities={k:id(pool[k]) for k in ("data","weight","output")}
    buffers.prepare_reused_buffers(pool,[],0);buffers.prepare_reused_buffers(pool,[],1)
    assert all(id(pool[k])==v for k,v in identities.items())
    assert np.all(pool["output"].value==-113)
    assert np.all(pool["output"].value!=expected)


def test_analysis_rejects_same_boot_twice(tmp_path):
    from types import SimpleNamespace
    import run_vta_c3_reuse_supplement as s
    from c3_thread_sync_results import analyze
    prepared=tmp_path/"prepared";prepared.mkdir()
    s.write(prepared/"contract.json",{"phase":"prepare"})
    s.write(prepared/"candidates.json",[]);s.write(prepared/"controls.json",{})
    s.seal(prepared)
    frozen=tmp_path/"frozen";frozen.mkdir()
    s.write(frozen/"contract.json",dict(phase="freeze",schema="c3_thread_sync_ablation_v1",
        prepared_directory=str(prepared),prepared_manifest_sha256=s.file_hash(prepared/"artifact_hashes.json")))
    s.seal(frozen)
    dirs=[]
    for index in (1,2):
        directory=tmp_path/f"boot{index}";directory.mkdir();dirs.append(directory)
        s.write(directory/"contract.json",dict(boot_index=index,frozen_manifest_sha256=s.file_hash(frozen/"artifact_hashes.json")))
        s.write(directory/"summary.json",dict(boot_id="unchanged-boot",timing_valid=True,accepted=["T8_A"]))
        for row in fake_times(["T8_A"]): s.append(directory/"timing.jsonl",row)
        s.seal(directory)
    out=tmp_path/"analysis";out.mkdir()
    with pytest.raises(ValueError,match="distinct boots"):
        analyze(SimpleNamespace(contract=str(frozen/"contract.json"),boot_dirs=dirs),out)


def test_manifest_change_fails_before_reuse(tmp_path):
    import run_vta_c3_reuse_supplement as s
    s.write(tmp_path/"contract.json",{"schema":"test"})
    (tmp_path/"binary.so").write_bytes(b"frozen")
    s.seal(tmp_path)
    (tmp_path/"binary.so").write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash"):
        s.verify(tmp_path)
