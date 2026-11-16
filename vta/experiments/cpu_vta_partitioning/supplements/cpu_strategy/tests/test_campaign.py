"""End-to-end synthetic native evidence; no board, SSH or hardware timing claims."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from cpu_vta_partitioning.common import read, write
from cpu_strategy import campaign
from cpu_strategy.domain import components, matrix


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from cpu_strategy import prepare
    monkeypatch.setattr(campaign, "_identity", lambda: "test-controller")
    monkeypatch.setattr(prepare, "validate", lambda bundle: read(Path(bundle) / "manifest.json"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write(bundle / "manifest.json", {"bundle_id": "unit-bundle", "controller_identity": "test-controller"})
    programs = {}
    for c in components():
        s = c["stage"]
        programs[s["program"]] = {"device": s["device"], "begin": s["begin"], "end": s["end"],
                                   "output_node_ids": [s["end"]]}
    write(bundle / "plans/programs.json", programs)
    (bundle / "references").mkdir()
    np.save(bundle / "references/value.npy", np.array([7], dtype="int8"))
    records = [{"index": i, "frontiers": {str(j): {"npy": "references/value.npy"} for j in range(22)}}
               for i in range(67)]
    write(bundle / "plans/inputs.json", {"records": records})
    root = tmp_path / "run"
    campaign.create(bundle, root, {"host": "fake"}, excluded_boot="preflight")
    return root, bundle, FakeBoard()


class FakeBoard:
    def __init__(self):
        self.boot = "explore"
        self.calls, self.acknowledged = [], []
        self.fail_on_call = None
        self.bad_output = False
        self.duplicate_frame = False
        self.change_boot_after_measure = False
        self.drift_attempts = set()
        self.extend_warmup = False
        self.unstable_extended = False

    def boot_id(self): return self.boot
    def wait_idle(self, boot): assert boot == self.boot
    def mount_storage(self): pass
    def deploy(self, bundle, root): return "/fake/bundle"
    def verify_payload(self, remote, boot): assert boot == self.boot
    def qualify_environment(self, remote, boot, root, bundle): assert boot == self.boot
    def idle_command(self, *args, **kwargs): pass
    def recover_abandoned(self, root, boot): pass
    def acknowledge(self, document, root, boot): self.acknowledged.append((boot, document["task_id"]))

    def measure(self, document, bundle, remote, root, boot):
        self.calls.append((boot, document["task_id"], copy.deepcopy(document)))
        if self.fail_on_call == len(self.calls):
            self.fail_on_call = None
            raise RuntimeError("simulated SSH disconnect")
        raw = Path(root) / "raw" / boot / document["task_id"] / "collected"
        raw.mkdir(parents=True, exist_ok=True)
        stages = document["stages"]
        def run_time(s):
            base = 200. if s["begin"] == 0 else 140.
            if s["end"] - s["begin"] == 1:
                base = {0: 84., 2: .8, 20: 2.8}[s["begin"]]
            return 30. if s["device"] == "vta" else base / s["threads"]
        ii = max(run_time(s) for s in stages)
        if len(stages) == 3:
            ii += 8 * len(set(stages[0]["mask"]) & set(stages[2]["mask"]))
        if document["kind"] == "warmup_qualification":
            if self.extend_warmup and document["warmup"] == 8:
                ii *= 1.2
            if self.unstable_extended and document.get("warmup_repeat") == 1:
                ii *= 1.2
        if (document.get("position") == "after"
                and (document.get("round"), document.get("attempt")) in self.drift_attempts):
            ii *= 1.2
        affinities = []
        for j, s in enumerate(stages):
            tids = [1000 + 10*j + t for t in range(s["threads"])]
            affinities.append({"requested_cpus": s["mask"], "thread_ids": tids,
                               "active_runtime_threads": s["threads"],
                               "effective_affinities": [{"tid": tid, "cpus": s["mask"],
                                                        "cpu_time_delta_ms": 10. if k == 0 else 0.}
                                                       for k, tid in enumerate(tids)]})
        write(raw / "affinity.json", {"stages": affinities})
        rows = []
        for i in range(sum(document[k] for k in ("warmup", "scored", "tail"))):
            row = {"frame_id": i, "completion_index": i,
                   "input_index": i % len(document["input_indices"]), "completion_ms": 10. + ii*i,
                   "completion_process_cpu_ms": 100. + ii*2*i, "total_latency_ms": ii*3,
                   "cpu_stage_gate_mode": "off", "stage_cpu_time_scope": "exclusive_process_cpu_time"
                   if len(stages) == 1 else "overlapping_process_cpu_time"}
            for j, s in enumerate(stages):
                row.update({f"stage{j}_set_ms": .5, f"stage{j}_run_ms": run_time(s), f"stage{j}_get_ms": .25,
                            f"stage{j}_set_process_cpu_ms": .3,
                            f"stage{j}_run_process_cpu_ms": run_time(s) * min(s["threads"], 3),
                            f"stage{j}_get_process_cpu_ms": .15})
            if document["kind"].startswith("qualify"):
                row["stage_raw_outputs"] = []
                for j in range(len(stages)):
                    output = raw / "outputs" / f"run_{i}" / f"stage{j}.bin"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(bytes([8 if self.bad_output else 7]))
                    row["stage_raw_outputs"].append({"output_files": [{"path": "/remote/" + output.name,
                                                                         "dtype": "int8", "shape": [1]}]})
            rows.append(row)
        if self.duplicate_frame:
            rows[1]["frame_id"] = rows[0]["frame_id"]
        (raw / "result.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if self.change_boot_after_measure:
            self.boot = "unexpected"
        return raw


def complete_exploration(root, board):
    state = campaign.run_current(root, board)
    assert state["boot_index"] == 1 and state["status"] == "WAITING_NEW_BOOT"
    return state


def test_full_four_boot_flow_profile_precedes_feedback_and_replays_evidence(setup):
    root, bundle, board = setup
    complete_exploration(root, board)
    tasks = [doc for _, _, doc in board.calls]
    first_perf = next(i for i, t in enumerate(tasks) if t["kind"] == "performance")
    assert all(t["kind"] != "calibration" for t in tasks[first_perf:])
    events = [json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()]
    predictive_event = next(i for i, e in enumerate(events) if e["event"] == "profile_selection_frozen_before_pipeline")
    first_grid_event = next(i for i, e in enumerate(events) if e["event"] == "session_start" and e["task_kind"] == "performance")
    assert predictive_event < first_grid_event
    selected = read(root / "selection.json")
    assert len(selected["pool"]) <= 10
    assert set(selected["methods"]) == {"local_fastest", "profile_disjoint", "dp_static", "dp_budget5"}
    ranking = campaign.validated_ranking(root)
    assert len(ranking["rankings"]) == 20 and ranking["candidate_count"] == 32
    assert not ranking["ranking_uses_pipeline_feedback"]
    assert selected["methods"]["dp_static"] == ranking["rankings"][0]["id"]
    assert selected["methods"]["dp_budget5"] in {r["id"] for r in ranking["rankings"][:5]}
    assert selected["measured_exploration_configurations"] <= 13
    performance = [t for t in tasks if t["kind"] == "performance"]
    assert len({t["candidate_id"] for t in performance}) <= 13
    assert all(t["stages"] == matrix()["configs"][matrix()["references"]["single_thread"]]["stages"]
               for t in tasks if t["kind"] == "anchor")
    assert all(t.get("measurement_role") == "screening" for t in performance)
    original = (root / "selection.json").read_bytes()
    calls = len(board.calls)
    campaign.run_current(root, board)  # completed boot is not a new replicate
    assert len(board.calls) == calls
    for i in range(1, 4):
        board.boot = f"confirm{i}"
        state = campaign.run_current(root, board)
    assert state["status"] == "COMPLETE" and state["boot_index"] == 4
    assert len(set(state["completed_boots"])) == 4
    assert (root / "selection.json").read_bytes() == original
    assert len(campaign.validated_rounds(root, "confirmation")) == 6
    calls = len(board.calls)
    campaign.run_current(root, board)
    assert len(board.calls) == calls


def test_disconnect_resume_never_remeasures_accepted_task(setup):
    root, _, board = setup
    board.fail_on_call = 6
    with pytest.raises(RuntimeError, match="disconnect"):
        campaign.run_current(root, board)
    accepted = {task for _, task in board.acknowledged}
    before = len(board.calls)
    complete_exploration(root, board)
    assert not accepted & {task for _, task, _ in board.calls[before:]}
    assert len(board.acknowledged) >= len(accepted)


def test_unexpected_boot_preserves_old_raw_and_restarts_replicate(setup):
    root, _, board = setup
    board.fail_on_call = 5
    with pytest.raises(RuntimeError): campaign.run_current(root, board)
    assert (root / "raw/explore").is_dir()
    board.boot = "replacement"
    state = complete_exploration(root, board)
    assert state["completed_boots"] == ["replacement"]
    assert state["abandoned_boots"] == ["explore"]
    assert read(root / "costs.json")["boot_id"] == "replacement"
    assert (root / "raw/explore").is_dir()


@pytest.mark.parametrize("fault,match", [("bad_output", "incorrect full output"),
                                         ("duplicate_frame", "duplicate/missing"),
                                         ("change_boot_after_measure", "before result acceptance")])
def test_invalid_native_data_is_not_accepted_or_acknowledged(setup, fault, match):
    root, _, board = setup
    setattr(board, fault, True)
    with pytest.raises((ValueError, RuntimeError), match=match): campaign.run_current(root, board)
    assert not list((root / "accepted").glob("*/*.json"))
    assert not board.acknowledged


def test_tampered_accepted_raw_and_frozen_selection_are_rejected(setup):
    root, _, board = setup
    complete_exploration(root, board)
    file = next((root / "raw").glob("*/*/collected/result.jsonl"))
    original = file.read_bytes()
    file.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="missing or changed"): campaign.validated_rounds(root)
    file.write_bytes(original)
    selected = root / "selection.json"
    selected.write_text(selected.read_text() + "\n")
    with pytest.raises(ValueError, match="missing or changed"): campaign.validate_state(root)


def test_drift_replaces_whole_round_once_and_preserves_failure(setup):
    root, _, board = setup
    board.drift_attempts = {(0, 0)}
    complete_exploration(root, board)
    assert read(root / "exploration/explore/screening/round0.json")["attempt"] == 1
    assert (root / "exploration/explore/screening/invalid-round0-attempt0.json").is_file()
    rounds = campaign.validated_rounds(root, "exploration")
    controls = [r for r in rounds if r["role"] == "screening"]
    assert len(controls) == 2 and controls[0]["attempt"] == 1


def test_preflight_boot_does_not_count_as_exploration(setup):
    root, _, board = setup
    board.boot = "preflight"
    state = campaign.run_current(root, board)
    assert state["boot_index"] == 0 and not board.calls


def test_profile_selector_rejects_historical_cost_source_and_uses_stable_minmax():
    costs = {c["stage"]["signature"]: {"service_ms": 5., "run_core_ms": 1., "boundary_core_ms": .1} for c in components()}
    result = campaign.profile_selection({"source": "fresh_isolated_board_calibration", "costs": costs})
    assert result["profile_disjoint"] == min(matrix()["families"]["disjoint"])
    assert result["source"].endswith("before_pipeline")
    with pytest.raises(ValueError, match="current-board"):
        campaign.profile_selection({"source": "historical", "costs": costs})


def test_screening_disconnect_keeps_static_dp_and_never_remeasures_accepted_tasks(setup):
    root, _, board = setup
    original = board.measure
    failed = False
    def disconnect(document, *args):
        nonlocal failed
        if document["kind"] == "performance" and document.get("round") == 1 and not failed:
            failed = True
            raise RuntimeError("screening disconnect")
        return original(document, *args)
    board.measure = disconnect
    with pytest.raises(RuntimeError, match="screening disconnect"):
        campaign.run_current(root, board)
    frozen = (root / "profile_selection.json").read_bytes()
    accepted = {task for _, task in board.acknowledged}
    call_count = len(board.calls)
    complete_exploration(root, board)
    assert (root / "profile_selection.json").read_bytes() == frozen
    assert not accepted & {task for _, task, _ in board.calls[call_count:]}
    assert len(campaign.validated_ranking(root)["rankings"]) == 20


def test_exhausted_anchor_replacement_marks_run_stopped_and_preserves_raw(setup):
    root, _, board = setup
    board.drift_attempts = {(0,0),(0,1)}
    with pytest.raises(RuntimeError,match="single replacement"):
        campaign.run_current(root,board)
    state = campaign.validate_state(root)
    assert state["status"] == "STOPPED_ANCHOR_DRIFT" and state["boot_index"] == 0
    assert len(list((root/"exploration/explore/screening").glob("invalid-*.json"))) == 2
    assert not (root/"selection.json").exists()
    assert list((root/"raw").glob("*/*/collected/result.jsonl"))


@pytest.mark.parametrize("unstable", [False, True])
def test_extended_warmup_is_uniform_and_unstable_repeat_stops(setup, unstable):
    root, _, board = setup
    state = campaign.validate_state(root)
    state["active_boot"] = board.boot
    board.extend_warmup = True
    board.unstable_extended = unstable
    recorder = campaign.Recorder(root, state, board, "/fake")
    if unstable:
        with pytest.raises(RuntimeError, match="remains unstable"):
            campaign.warmup_qualification(root, state, recorder)
        assert (root / "warmup_failure.json").is_file()
        assert not (root / "warmup.json").exists()
    else:
        value = campaign.warmup_qualification(root, state, recorder)
        assert value["warmup"] == 24
        assert all("24_repeat" in x for x in value["checks"].values())


def test_confirmation_reboot_archives_partial_round_without_reusing_it(setup):
    root, _, board = setup
    complete_exploration(root, board)
    selection = (root / "selection.json").read_bytes()
    count = len(read(root / "selection.json")["pool"])
    board.boot = "confirm-old"
    board.fail_on_call = len(board.calls) + 2*count + 4
    with pytest.raises(RuntimeError, match="disconnect"):
        campaign.run_current(root, board)
    assert (root / "confirmation/boot1/round0.json").is_file()
    board.boot = "confirm-replacement"
    state = campaign.run_current(root, board)
    assert state["completed_boots"] == ["explore", "confirm-replacement"]
    assert (root / "abandoned/confirm-old/confirmation/boot1/round0.json").is_file()
    assert (root / "selection.json").read_bytes() == selection
    assert len(campaign.validated_rounds(root, "confirmation")) == 2


def test_copying_a_previous_confirmation_round_cannot_count_as_new_boot(setup):
    root, _, board = setup
    complete_exploration(root, board)
    board.boot = "confirm1"
    campaign.run_current(root, board)
    path = root / "confirmation/boot2/round0.json"
    path.parent.mkdir(parents=True)
    path.write_bytes((root / "confirmation/boot1/round0.json").read_bytes())
    board.boot = "confirm2"
    with pytest.raises(ValueError, match="different replicate"):
        campaign.run_current(root, board)
    assert campaign.validate_state(root)["boot_index"] == 2


def test_packaged_entrypoint_imports_its_own_base_controller_from_unrelated_directory(tmp_path):
    from cpu_strategy.domain import ROOT, PROJECT
    snapshot = tmp_path / "relocated_bundle/sources"
    experiment = snapshot / "cpu_strategy"
    experiment.mkdir(parents=True)
    shutil.copy2(ROOT / "run.py", experiment / "run.py")
    shutil.copytree(ROOT / "cpu_strategy", experiment / "cpu_strategy",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT / "src", snapshot / "base_controller/src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    script = """import pathlib,runpy,sys
entry=pathlib.Path(sys.argv[1]);sys.argv=[str(entry),'--help']
try:runpy.run_path(str(entry),run_name='__main__')
except SystemExit as result:
 assert result.code == 0
import cpu_vta_partitioning.common as common
assert pathlib.Path(common.__file__).resolve().is_relative_to(entry.parent.parent/'base_controller/src'),common.__file__
"""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", script, str(experiment / "run.py")],
                            cwd=unrelated, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "board" in result.stdout


def test_native_command_transmits_one_thread_with_four_allowed_cores(tmp_path):
    from cpu_strategy.transport import native_command
    from cpu_strategy.measurement import make_task
    from cpu_strategy.evidence import identified
    from cpu_strategy.domain import configuration
    bundle = tmp_path / "command-bundle"
    candidate = configuration((1, 4), ((0, 1, 2, 3), (0, 1, 2, 3)))
    programs = {}
    for index, stage in enumerate(candidate["stages"]):
        programs[stage["program"]] = {"begin": stage["begin"], "input_names": ["data"],
                                      "input_node_ids": [index], "output_node_ids": [index + 1]}
    write(bundle / "plans/programs.json", programs)
    write(bundle / "plans/inputs.json", {"records": [{"input": "inputs/image.bin"}] * 19})
    task = identified(make_task("performance", candidate["stages"], warmup=8, scored=32,
                                tail=4, indices=[16], candidate_id=candidate["id"]), "command-test")
    argv, inputs = native_command(task, bundle, "/bundle", "/job")
    assert argv[argv.index("--stage0-runtime-num-threads") + 1] == "1"
    assert argv[argv.index("--stage0-cpu-affinity") + 1] == "0,1,2,3"
    assert argv[argv.index("--stage2-runtime-num-threads") + 1] == "4"
    assert argv.count("--allow-cpu-affinity-superset") == 1
    assert argv[argv.index("--cpu-stage-gate") + 1] == "off"
    assert inputs == "/bundle/inputs/image.bin\n"
    task["allow_cpu_affinity_superset"] = False
    with pytest.raises(ValueError, match="focused"):
        native_command(task, bundle, "/bundle", "/job")


def test_acknowledgement_disconnect_keeps_receipt_and_retries_without_measurement(setup, monkeypatch):
    root, bundle, board = setup
    state = campaign.validate_state(root)
    state["active_boot"] = board.boot
    stage = components()[0]["stage"]
    task = campaign._task("qualify_component", [stage], state, component_signature=stage["signature"])
    original = board.acknowledge
    def disconnected_ack(*args):
        raise RuntimeError("simulated acknowledgement disconnect")
    monkeypatch.setattr(board, "acknowledge", disconnected_ack)
    first = campaign.measured(task, bundle, "/fake", root, board.boot, board)
    assert len(list((root / "cleanup_pending").glob("*/*.json"))) == 1
    measured_count = len(board.calls)
    monkeypatch.setattr(board, "acknowledge", original)
    campaign._retry_cleanup(root, state, board)
    again = campaign.measured(task, bundle, "/fake", root, board.boot, board)
    assert again == first and len(board.calls) == measured_count
    assert not list((root / "cleanup_pending").glob("*/*.json"))
