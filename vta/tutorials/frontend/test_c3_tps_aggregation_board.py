"""Board-contract checks using synthetic records/mocks; never contact a board."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import c3_tps_aggregation_board as board


def submission(mode="detailed"):
    return {"schema": "c3_official_window_submit_ledger_v1", "mode": mode,
            "overflow": False, "attempted_submits": 1, "recorded_submits": 1,
            "submits": [{"ordinal": 1, "reason": "explicit_sync", "insns": 76,
                         "insn_bytes": 1216, "serialized_uop_bytes": 37760,
                         "completion": True, "timeout": 0}]}


class SubmissionTests(unittest.TestCase):
    def test_detailed_and_minimal_have_same_instruction_signature(self):
        self.assertEqual(board.submit_signature(submission(), "detailed"),
                         board.submit_signature(submission("minimal"), "minimal"))

    def test_incomplete_overflow_wrong_mode_or_failed_completion_rejected(self):
        changes = [dict(mode="minimal"), dict(overflow=True), dict(recorded_submits=0),
                   dict(attempted_submits=2), dict(submits=[])]
        for change in changes:
            record = submission()
            record.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                board.submit_signature(record, "detailed")
        for change in (dict(completion=False), dict(timeout=1), dict(ordinal=2)):
            record = submission()
            record["submits"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                board.submit_signature(record, "detailed")

    def test_missing_or_invalid_instruction_and_uop_units_rejected(self):
        changes = [dict(insns=0, insn_bytes=0), dict(insns=-1, insn_bytes=-16),
                   dict(insns=True, insn_bytes=16), dict(serialized_uop_bytes=-4),
                   dict(serialized_uop_bytes=1), dict(serialized_uop_bytes=True)]
        for change in changes:
            record = submission()
            record["submits"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                board.submit_signature(record, "detailed")
        record = submission()
        del record["submits"][0]["serialized_uop_bytes"]
        with self.assertRaises(ValueError):
            board.submit_signature(record, "detailed")


class OwnershipTests(unittest.TestCase):
    def make_board(self):
        args = SimpleNamespace(host="invalid.test", known_hosts="/tmp/unused-known-hosts")
        return board.Board(args, "/var/volatile/owned_attempt", "f" * 64)

    def test_stop_requires_both_owned_executable_and_session_directory(self):
        instance = self.make_board()
        instance.ssh = Mock(return_value="")
        instance.stop()
        instance.ssh.assert_not_called()
        instance.session = "/var/volatile/owned_attempt/session_01"
        instance.stop()
        commands = [str(c.args[0]) for c in instance.ssh.call_args_list]
        kill_commands = [c for c in commands if "kill " in c]
        self.assertEqual(len(kill_commands), 1)
        command = kill_commands[0]
        self.assertIn('readlink /proc/$c3_pid/cwd', command)
        self.assertIn('readlink /proc/$c3_pid/exe', command)
        self.assertIn('/var/volatile/owned_attempt/session_01', command)
        self.assertIn('/var/volatile/owned_attempt/tvm_rpc', command)
        self.assertIn('; then kill $c3_pid;', command)
        self.assertNotIn('killall', command)
        self.assertNotIn('pkill', command)

    def test_readonly_preflight_refuses_external_rpc(self):
        instance = self.make_board()
        raw = "\n".join(["boot01", "operating", "201326592", "f" * 64 + "  /lib/firmware/vta_hpc.bit",
                         "/var/volatile/tmp", "0-3", "tmpfs on /var/volatile type tmpfs",
                         "  123 root /var/volatile/other/tvm_rpc server"])
        instance.ssh = Mock(return_value=raw)
        with self.assertRaisesRegex(ValueError, "EXTERNAL_RPC_PRESENT"):
            instance.preflight()
        self.assertEqual(instance.ssh.call_count, 1)


class ReferenceCaseTests(unittest.TestCase):
    def test_weight_roundtrip_is_observable_and_sentinels_differ_everywhere(self):
        from c3_official_window_reference import array_sha, from_workload

        xshape, wshape = [1, 1, 2, 2, 1, 16], [1, 1, 1, 1, 16, 16]
        yshape, addshape = [1, 1, 2, 2, 1, 16], [1, 1, 1, 1, 16]
        abi = [{"kind": "buffer", "shape": shape, "dtype": dtype, "strides": [], "elem_offset": "0"}
               for shape, dtype in ((xshape, "int8"), (wshape, "int8"),
                                    (addshape, "int32"), (yshape, "int8"))]
        tail = [{"op": "add", "source": {"kind": "buffer", "param": 2}},
                {"op": "shr", "source": {"kind": "imm", "value": 7}},
                {"op": "min", "source": {"kind": "imm", "value": 127}},
                {"op": "max", "source": {"kind": "imm", "value": -127}}]
        spec = from_workload(["conv2d_packed.vta", ["TENSOR", xshape, "int8"],
                              ["TENSOR", wshape, "int8"], [1, 1], [0, 0], [1, 1],
                              "NCHW1n16c", "int32"], abi, tail)
        cases, identities = board.make_cases(spec.contract())
        self.assertEqual(set(cases), {(0, "W0"), (20250901, "W0"), (20260910, "W0"), (0, "W1")})
        for (seed, weight), (arrays, expected) in cases.items():
            self.assertTrue(np.all(arrays[-1] != expected))
            self.assertTrue(np.array_equal(spec.reference(arrays), expected))
            self.assertEqual(identities[str(seed) + ":" + weight]["reference"], array_sha(expected))
        w0, y0 = cases[0, "W0"]
        w1, y1 = cases[0, "W1"]
        self.assertFalse(np.array_equal(w0[1], w1[1]))
        self.assertFalse(np.array_equal(y0, y1))
        self.assertTrue(np.array_equal(w0[0], w1[0]))
        self.assertTrue(np.array_equal(w0[2], w1[2]))


class FailureTests(unittest.TestCase):
    def test_preflight_failure_is_recorded_once_and_no_rpc_target_is_opened(self):
        class Parent(dict):
            def get_global_var(self, name):
                return name

        fake_rpc = SimpleNamespace(connect=Mock(side_effect=AssertionError("RPC must not open")))
        fake_tvm = SimpleNamespace(ir=SimpleNamespace(load_json=lambda value: Parent(sym=None)), rpc=fake_rpc)
        modules = {
            "tvm": fake_tvm,
            "c3_official_window_reference": SimpleNamespace(array_sha=lambda value: "unused"),
            "c3_official_window_method": SimpleNamespace(audit_function=lambda value: {"events": []}),
            "c3_valid_window": SimpleNamespace(certify_trace=lambda events, **kw: {"output_terms": []}),
            "c3_tps_window_native": SimpleNamespace(certify_native=None),
            "c3_tps_window_progress": SimpleNamespace(certify_progress=None,
                LANES=((1, 2), (2, 1), (2, 3), (3, 2))),
            "c3_five_tradeoff_commands": SimpleNamespace(decode_submissions=None),
            "c3_official_window_workflow": SimpleNamespace(native_certificate_for_storage=None),
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "call_manifest.json").write_text("{}")
            (root / "known_hosts").write_text("not a live host")
            (root / "reference.json").write_text("{}")
            (root / "parent.json").write_text("{}")
            (root / "hardware.json").write_text(json.dumps({"fifo_depths": {
                "load_queue": 512, "gemm_queue": 512, "store_queue": 512,
                "l2g_queue": 1024, "g2l_queue": 1024, "g2s_queue": 1024, "s2g_queue": 1024}}))
            instance = SimpleNamespace(root="/var/volatile/mock", session=None,
                                      preflight=Mock(side_effect=ValueError("TEST_PREFLIGHT_STOP")))
            artifacts = {"bitstream": {"sha256": "f" * 64},
                         "programs": {"B": {"reference_contract": str(root / "reference.json"),
                                             "qualified_directory": str(root), "symbol": "sym"}},
                         "hardware_archive": {"path": str(root / "hardware.json")}}
            args = SimpleNamespace(host="invalid.test", known_hosts=str(root / "known_hosts"))
            with patch.dict("sys.modules", modules), patch.object(board, "Board", return_value=instance), \
                    patch.object(board, "make_cases", return_value=({}, {})):
                with self.assertRaisesRegex(ValueError, "TEST_PREFLIGHT_STOP"):
                    board.execute(args, artifacts, {"sessions": [], "calls": []}, root / "board", {})
            result = json.loads((root / "board" / "summary.json").read_text())
            self.assertEqual(result["status"], "STOP_ATTEMPT_FAILED")
            self.assertEqual(result["begun_calls"], 0)
            self.assertFalse(result["performance_conclusion_allowed"])
            self.assertTrue(result["no_retry_or_sample_splicing"])
            self.assertEqual(instance.preflight.call_count, 1)
            fake_rpc.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
