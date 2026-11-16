"""Local schedule/descriptor checks; never instantiate a board runner."""

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import c3_valid_window_board as board


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261014_c3_valid_window_local_run01"


class FixedBoardProtocol(unittest.TestCase):
    def test_full_fixed_orders_and_call_ledgers(self):
        result = board.plan()
        self.assertEqual(result["orders"],[list(row) for row in board.ORDERS])
        self.assertEqual(result["seed_orders"],[list(row) for row in board.SEED_ORDERS])
        self.assertEqual(result["expected_calls"],{"correctness":15,"new_Q_weight_ABA":12,
            "diagnostic_prefix":45,"diagnostic_positions_4_5_6":45,
            "formal_explicit_warmup":90,"formal_timing":90})
        self.assertEqual(result["formal_round_to_diagnostic_rotation"],[0,1,2,0,1,2])

    def test_dedup_preserves_first_occurrence_without_replacement(self):
        result = board.plan({"Q4":"Q2"})
        for original,reduced in zip(board.ORDERS,result["orders"]):
            expected = list(dict.fromkeys("Q2" if name == "Q4" else name for name in original))
            self.assertEqual(reduced,expected)
        self.assertEqual(result["expected_calls"]["formal_timing"],72)
        self.assertEqual(result["expected_calls"]["diagnostic_prefix"],36)

    def test_cyclic_alias_and_TPS_replacement_rejected(self):
        for aliases in ({"Q1":"Q2","Q2":"Q1"},{"TPS":"Q8"}):
            with self.assertRaises(ValueError):
                board.plan(aliases)

    def test_expected_trace_filters_events_without_losing_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"trace.json"
            kernel = {"kind":"kernel","cache_identity":{"scope":1,"signature":[7]},"uops":[]}
            p.write_text(json.dumps([{"kind":"load"},kernel,{"kind":"sync"}]))
            self.assertEqual(board.expected_kernels({"expected_trace":{"path":str(p)}}),[kernel])

    def test_physical_DDR_relocation_keeps_descriptor_identity(self):
        text = (ROOT/"sealed_tps_replay_attempt01/native_call_01.txt").read_text()
        moved = re.sub(r"DRAM: 0x([0-9a-f]+)",lambda m:"DRAM: 0x%08x" % (int(m[1],16)+8192),text)
        self.assertEqual(board.command_fingerprint(text),board.command_fingerprint(moved))

    def test_dma_shape_or_token_change_is_detected(self):
        text = (ROOT/"sealed_tps_replay_attempt01/native_call_01.txt").read_text()
        original = board.command_fingerprint(text)
        changed_shape = re.sub(r"(x: size=\d+, stride=)(\d+)",lambda m:m[1]+str(int(m[2])+1),text,count=1)
        changed_token = text.replace("dep - pop prev: 0","dep - pop prev: 1",1)
        self.assertNotEqual(original,board.command_fingerprint(changed_shape))
        self.assertNotEqual(original,board.command_fingerprint(changed_token))

    def test_old_pool_is_rejected_before_hardware_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root/"pool.json"; pool.write_text(json.dumps({"status":"OLD_SIX_PROGRAM_POOL","TPS_modified":False}))
            arm = root/"arm.json";arm.write_text("{}")
            runtime = root/"runtime.json";runtime.write_text("{}")
            document = root/"plan.md";document.write_text("local fixture")
            with patch.object(board,"_base",side_effect=AssertionError("hardware dependencies must stay unloaded")):
                with self.assertRaisesRegex(ValueError,"fixed, fully gated local Q pool"):
                    board.freeze(root,pool,arm,runtime,document)

    def test_readiness_requires_owned_socket_and_current_boot(self):
        class FakeBoard:
            runtime,port = "/var/volatile/owned-test",9090
            def ssh(self,command,timeout):
                self.command,self.timeout = command,timeout
                return "BOOT same-boot\nOWNED 123\nLISTEN 123 456\n"
        fake = FakeBoard()
        result = board._wait_owned_rpc(fake,"same-boot")
        self.assertEqual(result["status"],"OWNED_RPC_LISTEN_READY")
        self.assertEqual(result["connect_attempts_so_far"],0)
        self.assertIn("/proc/net/tcp",fake.command)
        self.assertIn("socket:[$c3_ready_inode]",fake.command)
        self.assertLessEqual(fake.timeout,8.0)
        with self.assertRaisesRegex(ValueError,"boot changed"):
            board._wait_owned_rpc(fake,"different-boot")

    def test_native_gate_failure_writes_one_final_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = ROOT/"sealed_tps_replay_attempt01"
            history = ROOT.parent/"20261013_c3_banerjee_baseline_completion_run02"
            (root/"dump.txt").write_text((source/"native_call_01.txt").read_text())
            saved = []
            runner = board.Runner.__new__(board.Runner)
            runner.boot,runner.out,runner.rows = "local-test",root,[]
            runner._defer_record,runner._pending_record = False,None
            runner._cache_definitions = {}
            runner._base = SimpleNamespace(append=lambda path,row:saved.append((Path(path).name,copy.deepcopy(row))))
            runner.prepared = {"programs":{"TPS":{"expected_trace":{"path":str(history/"local_final06/TPS-oc-reuse-ci4/address_audit.json")},
                "command_fingerprints":{"first":"wrong-fingerprint","warmed":[]}}}}
            def raw(*args,**kwargs):
                row = {"program":"TPS","session_id":"local-test","correct":True,"arguments_unchanged":True,
                       "dump_file":"dump.txt","dump_sha256":"fixture","call_position":1}
                return runner.record(row)
            runner._raw_call = raw
            with self.assertRaisesRegex(ValueError,"normalized actual command"):
                runner.call()
            calls = [row for name,row in saved if name == "calls.jsonl"]
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]["status"],"FAIL_NATIVE_OR_EXECUTION")
            self.assertEqual(calls[0]["native_gate_status"],"FAIL")
            self.assertTrue(calls[0]["correct"])
            self.assertEqual(len(runner.rows),1)
            self.assertIsNone(runner._pending_record)

    def test_unproven_ownership_never_stops_reads_or_removes(self):
        class UntouchedBoard:
            def stop(self):
                raise AssertionError("foreign process must stay untouched")
            def ssh(self,*args,**kwargs):
                raise AssertionError("unowned directory must stay untouched")
        saved,sessions = [],[]
        data = {"runtime_directory":"/var/volatile/c3_tradeoff_existing", "owned_runtime_directory":False}
        board._cleanup_owned(UntouchedBoard(),data,Path("/tmp/local-no-write"),sessions,
            data["runtime_directory"],"existing",lambda path,row:saved.append(copy.deepcopy(row)))
        self.assertEqual(data["cleanup_status"],"SKIPPED_UNPROVEN_DIRECTORY_OWNERSHIP_RETAINED")
        self.assertEqual(saved,[data])
        self.assertEqual(sessions,[data])

    def test_fresh_inventory_binds_capacity_replay_and_no_prior_work(self):
        inventory = {"queue_capacity_initial":{"insn_capacity_bytes":33554432,"uop_capacity_bytes":33554432,
            "insn_submit_threshold_bytes":33554432,"insn_submit_threshold_configured":True,
            "command_manifest_id":"","replay_policy":"disabled","submissions":0},
            "replay_initial":{"mode":"disabled","template_count":0},
            "profiler_initial":{"driver_run_calls":0}}
        functions = ["vta.runtime."+name for name in
            ("profiler_clear","profiler_status","queue_capacity_status","replay_status")]
        board._verify_inventory(inventory,functions)
        for path,value in ((('queue_capacity_initial','uop_capacity_bytes'),8192),
                           (('queue_capacity_initial','command_manifest_id'),'existing'),
                           (('queue_capacity_initial','replay_policy'),'enabled'),
                           (('replay_initial','template_count'),1),
                           (('profiler_initial','driver_run_calls'),1)):
            changed = copy.deepcopy(inventory)
            changed[path[0]][path[1]] = value
            with self.assertRaises(ValueError):
                board._verify_inventory(changed,functions)
        with self.assertRaisesRegex(ValueError,"introspection"):
            board._verify_inventory(inventory,functions[:-1])


if __name__ == "__main__":
    unittest.main()
