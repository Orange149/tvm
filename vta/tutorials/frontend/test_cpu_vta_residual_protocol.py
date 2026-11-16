import json
from pathlib import Path
import tempfile
import unittest

from cpu_vta_residual_protocol import (
    PROGRAM_NAMES, freeze_protocol, load_protocol, matrix, protocol_paths,
    stages_for, verify_assets, verify_source,
)
from thesis_supplement_run import RunStore, sha256, write_new_json


class ResidualProtocolTests(unittest.TestCase):
    def source(self, parent):
        run = RunStore.create(parent / "source", {
            "run_id": "source", "stage": "06", "campaign_id": "test_only",
            "canonical_qir_sha256": "a" * 64, "bitstream_sha256": sha256(self.binary(parent / "firmware", b"bit")),
            "numeric": {"integer": "exact", "float_atol": 1e-5, "float_rtol": 1e-4},
            "board_host": "test.invalid", "udmabuf_bytes": 201326592,
        })
        r = run.directory

        def put(relative, data):
            p = r / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, dict):
                write_new_json(p, data)
            else:
                p.write_bytes(data)
            return p

        put("summary.json", {"status": "SINGLE_IMAGE_SMOKE_PASS"})
        frontiers = {"cpu_00_02": ([13], [56]), "vta_03_17": ([56], [269]),
                     "cpu_18_20": ([269], [308]), "vta_03_15": ([56], [228, 263]),
                     "cpu_16_20": ([228, 263], [308])}
        rows = []
        for name in PROGRAM_NAMES:
            inputs, outputs = frontiers[name]
            files = {}
            for filename in ("graph.json", "module.so", "params.bin", "lowering_qir.json"):
                p = put(f"raw/c1_components_int8/{name}/{filename}", (name + filename).encode())
                files[filename] = {"sha256": sha256(p), "bytes": p.stat().st_size}
            for index in range(len(inputs)):
                put(f"raw/c1_component_inputs/{name}/input{index}.bin", b"tensor")
            rows.append({"name": name, "device": name.split("_")[0], "files": files,
                         "input_names": [f"edge{i}" for i in range(len(inputs))],
                         "input_node_ids": inputs, "output_node_ids": outputs,
                         "output_sources": [{"kind": "executor_output", "live_input_index": None} for _ in outputs]})
        put("raw/c1_components_int8/component_manifest.json", {
            "canonical_qir_sha256": "a" * 64, "extra_quantize_calls": 0, "programs": rows})
        binary = put("raw/inputs/input.bin", b"input")
        numeric = put("raw/inputs/input.npy", b"numpy input")
        put("analysis/input_contract.json", {"input_sha256": sha256(numeric), "input_bin_sha256": sha256(binary)})
        for name in ("reference_logits.npy", "reference_boundaries.npz"):
            put("raw/host_reference/" + name, b"reference")
        bindings = {}
        for name in ("libtvm_runtime.so", "libvta.so", "tvm_rpc", "vta_hpc.bit", "u-dma-buf.ko"):
            p = self.binary(parent / "external" / name, b"bit" if name.endswith(".bit") else name.encode())
            bindings[name] = {"host_path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
        put("analysis/board_artifact_bindings.json", bindings)
        selected = put("raw/selected_runner", b"original selected executable")
        proxy_runner = put("raw/proxy_runner", b"old proxy executable")
        put("analysis/native_runner_selection.json", {
            "name": "selected_runner", "sha256": sha256(selected),
            "source_C1_runner_selection": {"name": "proxy_runner", "sha256": sha256(proxy_runner)}})
        put("raw/c1_frozen/fair_support_contract.json", {
            "label_free": True, "actual_policy": "ordinary_copy", "support": [{
                "full_identity": {"runner_sha256": sha256(proxy_runner), "canonical_qir_sha256": "a" * 64,
                                  "input_sha256": sha256(numeric)}}]})
        self.seal(run)
        return run

    @staticmethod
    def binary(path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    @staticmethod
    def seal(run):
        files = {str(p.relative_to(run.directory)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                 for p in run.directory.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
        path = run.directory / "artifact_hashes.json"
        path.write_text(json.dumps(files))

    def test_freezes_complete_holdout_frontier_and_new_runtime_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = self.source(parent)
            store = freeze_protocol(source.directory, parent / "run")
            self.assertTrue(verify_assets(store))
            self.assertEqual(load_protocol(store.directory).contract, store.contract)
            holdout = stages_for(store, "03:15")
            self.assertEqual(holdout[1]["output_node_ids"], [228, 263])
            self.assertEqual(holdout[2]["input_node_ids"], [228, 263])
            self.assertEqual(len(holdout[2]["input_files"]), 2)
            self.assertEqual(protocol_paths(store)["input"], store.directory / "raw/inputs/input.bin")
            self.assertEqual(len(store.contract["programs"]), 5)
            self.assertTrue(store.contract["completion"]["execution_context_reset"]["required_before_each_new_command_queue"])
            for identity in json.loads((store.directory / "analysis/board_artifact_bindings.json").read_text()).values():
                self.assertTrue(Path(identity["host_path"]).is_relative_to(store.directory))
            holdout[2]["input_node_ids"].clear()
            self.assertEqual(len(stages_for(store, "03:15")[2]["input_node_ids"]), 2)

    def test_source_asset_drift_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = self.source(parent)
            (source.directory / "raw/inputs/input.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                freeze_protocol(source.directory, parent / "run")
            self.assertFalse((parent / "run").exists())

    def test_current_runtime_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = self.source(parent)
            (parent / "external/libvta.so").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_source(source.directory)

    def test_existing_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = self.source(parent)
            output = parent / "run"
            output.mkdir()
            marker = output / "paid_evidence"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                freeze_protocol(source.directory, output)
            self.assertEqual(marker.read_text(), "keep")

    def test_contract_and_frozen_file_changes_block_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            store = freeze_protocol(self.source(parent).directory, parent / "run")
            protocol_paths(store)["baseline_runner"].write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_protocol(store.directory)
            (store.directory / "contract.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "contract changed"):
                store.verify_contract()

    def test_main_and_holdout_retain_live_input_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = self.source(parent)
            manifest_path = source.directory / "raw/c1_components_int8/component_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            row = next(x for x in manifest["programs"] if x["name"] == "vta_03_15")
            row["output_sources"][0] = {"kind": "live_input", "live_input_index": 0}
            row["output_node_ids"][0] = row["input_node_ids"][0]
            suffix = next(x for x in manifest["programs"] if x["name"] == "cpu_16_20")
            suffix["input_node_ids"][0] = row["input_node_ids"][0]
            manifest_path.write_text(json.dumps(manifest))
            self.seal(source)
            store = freeze_protocol(source.directory, parent / "run")
            self.assertEqual(stages_for(store, "03:15")[1]["output_sources"][0]["kind"], "live_input")

    def test_matrix_values_and_independence(self):
        rows = matrix()
        self.assertEqual(rows[1]["threads"], [4, 1, 1])
        self.assertEqual(rows[3]["affinity"][0], [0, 1, 2])
        self.assertEqual(rows[4]["affinity"][2], [3])
        self.assertEqual(rows[5]["spin_count"], 300000)
        rows[0]["affinity"][0].clear()
        self.assertEqual(matrix()[0]["affinity"][0], [0, 1, 2, 3])

    def test_preregistration_separates_pilot_and_independent_boots(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            store = freeze_protocol(self.source(parent).directory, parent / "run")
            window = store.contract["performance_window"]
            self.assertEqual(window["pipeline_rows"], 307)
            self.assertEqual(window["scored_intervals"], 299)
            self.assertEqual(window["serial_scored_rows"], 15)
            self.assertEqual(store.contract["correctness"]["repeats"], 10)
            replication = store.contract["replication"]
            self.assertEqual(replication["independent_boots_required"], 3)
            self.assertEqual(replication["blocks_per_boot"], 2)
            self.assertFalse(replication["pilot_rows_used_for_replication"])
            self.assertFalse(replication["rpc_restart_is_new_boot"])
            self.assertIsNone(store.contract["selection"]["cpu_stage_preference"])
            self.assertFalse(store.contract["selection"]["fps_used_for_selection"])
            self.assertEqual(store.contract["thresholds"]["timing_telemetry_ii_relative_difference_max"], 0.02)
            self.assertNotIn("unperturbed_runner_ii_relative_difference_max", store.contract["thresholds"])
            self.assertEqual(store.contract["thresholds"]["replicate_median_absolute_effect_fraction_min"], 0.05)
            self.assertNotIn("replicate_same_direction_effect_fraction_each_boot_min", store.contract["thresholds"])
            self.assertEqual(len(store.contract["missing_baselines"]["isolated_missing_components"]), 3)
            self.assertEqual(store.contract["board_state"]["cpu_frequency_khz"],
                             "actual_unknown_until_each_session_preflight")

    def test_missing_deployment_binding_blocks_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            store = freeze_protocol(self.source(parent).directory, parent / "run")
            path = store.directory / "analysis/board_artifact_bindings.json"
            bindings = json.loads(path.read_text())
            bindings.pop("tvm_rpc")
            path.write_text(json.dumps(bindings))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_protocol(store.directory)


if __name__ == "__main__":
    unittest.main()
