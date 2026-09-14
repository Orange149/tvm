# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tune packed conv2d tasks for VTA ResNet on AXU5EVB."""

from __future__ import absolute_import, print_function

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mxnet.gluon.model_zoo import vision

import tvm
from tvm import te, relay, autotvm, topi
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.top import graph_pack


PACK_DICT = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18_v1", choices=sorted(PACK_DICT.keys()))
    parser.add_argument("--tracker-host", default="127.0.0.1")
    parser.add_argument("--tracker-port", type=int, default=9190)
    parser.add_argument("--rpc-key", default="axu5evb")
    parser.add_argument("--log-file", default="vta.resnet18_v1.axu5evb.tuning.log")
    parser.add_argument("--tuner", default="random", choices=["xgb", "ga", "random", "gridsearch"])
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--measure-number", type=int, default=3)
    parser.add_argument("--measure-timeout", type=int, default=120)
    parser.add_argument("--rpc-host", default="", help="Direct RPC server; bypass the tracker")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--measure-repeat", type=int, default=3)
    parser.add_argument("--cases", default="", help="Representative conv case IDs from benchmark_resnet18_single_ops")
    parser.add_argument("--config-indices", default="", help="Optional comma-separated config indices for a single-task smoke test")
    parser.add_argument("--task-indices", default="", help="Optional extracted task indices")
    parser.add_argument("--task-file", default="", help="JSON workload rows emitted by run_stage_tile_iteration")
    parser.add_argument("--artifact-dir", default="", help="Direct measurement artifacts; default: <log-file>.artifacts")
    parser.add_argument(
        "--certificate-dispatch",
        default="",
        help="Hardware-certified dispatch.json; only allow_timing ConfigEntities may be measured",
    )
    parser.add_argument(
        "--command-resource-certificate",
        default="",
        help=(
            "Exact allowlist-derived command-memory certificate. It is validated and recorded "
            "as the deployment runtime contract; it does not mutate an already-running RPC server."
        ),
    )
    return parser.parse_args()


def compile_network(env, model):
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    start_name, stop_name = PACK_DICT[model]
    relay_prog = graph_pack(
        mod["main"],
        env.BATCH,
        env.BLOCK_OUT,
        env.WGT_WIDTH,
        start_name=start_name,
        stop_name=stop_name,
    )
    return relay_prog, params


def register_vta_conv2d_template():
    if getattr(register_vta_conv2d_template, "registered", False):
        return
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "kwargs are not supported in this template"
        a, w, strides, padding, dilation, layout, out_dtype = args

        # Newer Relay may pass 4D padding (top, left, bottom, right).
        # VTA packed conv2d template expects symmetric 2D (h, w).
        if isinstance(padding, (tuple, list)) and len(padding) == 4:
            pt, pl, pb, pr = padding
            if pt == pb and pl == pr:
                padding = (pt, pl)
            else:
                raise ValueError("Asymmetric padding is not supported for conv2d_packed.vta")

        norm_args = (a, w, strides, padding, dilation, layout, out_dtype)

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*norm_args)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            sch = vta.top.schedule_conv2d_packed([res])
        else:
            sch = te.create_schedule([res.op])
        return sch, [a, w, res]

    register_vta_conv2d_template.registered = True


def create_tuner(task, tuner_name):
    if tuner_name == "xgb":
        try:
            return XGBTuner(task, loss_type="reg")
        except ImportError:
            print("[TUNE] xgboost is not installed, fallback to random tuner")
            return RandomTuner(task)
    if tuner_name == "ga":
        return GATuner(task, pop_size=50)
    if tuner_name == "gridsearch":
        return GridSearchTuner(task)
    return RandomTuner(task)


def tune_tasks(tasks, args, measure_option):
    tmp_log = args.log_file + ".tmp"
    if os.path.exists(tmp_log) or os.path.exists(args.log_file):
        raise FileExistsError("Use a new log path; existing tuning evidence is preserved: " + args.log_file)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner = create_tuner(task, args.tuner)
        if os.path.isfile(tmp_log):
            tuner.load_history(autotvm.record.load_from_file(tmp_log))

        n_trial = min(args.trials, len(task.config_space))
        tuner.tune(
            n_trial=n_trial,
            early_stopping=(args.early_stopping or None),
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log),
            ],
        )

    records = list(autotvm.record.load_from_file(tmp_log)) if os.path.exists(tmp_log) else []
    successful = [(inp, result) for inp, result in records if result.error_no == 0
                  and len(result.costs) and all(np.isfinite(x) and x > 0 for x in result.costs)]
    summary = {"measurement_count": len(records), "successful_measurements": len(successful),
               "successful_workloads": len({str(inp.task.workload) for inp, _ in successful}),
               "requested_workloads": len(tasks), "raw_log": tmp_log,
               "best_log": args.log_file, "measurement_path": "direct_rpc" if args.rpc_host else "tracker",
               "certificate_dispatch": str(Path(args.certificate_dispatch).resolve()) if args.certificate_dispatch else None,
               "certificate_dispatch_sha256": getattr(args, "certificate_dispatch_sha256", None),
               "command_resource_certificate": (
                   str(Path(args.command_resource_certificate).resolve())
                   if args.command_resource_certificate else None
               ),
               "command_resource_certificate_sha256": getattr(
                   args, "command_resource_certificate_sha256", None
               ),
               "command_resource_certificate_key": getattr(
                   args, "command_resource_certificate_key", None
               ),
               "deployment_runtime_environment": getattr(
                   args, "deployment_runtime_environment", None
               )}
    Path(args.log_file + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not successful:
        raise RuntimeError("No successful measurements; retained raw log and diagnostics, no best log generated")
    autotvm.record.pick_best(tmp_log, args.log_file)
    print("[TUNE] summary:", summary)


def filter_buildable_tasks(tasks):
    """Keep every workload: one invalid configuration does not invalidate a task.

    Individual configurations are validated by the measurement builder, and failed
    measurements remain in the raw log rather than silently dropping coverage.
    """
    return list(tasks)


def config_signature(entities):
    return tuple((name, tuple(value.size) if hasattr(value, "size") else value.val)
                 for name, value in entities.items())


def canonical_json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def serialized_config_signature(config_entity):
    """Normalize a serialized ConfigEntity without relying on its numeric index."""

    if not isinstance(config_entity, dict) or "entity" not in config_entity:
        raise ValueError("dispatch lacks a complete serialized ConfigEntity")
    normalized = []
    for row in config_entity["entity"]:
        if len(row) != 3:
            raise ValueError("malformed serialized ConfigEntity entry")
        name, kind, value = row
        normalized.append((name, tuple(value) if isinstance(value, list) else value))
    return tuple(normalized)


def load_certified_dispatch(path, expected_workload_id=None, expected_mode=None):
    path = Path(path).resolve()
    dispatch = json.loads(path.read_text(encoding="utf-8"))
    if dispatch.get("schema") not in {
        "c3_vta_hardware_certified_dispatch_v1",
        "c3_vta_hardware_certified_dispatch_v2",
    }:
        raise ValueError("unexpected hardware-certified dispatch schema")
    if dispatch["schema"] == "c3_vta_hardware_certified_dispatch_v2":
        expected_dispatch_id = canonical_json_hash(
            {key: value for key, value in dispatch.items() if key != "dispatch_id"}
        )
        if dispatch.get("dispatch_id") != expected_dispatch_id:
            raise ValueError("hardware-certified dispatch ID mismatch")
    if dispatch.get("performance_labels_used") is not False:
        raise ValueError("dispatch routing must be label-free")
    allowed_rows = [row for row in dispatch["records"] if row["route"] == "allow_timing"]
    if not allowed_rows:
        raise ValueError("dispatch contains no exact hardware-certified configuration to time")
    workload_ids = {row["workload_id"] for row in allowed_rows}
    modes = {row["mode"] for row in allowed_rows}
    if len(workload_ids) != 1 or len(modes) != 1:
        raise ValueError("allow_timing rows must share one workload and residence mode")
    workload_id, mode = next(iter(workload_ids)), next(iter(modes))
    if expected_workload_id is not None and workload_id != expected_workload_id:
        raise ValueError("certificate dispatch workload does not match task file")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("certificate dispatch mode does not match task file")
    return dispatch, allowed_rows, hashlib.sha256(path.read_bytes()).hexdigest()


def load_certified_config_indices(path, expected_workload_id=None, expected_mode=None):
    """Load the pre-measure allowlist; unknown/failed candidates never reach AutoTVM."""

    _, allowed_rows, digest = load_certified_dispatch(
        path, expected_workload_id, expected_mode
    )
    allowed = {int(row["config_index"]) for row in allowed_rows}
    return allowed, digest


def load_certified_candidate_ids(path):
    """Return exact IDs whose hardware certificates allow timing."""

    _, rows, _ = load_certified_dispatch(path)
    missing = [row.get("config_index") for row in rows if not row.get("candidate_id")]
    if missing:
        raise ValueError("allow_timing dispatch rows lack exact candidate IDs: {}".format(missing))
    return {row["candidate_id"] for row in rows}


def certified_config_signatures(path, expected_workload_id=None, expected_mode=None):
    """Load semantic ConfigEntity identities from a v2 dispatch."""

    dispatch, rows, _ = load_certified_dispatch(path, expected_workload_id, expected_mode)
    if dispatch["schema"] != "c3_vta_hardware_certified_dispatch_v2":
        return None
    missing = [row.get("config_index") for row in rows if "complete_config_entity" not in row]
    if missing:
        raise ValueError("v2 dispatch rows lack complete ConfigEntity: {}".format(missing))
    return {serialized_config_signature(row["complete_config_entity"]) for row in rows}


def load_command_resource_contract(path, required_candidate_ids, repo_root=None):
    """Fail closed if a command certificate no longer matches source or allowlist identity."""

    from build_vta_command_resource_certificate import validate_certificate

    path = Path(path).resolve()
    certificate = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_certificate(
        certificate,
        required_candidate_ids=required_candidate_ids,
        repo_root=repo_root,
    )
    validated["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return validated


def task_from_entry(entry, target, target_host):
    workload = entry["workload"]
    mode = entry.get("residence_mode")
    if mode is None or mode == "original_template":
        return autotvm.task.create(
            workload[0], args=workload[1:], target=target, target_host=target_host
        )
    mode_numbers = {
        "original": 0,
        "input_stationary": 1,
        "weight_stationary": 2,
        "paper_inspired_hybrid": 3,
    }
    if mode not in mode_numbers:
        raise ValueError("unsupported residence_mode in task file: " + str(mode))
    import vta.top.vta_conv2d_residency  # pylint: disable=unused-import,import-outside-toplevel

    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (mode_numbers[mode],),
        target=target,
        target_host=target_host,
    )


def representative_tasks(env, case_ids):
    from benchmark_resnet18_single_ops import selected_cases
    cases = selected_cases(case_ids)
    if {c.case_id for c in cases} != set(case_ids.split(",")):
        raise ValueError("Unknown case ID in " + case_ids)
    tasks = []
    for case in cases:
        if case.op_name != "conv2d" or case.channels_in % env.BLOCK_IN or case.channels_out % env.BLOCK_OUT:
            raise ValueError("A packed VTA conv case is required: " + case.case_id)
        a = te.placeholder((1, case.channels_in // env.BLOCK_IN, case.height, case.width,
                            env.BATCH, env.BLOCK_IN), name="data", dtype=env.inp_dtype)
        w = te.placeholder((case.channels_out // env.BLOCK_OUT, case.channels_in // env.BLOCK_IN,
                            case.kernel_h, case.kernel_w, env.BLOCK_OUT, env.BLOCK_IN),
                           name="kernel", dtype=env.wgt_dtype)
        tasks.append(autotvm.task.create("conv2d_packed.vta", args=(a, w,
            (case.stride_h, case.stride_w), (case.pad_h, case.pad_w, case.pad_h, case.pad_w),
            (1, 1), "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), env.acc_dtype),
            target=env.target, target_host=env.target_host))
    return tasks


def main():
    args = parse_args()
    env = vta.get_env()
    target = env.target

    print("========== Tuning Configuration ==========")
    print("env.TARGET      =", env.TARGET)
    print("model           =", args.model)
    print("target          =", target)
    print("tracker_host    =", args.tracker_host)
    print("tracker_port    =", args.tracker_port)
    print("rpc_key         =", args.rpc_key)
    print("tuner           =", args.tuner)
    print("trials/task     =", args.trials)
    print("early_stopping  =", args.early_stopping if args.early_stopping else "<none>")
    print("log_file        =", args.log_file)
    print("==========================================")

    register_vta_conv2d_template()
    task_entries = None
    if args.task_file:
        task_entries = json.loads(Path(args.task_file).read_text())
        tasks = [task_from_entry(entry, target, env.target_host) for entry in task_entries]
    elif args.cases:
        tasks = representative_tasks(env, args.cases)
    else:
        relay_prog, params = compile_network(env, args.model)
        print("[TUNE] extracting packed conv2d tasks ...")
        tasks = autotvm.task.extract_from_program(tvm.IRModule.from_expr(relay_prog), params=params,
            ops=(relay.op.get("nn.conv2d"),), target=target, target_host=env.target_host)
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, tasks))
    if args.task_indices:
        tasks = [tasks[int(i)] for i in args.task_indices.split(",")]
    if args.config_indices:
        if len(tasks) != 1:
            raise ValueError("--config-indices requires exactly one selected task")
        space = tasks[0].config_space
        allowed = {config_signature(space.get(int(i))._entity_map) for i in args.config_indices.split(",")}
        space.multi_filter(lambda entities: config_signature(entities) in allowed)
    certificate_dispatch_sha256 = None
    command_resource_contract = None
    if args.command_resource_certificate and not args.certificate_dispatch:
        raise ValueError("--command-resource-certificate requires --certificate-dispatch")
    if args.certificate_dispatch:
        if args.config_indices:
            raise ValueError("--config-indices cannot bypass --certificate-dispatch")
        if len(tasks) != 1:
            raise ValueError("--certificate-dispatch requires exactly one selected task")
        if task_entries is None or len(task_entries) != 1:
            raise ValueError("--certificate-dispatch requires a single-entry --task-file")
        expected_workload_id = task_entries[0].get("workload_id")
        expected_mode = task_entries[0].get("residence_mode")
        if expected_workload_id is None or expected_mode is None:
            raise ValueError("certified task entry requires workload_id and residence_mode")
        indices, certificate_dispatch_sha256 = load_certified_config_indices(
            args.certificate_dispatch, expected_workload_id, expected_mode
        )
        space = tasks[0].config_space
        semantic_allowed = certified_config_signatures(
            args.certificate_dispatch, expected_workload_id, expected_mode
        )
        if semantic_allowed is None:
            invalid = sorted(index for index in indices if index < 0 or index >= len(space))
            if invalid:
                raise ValueError("certified config indices outside task ConfigSpace: {}".format(invalid))
            allowed = {config_signature(space.get(index)._entity_map) for index in indices}
        else:
            available = {
                config_signature(space.get(index)._entity_map)
                for index in range(len(space))
            }
            missing = semantic_allowed - available
            if missing:
                raise ValueError("certified semantic ConfigEntity is absent from current ConfigSpace")
            allowed = semantic_allowed
        space.multi_filter(lambda entities: config_signature(entities) in allowed)
        print(
            "[TUNE] hardware certificate allowlist: {} configs; dispatch_sha256={}".format(
                len(indices), certificate_dispatch_sha256
            )
        )
        if args.command_resource_certificate:
            command_resource_contract = load_command_resource_contract(
                args.command_resource_certificate,
                load_certified_candidate_ids(args.certificate_dispatch),
                repo_root=Path(__file__).resolve().parents[3],
            )
            print(
                "[TUNE] command-resource certificate: key={} runtime_env={}".format(
                    command_resource_contract["certificate_key"],
                    command_resource_contract["environment"],
                )
            )
    args.certificate_dispatch_sha256 = certificate_dispatch_sha256
    args.command_resource_certificate_sha256 = (
        command_resource_contract["sha256"] if command_resource_contract else None
    )
    args.command_resource_certificate_key = (
        command_resource_contract["certificate_key"] if command_resource_contract else None
    )
    args.deployment_runtime_environment = (
        command_resource_contract["environment"] if command_resource_contract else None
    )
    print("[TUNE] extracted %d tasks" % len(tasks))
    if not tasks:
        raise RuntimeError("No VTA packed conv2d tasks extracted")
    tasks = filter_buildable_tasks(tasks)
    print("[TUNE] retained tasks: %d; configurations are validated individually" % len(tasks))
    if not tasks:
        raise RuntimeError("No buildable VTA conv2d tasks; tuning log would be empty")

    if args.rpc_host:
        from vta_autotvm_measure import VTASequentialBuilder, VTADirectRunner
        artifact_dir = args.artifact_dir or args.log_file + ".artifacts"
        measure_option = autotvm.measure_option(
            builder=VTASequentialBuilder(artifact_dir),
            runner=VTADirectRunner(args.rpc_host, args.rpc_port, args.measure_number,
                                  args.measure_repeat, args.measure_timeout, artifact_dir))
    else:
        measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=autotvm.RPCRunner(
            args.rpc_key,
            host=args.tracker_host,
            port=args.tracker_port,
            number=args.measure_number,
            timeout=args.measure_timeout,
        ))

    tune_tasks(tasks, args, measure_option)
    print("[TUNE] done, best records saved to:", args.log_file)


if __name__ == "__main__":
    main()
