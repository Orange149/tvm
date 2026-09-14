"""Sequential, correctness-checked AutoTVM measurement on the existing VTA RPC server.

This adapter uses the same VTA passes and host cross compiler as native deployment.
It deliberately avoids tracker setup, on-board compilation, and random-fill RPC hooks.
"""

import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.autotvm.measure.measure import Builder, Runner, MeasureErrorNo, MeasureResult
from tvm.autotvm.measure.measure_methods import BuildResult
from tvm.contrib import cc
import vta


def build_config_module(measure_input):
    target, task, config = measure_input
    with target:
        schedule, tensors = task.instantiate(config)
    if not config.valid():
        raise RuntimeError("Invalid configuration: {}".format(config.errors))
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = tvm.build(
            schedule, tensors,
            target=tvm.target.Target(target, host=task.target_host or vta.get_env().target_host),
            name="main",
        )
    return module, tuple((tuple(int(x) for x in t.shape), t.dtype) for t in tensors)


class VTASequentialBuilder(Builder):
    """A synchronous builder; timeout is not a process-level compilation watchdog."""

    def __init__(self, artifact_dir, verbose_errors=True):
        super().__init__(n_parallel=1)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.verbose_errors = bool(verbose_errors)

    def build(self, measure_inputs):
        results = []
        for inp in measure_inputs:
            start = time.time()
            try:
                module, arg_info = build_config_module(inp)
                directory = Path(tempfile.mkdtemp(prefix="config_", dir=self.artifact_dir))
                path = directory / "kernel.so"
                sysroot = os.environ["SDKTARGETSYSROOT"]
                module.export_library(
                    str(path), fcompile=cc.cross_compiler("aarch64-xilinx-linux-g++"),
                    options=["--sysroot=" + sysroot,
                             "-Wl,-rpath-link," + sysroot + "/lib",
                             "-Wl,-rpath-link," + sysroot + "/usr/lib",
                             "-L" + sysroot + "/lib", "-L" + sysroot + "/usr/lib"],
                )
                results.append(BuildResult(str(path), arg_info, None, time.time() - start))
            except Exception:
                error = traceback.format_exc()
                print(error if self.verbose_errors else error.splitlines()[-1], flush=True)
                results.append(MeasureResult((error, "VTA build failed"), MeasureErrorNo.COMPILE_HOST,
                                             time.time() - start, time.time()))
        return results


def reference_data(task, seed=0):
    """Independent int32 NCHW convolution reference for the tuning template."""
    task_args = task.args
    if task.name == "conv2d_packed_residency.vta":
        task_args = task_args[:-1]
    data_arg, weight_arg, strides, padding, dilation, _, out_dtype = task_args
    if tuple(dilation) != (1, 1) or out_dtype != "int32":
        raise ValueError("Smoke reference supports int8 convolution with dilation=1 only")
    ds, ws = tuple(data_arg[1]), tuple(weight_arg[1])
    if data_arg[2] != "int8" or weight_arg[2] != "int8":
        raise ValueError("Smoke reference requires int8 inputs and weights")
    nb, ci, h, w, batch, bi = ds
    co, ciw, kh, kw, bo, biw = ws
    if ci != ciw or bi != biw:
        raise ValueError("Packed channel mismatch")
    rng = np.random.default_rng(seed)
    data = rng.integers(-32, 33, ds, dtype=np.int8)
    weights = rng.integers(-8, 9, ws, dtype=np.int8)
    nchw = data.transpose(0, 4, 1, 5, 2, 3).reshape(nb * batch, ci * bi, h, w)
    kernel = weights.transpose(0, 4, 1, 5, 2, 3).reshape(co * bo, ci * bi, kh, kw)
    if len(padding) == 2:
        pt, pl = padding
        pb, pr = pt, pl
    else:
        pt, pl, pb, pr = padding
    padded = np.pad(nchw.astype("int32"), ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::strides[0], ::strides[1], :, :]
    accum = np.einsum("ncyxij,ocij->noyx", windows, kernel.astype("int32"), optimize=True)
    result = np.clip(accum >> 8, 0, 127).astype("int8")
    oh, ow = result.shape[2:]
    packed = result.reshape(nb, batch, co, bo, oh, ow).transpose(0, 2, 4, 5, 1, 3)
    return data, weights, np.ascontiguousarray(packed)


class VTADirectRunner(Runner):
    """Use deterministic host inputs and validate every measured configuration."""

    def __init__(self, host, port, number=1, repeat=3, timeout=120, artifact_dir=None,
                 correctness_seeds=(0,), before_measure=None, verbose_errors=True):
        super().__init__(timeout=timeout, n_parallel=1)
        self.host, self.port = host, port
        self.number, self.repeat = number, repeat
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.correctness_seeds = tuple(int(seed) for seed in correctness_seeds)
        if not self.correctness_seeds:
            raise ValueError("at least one correctness seed is required")
        self.references = None
        self.before_measure = before_measure
        self.verbose_errors = bool(verbose_errors)

    def set_task(self, task):
        super().set_task(task)
        self.references = {
            seed: reference_data(task, seed=seed) for seed in self.correctness_seeds
        }

    def get_build_kwargs(self):
        return {}

    def run(self, measure_inputs, build_results):
        results = []
        for inp, built in zip(measure_inputs, build_results):
            if isinstance(built, MeasureResult):
                results.append(built)
                continue
            start = time.time()
            detail = {"workload": inp.task.workload, "config": inp.config.to_json_dict(),
                      "host": self.host, "port": self.port}
            try:
                if self.before_measure is not None:
                    detail["clean_start"] = self.before_measure()
                remote = rpc.connect(self.host, self.port, session_timeout=self.timeout)
                remote.upload(built.filename)
                module = remote.load_module(Path(built.filename).name)
                dev = remote.ext_dev(0)
                correctness = []
                buffers = None
                for seed in self.correctness_seeds:
                    data, weights, expected = self.references[seed]
                    buffers = [tvm.nd.array(data, dev), tvm.nd.array(weights, dev),
                               tvm.nd.empty(expected.shape, "int8", dev)]
                    module["main"](*buffers)
                    actual = buffers[-1].numpy()
                    mismatch = int(np.count_nonzero(actual != expected))
                    correctness.append({
                        "seed": seed,
                        "correct": mismatch == 0,
                        "mismatch_count": mismatch,
                        "output_range": [int(actual.min()), int(actual.max())],
                        "expected_range": [int(expected.min()), int(expected.max())],
                    })
                    if mismatch:
                        raise AssertionError(
                            "seed {}: {} incorrect elements; max absolute error {}".format(
                                seed, mismatch,
                                np.max(np.abs(actual.astype("int32") - expected.astype("int32"))),
                            )
                        )
                detail["correctness"] = correctness
                detail["correctness_seeds"] = list(self.correctness_seeds)
                detail["output_range"] = correctness[0]["output_range"]
                detail["expected_range"] = correctness[0]["expected_range"]
                costs = module.time_evaluator("main", dev, number=self.number,
                                              repeat=self.repeat)(*buffers).results
                if not all(np.isfinite(x) and x > 0 for x in costs):
                    raise RuntimeError("Invalid timing samples: {}".format(costs))
                detail.update(correct=True, costs_s=list(costs))
                try:
                    clear = remote.get_function("vta.runtime.profiler_clear")
                    status = remote.get_function("vta.runtime.profiler_status")
                except AttributeError:
                    detail["runtime_profile"] = None
                else:
                    clear()
                    module["main"](*buffers)
                    detail["runtime_profile"] = json.loads(status())
                result = MeasureResult(costs, MeasureErrorNo.NO_ERROR,
                                       built.time_cost + time.time() - start, time.time())
                print("[MEASURE] config={} correct=True median_ms={:.6f}".format(
                    inp.config.index, float(np.median(costs)) * 1000), flush=True)
            except Exception as err:
                error = traceback.format_exc()
                detail.update(correct=False, error=error)
                code = MeasureErrorNo.WRONG_ANSWER if isinstance(err, AssertionError) else MeasureErrorNo.RUNTIME_DEVICE
                result = MeasureResult((error, str(err)), code, built.time_cost + time.time() - start, time.time())
                print(error if self.verbose_errors else error.splitlines()[-1], flush=True)
            if self.artifact_dir:
                # Each compiled artifact has its own directory, including across tasks.
                (Path(built.filename).parent / "measurement.json").write_text(
                    json.dumps(detail, indent=2) + "\n", encoding="utf-8")
            results.append(result)
        return results
