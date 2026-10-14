"""Opt-in board backend; imported only after the explicit resume flag.

All program transitions use the historical clean-bitstream/fresh-RPC boundary.
Failures terminate the attempt; there is no replacement or sample splice path.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
from statistics import mean, median
import tempfile
import traceback

import numpy as np
import tvm
from tvm.contrib import graph_executor

import c3_yolo224_incremental as p
from c3_thread_sync_board import ExperimentBoard
import run_vta_c3_reuse_board as reuse
from run_vta_c3_mode11_graph_context_diagnostic import connect, upload_module, local_params
from run_vta_c3_yolo608_graph_boot import random_input, output_summary
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory


def equal(left, right):
    return len(left) == len(right) and all(
        a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        for a, b in zip(left, right))


def output_sentinel(reference):
    # -113 is a valid clipped output: avoid falsely proving those cells were written.
    return np.where(reference == -113, 112, -113).astype(reference.dtype)


class SessionRunner:
    def __init__(self, args, ledger):
        self.args, self.output, self.ledger = args, args.output_dir, ledger
        self.runtime = p.read(args.runtime_contract)
        self.helper = p.read(args.helper_build_dir / "contract.json")
        self.policy = p.read(args.completion_dir / "completion_policy.json")
        if self.policy.get("status") != "COMPLETION_POLICY_FROZEN" or self.policy.get("mode") != "run_only":
            raise ValueError("run-only completion audit required")
        if (p.sha(self.helper["binary"]) != self.helper["binary_sha256"] or
                self.policy["helper_sha256"] != self.helper["binary_sha256"]):
            raise ValueError("timer helper identity drift")
        for name, path in self.runtime["runtime_paths"].items():
            if p.sha(path) != self.runtime["runtime_sha256"][name]:
                raise ValueError("runtime drift")
        reuse.RUNTIME = self.runtime["runtime_sha256"]
        self.board = ExperimentBoard(args, "/var/volatile/" + self.output.name, args.port)
        self.initial = self.board.state()
        reuse.validate_state(self.initial, exclusive=True)
        self.environment = self.board.environment()
        self.boot = self.initial["boot_id"]
        if args.stage == "graph" and args.boot_index == 2:
            first = p.read(args.boot1 / "summary.json")
            if self.boot == first["boot_id"]:
                raise ValueError("second graph boot must have a different Linux boot ID")
        if args.stage == "qualify" or (args.stage == "graph" and args.boot_index == 1):
            previous = args.profile_result if args.stage == "qualify" else args.qualification
            if p.read(previous / "summary.json")["boot_id"] != self.boot:
                raise ValueError("same-boot prerequisite required")
        p.write_new(self.output / "preflight.json", {"state": self.initial, "environment": self.environment})
        p.write_new(self.output / "runtime_attestation.json", self.board.deploy(self.runtime["runtime_paths"]))
        self.rows, self.sessions = [], []

    def record(self, **row):
        row.update(boot_id=self.boot, invocation_index=len(self.rows),
                   contract_sha256=p.sha(self.args.prepared / "contract.json"),
                   build_ledger=self.ledger, attempt_id=self.output.name)
        self.rows.append(row)
        with (self.output / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

    @contextmanager
    def session(self, variant, graph=None, observe=False):
        state = self.board.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("boot changed during attempt")
        index = len(self.sessions)
        attestation = self.board.reload()
        self.board.port, self.board.pid = self.args.port + 2 * index, None
        remote = None
        refs = []
        try:
            self.board.start()
            remote = connect(self.args.host, self.board.port, self.args.session_timeout)
            inventory, funcs = runtime_inventory(remote)
            current = self.board.environment(require_affinity=True)
            if current["governor"] != self.environment["governor"] or current["boot_id"] != self.boot:
                raise ValueError("environment changed")
            refs.append(load_ephemeral(remote, Path(self.helper["binary"])))
            symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
            for name, observed in symbols.items():
                if Path(observed["dso"]).name != Path(self.policy["runtime_symbols"][name]["dso"]).name:
                    raise ValueError("runtime symbol DSO mismatch")
            with tempfile.TemporaryDirectory(prefix="c3_incremental_") as temporary:
                root = self.args.prepared / variant
                lib = upload_module(remote, root / "graphlib.so", temporary, variant)
                refs.append(lib)
                module = lib
                if observe:
                    refs.append(load_ephemeral(remote, self.args.prepared / "c3_yolo224_graph_observer.so"))
                    inventory_rows = p.graph_inventory(p.read(root / "graph.json"))
                    spec = "\n".join(str(r["node"]) + " " + r["symbol"] for r in inventory_rows if r["wrapped"])
                    module = remote.get_function("c3.incremental.observe_module")(lib, spec)
                    refs.append(module)
                executor = None
                if graph is not False:
                    text = (root / "graph.json").read_text() if graph is None else json.dumps(graph)
                    executor = graph_executor.create(text, module, [remote.ext_dev(0), remote.cpu(0)])
                    executor.load_params((root / "params.bin").read_bytes())
                timer = remote.get_function("c3.v5.graph_run_only_timed_once")
                remote.get_function("c3.v5.reset_counts")()
                yield remote, lib, module, executor, timer, funcs
                counts = json.loads(str(remote.get_function("c3.v5.counts")()))
                self.sessions.append({"index": index, "variant": variant, "observed": observe,
                    "bitstream": attestation, "environment": current, "runtime_symbols": symbols,
                    "helper_counts": counts, "runtime_inventory": inventory})
                p.write_new(self.output / ("session_%03d.json" % index), self.sessions[-1])
                del executor, module, lib
        finally:
            refs.clear()
            remote = None
            gc.collect()
            self.board.stop()
            self.board.pid = None

    def call(self, ctx, seed, phase, variant, block=None, index=0, reference=None, diagnostic=False):
        remote, lib, module, executor, timer, funcs = ctx
        executor.set_input("data", random_input(seed, resolution=224))
        funcs["vta.runtime.profiler_clear"]()
        measured = phase in ("timing", "profile")
        proxy = None
        if diagnostic:
            proxy = module.get_function("c3_observer_begin", False)
            if proxy is not None:
                proxy()
        latency = float(timer(executor.module)) if measured else None
        if not measured:
            executor.run()
        events = None
        if proxy is not None:
            events = json.loads(str(module["c3_observer_end"]()))
        if measured and (not math.isfinite(latency) or latency <= 0):
            raise ValueError("invalid graph duration")
        profile = json.loads(str(funcs["vta.runtime.profiler_status"]()))
        outputs = [np.array(executor.get_output(i).numpy(), copy=True) for i in range(executor.get_num_outputs())]
        correct = reference is None or equal(outputs, reference)
        self.record(phase=phase, variant=variant, block=block, index=index, seed=seed,
                    latency_ms=latency, diagnostic=diagnostic, correct=correct,
                    outputs=output_summary(outputs), events=events, runtime_profile=profile)
        if (not correct or profile.get("driver_timeout_calls", 0) != 0 or profile.get("driver_run_calls", 0) <= 0):
            if reference is not None:
                np.savez_compressed(self.output / "failed_outputs.npz", *outputs)
            raise ValueError("output or device completion check failed")
        return outputs

    def references(self):
        result = {}
        with self.session("stock") as ctx:
            for seed in p.SEEDS:
                result[seed] = self.call(ctx, seed, "qualification", "stock")
                np.savez_compressed(self.output / ("stock_%d.npz" % seed), *result[seed])
        return result

    def finish(self, status, **extra):
        state = self.board.state()
        reuse.validate_state(state, exclusive=True)
        environment = self.board.environment()
        if state["boot_id"] != self.boot or environment["governor"] != self.environment["governor"]:
            raise ValueError("final environment changed")
        p.write_new(self.output / "postflight.json", {"state": state, "environment": environment})
        result = {"status": status, "boot_id": self.boot, "all_correct": True,
                  "contract_sha256": p.sha(self.args.prepared / "contract.json"),
                  "build_ledger": self.ledger, "calls": len(self.rows),
                  "runtime_contract_sha256": p.sha(self.args.runtime_contract),
                  "helper_sha256": self.helper["binary_sha256"], **extra}
        p.write_new(self.output / "summary.json", result)


def profile(r):
    references = r.references()
    for block, enabled in enumerate((False, True, True, False)):
        with r.session("G0", observe=enabled) as ctx:
            r.call(ctx, p.GRAPH_SEEDS[0], "warmup", "G0", block=block,
                   reference=references[p.GRAPH_SEEDS[0]], diagnostic=enabled)
            for i, seed in enumerate(p.GRAPH_SEEDS[:5]):
                r.call(ctx, seed, "profile", "G0", block=block, index=i,
                       reference=references[seed], diagnostic=enabled)
        if r.sessions[-1]["helper_counts"]["graph_calls"] != 5:
            raise ValueError("profile block hidden graph call")
    blocks = {i: median(x["latency_ms"] for x in r.rows if x["phase"] == "profile" and x["block"] == i)
              for i in range(4)}
    perturbation = [blocks[1] / blocks[0] - 1, blocks[2] / blocks[3] - 1]
    observed = [x for x in r.rows if x["phase"] == "profile" and x["diagnostic"]]
    mean_total = mean(x["latency_ms"] for x in observed)
    inventory = p.read(r.args.prepared / "graph_inventory.json")
    nodes = []
    for item in inventory:
        values = [event["host_call_ms"] for row in observed for event in row["events"] or []
                  if event["node"] == item["node"]]
        nodes.append({**item, "median_host_call_ms": median(values) if values else None,
                      "mean_host_call_ms": mean(values) if values else None,
                      "percent_of_observed_graph_mean": 100 * mean(values) / mean_total if values else None,
                      "observed_invocations": len(values), "hardware_time_claim": False})
    residuals = [x["latency_ms"] - sum(e["host_call_ms"] for e in x["events"] or []) for x in observed]
    if any(value < 0 for value in residuals):
        raise ValueError("negative unassigned graph duration")
    p.write_new(r.output / "node_times.json", {"nodes": nodes, "unassigned_median_ms": median(residuals),
        "unassigned_mean_ms": mean(residuals), "observed_graph_mean_ms": mean_total,
        "interpretation": "host function-call wall time; built-in copy and executor overhead remain unassigned"})
    lines = ["# 当前 YOLO-224 图的调用耗时分布", "",
        "下表为诊断开启时的主机函数调用墙钟，不是逐周期硬件耗时。",
        "正式 FPS 只取关闭诊断的独立整图评测。", "",
        "| 节点 | 函数 | 每图调用次数 | 平均 ms | 占同批整图均值 |", "|---|---|---:|---:|---:|"]
    for item in sorted((n for n in nodes if n["mean_host_call_ms"] is not None),
                       key=lambda n: -n["mean_host_call_ms"]):
        lines.append(f"| {item['node']} | {item['symbol']} | 1 | {item['mean_host_call_ms']:.3f} | "
                     f"{item['percent_of_observed_graph_mean']:.2f}% |")
    lines += [f"| — | 未分解：内建复制、执行器与包装开销 | — | {mean(residuals):.3f} | "
              f"{100 * mean(residuals) / mean_total:.2f}% |", "",
              "开关扰动：" + json.dumps(perturbation),
              "超过 2% 时，不能用此表精确分摊未插桩整图时间。"]
    (r.output / "PROFILE_REPORT.md").write_text("\n".join(lines) + "\n")
    r.finish("PROFILE_COMPLETE", perturbation=perturbation,
             quantitative_attribution_allowed=all(abs(x) <= .02 for x in perturbation),
             bottleneck_classification=None)


def qualify(r):
    graph = p.read(r.args.prepared / "G0/graph.json")
    params = local_params(r.args.prepared / "G0")
    snapshots = r.output / "snapshots"
    snapshots.mkdir()
    # Each input and output snapshot is a separate, ordinary prefix invocation.
    # No snapshot is taken after a successor has legally reused its storage.
    for node in p.TARGETS:
        target = graph["nodes"][node]
        xnode, wnode, bnode = [edge[0] for edge in target["inputs"]]
        weight, bias = params[graph["nodes"][wnode]["name"]], params[graph["nodes"][bnode]["name"]]
        xs, ys = {}, {}
        for end, dest in ((xnode, xs), (node, ys)):
            tapped = p.prefix_graph(graph, end, [[end, 0, 0]])
            with r.session("G0", graph=tapped) as ctx:
                for seed in p.SEEDS:
                    values = r.call(ctx, seed, "snapshot", "G0")
                    dest[seed] = values[0]
                    np.save(snapshots / ("node%d_seed%d.npy" % (end, seed)), values[0])
        np.save(snapshots / ("node%d_weight.npy" % node), weight)
        np.save(snapshots / ("node%d_add.npy" % node), bias)
        for variant in ("G0", "G1", "G2"):
            certificate = p.read(r.args.prepared / variant / "fusion_contracts.json")[str(node)]
            if (certificate["address_audit"]["status"] != "passed" or
                    certificate["fusion_semantics_status"] != "FULL_FUSION_CONTRACT_RESOLVED"):
                raise ValueError("full-fusion address audit missing")
            with r.session(variant, graph=False) as ctx:
                remote, lib, _, _, _, funcs = ctx
                dev = remote.ext_dev(0)
                # Reuse the exact handles across the three contents.
                ydev = tvm.nd.array(output_sentinel(ys[p.SEEDS[0]]), dev)
                inputs = [tvm.nd.array(a, dev) for a in (xs[p.SEEDS[0]], weight, bias)]
                for seed in p.SEEDS:
                    source = [xs[seed], weight, bias]
                    for dst, value in zip(inputs, source):
                        dst.copyfrom(value)
                    ydev.copyfrom(output_sentinel(ys[seed]))
                    funcs["vta.runtime.profiler_clear"]()
                    lib[target["attrs"]["func_name"]](*inputs, ydev)
                    actual = ydev.numpy()
                    runtime_profile = json.loads(str(funcs["vta.runtime.profiler_status"]()))
                    good = equal([actual], [ys[seed]]) and equal([a.numpy() for a in inputs], source)
                    np.save(snapshots / ("%s_node%d_seed%d.npy" % (variant, node, seed)), actual)
                    r.record(phase="fused_replay", variant=variant, node=node, seed=seed,
                             latency_ms=None, correct=good, diagnostic=True,
                             inputs_unchanged=equal([a.numpy() for a in inputs], source),
                             sentinel_differs_from_reference_at_every_element=True,
                             fused_program_id=certificate["fused_program_id"], runtime_profile=runtime_profile)
                    if (not good or runtime_profile.get("driver_timeout_calls", 0) != 0
                            or runtime_profile.get("driver_run_calls", 0) <= 0):
                        raise ValueError("full fusion replay failed; no target replacement allowed")
    references = r.references()
    for variant in ("G0", "G1", "G2"):
        with r.session(variant) as ctx:
            for seed in p.SEEDS:
                r.call(ctx, seed, "qualification", variant, reference=references[seed])
    r.finish("FUSION_AND_GRAPH_QUALIFIED", target_nodes=list(p.TARGETS), core_certificate_used=False)


def graph(r):
    references = r.references()
    for variant in ("G0", "G1", "G2"):
        with r.session(variant) as ctx:
            for seed in p.SEEDS:
                r.call(ctx, seed, "qualification", variant, reference=references[seed])
    for block, order in zip(("A", "B"), p.ORDERS[r.args.boot_index]):
        seeds = p.GRAPH_SEEDS[:5] if block == "A" else p.GRAPH_SEEDS[5:]
        for variant in order:
            with r.session(variant) as ctx:
                r.call(ctx, seeds[0], "warmup", variant, block=block, reference=references[seeds[0]])
                for index, seed in enumerate(seeds):
                    r.call(ctx, seed, "timing", variant, block=block, index=index, reference=references[seed])
            if r.sessions[-1]["helper_counts"]["graph_calls"] != 5:
                raise ValueError("hidden timed graph call")
    counts = {phase: sum(row["phase"] == phase for row in r.rows) for phase in ("timing", "warmup", "qualification")}
    if counts != {"timing": 30, "warmup": 6, "qualification": 12}:
        raise ValueError("graph call ledger drift")
    r.finish("GRAPH_COMPLETE", boot_index=r.args.boot_index, phase_counts=counts,
             evaluation=p.evaluation(r.rows, r.args.boot_index))


def run(args):
    if not args.enable_board:
        raise ValueError("board remains paused")
    ledger = p.verify(args.prepared)
    if p.read(args.prepared / "status.json")["status"] != "PREPARED_BOARD_PAUSED":
        raise ValueError("local builds incomplete")
    contract = p.read(args.prepared / "contract.json")
    for name, expected in contract["source_hashes"].items():
        if p.sha(Path(__file__).parent / name) != expected:
            raise ValueError("runner code changed after freeze")
    p.verify(args.helper_build_dir)
    p.verify(args.completion_dir)
    prior = args.profile_result if args.stage == "qualify" else args.qualification if args.stage == "graph" else None
    if args.stage in ("qualify", "graph"):
        if prior is None:
            raise ValueError("missing prerequisite evidence")
        p.verify(prior)
        cert = p.read(prior / "summary.json")
        expected = "PROFILE_COMPLETE" if args.stage == "qualify" else "FUSION_AND_GRAPH_QUALIFIED"
        if cert.get("status") != expected or cert["build_ledger"] != ledger or not cert["all_correct"]:
            raise ValueError("prerequisite certificate not bound to these binaries")
        if (cert["runtime_contract_sha256"] != p.sha(args.runtime_contract) or
                cert["helper_sha256"] != p.read(args.helper_build_dir / "contract.json")["binary_sha256"]):
            raise ValueError("prerequisite runtime/helper identity drift")
    if args.stage == "graph" and args.boot_index == 2:
        if args.boot1 is None:
            raise ValueError("missing first-boot gate")
        p.verify(args.boot1)
        first = p.read(args.boot1 / "summary.json")
        if first.get("status") != "GRAPH_COMPLETE" or first["build_ledger"] != ledger or not first["evaluation"]["request_second_boot"]:
            raise ValueError("second boot not eligible")
        if (first["runtime_contract_sha256"] != p.sha(args.runtime_contract) or
                first["helper_sha256"] != p.read(args.helper_build_dir / "contract.json")["binary_sha256"]):
            raise ValueError("second boot runtime/helper identity drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = None
    try:
        runner = SessionRunner(args, ledger)
        {"profile": profile, "qualify": qualify, "graph": graph}[args.stage](runner)
    except BaseException as error:
        p.write_new(args.output_dir / "invalid_session.json", {"status": "INVALID_NON_SPLICEABLE",
            "message": str(error), "traceback": traceback.format_exc(), "all_timing_invalid": True})
        raise
    finally:
        if runner is not None:
            runner.board.stop()
        p.seal(args.output_dir)
