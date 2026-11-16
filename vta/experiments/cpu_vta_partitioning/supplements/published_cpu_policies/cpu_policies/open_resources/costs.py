"""Matched isolated costs, explicit aliases, and physical-core demand vectors."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from math import isfinite
from numbers import Integral
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

from .domain import CORES, component, component_key, context, digest, stage, validate_candidate


FIELDS = ("service_ms", "run_core_ms", "boundary_core_ms", "set_ms", "run_ms",
          "get_ms", "set_core_ms", "get_core_ms")
NS_PER_MS = 1_000_000
WEIGHT_SCALE = 1_000_000_000_000
MAX_ACCOUNTING_NS = (1 << 63) - 1


def quantize_ns(value_ms):
    """Round a declared millisecond cost once, half-up, to integer nanoseconds.

    Original measured values remain untouched. Search payloads contain these
    integers, avoiding a second floating-point conversion in the C++ oracle.
    """
    if (type(value_ms) not in (int, float) or not isfinite(value_ms)
            or value_ms < 0):
        raise ValueError("invalid millisecond accounting input")
    value = int((Decimal(str(value_ms)) * NS_PER_MS).to_integral_value(rounding=ROUND_HALF_UP))
    if value > MAX_ACCOUNTING_NS:
        raise ValueError("integer nanosecond accounting overflow")
    return value


def integer_weights(weights=None, threads=None):
    """Freeze relative shares as integer units; their sum is the denominator."""
    if threads is None:
        threads = len(weights) if isinstance(weights, (list, tuple)) else None
    if type(threads) is not int or threads < 1:
        raise ValueError("invalid logical thread count for accounting")
    if weights is None:
        return (1,) * threads
    if (not isinstance(weights, (list, tuple)) or len(weights) != threads
            or any(type(w) not in (int, float) or not isfinite(w) or w < 0 for w in weights)
            or abs(sum(weights) - 1.) > 1e-8):
        raise ValueError("invalid normalized logical-thread accounting shares")
    result = tuple(int((Decimal(str(w)) * WEIGHT_SCALE).to_integral_value(rounding=ROUND_HALF_UP))
                   for w in weights)
    if not sum(result):
        raise ValueError("integer logical-thread shares have no weight")
    return result


def allocate_core_ns(run_ns, weights=None, threads=None):
    """Conserve total work exactly using largest remainders, then slot order."""
    return allocate_integer_ns(run_ns, integer_weights(weights, threads))


def allocate_integer_ns(run_ns, shares):
    """Allocate already-frozen integer weight units without re-quantization."""
    if isinstance(run_ns, bool) or not isinstance(run_ns, Integral) or not 0 <= run_ns <= MAX_ACCOUNTING_NS:
        raise ValueError("invalid total integer CPU work")
    run_ns = int(run_ns)
    if (not isinstance(shares, (list, tuple)) or not shares
            or any(isinstance(w, bool) or not isinstance(w, Integral) or w < 0 for w in shares)
            or not sum(shares)):
        raise ValueError("invalid integer logical-thread shares")
    shares = tuple(int(w) for w in shares)
    denominator = sum(shares)
    divisions = [divmod(run_ns * share, denominator) for share in shares]
    answer = [q for q, remainder in divisions]
    remaining = run_ns - sum(answer)
    for slot in sorted(range(len(shares)), key=lambda i: (-divisions[i][1], i))[:remaining]:
        answer[slot] += 1
    return tuple(answer)


def ceil_div_ns(work_ns, capacity):
    if (isinstance(work_ns, bool) or not isinstance(work_ns, Integral) or work_ns < 0
            or isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1):
        raise ValueError("invalid integer resource capacity")
    return (int(work_ns) + int(capacity) - 1) // int(capacity)


def file_sha256(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _validate_cost(value):
    for key in ("service_ms", "run_core_ms", "boundary_core_ms"):
        if key not in value:
            raise ValueError("component missing cost: " + key)
    for key in FIELDS:
        if key in value and (type(value[key]) not in (int, float)
                             or not isfinite(value[key]) or value[key] < 0):
            raise ValueError("invalid isolated cost: " + key)
    for whole, parts in (("service_ms", ("set_ms", "run_ms", "get_ms")),
                         ("boundary_core_ms", ("set_core_ms", "get_core_ms"))):
        if all(k in value for k in parts):
            if abs(value[whole] - sum(value[k] for k in parts)) > 1e-8 * max(1., value[whole]):
                raise ValueError("isolated cost scope inconsistent: " + whole)
    return value


class Costs:
    """Non-prefix masks are inferred only when their measured representative exists."""

    def __init__(self, records, programs=None, ddr=None, homogeneous=True,
                 distribution="balanced"):
        if distribution not in ("balanced", "session_weights"):
            raise ValueError("unknown CPU demand distribution")
        self.records = records.get("costs", records)
        self.programs = programs
        self.ddr = ddr
        self.homogeneous = homogeneous
        self.distribution = distribution
        self.aliases = {}

    def get(self, row, context_value="shared4"):
        if not isinstance(row, dict):
            row = row.to_dict()
        ctx = context(context_value)
        cfg = component(row, ctx)
        requested = component_key(cfg)
        key = requested
        assumptions = []
        if key not in self.records and row["device"] == "cpu" and self.homogeneous:
            source = stage("cpu", row["begin"], row["end"], row["threads"], CORES[:row["threads"]])
            key = component_key(component(source))
            assumptions.append("homogeneous_core_permutation_not_per_mask_measurement")
        if key not in self.records:
            raise ValueError("missing matched isolated component: " + requested)
        record = self.records[key]
        value = deepcopy(_validate_cost(record["cost"]))
        if row["device"] == "vta" and (ctx["reserved_cores"] or ctx["control_mask"] != cfg["control_mask"]):
            assumptions.append("isolated_VTA_has_no_concurrent_producer_or_collector")
        n = row["threads"]
        weights = [1. / n] * n
        if row["device"] == "cpu" and self.distribution == "session_weights":
            observed = record.get("session_thread_weights")
            if observed is not None:
                if (not isinstance(observed, (list, tuple)) or len(observed) != n
                        or any(type(w) not in (int, float) or not isfinite(w) or w < 0 for w in observed)
                        or abs(sum(observed) - 1.) > 1e-8):
                    raise ValueError("invalid normalized whole-session thread shares")
                weights = list(observed)
                assumptions.append("whole_session_thread_shares_rescaled_to_scored_run_core_not_scored_per_thread_measurement")
            else:
                assumptions.append("session_thread_shares_unavailable_uniform_fallback")
        value.update({"source_key": key, "requested_key": requested,
                      "kind": record.get("kind", "measured"), "assumptions": assumptions,
                      "cpu_core_weights": weights})
        if requested != key or assumptions:
            self.aliases[requested] = {"source_key": key, "assumptions": list(assumptions)}
        return value


def ddr_stage_service(ddr, row):
    if ddr is None:
        return 0.
    if hasattr(ddr, "stage"):
        value = ddr.stage(row)
    else:
        table = ddr.get("stage_service_ms", ddr)
        if row["program"] not in table:
            raise ValueError("missing DDR demand for " + row["program"])
        item = table[row["program"]]
        value = item[str(row["threads"])] if isinstance(item, dict) else item
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        raise ValueError("invalid stage DDR service")
    return float(value)


def score_configuration(configuration, costs, ddr=None):
    """A resource lower-bound estimate; no full-pipeline FPS is an input.

    CPU compute is fixed to cores. Host work is ideally movable within its
    allowed mask. Producer/collector cost is explicitly unmodelled, not zero.
    """
    if not isinstance(costs, Costs):
        costs = Costs(costs)
    # Graph sizes are generic here. The caller's protocol separately enforces
    # the frozen maximum island count and qualified program inventory.
    rows = configuration.get("stages", [])
    if not rows:
        raise ValueError("cannot score an empty configuration")
    validate_candidate(configuration, costs.programs, unit_count=rows[-1]["end"],
                       max_vta_islands=max(1, sum(r["device"] == "vta" for r in rows)))
    ddr = costs.ddr if ddr is None else ddr
    demand = [0] * len(CORES)
    cpu_max = vta = cpu_work = host_work = boundary_work = ddr_service = 0
    group_services = {}
    for row in configuration["stages"]:
        value = costs.get(row, configuration)
        service, run, boundary = (quantize_ns(value[k]) for k in ("service_ms", "run_core_ms", "boundary_core_ms"))
        boundary_work += boundary
        ddr_service += quantize_ns(ddr_stage_service(ddr, row))
        if row["device"] == "cpu":
            cpu_max = max(cpu_max, service)
            cpu_work += run
            shares = allocate_core_ns(run, value["cpu_core_weights"], threads=row["threads"])
            for core, work in zip(row["mask"], shares):
                demand[core] += work
            demand[row["mask"][0]] += boundary
            if configuration["execution_mode"] == "resource_group":
                group = ",".join(map(str, row["mask"]))
                group_services[group] = group_services.get(group, 0) + service
        else:
            vta += service
            host_work += run + boundary
    host_bits = sum(1 << c for c in configuration["host_mask"])
    capacities = []
    for subset in range(1, 16):
        fixed = sum(work for core, work in enumerate(demand) if subset & (1 << core))
        movable = host_work if subset & host_bits == host_bits else 0
        capacities.append(ceil_div_ns(fixed + movable, subset.bit_count()))
    score = max(cpu_max, vta, ddr_service, *capacities, *group_services.values())
    return {"score_ms": score / NS_PER_MS, "score_ns": score,
            "cpu_service_max_ms": cpu_max / NS_PER_MS, "cpu_service_max_ns": cpu_max,
            "vta_service_sum_ms": vta / NS_PER_MS, "vta_service_sum_ns": vta,
            "cpu_core_demand_ms": [v / NS_PER_MS for v in demand], "cpu_core_demand_ns": demand,
            "cpu_compute_core_ms": cpu_work / NS_PER_MS, "cpu_compute_core_ns": cpu_work,
            "host_core_ms": host_work / NS_PER_MS, "host_core_ns": host_work,
            "boundary_core_ms": boundary_work / NS_PER_MS, "boundary_core_ns": boundary_work,
            "group_services_ms": {k: v / NS_PER_MS for k, v in group_services.items()},
            "group_services_ns": group_services,
            "subset_capacity_ms": [v / NS_PER_MS for v in capacities], "subset_capacity_ns": capacities,
            "ddr_service_ms": ddr_service / NS_PER_MS, "ddr_service_ns": ddr_service,
            "ddr_available": ddr is not None, "cpu_distribution": costs.distribution,
            "control_cost_model_status": "not_separately_profiled",
            "host_migration_model": "ideal_movable_work_within_allowed_mask",
            "accounting": {"unit": "integer_nanosecond", "ms_scale": NS_PER_MS,
                           "source_rounding": "decimal_string_HALF_UP_once_per_component_field",
                           "CPU_split": "integer_weight_largest_remainder_ties_logical_slot_ascending",
                           "weight_scale": WEIGHT_SCALE, "capacity_rounding": "ceil_once_after_work_sum"}}


def _safe_file(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("unsafe or missing evidence file: " + str(relative))
    return path


def _source_manifest(source_root, source_bundle, target_bundle):
    state = read(source_root / "state.json")
    if source_bundle is not None:
        source = Path(source_bundle).resolve()
    elif target_bundle is not None and (Path(target_bundle) / "plans/source_parent_manifest.json").is_file():
        source = Path(target_bundle).resolve()
    else:
        source = Path(state["bundle"]).resolve()
    archived = source / "plans/source_parent_manifest.json"
    path = archived if archived.is_file() else source / "manifest.json"
    manifest = read(path)
    if digest({k: v for k, v in manifest.items() if k != "bundle_id"})[:24] != manifest["bundle_id"]:
        raise ValueError("source bundle manifest hash differs")
    if manifest["bundle_id"] != state["bundle_id"]:
        raise ValueError("source run/bundle identity differs")
    return state, source, manifest


def _verify_assets(manifest, source_bundle, target_bundle):
    source_bundle = Path(source_bundle)
    destination = Path(target_bundle) if target_bundle is not None else source_bundle
    required = {n: v for n, v in manifest["files"].items()
                if n.startswith(("programs/", "runtime/", "firmware/", "runner/", "inputs/", "references/"))}
    if not required:
        raise ValueError("source execution inventory is empty")
    for name, expected in required.items():
        path = _safe_file(destination, name)
        if file_sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("imported profile execution asset differs: " + name)
    return required


_REPLAY_SCRIPT = r'''
import json,sys
from pathlib import Path
snapshot,source_run,bundle=sys.argv[1:]
sys.path.insert(0,snapshot)
from cpu_policies.pilot_runtime import Evidence
root=Path(source_run)
records=json.loads((root/'costs.json').read_text())['costs']
evidence=Evidence(root,Path(bundle))
count=0
for value in records.values():
    for role in ('qualification','measurement'):
        proof=value[role]
        evidence.summary(proof['summary'],proof['document'],proof['boot_id'])
        count+=1
print(json.dumps({'replayed_evidence':count,'passed':True}))
'''


def import_verified_costs(source_run, target_bundle=None, source_bundle=None):
    """Replay frozen raw evidence; retain old tasks/boots in a new source index.

    A temporary source-bundle view permits relocation: it uses the original
    manifest and controller, while verified execution assets are in the new
    package. The view is never a replacement measurement or a deployed bundle.
    """
    root = Path(source_run).resolve()
    state, source, manifest = _source_manifest(root, source_bundle, target_bundle)
    destination = Path(target_bundle).resolve() if target_bundle is not None else source
    assets = _verify_assets(manifest, source, destination)
    source_costs = read(root / "costs.json")
    records = source_costs.get("costs", {})
    if len(records) != 514 or state.get("status") != "COMPLETE":
        raise ValueError("expected the complete 514-component source pilot")
    snapshot = source / "sources/published_cpu_policies"
    if not (snapshot / "cpu_policies/pilot_runtime.py").is_file():
        raise ValueError("frozen source evidence controller is missing")
    # Verify frozen code actually belongs to the archived source manifest.
    prefix = "sources/published_cpu_policies/"
    for name, expected in manifest["files"].items():
        if name.startswith(prefix):
            if file_sha256(_safe_file(source, name)) != expected["sha256"]:
                raise ValueError("frozen source controller hash differs: " + name)
    with tempfile.TemporaryDirectory(prefix="cpu-vta-profile-proof-") as folder:
        view = Path(folder)
        (view / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name in ("plans", "programs", "runtime", "firmware", "runner", "inputs", "references"):
            path = destination / name
            if path.exists():
                (view / name).symlink_to(path, target_is_directory=True)
        (view / "sources").symlink_to(source / "sources", target_is_directory=True)
        result = subprocess.run([sys.executable, "-c", _REPLAY_SCRIPT, str(snapshot), str(root), str(view)],
                                capture_output=True, text=True, timeout=600,
                                env={**os.environ, "PYTHONNOUSERSITE": "1"})
        if result.returncode:
            raise ValueError("frozen source evidence replay failed:\n" + result.stdout + result.stderr)
        replay = json.loads(result.stdout.strip().splitlines()[-1])
    answer, index = {}, {}
    costs_hash = file_sha256(root / "costs.json")
    for oldkey, record in sorted(records.items()):
        original = record["configuration"]
        row = original["stages"][0]
        if original["affinity_mode"] != ("pinned" if row["device"] == "cpu" else "group"):
            raise ValueError("source has an incompatible CPU binding mode")
        if row["device"] == "vta":
            host = tuple(row["mask"])
            ctx = "shared4" if host == CORES else f"host_shared_{host[0]}" if len(host) == 1 else None
            if ctx is None:
                raise ValueError("source VTA host placement unsupported")
        else:
            ctx = "shared4"
        cfg = component(stage(row["device"], row["begin"], row["end"], row["threads"], row["mask"]), ctx)
        key = component_key(cfg)
        value = _validate_cost(deepcopy(record["cost"]))
        measured = record["measurement"]
        doc = measured["document"]
        if (doc["kind"] != "calibration" or doc["input_indices"] != [0]
                or (doc["warmup"], doc["scored"], doc["tail"]) != (2, 8, 0)
                or doc["executor_sha256"] != read(destination / "plans/executors.json")["policy"]["sha256"]
                or original["execution_mode"] != "off"):
            raise ValueError("source component protocol or executor differs")
        if any(value[k] != measured["summary"][k] for k in FIELDS):
            raise ValueError("source cost differs from replayed measurement")
        if key in answer:
            raise ValueError("duplicate effective source component")
        equivalence = ["source_task_boot_and_raw_evidence_preserved"]
        if row["device"] == "vta" and original["control_mask"] != cfg["control_mask"]:
            equivalence.append("isolated_VTA_has_no_producer_or_collector_control_mask_does_not_change_executed_work")
        provenance = {"source_key": oldkey, "source_bundle_id": manifest["bundle_id"],
                      "source_task_id": doc["task_id"], "source_boot_id": measured["boot_id"],
                      "source_costs_sha256": costs_hash,
                      "source_receipt": deepcopy(measured["summary"]["_receipt"]),
                      "source_configuration": deepcopy(original), "semantic_equivalence": equivalence}
        imported = {"kind": "imported", "configuration": cfg, "cost": value,
                    "measurement": deepcopy(measured), "qualification": deepcopy(record["qualification"]),
                    "provenance": provenance}
        # These are coarse session shares, never scored-window per-thread costs.
        proof = read(_safe_file(root, measured["summary"]["_receipt"]["path"]))
        affinity = [n for n in proof["files"] if Path(n).name == "affinity.json"]
        if row["device"] == "cpu" and len(affinity) == 1:
            report = read(_safe_file(root, affinity[0]))
            entries = report["stages"][0]["effective_affinities"]
            ordered = {tuple(e["cpus"]): e.get("cpu_time_delta_ms", 0.) for e in entries}
            shares = [float(ordered.get((c,), 0.)) for c in row["mask"]]
            if any(not isfinite(v) or v < 0 for v in shares):
                raise ValueError("invalid whole-session thread counters")
            if sum(shares) > 0:
                imported["session_thread_weights"] = [v / sum(shares) for v in shares]
                imported["session_thread_weights_scope"] = report.get("cpu_time_delta_scope")
        answer[key] = imported
        index[key] = provenance
    return {"costs": dict(sorted(answer.items())), "source_index": index,
            "source_costs_sha256": costs_hash, "source_bundle_id": manifest["bundle_id"],
            "source_run_boot_id": state["boot_id"], "verified_execution_assets": assets,
            "evidence_replay": replay, "costs_relabelled_as_new_measurements": False}


def import_profiles(source_root, bundle=None, *, target_bundle=None, source_bundle=None):
    return import_verified_costs(source_root, target_bundle or bundle, source_bundle)["costs"]
