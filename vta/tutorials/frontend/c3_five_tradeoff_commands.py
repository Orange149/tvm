"""Certify actual native runtime encoded queue dependencies, not cycle timing."""

from collections import defaultdict, deque
import re

from c3_thread_sync_audit import instruction_dump


def decode_submissions(text):
    blocks = []
    pattern = r"There are (\d+) instructions\n(.*?)(?=There are \d+ instructions|\Z)"
    for block in re.finditer(pattern, text, re.S):
        records = []
        for item in re.finditer(r"INSTRUCTION (\d+): ([^\n]+)\n(.*?)(?=INSTRUCTION |\Z)", block[2], re.S):
            desc, detail = item[2], item[3]
            dep = re.search(r"dep - pop prev: ([01]), pop next: ([01]), push prev: ([01]), push next: ([01])", detail)
            if not dep and desc != "FINISH":
                raise ValueError("missing dependency bits")
            if desc.startswith("LOAD INP") or desc.startswith("LOAD WGT") or desc == "NOP-MEMORY-STAGE":
                stage = 1
            elif desc.startswith("STORE") or desc == "NOP-STORE-STAGE":
                stage = 3
            elif desc.startswith(("GEMM", "ALU", "LOAD UOP", "LOAD ACC")) or desc in ("FINISH", "NOP-COMPUTE-STAGE"):
                stage = 2
            else:
                raise ValueError("unknown native stage: " + desc)
            records.append({"index": int(item[1]), "description": desc, "detail": detail,
                            "stage": stage, "bits": tuple(map(int, dep.groups())) if dep else (0, 0, 0, 0)})
        if len(records) != int(block[1]):
            raise ValueError("incomplete native dump")
        blocks.append(records)
    if not blocks:
        raise ValueError("no native encoded instructions")
    return blocks


def completion_dag(records):
    edges, last, producers, consumers = defaultdict(set), {}, defaultdict(list), defaultdict(list)
    for i, rec in enumerate(records):
        stage = rec["stage"]
        if stage in last:
            edges[last[stage]].add(i)
        last[stage] = i
        pop_prev, pop_next, push_prev, push_next = rec["bits"]
        for flag, lane, push in ((pop_prev, (stage - 1, stage), False),
                                 (pop_next, (stage + 1, stage), False),
                                 (push_prev, (stage, stage - 1), True),
                                 (push_next, (stage, stage + 1), True)):
            if flag:
                if min(lane) < 1 or max(lane) > 3:
                    raise ValueError("invalid adjacent token lane")
                (producers if push else consumers)[lane].append(i)
    for lane in set(producers) | set(consumers):
        if len(producers[lane]) != len(consumers[lane]):
            raise ValueError("unmatched native tokens on " + str(lane))
        for source, target in zip(producers[lane], consumers[lane]):
            edges[source].add(target)
    indegree = [0] * len(records)
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    queue = deque(i for i, d in enumerate(indegree) if not d)
    ancestors, visited = [0] * len(records), 0
    while queue:
        i = queue.popleft()
        visited += 1
        for target in edges[i]:
            ancestors[target] |= ancestors[i] | (1 << i)
            indegree[target] -= 1
            if not indegree[target]:
                queue.append(target)
    if visited != len(records):
        raise ValueError("native queue/token circular wait")
    return ancestors


def certify_prefetch_dump(text):
    capacity = instruction_dump(text)
    proofs = []
    for records in decode_submissions(text):
        ancestors = completion_dag(records)
        before = lambda a, b: bool(ancestors[b] & (1 << a))
        weights = [r for r in records if r["description"] == "LOAD WGT"]
        macs = [r for r in records if r["description"] == "GEMM"
                and re.search(r"reset_out: 0", r["detail"])]
        resets = [r for r in records if r["description"] == "GEMM"
                  and re.search(r"reset_out: 1", r["detail"])]
        if (len(weights), len(macs), len(resets)) != (40, 112, 112):
            raise ValueError("not fixed complete R12 double-slot invocation")
        base = min(int(re.search(r"DRAM: 0x([0-9a-fA-F]+)", r["detail"])[1], 16) for r in weights)
        by_group, mapping = defaultdict(list), {}
        for r in weights:
            src = int(re.search(r"DRAM: 0x([0-9a-fA-F]+)", r["detail"])[1], 16)
            dst = int(re.search(r"SRAM:0x([0-9a-fA-F]+)", r["detail"])[1], 16)
            if (src - base) % 72 or dst % 72:
                raise ValueError("non-group aligned WGT command")
            group = (src - base) // 72
            slot = group if group < 12 else 12 + (group - 12) % 2
            if not 0 <= group < 16 or dst != slot * 72:
                raise ValueError("actual stream slot/source mapping differs")
            by_group[group].append(r["index"])
            mapping[r["index"]] = (group, slot)
        uses = defaultdict(list)
        windows = []
        for p in range(7):
            for g in range(16):
                load = by_group[g][0 if g < 12 else p]
                mac = macs[p * 16 + g]["index"]
                if not before(load, mac):
                    raise ValueError("WGT LOAD completion not guaranteed before MAC")
                uses[load].append(mac)
            for target, running in ((13, 12), (14, 13), (15, 14)):
                load = by_group[target][p]
                mac = macs[p * 16 + running]["index"]
                if before(mac, load) or before(load, mac):
                    raise ValueError("native encoding serializes claimed independent LOAD/MAC window")
                windows.append({"spatial": p, "load_group": target, "mac_group": running,
                                "load_instruction": load, "mac_instruction": mac})
        previous, overwrites = {}, []
        for rec in weights:
            load = rec["index"]
            group, slot = mapping[load]
            if slot in previous:
                old = previous[slot]
                if not all(before(mac, load) for mac in uses[old]):
                    raise ValueError("native WGT overwrite can race a previous GEMM read")
                overwrites.append({"slot": slot, "old_load": old, "new_load": load,
                                   "last_read": max(uses[old])})
            previous[slot] = load
        inputs = [r["index"] for r in records if r["description"] == "LOAD INP"]
        outputs = [r["index"] for r in records if r["description"].startswith("STORE")]
        if len(inputs) != 56 or len(outputs) != 112:
            raise ValueError("nonweight tile geometry changed")
        for p in range(7):
            tile_macs = [r["index"] for r in macs[p * 16:(p + 1) * 16]]
            if not all(before(inputs[p * 8 + 7], m) for m in tile_macs):
                raise ValueError("input tile not complete before MAC consumers")
            if not all(before(m, outputs[p * 16]) for m in tile_macs):
                raise ValueError("ACC result can be stored before reduction completion")
            if p < 6:
                if not all(before(m, inputs[(p + 1) * 8]) for m in tile_macs):
                    raise ValueError("next input overwrite races last tile MAC")
                if not before(outputs[p * 16 + 15], resets[(p + 1) * 16]["index"]):
                    raise ValueError("ACC RESET races preceding output STORE")
        uop_loads = [r for r in records if r["description"] == "LOAD UOP"]
        if not all(r["stage"] == 2 for r in uop_loads):
            raise ValueError("UOP overwrite is outside Compute FIFO protection")
        proofs.append({"status": "NATIVE_QUEUE_DAG_PASSED", "instructions": len(records),
                       "independent_windows": windows, "overwrite_edges": overwrites,
                       "all_WGT_reads_loaded_and_complete_before_overwrite": True,
                       "INP_ACC_spatial_lifetimes_and_Compute_FIFO_UOP_protection": True,
                       "not_hardware_trace": True})
    return {"status": "NATIVE_PREFETCH_ENCODING_CERTIFIED", "capacity": capacity,
            "invocations": proofs, "hardware_correctness_required": True}
