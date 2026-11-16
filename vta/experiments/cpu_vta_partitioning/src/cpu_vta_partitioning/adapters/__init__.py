"""Explicit access to historical sources; never execute legacy campaigns."""


def legacy_graph_class():
    import sys
    from cpu_vta_partitioning.common import REPO
    frontend = str(REPO / "vta/tutorials/frontend")
    if frontend not in sys.path:
        sys.path.insert(0, frontend)
    from thesis_canonical_graph_split import UnitGraph
    return UnitGraph


def legacy_domain(manifest):
    # Only topology identities are imported. Old timing, dtype and ranking fields
    # must never be transferred to the new canonical graph or cost table.
    return [{"name": f"{s['device']}_{s['start_index']:02}_{s['end_index']:02}",
             "device": s["device"], "begin": s["start_index"], "end": s["end_index"] + 1}
            for s in manifest["segments"]]
