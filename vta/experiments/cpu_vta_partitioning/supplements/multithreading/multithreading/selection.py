"""Freeze crossovers, interventions and historical configurations before confirmation."""
from cpu_vta_partitioning.domain import Stage
from .domain import configuration,row
from .search import rank


def interventions(environment):
    dedicated = environment["name"]=="dedicated3"
    first = Stage("cpu",0,3,2,(0,1))
    vta = Stage("vta",3,18,1,tuple(environment["host_cores"]))
    return {"overlap":row((first,vta,Stage("cpu",18,21,1 if dedicated else 2,(0,) if dedicated else (0,1)))),
            "separate":row((first,vta,Stage("cpu",18,21,1 if dedicated else 2,(2,) if dedicated else (2,3))))}


def historical(environment):
    if environment["name"]!="shared4": return {}
    def make(parts):
        return row([Stage(dev,b,e,t,tuple(range(t)) if dev=="cpu" else (0,1,2,3))
                    for dev,b,e,t in parts])
    return {"historical_three_islands":make([
        ("cpu",0,3,4),("vta",3,8,1),("cpu",8,10,3),("vta",10,13,1),
        ("cpu",13,15,3),("vta",15,18,1),("cpu",18,21,3)]),
        "historical_one_island":make([("cpu",0,3,4),("vta",3,18,1),("cpu",18,21,2)])}


def freeze(programs,costs,environment,ranking,selection):
    pool = {r["id"]:r for rows in ranking["rankings"].values() for r in rows}
    wanted = {key for picks in selection.values() for key in picks.values()}
    crossover = {}
    topology_names = {}
    for origin,method in (("single_partition","single_core"),("multi_partition","subset_capacity")):
        names = tuple(s["program"] for s in ranking["rankings"][method][0]["stages"])
        topology_names[origin] = list(names)
        local = rank(programs,costs,environment,k=1,topology_filter=names)
        for execution,other in (("single","single_core"),("multi","subset_capacity")):
            r = local["rankings"][other][0]
            pool[r["id"]] = r
            wanted.add(r["id"])
            crossover[origin+"_"+execution] = r["id"]
    ungated = {}
    for budget,key in selection["fixed_groups_serial"].items():
        r = row(pool[key]["stages"],"off")
        pool[r["id"]] = r
        wanted.add(r["id"])
        ungated[budget] = r["id"]
    diagnostic = interventions(environment)
    history = historical(environment)
    legacy = ranking["rankings"]["legacy_independent"][0]
    for r in [*diagnostic.values(),*history.values(),legacy]:
        pool[r["id"]]=r
        wanted.add(r["id"])
    return {"selection":selection,"pool":{k:pool[k] for k in sorted(wanted)},
        "crossovers":crossover,"topologies":topology_names,
        "same_crossover_partition":topology_names["single_partition"]==topology_names["multi_partition"],
        "ungated_serial_selections":ungated,"interventions":{k:r["id"] for k,r in diagnostic.items()},
        "historical":{k:r["id"] for k,r in history.items()},"legacy_independent":legacy["id"]}
