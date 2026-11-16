import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest
from cpu_vta_partitioning.common import read,write,REPO
from cpu_vta_partitioning.domain import Stage,topologies
from cpu_vta_partitioning.models import Cost
from multithreading.domain import ENVIRONMENTS,ROOT,PROJECT,partitions,masks,signatures,row,configuration
from multithreading.models import score,load_costs
from multithreading.search import rank,reference,topology_values


def programs_for(cuts=(0,3,8,12,16,21)):
    return {f"{'cpu' if i%2==0 else 'vta'}_{a:02}_{b-1:02}":{
        "name":f"{'cpu' if i%2==0 else 'vta'}_{a:02}_{b-1:02}","device":"cpu" if i%2==0 else "vta",
        "begin":a,"end":b} for i,(a,b) in enumerate(zip(cuts,cuts[1:]))}


def synthetic(programs,ties=False):
    return {key:Cost(50 if ties else 1+(s.end-s.begin)*10/(s.threads if s.device=="cpu" else 5)+sum(s.mask)/16,
                     (s.end-s.begin)*10 if s.device=="cpu" else (s.end-s.begin)/2,
                     .25+s.begin/16) for key,s in signatures(programs).items()}


def test_domain_and_counts():
    ps={p["name"]:p for p in read(PROJECT/"tests/fixtures/full_domain.json")}
    assert len(list(partitions(range(4))))==15
    assert len(list(partitions(range(3))))==5
    assert len(masks(range(4)))==15 and len(masks(range(3)))==7
    assert len(signatures(ps))==1449
    counts={n:0 for n in (2,3,4)}
    for t in topologies(ps): counts[sum(p["device"]=="cpu" for p in t)]+=1
    assert counts=={2:87,3:990,4:3546}
    assert sum(n*15**c for c,n in counts.items())==182877075
    assert sum(n*7**c for c,n in counts.items())==8857779


@pytest.mark.parametrize("environment",ENVIRONMENTS)
@pytest.mark.parametrize("ties",[False,True])
def test_vector_search_matches_independent_scalar(environment,ties):
    ps=programs_for();costs=synthetic(ps,ties)
    actual=rank(ps,costs,environment)
    expected=reference(ps,costs,environment)
    for method,values in expected.items():
        assert [(r["score_ms"],r["id"]) for r in actual["rankings"][method]]==values


def test_cpu_accumulation_boundaries_host_and_single_vta():
    ps=programs_for();env=ENVIRONMENTS[0];t=next(topologies(ps))
    candidate=configuration(t,[(0,),(0,),(1,)],env)
    costs={s.signature:Cost(20 if s.device=="cpu" else 7,10 if s.device=="cpu" else 2,1)
           for s in map(Stage.from_dict,candidate["stages"])}
    result=score(candidate,costs,env)
    assert result["single_core"]==40
    assert result["legacy_independent"]==20
    assert result["components"]["cpu_core_ms"]==33
    assert result["components"]["host_core_ms"]==6
    # Two VTA actors remain ONE physical accelerator, not independent devices.
    for s in candidate["stages"]:
        if s["device"]=="vta": costs[s["signature"]]=Cost(30,2,1)
    assert score(candidate,costs,env)["legacy_independent"]==60


def test_partial_group_overlap_rejected_but_shared_time_slicing_legal():
    ps=programs_for();t=next(topologies(ps))
    configuration(t,[(0,1),(1,2),(0,1,2,3)],ENVIRONMENTS[0])
    with pytest.raises(ValueError,match="partially overlap"):
        configuration(t,[(0,1),(1,2),(3,)],ENVIRONMENTS[0],"resource_group")


def test_fixed_mask_domain_equals_all_physical_set_partitions():
    from multithreading.domain import fixed_compatible
    for cores in (3,4):
        by_partitions={allocation for partition in partitions(range(cores))
                       for allocation in itertools.product(partition,repeat=4)}
        by_masks={allocation for allocation in itertools.product(masks(range(cores)),repeat=4)
                  if fixed_compatible(allocation)}
        assert by_partitions==by_masks


def test_old_or_synthetic_costs_cannot_drive_real_rankings():
    with pytest.raises(ValueError,match="two-boot"):
        load_costs({"source":"synthetic"},"new-bundle")


def oracle_payload(programs,costs,environment):
    options=masks(environment["cpu_cores"])
    out={}
    for name,p in programs.items():
        choices=options if p["device"]=="cpu" else [tuple(environment["host_cores"])]
        stages=[Stage(p["device"],p["begin"],p["end"],len(m) if p["device"]=="cpu" else 1,m) for m in choices]
        out[name]={"cpu":p["device"]=="cpu","keys":[s.signature for s in stages],
                   "costs":[list(costs[s.signature].__dict__.values()) for s in stages]}
    return {"programs":out,"topologies":[[p["name"] for p in t] for t in topologies(programs)],
            "options":[{"mask":sum(1<<c for c in m),"threads":len(m)} for m in options],
            "host_mask":sum(1<<c for c in environment["host_cores"]),"dedicated":environment["name"]=="dedicated3"}


@pytest.mark.full
def test_complete_expanded_domain_against_cpp_scalar_oracle(tmp_path):
    if os.environ.get("CPUVTA_FULL_AUDIT")!="1": pytest.skip("set CPUVTA_FULL_AUDIT=1 for complete local acceptance")
    ps={p["name"]:p for p in read(PROJECT/"tests/fixtures/full_domain.json")}
    costs=synthetic(ps)
    binary=tmp_path/"oracle"
    subprocess.run(["g++","-O3","-std=c++17",str(ROOT/"tests/search_oracle.cc"),
                    "-I"+str(REPO/"3rdparty/picojson"),"-o",str(binary)],check=True)
    audits=[]
    for env in ENVIRONMENTS:
        actual=rank(ps,costs,env)
        result=subprocess.run([str(binary)],input=json.dumps(oracle_payload(ps,costs,env)),text=True,
                              capture_output=True,check=True,timeout=900)
        expected=json.loads(result.stdout)
        assert actual["counts"]==expected["counts"]
        for method,rows in expected["rankings"].items():
            assert [(r["score_ms"],r["id"]) for r in actual["rankings"][method]]==[(r["score_ms"],r["id"]) for r in rows]
        audits.append({"environment":env["name"],"counts":actual["counts"],"all_top20_match_cpp_scalar":True})
    if os.environ.get("CPUVTA_SUPPLEMENT_WORKSPACE"):
        write(Path(os.environ["CPUVTA_SUPPLEMENT_WORKSPACE"])/"plans/full_domain_audit.json",audits,replace=True)
