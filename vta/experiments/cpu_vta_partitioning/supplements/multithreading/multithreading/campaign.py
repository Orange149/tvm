"""Ten actual kernel boots, two environments in each screening/confirmation boot."""
from collections import defaultdict
from pathlib import Path
import random
import statistics

from cpu_vta_partitioning.common import read,write,locked,event,sha256,verify_inventory
from cpu_vta_partitioning.campaign import measured,retry_cleanup
from cpu_vta_partitioning.domain import Stage,topologies
from .domain import ENVIRONMENTS,ENV_NAMES,METHODS,protocol,signatures,configuration,task,task_for,validate_row
from .evidence import Evidence
from .models import load_costs
from .search import rank,screening_pool,select
from .selection import freeze


def phase_at(index):
    if not 0<=index<=10: raise ValueError("invalid boot index")
    return "calibration" if index<2 else "screening" if index<4 else "confirmation" if index<10 else "complete"


def validate_state(root):
    from .prepare import identity,validate
    root = Path(root)
    state = read(root/"state.json")
    if state["experiment"]!=protocol()["experiment"] or state["controller_identity"]!=identity():
        raise ValueError("supplement controller identity mismatch; preserve the old batch and prepare a new one")
    manifest = validate(state["bundle"])
    if manifest["bundle_id"]!=state["bundle_id"]:
        raise ValueError("run/bundle identity mismatch")
    if len(state["completed_boots"])!=state["boot_index"] or len(set(state["completed_boots"]))!=state["boot_index"]:
        raise ValueError("independent boot history mismatch")
    verify_inventory(root,state.get("frozen_files",{}))
    return state


def pin(root,state,paths):
    """Freeze development inputs before any later data can influence selection."""
    frozen = state.setdefault("frozen_files",{})
    for path in paths:
        path = Path(path)
        key = str(path.relative_to(root))
        record = {"sha256":sha256(path),"bytes":path.stat().st_size}
        if key in frozen and frozen[key]!=record: raise ValueError("frozen development artifact changed: "+key)
        frozen[key] = record
    write(Path(root)/"state.json",state,replace=True)


def create(bundle,root,connection,*,excluded_boot=None):
    from .prepare import validate,identity
    root = Path(root).resolve()
    manifest = validate(bundle)
    if root.exists(): raise FileExistsError("run already exists; resume it")
    root.mkdir(parents=True)
    state = {"experiment":protocol()["experiment"],"bundle":str(Path(bundle).resolve()),
        "bundle_id":manifest["bundle_id"],"controller_identity":identity(),"connection":connection,
        "boot_index":0,"completed_boots":[],"active_boot":None,"status":"READY",
        "excluded_preflight_boot":excluded_boot}
    write(root/"state.json",state)
    return state


def anchor(programs,environment):
    topology = min(topologies(programs),key=lambda t:(len(t),tuple(p["name"] for p in t)))
    return configuration(topology,[(0,)]*sum(p["device"]=="cpu" for p in topology),environment)


def common(index):
    return {"campaign_phase":phase_at(index),"global_boot_index":index}


def warmup(root,environment):
    return read(Path(root)/environment["name"]/"warmup.json")["warmup"]


def validate_warmup(root,environment,programs,evidence):
    value = read(Path(root)/environment["name"]/"warmup.json")
    for name,count in (("short",20),("long",100)):
        document = task_for("warmup_check",anchor(programs,environment),environment,warmup=count,**common(0))
        evidence.summary(value[name],document,value["boot_id"])
    expected = 100 if abs(value["short"]["ii_ms"]/value["long"]["ii_ms"]-1)>.02 else 20
    if value["warmup"]!=expected: raise ValueError("warmup rule mismatch")
    return value


def pool_for(root,phase,environment,programs):
    envdir = Path(root)/environment["name"]
    if phase=="calibration":
        return {k:s for k,s in signatures(programs).items()}
    if phase=="screening": return screening_pool(read(envdir/"ranking.json"))
    return read(envdir/"frozen.json")["pool"]


def measurement_task(root,phase,environment,key,value,labels):
    if phase=="calibration":
        return task("calibration",[value],ENVIRONMENTS[0],warmup=5,scored=32,tail=0,
                    indices=list(range(16)),signature=key,**labels)
    return task_for(phase,value,environment,warmup=warmup(root,environment),**labels)


def anchor_task(root,programs,environment,position,labels):
    return task_for("anchor",anchor(programs,environment),environment,
                    warmup=warmup(root,environment),position=position,**labels)


def validate_round(root,path,programs,evidence):
    record = read(path)
    environment = next(e for e in ENVIRONMENTS if e["name"]==record["environment"])
    index,round_index,attempt = record["global_boot_index"],record["round"],record["attempt"]
    expected_path = Path(root)/environment["name"]/phase_at(index)/f"boot{index}"/f"round{round_index}.json"
    if Path(path)!=expected_path or round_index not in (0,1) or attempt not in range(3):
        raise ValueError("round identity mismatch")
    boot = record["boot_id"]
    labels = {**common(index),"round":round_index,"attempt":attempt}
    pool = pool_for(root,phase_at(index),environment,programs)
    if set(record["measurements"])!=set(pool): raise ValueError("round pool mismatch")
    for position in ("before","after"):
        evidence.summary(record[position],anchor_task(root,programs,environment,position,labels),boot)
    if abs(record["after"]["ii_ms"]/record["before"]["ii_ms"]-1)>.05:
        raise ValueError("round drift exceeds frozen limit")
    for key,value in pool.items():
        evidence.summary(record["measurements"][key],measurement_task(root,phase_at(index),environment,key,value,labels),boot)
    return record


def rounds(root,phase,environment,programs,evidence):
    indices = range(2) if phase=="calibration" else range(2,4) if phase=="screening" else range(4,10)
    result,boots = [],set()
    wanted = {Path(root)/environment["name"]/phase/f"boot{i}"/f"round{r}.json" for i in indices for r in (0,1)}
    if set((Path(root)/environment["name"]/phase).glob("boot[0-9]/round*.json"))!=wanted:
        raise ValueError("phase lacks the complete preregistered rounds")
    for i in indices:
        pair = [validate_round(root,Path(root)/environment["name"]/phase/f"boot{i}"/f"round{r}.json",programs,evidence) for r in (0,1)]
        boot = pair[0]["boot_id"]
        if pair[1]["boot_id"]!=boot or boot in boots: raise ValueError("duplicate/mismatched independent boot")
        boots.add(boot)
        result.extend(pair)
    return result


def grouped(records):
    result = defaultdict(lambda:defaultdict(list))
    for r in records:
        for key,value in r["measurements"].items(): result[key][r["boot_id"]].append(value["ii_ms"])
    return {key:{boot:statistics.mean(v) for boot,v in bs.items()} for key,bs in result.items()}


def run_round(root,index,environment,round_index,programs,execute,evidence,boot):
    phase = phase_at(index)
    directory = Path(root)/environment["name"]/phase/f"boot{index}"
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/f"round{round_index}.json"
    if path.exists(): return validate_round(root,path,programs,evidence)
    pool = pool_for(root,phase,environment,programs)
    for attempt in range(3):
        bad = directory/f"invalid-round{round_index}-attempt{attempt}.json"
        if bad.exists(): continue
        labels = {**common(index),"round":round_index,"attempt":attempt}
        before = execute(anchor_task(root,programs,environment,"before",labels))
        keys = sorted(pool)
        random.Random(20261102+index*31+round_index*7).shuffle(keys)
        values = {key:execute(measurement_task(root,phase,environment,key,pool[key],labels)) for key in keys}
        after = execute(anchor_task(root,programs,environment,"after",labels))
        record = {"boot_id":boot,"environment":environment["name"],"global_boot_index":index,
                  "round":round_index,"attempt":attempt,"before":before,"after":after,"measurements":values}
        if abs(after["ii_ms"]/before["ii_ms"]-1)>.05:
            write(bad,record)
            continue
        write(path,record)
        return validate_round(root,path,programs,evidence)
    raise RuntimeError("anchor drift exhausted replacement rounds; reboot and resume")


def diagnostic_documents(root,index,environment,attempt):
    frozen = read(Path(root)/environment["name"]/"frozen.json")
    labels = {**common(index),"diagnostic_attempt":attempt}
    conditions = [(placement,detailed) for placement in ("overlap","separate") for detailed in (False,True)]
    random.Random(20261102+(index-4)//2*101+ENVIRONMENTS.index(environment)*17).shuffle(conditions)
    if index%2: conditions.reverse()
    return [(placement+"_"+("profile" if detailed else "plain"),task_for("diagnostic",
        frozen["pool"][frozen["interventions"][placement]],environment,warmup=warmup(root,environment),
        detailed=detailed,diagnostic_placement=placement,**labels)) for placement,detailed in conditions]


def validate_diagnostic(root,path,programs,evidence):
    value = read(path)
    index,attempt,boot = value["global_boot_index"],value["attempt"],value["boot_id"]
    environment = next(e for e in ENVIRONMENTS if e["name"]==value["environment"])
    if index not in range(4,10) or attempt not in range(3): raise ValueError("invalid diagnostic identity")
    if Path(path)!=Path(root)/environment["name"]/"confirmation"/f"boot{index}"/"diagnostic.json":
        raise ValueError("diagnostic path identity mismatch")
    docs = dict(diagnostic_documents(root,index,environment,attempt))
    if set(value["measurements"])!=set(docs): raise ValueError("diagnostic coverage mismatch")
    for key,document in docs.items(): evidence.summary(value["measurements"][key],document,boot)
    labels = {**common(index),"diagnostic_attempt":attempt}
    for pos in ("before","after"):
        evidence.summary(value[pos],anchor_task(root,programs,environment,"diagnostic_"+pos,labels),boot)
    if abs(value["after"]["ii_ms"]/value["before"]["ii_ms"]-1)>.05: raise ValueError("diagnostic drift")
    return value


def run_diagnostic(root,index,environment,programs,execute,evidence,boot):
    directory = Path(root)/environment["name"]/"confirmation"/f"boot{index}"
    path = directory/"diagnostic.json"
    if path.exists(): return validate_diagnostic(root,path,programs,evidence)
    for attempt in range(3):
        bad = directory/f"invalid-diagnostic-attempt{attempt}.json"
        if bad.exists(): continue
        labels = {**common(index),"diagnostic_attempt":attempt}
        before = execute(anchor_task(root,programs,environment,"diagnostic_before",labels))
        values = {key:execute(doc) for key,doc in diagnostic_documents(root,index,environment,attempt)}
        after = execute(anchor_task(root,programs,environment,"diagnostic_after",labels))
        record = {"boot_id":boot,"global_boot_index":index,"environment":environment["name"],
                  "attempt":attempt,"before":before,"after":after,"measurements":values}
        if abs(after["ii_ms"]/before["ii_ms"]-1)>.05:
            write(bad,record)
            continue
        write(path,record)
        return validate_diagnostic(root,path,programs,evidence)
    raise RuntimeError("diagnostic drift exhausted replacement blocks")


def finish_calibration(root,state,programs,evidence):
    records = rounds(root,"calibration",ENVIRONMENTS[0],programs,evidence)
    boot_ids = sorted({r["boot_id"] for r in records})
    if boot_ids!=sorted(state["completed_boots"][:2]): raise ValueError("calibration boot history mismatch")
    costs = {key:{field:statistics.mean(r["measurements"][key][field] for r in records)
                  for field in ("service_ms","run_core_ms","boundary_core_ms")} for key in signatures(programs)}
    for key,stage in signatures(programs).items():
        details = []
        for r in records:
            labels = {**common(r["global_boot_index"]),"round":r["round"],"attempt":r["attempt"]}
            document = measurement_task(root,"calibration",ENVIRONMENTS[0],key,stage,labels)
            details.append(evidence.component_details(r["measurements"][key],document,r["boot_id"]))
        costs[key].update({field:statistics.mean(d[field] for d in details) for field in details[0]})
    document = {"source":"board_calibration","experiment":protocol()["experiment"],
                "bundle_id":state["bundle_id"],"boot_ids":boot_ids,"costs":costs}
    write(Path(root)/"costs.json",document,replace=True)
    costs = load_costs(document,state["bundle_id"])
    for env in ENVIRONMENTS:
        destination = Path(root)/env["name"]/"ranking.json"
        if not destination.exists():
            print("电脑端完整搜索："+ENV_NAMES[env["name"]],flush=True)
            ranking = rank(programs,costs,env)
            actual = ranking["counts"]["subset_capacity"]
            if actual!=protocol()["expected"][env["name"]+"_configurations"]:
                raise ValueError("complete candidate domain changed")
            ranking["development_provenance"] = {"cost_sha256":sha256(Path(root)/"costs.json"),
                "programs_sha256":sha256(Path(state["bundle"])/"plans/programs.json"),
                "controller_identity":state["controller_identity"]}
            write(destination,ranking)
        elif read(destination).get("development_provenance",{}).get("cost_sha256")!=sha256(Path(root)/"costs.json"):
            raise ValueError("ranking does not match new calibration costs")
    pin(Path(root),state,[Path(root)/"costs.json",*[Path(root)/e["name"]/"ranking.json" for e in ENVIRONMENTS]])


def finish_screening(root,state,programs,evidence):
    costs = load_costs(read(Path(root)/"costs.json"),state["bundle_id"])
    for env in ENVIRONMENTS:
        directory = Path(root)/env["name"]
        records = rounds(root,"screening",env,programs,evidence)
        if {r["boot_id"] for r in records}!=set(state["completed_boots"][2:4]):
            raise ValueError("screening boot history mismatch")
        ranking = read(directory/"ranking.json")
        selected = select(ranking,grouped(records))
        frozen = freeze(programs,costs,env,ranking,selected)
        frozen["development_provenance"] = {"cost_sha256":sha256(Path(root)/"costs.json"),
            "ranking_sha256":sha256(directory/"ranking.json"),"screening_boot_ids":state["completed_boots"][2:4]}
        for value in frozen["pool"].values(): validate_row(value,programs,env)
        write(directory/"frozen.json",frozen,replace=True)
    pin(Path(root),state,[Path(root)/e["name"]/"frozen.json" for e in ENVIRONMENTS])


def run_current(root,transport):
    root = Path(root).resolve()
    with locked(root):
        state = validate_state(root)
        if state["status"]=="COMPLETE": return state
        # Complete local transitions before accepting a new kernel boot. A crash
        # during a long ranking must never turn the next boot into calibration.
        bundle = Path(state["bundle"])
        programs = read(bundle/"plans/programs.json")
        evidence = Evidence(root,state["bundle_id"])
        if state["boot_index"]>=2 and not all((root/e["name"]/"ranking.json").exists() for e in ENVIRONMENTS):
            finish_calibration(root,state,programs,evidence)
        if state["boot_index"]>=4 and not all((root/e["name"]/"frozen.json").exists() for e in ENVIRONMENTS):
            finish_screening(root,state,programs,evidence)
        if state["boot_index"]==10:
            from .analysis import analyze
            analyze(root)
            state["status"]="COMPLETE"
            write(root/"state.json",state,replace=True)
            return state
        boot = transport.boot_id()
        if boot in state["completed_boots"] or boot==state.get("excluded_preflight_boot"):
            state["status"]="WAITING_NEW_BOOT"
            write(root/"state.json",state,replace=True)
            return state
        transport.wait_idle(boot)
        if hasattr(transport,"mount_storage"): transport.mount_storage()
        retry_cleanup(root,transport)
        if state["active_boot"] and state["active_boot"]!=boot:
            previous = state["active_boot"]
            transport.recover_abandoned(root,previous)
            for env in ENVIRONMENTS:
                directory = root/env["name"]/phase_at(state["boot_index"])/f"boot{state['boot_index']}"
                if directory.exists(): directory.rename(directory.with_name(directory.name+"-abandoned-"+previous))
            event(root,"incomplete_boot_abandoned",boot_id=previous,replacement=boot)
        index = state["boot_index"]
        state.update(active_boot=boot,status="RUNNING")
        write(root/"state.json",state,replace=True)
        remote = transport.deploy(bundle,root)
        if hasattr(transport,"verify_payload"): transport.verify_payload(remote,boot)
        transport.qualify_environment(remote,boot,root,bundle)
        def execute(document):
            print(f"[启动 {index+1}/10] {ENV_NAMES[document['environment']['name']]} {document['kind']} {document['task_id']}",flush=True)
            summary = measured(document,bundle,remote,root,boot,transport)
            evidence.summary(summary,document,boot)
            return summary
        envs = list(ENVIRONMENTS)
        if index%2: envs.reverse()
        for env in envs:
            execute(task_for("qualify_pipeline",anchor(programs,env),env,warmup=0,scored=67,tail=0,
                             indices=list(range(67)),**common(index)))
            warmup_path = root/env["name"]/"warmup.json"
            if not warmup_path.exists():
                short = execute(task_for("warmup_check",anchor(programs,env),env,warmup=20,**common(index)))
                long = execute(task_for("warmup_check",anchor(programs,env),env,warmup=100,**common(index)))
                write(warmup_path,{"boot_id":boot,"warmup":100 if abs(short["ii_ms"]/long["ii_ms"]-1)>.02 else 20,
                                   "short":short,"long":long})
            validate_warmup(root,env,programs,evidence)
        pin(root,state,[root/e["name"]/"warmup.json" for e in ENVIRONMENTS])
        phase = phase_at(index)
        if index==0:
            for key,s in signatures(programs).items():
                execute(task("qualify_component",[s],ENVIRONMENTS[0],warmup=0,scored=19,tail=0,
                             indices=[*range(16),64,65,66],signature=key,**common(index)))
        if phase=="calibration":
            for r in (0,1): run_round(root,index,ENVIRONMENTS[0],r,programs,execute,evidence,boot)
        else:
            for env in envs:
                pool = pool_for(root,phase,env,programs)
                for candidate in pool.values():
                    execute(task_for("qualify_pipeline",candidate,env,warmup=0,scored=3,tail=0,
                                     indices=[64,65,66],**common(index)))
                for r in (0,1): run_round(root,index,env,r,programs,execute,evidence,boot)
                if phase=="confirmation": run_diagnostic(root,index,env,programs,execute,evidence,boot)
        if transport.boot_id()!=boot: raise RuntimeError("boot changed before batch completion")
        fresh = Evidence(root,state["bundle_id"])
        check_envs = [ENVIRONMENTS[0]] if phase=="calibration" else ENVIRONMENTS
        completed_paths = []
        for env in check_envs:
            directory = root/env["name"]/phase/f"boot{index}"
            for r in (0,1):
                path = directory/f"round{r}.json"
                validate_round(root,path,programs,fresh)
                completed_paths.append(path)
            if phase=="confirmation":
                path = directory/"diagnostic.json"
                validate_diagnostic(root,path,programs,fresh)
                completed_paths.append(path)
        if hasattr(transport,"verify_payload"): transport.verify_payload(remote,boot)
        transport.idle_command("sync",boot=boot,timeout=120)
        state["completed_boots"].append(boot)
        state.update(active_boot=None,boot_index=index+1,status="WAITING_NEW_BOOT")
        pin(root,state,completed_paths)
        if index==1: finish_calibration(root,state,programs,evidence)
        if index==3: finish_screening(root,state,programs,evidence)
        if index==9:
            from .analysis import analyze
            analyze(root)
            state["status"]="COMPLETE"
            write(root/"state.json",state,replace=True)
        return state
