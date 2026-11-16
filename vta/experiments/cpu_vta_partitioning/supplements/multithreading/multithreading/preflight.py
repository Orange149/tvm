"""Operational qualification on an excluded boot before formal calibration."""
from pathlib import Path

from cpu_vta_partitioning.common import read,write,locked
from cpu_vta_partitioning.campaign import measured
from cpu_vta_partitioning.domain import Stage
from .domain import ROOT,ENVIRONMENTS,row,task_for
from .evidence import Evidence
from .prepare import validate,identity


def qualify(bundle,transport):
    bundle = Path(bundle).resolve()
    manifest = validate(bundle)
    boot = transport.boot_id()
    root = ROOT/"artifacts/qualification"/("preflight_"+manifest["bundle_id"]+"_"+boot)
    root.mkdir(parents=True,exist_ok=True)
    with locked(root):
        transport.wait_idle(boot)
        transport.mount_storage()
        remote = transport.deploy(bundle,root)
        transport.verify_payload(remote,boot)
        transport.qualify_environment(remote,boot,root,bundle)
        evidence = Evidence(root,manifest["bundle_id"])
        reports = []
        for env in ENVIRONMENTS:
            for mode in ("off","resource_group"):
                stages = [Stage("cpu",0,3,2,(0,1)),Stage("vta",3,18,1,tuple(env["host_cores"])),
                          Stage("cpu",18,21,2,(0,1))]
                candidate = row(stages,mode)
                document = task_for("qualify_pipeline",candidate,env,warmup=0,scored=67,tail=0,
                                    indices=list(range(67)),campaign_phase="operational_preflight",global_boot_index=-1)
                summary = measured(document,bundle,remote,root,boot,transport)
                evidence.summary(summary,document,boot)
                reports.append({"environment":env["name"],"execution_mode":mode,"summary":summary})
        transport.verify_payload(remote,boot)
        transport.idle_command("sync",boot=boot,timeout=120)
        result = {"status":"PREFLIGHT_PASSED","excluded_boot_id":boot,"bundle_id":manifest["bundle_id"],
                  "controller_identity":identity(),"reports":reports,"evidence_directory":str(root)}
        write(root/"PREFLIGHT.json",result,replace=True)
        write(ROOT/"artifacts/LATEST_PREFLIGHT.json",result,replace=True)
        return result
