"""Local-only package assembly: reuse frozen modules, rebuild only the runner."""
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from cpu_vta_partitioning.common import read,write,digest,sha256,inventory,verify_inventory,locked,REPO
from cpu_vta_partitioning.campaign import controller_identity as base_identity
from cpu_vta_partitioning.prepare.build import sdk_command
from cpu_vta_partitioning.prepare.package import elf
from cpu_vta_partitioning.prepare.payload import execution_files
from cpu_vta_partitioning.prepare.toolchain import inspect_toolchain
from .domain import ROOT,PROJECT,protocol,signatures


def identity():
    files = [ROOT/"run.py",*sorted((ROOT/"multithreading").glob("*.py"))]
    native = ROOT.parent/"vta/apps/native_deploy"
    if not native.is_dir(): native = REPO/"vta/apps/native_deploy"
    return digest({"base_controller":base_identity(),"sources":{str(p.relative_to(ROOT)):sha256(p) for p in files},
        "native":{p.name:sha256(p) for p in [native/"vta_stage_pipeline_runner.cc",*sorted(native.glob("*.h"))]},
        "protocol":protocol(),"python":sys.version,
        "dependencies":{name:version(name) for name in ("numpy","scipy","matplotlib")}})


def validate(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle/"manifest.json")
    if digest({k:v for k,v in manifest.items() if k!="bundle_id"})[:24]!=manifest["bundle_id"]:
        raise ValueError("bundle manifest identity mismatch")
    if manifest["controller_identity"]!=identity(): raise ValueError("supplement controller changed")
    if manifest.get("experiment")!=protocol()["experiment"]: raise ValueError("wrong experiment bundle")
    verify_inventory(bundle,manifest["files"])
    if read(bundle/"plans/supplement_protocol.json")!=protocol(): raise ValueError("unsupported protocol change")
    payload = read(bundle/"plans/board_payload.json")
    if payload["files"]!=execution_files(bundle): raise ValueError("board dependency closure mismatch")
    if payload["unpacked_bytes"]!=sum(v["bytes"] for v in payload["files"].values()): raise ValueError("payload size mismatch")
    if read(bundle/"OFFLINE_READINESS.json")["status"]!="PREPARED_OFFLINE": raise ValueError("offline preparation incomplete")
    return manifest


def build_runner(workspace,local):
    source = REPO/"vta/apps/native_deploy"
    snapshot = workspace/"sources/vta/apps/native_deploy"
    snapshot.mkdir(parents=True,exist_ok=True)
    for path in source.iterdir():
        if path.suffix in (".cc",".h"): shutil.copy2(path,snapshot/path.name)
    pico = workspace/"sources/3rdparty/picojson/picojson.h"
    pico.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPO/"3rdparty/picojson/picojson.h",pico)
    output = workspace/"runner/vta_stage_pipeline_runner"
    output.parent.mkdir(exist_ok=True)
    recipe = 'exec $CXX -std=c++17 -O2 "$1" -o "$2" -I"$3/include" -I"$3/3rdparty/dlpack/include" -I"$3/3rdparty/dmlc-core/include" -I"$3/3rdparty/vta-hw/include" -L"$4" -Wl,-rpath,\'$ORIGIN/../runtime\' -ltvm_runtime -lvta -ldl -pthread'
    sdk_command(local,["bash","-c",recipe,"supplement-runner",snapshot/"vta_stage_pipeline_runner.cc",
                       output,REPO,workspace/"runtime"],workspace/"build_logs/runner.log")
    return {"recipe":recipe,"runner_sha256":sha256(output),"elf":elf(output)}


def prepare(base_bundle,local_config,config,*,full_audit=True):
    base_bundle = Path(base_bundle).resolve()
    base = read(base_bundle/"manifest.json")
    if digest({k:v for k,v in base.items() if k!="bundle_id"})[:24]!=base["bundle_id"]:
        raise ValueError("base manifest identity mismatch")
    if base["controller_identity"]!=base_identity(): raise ValueError("base controller no longer matches frozen source")
    verify_inventory(base_bundle,base["files"])
    if read(config)!=protocol(): raise ValueError("config differs from implemented preregistration")
    local = read(local_config)
    controller = identity()
    key = digest({"parent":base["bundle_id"],"controller":controller})[:24]
    workspace = ROOT/"artifacts/builds"/key
    with locked(workspace):
        for folder in ("programs","runtime","runner","firmware","inputs","references","plans"):
            source,dest = base_bundle/folder,workspace/folder
            if not dest.exists(): shutil.copytree(source,dest)
        programs = read(workspace/"plans/programs.json")
        if len(programs)!=172 or len(signatures(programs))!=1449: raise ValueError("compiled coverage/signature count changed")
        actual_toolchain = inspect_toolchain(workspace,local)
        if actual_toolchain!=read(base_bundle/"plans/compiler_identity.json")["toolchain"]:
            raise ValueError("SDK changed; cannot silently bind a new runner to old program identity")
        native = build_runner(workspace,local)
        if native["elf"]["architecture"]!="aarch64": raise ValueError("runner is not aarch64")
        write(workspace/"plans/supplement_protocol.json",protocol(),replace=True)
        write(workspace/"plans/parent_identity.json",{"bundle_id":base["bundle_id"],"manifest_sha256":sha256(base_bundle/"manifest.json"),
            "modules":{name:record["files"] for name,record in programs.items()},
            "runtime":inventory(workspace/"runtime"),"native_build":native,"toolchain":actual_toolchain},replace=True)
        policy = {"revision":"multithreading-1","udmabuf_bytes":201326592,"storage_margin_bytes":256*1024**2,
                  "boot_policy":"manual_reboot_only","recovery_io":"boot_session_lock",
                  "cpu_gate_modes":["off","resource_group"],"analysis_evidence":"native_files_and_hashed_receipts"}
        write(workspace/"plans/board_execution.json",policy,replace=True)
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   CPUVTA_SUPPLEMENT_WORKSPACE=str(workspace),CPUVTA_FULL_AUDIT="1" if full_audit else "0")
        log = workspace/"build_logs/local_tests.log"
        with log.open("w") as stream:
            result = subprocess.run([sys.executable,"-m","pytest","-c",str(ROOT/"pyproject.toml"),
                "--confcutdir="+str(ROOT),str(ROOT/"tests"),"-q"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError("local acceptance failed: "+str(log))
        if not full_audit: raise ValueError("PREPARED_OFFLINE requires the complete domain audit")
        if identity()!=controller: raise ValueError("source changed during preparation")
        shutil.copy2(log,workspace/"plans/local_tests.log")
        snapshot = workspace/"sources/supplement"
        if snapshot.exists(): shutil.rmtree(snapshot)
        shutil.copytree(ROOT/"multithreading",snapshot/"multithreading",ignore=shutil.ignore_patterns("__pycache__"))
        for folder in ("tests","configs","docs"):
            shutil.copytree(ROOT/folder,snapshot/folder,ignore=shutil.ignore_patterns("__pycache__","local.json"))
        for filename in ("run.py","pyproject.toml","README.md"): shutil.copy2(ROOT/filename,snapshot/filename)
        # Preserve the old controller as a read-only source dependency snapshot.
        if not (workspace/"sources/base_controller").exists():
            shutil.copytree(PROJECT/"src",workspace/"sources/base_controller/src",ignore=shutil.ignore_patterns("__pycache__"))
        files = execution_files(workspace)
        write(workspace/"plans/board_payload.json",{"revision":1,"files":files,
            "unpacked_bytes":sum(v["bytes"] for v in files.values()),"transfer":"gzip_tar_stream_no_remote_archive"},replace=True)
        write(workspace/"OFFLINE_READINESS.json",{"status":"PREPARED_OFFLINE","controller_identity":controller,
            "base_bundle_id":base["bundle_id"],"compile_coverage":{"cpu":85,"vta":87,"signatures":1449},
            "tests":{"passed":True,"log_sha256":sha256(log),"result":log.read_text()[-5000:]},
            "board_validation_pending":["native correctness","actual affinity","new costs","steady performance","independent boots"],
            "performance_claim":None},replace=True)
        # A self-contained copy; no links to mutable historical paths.
        parent = ROOT/"artifacts/bundles";parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".sealing-",dir=parent))
        for folder in ("programs","runtime","runner","firmware","inputs","references","plans","sources"):
            shutil.copytree(workspace/folder,staging/folder)
        shutil.copy2(workspace/"OFFLINE_READINESS.json",staging/"OFFLINE_READINESS.json")
        manifest = {"schema_version":1,"experiment":protocol()["experiment"],"controller_identity":controller,
                    "parent_bundle_id":base["bundle_id"],"files":inventory(staging)}
        manifest["bundle_id"] = digest(manifest)[:24]
        write(staging/"manifest.json",manifest)
        destination = parent/manifest["bundle_id"]
        # Execute the bundled controller after copying the whole closure to a
        # different directory; no checkout-relative source dependency is allowed.
        with tempfile.TemporaryDirectory(prefix="cpuvta-relocation-") as temporary:
            relocated = Path(temporary)/"bundle"
            shutil.copytree(staging,relocated)
            result = subprocess.run([sys.executable,str(relocated/"sources/supplement/run.py"),
                "validate","--bundle",str(relocated)],text=True,capture_output=True,timeout=120)
            if result.returncode: raise RuntimeError("relocated bundle failed validation: "+result.stderr[-2000:])
        if destination.exists(): shutil.rmtree(staging)
        else: staging.rename(destination)
        validate(destination)
        write(workspace/"plans/relocation_validation.json",{"passed":True,"bundled_controller":True,
            "bundle_id":manifest["bundle_id"],"stdout":result.stdout},replace=True)
        write(ROOT/"artifacts/LATEST_BUNDLE.json",{"path":str(destination),"bundle_id":manifest["bundle_id"]},replace=True)
        return destination
