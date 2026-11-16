import os
from pathlib import Path
import shutil
import subprocess
import pytest

from cpu_vta_partitioning.common import REPO,read
from multithreading.domain import ROOT


def test_fixed_groups_concurrency_and_cancellation(tmp_path):
    output=tmp_path/"groups"
    subprocess.run(["g++","-std=c++17","-O2","-pthread",str(ROOT/"tests/resource_groups.cc"),
                    "-I"+str(REPO/"vta/apps/native_deploy"),"-o",str(output)],check=True)
    subprocess.run([str(output)],check=True,timeout=10)


@pytest.mark.native
def test_cross_compiled_runner_loader_and_dependencies():
    value=os.environ.get("CPUVTA_SUPPLEMENT_WORKSPACE")
    if not value: pytest.skip("requires locally cross-compiled supplement")
    workspace=Path(value)
    binary=workspace/"runner/vta_stage_pipeline_runner"
    header=subprocess.check_output(["readelf","-h",str(binary)],text=True)
    assert "AArch64" in header
    dynamic=subprocess.check_output(["readelf","-d",str(binary)],text=True)
    import re
    for dependency in re.findall(r"Shared library: \[([^]]+)\]",dynamic):
        assert (workspace/"runtime"/dependency).is_file(),dependency
    emulator=shutil.which("qemu-aarch64-static")
    assert emulator,"QEMU is present in the preparation environment and required for parser acceptance"
    result=subprocess.run([emulator,str(workspace/"runtime/ld-linux-aarch64.so.1"),"--library-path",
        str(workspace/"runtime"),str(binary),"--help"],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "resource_group" in result.stdout+result.stderr


def test_no_tvm_import_for_models_search_or_cli():
    script="import sys; from multithreading import domain,models,search,cli; assert 'tvm' not in sys.modules"
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT.parents[1]/"src")]))
    subprocess.run([os.sys.executable,"-c",script],env=env,check=True,timeout=30)
