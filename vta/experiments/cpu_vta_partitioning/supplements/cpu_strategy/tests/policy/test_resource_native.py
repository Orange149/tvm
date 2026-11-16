import copy
import os
from pathlib import Path
import subprocess
import shutil
import pytest

from cpu_vta_partitioning.common import read,REPO
from cpu_strategy.policy import evidence


def workspace():
    value=os.environ.get('CPUVTA_RESOURCE_WORKSPACE')
    if not value:pytest.skip('requires prepared resource executors and imported evidence')
    return Path(value)


def test_native_pinned_probe_and_full_elf_closure():
    w=workspace();accept=read(w/'plans/policy_native_acceptance.json')
    assert accept['passed']
    slots=accept['worker_slots'];assert len(slots)==2 and len({s['tid'] for s in slots})==2
    assert sorted(s['cpus'][0] for s in slots)==accept['qemu_cores']
    assert set(accept['extra_pinned_widths'])=={'1','3','4'} and accept['control_role_selftest']
    closure=read(w/'plans/elf_dependencies.json')
    assert sum(name.startswith('programs/') for name in closure)==172
    assert read(w/'plans/executors.json')['legacy']['sha256']!=read(w/'plans/executors.json')['policy']['sha256']


def test_real_34_signature_import_and_source_tampering_refused(monkeypatch):
    w=workspace();rows=evidence.validate_import(w)
    assert len(rows)==34 and sum(len(v['blocks']) for v in rows.values())==58
    original=evidence.read
    plan=copy.deepcopy(original(w/'plans/imported_profiles.json'))
    key=next(iter(plan['records']));plan['records'][key]['blocks'][0]['summary']['_receipt']['sha256']='0'*64
    monkeypatch.setattr(evidence,'read',lambda p:plan if Path(p)==w/'plans/imported_profiles.json' else original(p))
    with pytest.raises(ValueError,match='receipt missing or changed'):evidence.validate_import(w)


def test_complete_frozen_172_module_dp_top20_equals_independent_enumeration():
    from cpu_strategy.policy.domain import complete_components
    from cpu_strategy.policy.search import rank_dp,enumerate_reference
    w=workspace();fixture=read(w/'plans/search_test_fixture.json')
    assert fixture['scope']=='offline_search_test_only_not_campaign_calibration'
    p=read(w/'plans/programs.json');r={}
    for key,c in complete_components(p,'group').items():
        r[key]={'configuration':c,'cost':fixture['costs'][c['stages'][0]['signature']]}
    rule={'affinity_mode':'group','reserved_host':False,'homogeneous':True}
    for method in ('single_core','prefix','enhanced'):
        a=rank_dp(p,r,rule,method);b=enumerate_reference(p,r,rule,method)
        assert len(a['rankings'])==20
        assert [(v['id'],v['score_ms']) for v in a['rankings']]==[(v['id'],v['score_ms']) for v in b['rankings']]
        assert b['visited_terminals']>900000


def test_same_group_serial_other_group_parallel_and_cancel(tmp_path):
    compiler=shutil.which('g++')
    if not compiler:pytest.skip('host C++ compiler unavailable')
    source=tmp_path/'groups.cc';output=tmp_path/'groups'
    source.write_text(r'''
#include "vta_cpu_resource_groups.h"
#include <atomic>
#include <thread>
#include <condition_variable>
#include <chrono>
int main() {
  vta_deploy::CpuResourceGroups groups({{0,1},{0,1},{2,3}});
  auto held=groups.Acquire(0);
  std::atomic<bool> other{false},same{false};
  std::mutex m;std::condition_variable cv;
  std::thread a([&]{auto x=groups.Acquire(1);same=true;cv.notify_all();});
  std::thread b([&]{auto x=groups.Acquire(2);other=true;cv.notify_all();});
  bool ok=false;
  { std::unique_lock<std::mutex> lock(m);ok=cv.wait_for(lock,std::chrono::seconds(2),[&]{return other.load();}); }
  if (!ok || same) { groups.Cancel();held.reset();a.join();b.join();return 10; }
  held.reset();a.join();b.join();if(!same)return 11;
  held=groups.Acquire(0);std::atomic<bool> cancelled{false};
  std::thread c([&]{try {auto x=groups.Acquire(1);}catch(const std::runtime_error&){cancelled=true;}});
  groups.Cancel();c.join();held.reset();if(!cancelled)return 12;
  try {vta_deploy::CpuResourceGroups invalid({{0,1},{1,2}});return 13;}catch(const std::invalid_argument&){}
  return 0;
}
''')
    result=subprocess.run([compiler,'-std=c++17','-pthread','-I'+str(REPO/'vta/apps/native_deploy'),str(source),'-o',str(output)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(output)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
