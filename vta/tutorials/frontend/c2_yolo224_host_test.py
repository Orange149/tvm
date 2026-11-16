#!/usr/bin/env python3
"""Exercise actual lease and bundle admission methods with CPU-only resources."""
import argparse
from pathlib import Path
import subprocess
from c2_yolo224_prepare import ROOT,HERE,write,sha
from thesis_c2_host_lifecycle import extract


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--label',default='host_protocol')
    options=parser.parse_args();out=options.out/options.label;out.mkdir()
    old=(HERE/'c3_input_early_release_runner.cc').read_text()
    source=(HERE/'c2_yolo224_runner.cc').read_text()
    token=extract((ROOT/'vta/apps/native_deploy/vta_stage_pipeline_runner.cc').read_text(),'struct P8SlotToken {')
    book=extract(old,'class LeaseBook {')
    bundle=extract(source,'class Bundle {')
    # Resource allocation and views are replaced, while protocol method bodies
    # are copied verbatim. This test makes no DMA/address correctness claim.
    protocol=bundle[bundle.index('  uint64_t Full()'):bundle.index('  const NDArray& Producer(')]
    code='''#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
void Require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
'''+token+'\n'+book+'''
class Bundle { public:
  LeaseBook book;size_t tensors;
  std::vector<uint64_t> generation,bound,written;
  Bundle(int count,int arity):book(count,0,100),tensors(arity),generation(count),bound(count),written(count){}
'''+protocol+'''};
int main(){int tests=0;
auto rejects=[&](auto action){bool bad=false;try{action();}catch(const std::exception&){bad=true;}Require(bad,"expected refusal");++tests;};
for(int arity:{1,2}){Bundle b(2,arity);
 for(int frame=0;frame<64;++frame){auto t=b.Acquire(frame);
  rejects([&]{b.Publish(t);});
  for(int i=0;i<arity;++i){b.Bound(t,i);b.Written(t,i);if(i+1<arity)rejects([&]{b.Publish(t);});}
  b.Publish(t);rejects([&]{b.Publish(t);});b.book.Consume(t);b.book.Release(t);rejects([&]{b.book.Release(t);});
 }b.book.AllFree();++tests;}
{Bundle b(1,2);auto old=b.Acquire(0);b.Bound(old,0);b.Written(old,0);
 rejects([&]{b.Publish(old);});b.Bound(old,1);b.Written(old,1);b.Publish(old);b.book.Consume(old);b.book.Release(old);
 auto fresh=b.Acquire(1);rejects([&]{b.Bound(old,0);});rejects([&]{b.book.Consume(old);});
 auto wrong=fresh;++wrong.edge_index;rejects([&]{b.book.Release(wrong);});
 b.Bound(fresh,0);b.Written(fresh,0);b.Bound(fresh,1);b.Written(fresh,1);b.Publish(fresh);b.book.Consume(fresh);
 // The correct fresh token is now releasable: each mutation below isolates
 // an identity check rather than relying on an already-invalid state.
 wrong=fresh;++wrong.edge_index;rejects([&]{b.book.Release(wrong);});
 wrong=fresh;++wrong.slot_id;rejects([&]{b.book.Release(wrong);});
 wrong=fresh;wrong.generation=old.generation;rejects([&]{b.book.Release(wrong);});
 wrong=fresh;wrong.frame_id=old.frame_id;rejects([&]{b.book.Release(wrong);});
 rejects([&]{b.book.Release(old);});b.book.Release(fresh);b.book.AllFree();++tests;}
for(bool abort:{false,true}){
 LeaseBook b(1,0,1000);auto t=b.Acquire(0);b.Publish(t);b.Consume(t);
 std::mutex mutex;std::condition_variable cv;bool waiting=false,acquired=false,failed=false;
 b.observer=[&](const char* name,const P8SlotToken&){if(std::strcmp(name,"producer_wait")==0){std::lock_guard<std::mutex> l(mutex);waiting=true;cv.notify_all();}};
 std::thread writer([&]{try{auto next=b.Acquire(1);{std::lock_guard<std::mutex> l(mutex);acquired=true;}b.Publish(next);b.Consume(next);b.Release(next);}catch(...){std::lock_guard<std::mutex> l(mutex);failed=true;}});
 {std::unique_lock<std::mutex> l(mutex);Require(cv.wait_for(l,std::chrono::seconds(1),[&]{return waiting;}),"real wait predicate not seen");Require(!acquired,"overwrote live input");}
 if(abort)b.Abort();else b.Release(t);writer.join();
 Require(abort ? failed&&!acquired : acquired&&!failed,"release/abort did not wake waiter correctly");
 if(!abort)b.AllFree();++tests;
}
std::cout<<"{\\"passed\\":true,\\"checks\\":"<<tests<<"}\\n";
}
'''
    path=out/'protocol.cc';path.write_text(code)
    command=['g++','-std=c++17','-O1','-g','-pthread','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(path),'-o',str(out/'protocol')]
    subprocess.run(command,check=True,capture_output=True)
    import os
    result=subprocess.run([str(out/'protocol')],check=True,capture_output=True,text=True,
                           env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'})
    import json
    write(out/'result.json',{'actual_core_methods_verbatim':True,'resource_adapter':'CPU only, no VTA allocation or graph execution',
        'bundle_method_sha256':__import__('hashlib').sha256(protocol.encode()).hexdigest(),
        'runner_source_sha256':sha(HERE/'c2_yolo224_runner.cc'),'sanitizers':['address','undefined'],
        'leak_sanitizer':False,'native_result':json.loads(result.stdout),'command':command})
    print(result.stdout,end='')


if __name__=='__main__':main()
