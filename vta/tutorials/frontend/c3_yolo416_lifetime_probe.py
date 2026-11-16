"""CPU-only helper lifetime probe; never loads VTA or a DNN graph."""
import argparse
import gc
import json
from pathlib import Path
import sys


def probe(observer, fake, order):
    import faulthandler
    import tvm
    faulthandler.enable()
    backend = tvm.runtime.load_module(str(fake))
    def local():
        library = tvm.runtime.load_module(str(observer))
        inner = tvm.get_global_func('c3.yolo416.observer_test_fake')()
        proxy = tvm.get_global_func('c3.yolo416.observe_module')(inner, '9 fake_conv 3 1')
        print('PROXY_CREATED', flush=True)
        if order == 'drop_library_first':
            library = None
            gc.collect()
            print('LIBRARY_REFERENCE_RELEASED', flush=True)
            print('PROXY_TYPE=' + proxy.type_key, flush=True)
            print('EMPTY_ROWS=' + str(len(proxy['c3_yolo416_observer_partial']())), flush=True)
        elif order == 'objects_first':
            proxy = inner = None
            gc.collect()
            print('OBJECTS_RELEASED_BEFORE_LIBRARY', flush=True)
        elif order != 'frame_return':
            raise ValueError('UNKNOWN_RELEASE_ORDER')
        print('FUNCTION_RETURN', flush=True)
    local()
    gc.collect()
    print('NORMAL_RETURN DNN_CALLS=0 VTA_CALLS=0 GRAPH_CALLS=0', flush=True)
    # Keep the fake implementation alive through all proxy destruction.
    if backend is None:
        raise ValueError('FAKE_LIBRARY_DROPPED')


def build_fake(destination):
    from tvm.contrib import cc
    from c3_tps_sync_release_local import REPO
    from test_c3_yolo416_graph_observer import FAKE
    source = destination.with_suffix('.cc')
    source.write_text('#include <cstdio>\n' + FAKE.replace(
        'int calls{0};', 'int calls{0};\n  ~FakeConv() final { std::puts("FAKE_MODULE_DESTROYED"); std::fflush(stdout); }'))
    cc.create_shared(str(destination), [str(source)], options=['-std=c++17', '-O2',
        *['-I'+str(REPO/p) for p in ('include', '3rdparty/dmlc-core/include', '3rdparty/dlpack/include')]])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('build_fake', 'probe'))
    p.add_argument('--fake', type=Path, required=True)
    p.add_argument('--observer', type=Path)
    p.add_argument('--order', choices=('drop_library_first', 'objects_first', 'frame_return'))
    a = p.parse_args()
    if a.action == 'build_fake':
        build_fake(a.fake.resolve())
    else:
        probe(a.observer.resolve(), a.fake.resolve(), a.order)
