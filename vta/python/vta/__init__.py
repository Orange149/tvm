# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""VTA Package is a TVM backend extension to support VTA hardware.

Besides the compiler toolchain, it also includes utility functions to
configure the hardware environment and access remote device through RPC.
"""
import sys
import tvm
import tvm._ffi.base

from .autotvm import module_loader
from .bitstream import get_bitstream_path, download_bitstream
from .environment import get_env, Environment
from .rpc_client import reconfig_runtime, program_fpga

__version__ = "0.1.0"


# do not from tvm import topi when running vta.exec.rpc_server
# in lib tvm runtime only mode
if not tvm._ffi.base._RUNTIME_ONLY:
    from . import top
    from .build_module import build_config, lower, build


def _runtime_func(name, remote=None):
    if remote is not None:
        return remote.get_function(name)
    return tvm.get_global_func(name)


def shared_cpu_view(arr, remote=None):
    """Create a CPU-visible alias over a VTA ext_dev NDArray."""

    return _runtime_func("vta.runtime.ndarray_shared_cpu_view", remote)(arr)


def mark_shared_buffer_host_write(arr, remote=None):
    """Mark a VTA shared buffer dirty after CPU writes through its alias."""

    return _runtime_func("vta.runtime.ndarray_mark_host_write", remote)(arr)


def sync_shared_buffer_host_read(arr, remote=None):
    """Make device-written data visible to CPU before reading through its alias."""

    return _runtime_func("vta.runtime.ndarray_sync_host_read", remote)(arr)


def replay_begin_capture(label, remote=None):
    """Enable VTA command template capture for the next run."""

    return _runtime_func("vta.runtime.replay_begin_capture", remote)(label)


def replay_begin_replay(label, remote=None):
    """Enable VTA command template replay for the next run."""

    return _runtime_func("vta.runtime.replay_begin_replay", remote)(label)


def replay_reset(remote=None):
    """Clear captured VTA command templates and disable replay mode."""

    return _runtime_func("vta.runtime.replay_reset", remote)()


def replay_status(remote=None):
    """Get VTA command replay status as a JSON string."""

    return _runtime_func("vta.runtime.replay_status", remote)()
