"""Inject a frozen RPC device/ledger into the existing CPU-only reference.

Uses the same ARM CPU binary as the graph, in a separate loaded SO. It never
uses GraphExecutor.run or a VTA convolution to establish reference outputs.
All inherited ABI, read-only and zero-submission guards remain active.
"""
from c3_yolo416_graph_cpu_reference import CPUCallback, require


class RemoteCPUCallback(CPUCallback):
    def __init__(self, module, manifest, device, ledger):
        require(type(device.device_type) is int and device.device_type % 128 == 12,
                'REFERENCE_DEVICE_NOT_EXT_DEV')
        require(callable(ledger),'REFERENCE_SUBMISSION_LEDGER_REQUIRED')
        self.module,self.manifest,self.device,self.ledger=module,manifest,device,ledger
        self.calls=[]
