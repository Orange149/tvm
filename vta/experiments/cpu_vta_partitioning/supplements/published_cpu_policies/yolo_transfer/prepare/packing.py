"""Local bridge fixes for tuple routing and unpackable live passthroughs.

The repository VTA packer and runtime are unchanged. Only explicit frontier
packing uses this subclass: concatenate executes on normal tensor operands,
and already-computed 255-channel detection tensors remain ordinary NCHW.
"""


def pack_frontiers(function, bfactor, cfactor, weight_bits):
    import tvm
    from tvm import relay
    from vta.top.graphpack import (ExprBoundaryBridgePack, _LayoutInfo,
                                   _get_tensor_shape, run_opt_pass)

    class TypedRoutingBridge(ExprBoundaryBridgePack):
        def _can_pack_shape(self, shape):
            return (super()._can_pack_shape(shape) and int(shape[0]) % self.bfactor == 0
                    and int(shape[1]) % self.cfactor == 0)

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op) and call.op.name == 'concatenate':
                args = [self.visit(arg) for arg in call.args]
                # A tuple layout can contain packed fields even when the tuple
                # itself is not marked packed. Normalize recursively before concat.
                args = [self._ensure_normal(arg)[0] for arg in args]
                result = relay.Call(call.op, args, call.attrs, call.type_args, call.span)
                return self._set_layout(result, _LayoutInfo.normal(_get_tensor_shape(call)))
            return super().visit_call(call)

    function = run_opt_pass(function, relay.transform.InferType())
    result = TypedRoutingBridge(bfactor, cfactor, weight_bits).visit(function)
    return run_opt_pass(result, relay.transform.InferType())
