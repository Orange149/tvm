"""Isolated, barrier-preserving reload ablations; not production mode aliases."""
import tvm
from tvm import autotvm, topi
from vta.top.vta_conv2d import conv2d_packed, _schedule_conv2d_packed_impl
from vta.top.vta_conv2d_residency import _clip


@autotvm.template("conv2d_packed_reuse_control.vta")
def reuse_control(data, kernel, strides, padding, dilation, layout, out_dtype, mode):
    if mode not in (4, 5):
        raise ValueError("reload control only accepts modes 4 and 5")
    if len(padding) == 4:
        top, left, bottom, right = padding
        if (top, left) != (bottom, right):
            raise ValueError("asymmetric padding is unsupported")
        padding = (top, left)
    with tvm.target.vta():
        result = conv2d_packed(data, kernel, strides, padding, dilation, layout, out_dtype)
        result = topi.right_shift(result, 8)
        result = _clip(result, 0, 127)
        result = topi.cast(result, "int8")
    schedule = _schedule_conv2d_packed_impl(
        autotvm.get_config(), [result], residency_mode=mode, _experimental_reload_weights=True
    )
    return schedule, [data, kernel, result]
