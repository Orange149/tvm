"""Restricted TPS output-copy representation adapter, outside production TVM.

Expose CO as host iterations only when a compact ACC tile maps to two
independent DRAM strides and cannot be one native two-dimensional STORE.
No scalar memory access, loop order, or compute stage is changed.
"""
import tvm
from tvm import tir


def strip_dma(stmt):
    return tir.stmt_functor.ir_transform(stmt, None,
        lambda n: n.body if isinstance(n, tir.AttrStmt)
        and n.attr_key == 'pragma_dma_copy' else None, ['tir.AttrStmt'])


def SplitStridedOutputChannels():
    @tir.transform.prim_func_pass(opt_level=0, name='tir.c3.SplitStridedOutputChannels')
    def transform(fn, _mod, _ctx):
        analyzer = tvm.arith.Analyzer()
        changed = []

        def constant(expr):
            value = analyzer.simplify(expr)
            return int(value) if isinstance(value, tir.IntImm) else None

        def steps(index, loops):
            result = []
            for loop in loops:
                v = loop.loop_var
                delta = tir.stmt_functor.substitute(index, {v: v + 1}) - index
                result.append(constant(delta))
            return result

        def post(node):
            if not isinstance(node, tir.AttrStmt) or node.attr_key != 'pragma_dma_copy':
                return None
            loops = []
            body = node.body
            while isinstance(body, tir.For):
                loops.append(body)
                body = body.body
            if len(loops) != 4 or not isinstance(body, tir.BufferStore):
                return None
            source = body.value
            if not (isinstance(source, tir.Cast) and source.dtype == 'int8'
                    and isinstance(source.value, tir.BufferLoad)):
                return None
            source = source.value
            if body.buffer.scope() != 'global' or source.buffer.scope() != 'local.acc_buffer':
                return None
            extents = [constant(loop.extent) for loop in loops]
            if (any(x is None or x <= 0 for x in extents) or extents[-1] != 16
                    or extents[0] <= 1 or len(body.indices) != 1 or len(source.indices) != 1):
                return None
            if any(constant(loop.min) != 0 or loop.kind != tir.ForKind.SERIAL
                   or loop.thread_binding is not None or len(loop.annotations) for loop in loops):
                raise ValueError('unsupported output copy loop semantics')
            co, height, width, lanes = extents
            dst = steps(body.indices[0], loops)
            src = steps(source.indices[0], loops)
            if (src != [height * width * lanes, width * lanes, lanes, 1]
                    or dst[-2:] != [lanes, 1] or any(x is None or x <= 0 for x in dst)):
                raise ValueError('unsupported output copy address pattern')
            # Either equality allows the stock folder to represent one 2-D DMA.
            if dst[1] == width * lanes or dst[0] == height * dst[1]:
                return None
            if dst[1] < width * lanes or dst[0] < height * dst[1]:
                raise ValueError('overlapping output copy slices')
            outer = loops[0]
            inside = tir.AttrStmt(node.node, node.attr_key, node.value, outer.body, node.span)
            result = tir.For(outer.loop_var, outer.min, outer.extent, outer.kind,
                             inside, outer.thread_binding, outer.annotations, outer.span)
            if not tvm.ir.structural_equal(strip_dma(node), strip_dma(result)):
                raise ValueError('output copy changed scalar address sequence')
            changed.append({'extent': co, 'shape': extents, 'dst_steps': dst, 'src_steps': src})
            return result

        body = tir.stmt_functor.ir_transform(fn.body, None, post, ['tir.AttrStmt'])
        return fn.with_body(body).with_attr('c3_output_channel_dma_split', len(changed))
    return transform
