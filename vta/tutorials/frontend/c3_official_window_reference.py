"""Independent NumPy workload reference for official packed fused convolutions.

ALU tail information comes from an independently certified source contract;
convolution numbers and expected products are derived from workload geometry,
never by evaluating the lowered microcode. This module executes no TVM target.
"""

from collections import Counter
from copy import deepcopy
import hashlib
from math import prod
import numpy as np


class SupportUnresolved(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SupportUnresolved("SUPPORT_UNRESOLVED: " + message)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pair(value):
    return (int(value), int(value)) if isinstance(value, int) else tuple(map(int, value))


def _broadcast_map(shape, output_shape):
    require(shape[-1] == output_shape[-1] == 16, "unsupported lane broadcast")
    grid = np.arange(prod(shape[:-1]), dtype=np.int64).reshape(shape[:-1])
    try:
        return np.broadcast_to(grid, output_shape[:-1]).reshape(-1)
    except ValueError as error:
        raise SupportUnresolved("SUPPORT_UNRESOLVED: buffer broadcast shape") from error


def postprocess_from_output_terms(output_terms, abi):
    """Recover one uniform tail; independently check every buffer broadcast.

    output_terms is the certified serial list [logical_output, terms, phases].
    Phases use (opcode,use_imm,('imm',v)|('self',)|('tensor',(param,vector))).
    Only tail extraction uses this list; no convolution value is read from it.
    """
    out = len(abi)-1
    shape = tuple(abi[out]["shape"])
    require(len(shape) == 6 and shape[-1] == 16, "packed output ABI")
    expected_count = prod(shape[:-1])
    require(len(output_terms) == expected_count, "tail output coverage")
    names = {0:"min", 1:"max", 2:"add", 3:"shr"}
    tail, visited, maps = None, set(), {}
    for logical, _terms, phases in output_terms:
        require(logical[0] == out and 0 <= logical[1] < expected_count and logical[1] not in visited,
                "tail output identity/coverage")
        index = logical[1]
        visited.add(index)
        ops = []
        for opcode, use_imm, operand in phases:
            require(opcode in names and use_imm in (0,1), "unknown ALU stage")
            kind = operand[0]
            if use_imm:
                require(kind == "imm" and len(operand) == 2, "ALU immediate origin")
                source = {"kind":"imm", "value":int(operand[1])}
            elif kind == "self":
                source = {"kind":"self"}
            else:
                require(kind == "tensor" and len(operand) == 2, "ALU source origin")
                param, offset = operand[1]
                require(isinstance(param, int) and 2 <= param < out, "ALU external parameter origin")
                if param not in maps:
                    maps[param] = _broadcast_map(tuple(abi[param]["shape"]), shape)
                require(int(offset) == int(maps[param][index]), "ALU logical source is not ABI broadcast")
                source = {"kind":"buffer", "param":param}
            ops.append({"op":names[opcode], "source":source})
        require(tail is None or tail == ops, "different postprocess sequences across logical outputs")
        tail = ops
    return tail


def from_workload(workload, abi, postprocess):
    require(workload[0] == "conv2d_packed.vta" and len(workload) == 8, "official conv workload")
    xshape, wshape = tuple(workload[1][1]), tuple(workload[2][1])
    require(len(abi) >= 3 and all(row.get("kind", "buffer") == "buffer" for row in abi), "buffer-only ABI")
    require(all(not row.get("strides") and str(row.get("elem_offset", "0")) == "0" for row in abi), "contiguous zero-offset ABI")
    require(tuple(abi[0]["shape"]) == xshape and tuple(abi[1]["shape"]) == wshape, "workload X/W ABI binding")
    require(abi[0]["dtype"] == abi[1]["dtype"] == "int8" and abi[-1]["dtype"] == "int8", "int8 operands/output")
    require(workload[1][2] == workload[2][2] == "int8" and workload[7] == "int32", "workload integer types")
    require(len(xshape) == len(wshape) == 6 and xshape[-1] == wshape[-1] == wshape[-2] == 16,
            "NCHW1n16c/OIHW16o16i layouts")
    require(workload[6] == "NCHW1n16c" and xshape[4] == 1 and xshape[1] == wshape[1], "batch/lane/CI layout")
    strides, dilation = _pair(workload[3]), _pair(workload[5])
    padding = tuple(map(int, workload[4]))
    if len(padding) == 2:
        padding = (padding[0],padding[1],padding[0],padding[1])
    require(len(strides) == len(dilation) == 2 and len(padding) == 4
            and min(*strides,*dilation) > 0 and min(padding) >= 0, "convolution geometry")
    kh, kw = wshape[2:4]
    oh = (xshape[2]+padding[0]+padding[2]-(kh-1)*dilation[0]-1)//strides[0]+1
    ow = (xshape[3]+padding[1]+padding[3]-(kw-1)*dilation[1]-1)//strides[1]+1
    output_shape = (xshape[0], wshape[0], oh, ow, 1, 16)
    require(oh > 0 and ow > 0 and tuple(abi[-1]["shape"]) == output_shape, "output geometry ABI")
    tail = deepcopy(postprocess)
    require(isinstance(tail, list), "postprocess contract required")
    for step in tail:
        require(step.get("op") in ("add","min","max","shr") and isinstance(step.get("source"),dict), "postprocess operation")
        source = step["source"]
        require(source.get("kind") in ("imm","self","buffer"), "postprocess source")
        if source["kind"] == "buffer":
            param = source.get("param")
            require(isinstance(param,int) and 2 <= param < len(abi)-1, "broadcast parameter")
            require(abi[param]["dtype"] in ("int8","int32"), "broadcast integer dtype")
            _broadcast_map(tuple(abi[param]["shape"]),output_shape)
        elif source["kind"] == "imm":
            require(isinstance(source.get("value"),int) and -32768 <= source["value"] <= 32767, "ALU immediate width")
        if step["op"] == "shr":
            # Frozen 32-bit ACC has LOG_ACC_WIDTH=5: Xilinx uses ap_int<5>
            # for the shift argument. Do not extrapolate FSim's unmasked int.
            require(source["kind"] == "imm" and -16 <= source["value"] <= 15, "fixed signed five-bit shift contract")
    bound = xshape[1]*16*kh*kw*128*128
    require(bound < 2**31, "int8 reduction partial-sum bound")
    return ReferenceSpec(deepcopy(abi), xshape,wshape,output_shape,strides,padding,dilation,tail,bound)


class ReferenceSpec:
    def __init__(self,abi,xshape,wshape,output_shape,strides,padding,dilation,tail,bound):
        self.abi,self.xshape,self.wshape,self.output_shape = abi,xshape,wshape,output_shape
        self.strides,self.padding,self.dilation,self.postprocess = strides,padding,dilation,tail
        self.int8_partial_sum_bound = bound

    def contract(self):
        return {"schema":"c3_official_window_reference_v1", "abi":deepcopy(self.abi),
                "input_shape":list(self.xshape), "weight_shape":list(self.wshape),
                "output_shape":list(self.output_shape), "strides":list(self.strides),
                "padding":list(self.padding), "dilation":list(self.dilation),
                "postprocess":deepcopy(self.postprocess), "int8_partial_sum_bound":self.int8_partial_sum_bound,
                "convolution_source":"independent workload geometry and NumPy matrix arithmetic",
                "ADD_overflow_bound_claimed":False}

    def raw_convolution(self, arrays):
        if len(arrays) == len(self.abi)-1:
            arrays = [*arrays,np.empty(self.output_shape,dtype=np.int8)]
        require(len(arrays) == len(self.abi), "complete input ABI")
        for value, descriptor in zip(arrays,self.abi):
            require(tuple(value.shape) == tuple(descriptor["shape"]) and str(value.dtype) == descriptor["dtype"], "array ABI")
        x = arrays[0].transpose(0,4,1,5,2,3).reshape(self.xshape[0],self.xshape[1]*16,*self.xshape[2:4]).astype(np.int64)
        w = arrays[1].transpose(0,4,1,5,2,3).reshape(self.wshape[0]*16,self.wshape[1]*16,*self.wshape[2:4]).astype(np.int64)
        top,left,bottom,right = self.padding
        x = np.pad(x,((0,0),(0,0),(top,bottom),(left,right)))
        oh,ow = self.output_shape[2:4]
        sy,sx = self.strides
        dy,dx = self.dilation
        sums = np.zeros((self.xshape[0]*oh*ow,self.wshape[0]*16),dtype=np.int64)
        for ky in range(self.wshape[2]):
            for kx in range(self.wshape[3]):
                patch = x[:,:,ky*dy:ky*dy+oh*sy:sy,kx*dx:kx*dx+ow*sx:sx]
                patch = patch.transpose(0,2,3,1).reshape(-1,self.xshape[1]*16)
                sums += patch @ w[:,:,ky,kx].T
        normal = sums.reshape(self.xshape[0],oh,ow,self.wshape[0],16)
        return normal.transpose(0,3,1,2,4)[:,:,:,:,None,:].astype(np.int32)

    def reference(self, arrays, return_raw=False):
        raw = self.raw_convolution(arrays)
        value = raw.copy()
        for step in self.postprocess:
            source = step["source"]
            operand = (np.int32(source["value"]) if source["kind"] == "imm" else
                       value if source["kind"] == "self" else
                       np.broadcast_to(arrays[source["param"]].astype(np.int32),self.output_shape))
            if step["op"] == "add":
                value = np.add(value,operand,dtype=np.int32)
            elif step["op"] == "max":
                value = np.maximum(value,operand)
            elif step["op"] == "min":
                value = np.minimum(value,operand)
            else:
                shift = source["value"]
                value = np.right_shift(value,shift) if shift >= 0 else np.left_shift(value,-shift).astype(np.int32)
        output = value.astype(np.int8)
        return (output,raw) if return_raw else output

    def arrays(self, seed, directed=False):
        rng = np.random.default_rng(seed)
        amplitude = 2 if directed else 8
        values = []
        for index,descriptor in enumerate(self.abi):
            shape,dtype = tuple(descriptor["shape"]),descriptor["dtype"]
            if index == len(self.abi)-1:
                values.append(np.zeros(shape,dtype=np.int8))
            else:
                limit = amplitude if index < 2 or dtype == "int8" else 16
                values.append(rng.integers(-limit,limit+1,size=shape,dtype=np.dtype(dtype)))
        values[-1] = output_sentinel(self.reference(values))
        return values

    def observable_weights(self, arrays):
        """Choose from a fixed local list; refuse if final outputs cannot differ."""
        baseline = self.reference(arrays)
        candidates = [np.negative(arrays[1].astype(np.int16)).clip(-128,127).astype(np.int8),
                      np.full(self.wshape,127,dtype=np.int8),np.full(self.wshape,-128,dtype=np.int8)]
        for weight in candidates:
            changed = [value.copy() for value in arrays]
            changed[1] = weight
            reference = self.reference(changed)
            if not np.array_equal(weight,arrays[1]) and not np.array_equal(reference,baseline):
                return weight,reference
        raise SupportUnresolved("SUPPORT_UNRESOLVED: W0/W1 final outputs not observably distinct")

    def qualification_cases(self, seeds=(0,20250901,20260910)):
        require(len(seeds)==3, "three frozen seed cases required")
        seeded = [self.arrays(seed) for seed in seeds]
        common = seeded[0]
        weight1,reference1 = self.observable_weights(common)
        cases = [{"kind":"three_seed", "seed":seed,"arrays":values,"expected":self.reference(values)}
                 for seed,values in zip(seeds,seeded)]
        for label,weight in (("W0",common[1]),("W1",weight1),("W0_return",common[1])):
            values = [value.copy() for value in common]
            values[1] = weight.copy()
            expected = reference1 if label=="W1" else self.reference(values)
            values[-1] = output_sentinel(expected)
            cases.append({"kind":"weight_ABA_"+label,"seed":seeds[0],"arrays":values,"expected":expected})
        base = self.arrays(seeds[-1],directed=True)
        directed = None
        for scale in (1,2,4,8,16,32,64):
            candidate = [value.copy() for value in base]
            for index in (0,1):
                candidate[index] = (base[index].astype(np.int16)*scale).clip(-128,127).astype(np.int8)
            expected,raw = self.reference(candidate,return_raw=True)
            if np.any((expected != 0) & (expected != 127) & (expected != -127)):
                directed = candidate
                directed[-1] = output_sentinel(expected)
                break
        require(directed is not None, "no nonzero unsaturated directed output in fixed local input list")
        cases.append({"kind":"directed","seed":seeds[-1],"arrays":directed,"expected":expected,"raw_conv_int32":raw,
                      "directed_scale":scale,
                      "nonzero_outputs":int(np.count_nonzero(expected)),
                      "clip_boundary_outputs":int(np.count_nonzero((expected==127)|(expected==-127)))})
        for case in cases:
            case["inputs"] = case["arrays"][:-1]
        return cases


def output_sentinel(expected):
    require(expected.dtype == np.int8, "int8 output sentinel")
    sentinel = np.bitwise_xor(expected,np.int8(1))
    require(np.all(sentinel != expected), "sentinel must differ at every output")
    return sentinel


def validate_product_terms(spec, output_terms):
    """Geometry-derived convolution products, separate from TIR interpretation."""
    output_shape = spec.output_shape[:-1]
    require(len(output_terms) == prod(output_shape), "product output coverage")
    visited = set()
    for logical, terms, _phases in output_terms:
        param,index = logical
        require(param == len(spec.abi)-1 and index not in visited and 0<=index<prod(output_shape), "product output identity")
        visited.add(index)
        n,co,y,x,b = np.unravel_index(index,output_shape)
        expected = Counter()
        for ci in range(spec.xshape[1]):
            for ky in range(spec.wshape[2]):
                for kx in range(spec.wshape[3]):
                    ih=y*spec.strides[0]+ky*spec.dilation[0]-spec.padding[0]
                    iw=x*spec.strides[1]+kx*spec.dilation[1]-spec.padding[1]
                    inp = ("zero",) if not(0<=ih<spec.xshape[2] and 0<=iw<spec.xshape[3]) else (0,int(np.ravel_multi_index((n,ci,ih,iw,b),spec.xshape[:-1])))
                    weight=(1,int(np.ravel_multi_index((co,ci,ky,kx),spec.wshape[:-2])))
                    expected[inp,weight]+=1
        actual=Counter()
        for inp,weight,count in terms:
            actual[tuple(inp),tuple(weight)]+=count
        require(actual==expected, "wrong/missing/duplicated workload product at output %d"%index)
    return {"status":"INDEPENDENT_WORKLOAD_PRODUCTS_PASSED","packed_outputs":len(visited),
            "packed_terms_per_output":spec.xshape[1]*spec.wshape[2]*spec.wshape[3],
            "int8_partial_sum_bound":spec.int8_partial_sum_bound,"ADD_overflow_bound_claimed":False}


def semantic_gates(spec, output_terms):
    result = validate_product_terms(spec,output_terms)
    require(postprocess_from_output_terms(output_terms,spec.abi) == spec.postprocess,
            "postprocess differs from frozen original contract")
    result["uniform_postprocess_and_ABI_broadcast_passed"] = True
    return result


def arrays(spec, seed, directed=False):
    return spec.arrays(seed,directed)


def reference(spec, values, return_raw=False):
    return spec.reference(values,return_raw)
