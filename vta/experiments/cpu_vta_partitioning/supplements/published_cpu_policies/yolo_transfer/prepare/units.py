"""Typed convolution-block units and dependency-complete YOLO frontiers.

One convolution plus its scalar/activation/pool epilogue forms a fixed unit.
Routing upsample/concat operations are assigned to the consuming convolution.
The two decode heads form the final CPU unit. This is a declared block domain,
not a claim to enumerate every primitive-operator cut or fusion variant.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import tvm
from tvm import relay
from .common_graph import file_hash,write

CONV_ORDER=(0,1,2,3,4,5,6,7,8,11,9,10,12)
POSTOPS={'add','nn.bias_add','multiply','divide','round','floor','ceil','clip','cast',
    'right_shift','left_shift','annotation.stop_fusion','nn.leaky_relu','nn.relu','nn.max_pool2d','nn.pad'}
LEGACY_CONV_PREFIXES=((),(0,),(0,1),(0,1,2),(0,1,2,3),(0,1,2,3,4),
    tuple(range(7)),tuple(range(8)),tuple(range(9)),(*range(9),11),(*range(10),11),tuple(range(13)))

def children(node):
    if isinstance(node,relay.Call):return list(node.args)
    if isinstance(node,relay.Tuple):return list(node.fields)
    if isinstance(node,relay.TupleGetItem):return [node.tuple_value]
    return []

def outputs(body):
    if isinstance(body,relay.Tuple):return [leaf for field in body.fields for leaf in outputs(field)]
    if not isinstance(body.checked_type,relay.TensorType):raise ValueError('non-tensor terminal output')
    return [body]

class Replace(relay.ExprMutator):
    def __init__(self,replacements):super().__init__();self.replacements=replacements
    def visit(self,node):
        if node in self.replacements:return self.replacements[node]
        return super().visit(node)

class UnitGraph:
    def __init__(self,mod,*,conv_order=CONV_ORDER):
        self.mod=relay.transform.InferType()(mod);self.nodes=[]
        relay.analysis.post_order_visit(self.mod['main'].body,self.nodes.append)
        if any(isinstance(n,(relay.Let,relay.If,relay.Function)) for n in self.nodes):
            raise ValueError('a first-order expression DAG is required')
        self.ids={n:i for i,n in enumerate(self.nodes)};self.users=defaultdict(list)
        for node in self.nodes:
            for arg in children(node):self.users[arg].append(node)
        self.calls=[n for n in self.nodes if isinstance(n,relay.Call)]
        if any(not isinstance(n.op,tvm.ir.Op) for n in self.calls):raise ValueError('opaque function call')
        self.convs=[n for n in self.calls if n.op.name=='nn.conv2d']
        if sorted(conv_order)!=list(range(len(self.convs))):raise ValueError('convolution order must cover each convolution once')
        self.owner={};self.units=[];self.conv_order=tuple(conv_order)
        for number in conv_order:
            index=len(self.units);conv=self.convs[number];self.assign(conv,index)
            for node in self.calls:
                if node in self.owner or node.op.name not in POSTOPS:continue
                args=[x for arg in node.args for x in (list(arg.fields) if isinstance(arg,relay.Tuple) else [arg])]
                if not any(self.owner.get(arg)==index for arg in args):continue
                if self.absorbable(node):self.assign(node,index)
            self.units.append({'index':index,'name':'conv_block_'+str(number),'original_conv_index':number})
        self.assign(self.mod['main'].body,len(self.units))
        self.units.append({'index':len(self.units),'name':'decode_heads','original_conv_index':None})
        if any(node not in self.owner for node in self.calls):raise ValueError('unowned graph computation')
        for node in self.nodes:
            if node not in self.owner:continue
            if any(arg in self.owner and self.owner[arg]>self.owner[node] for arg in children(node)):
                raise ValueError('unit ordering requires backwards execution')
        for unit in self.units:
            calls=[n for n in self.calls if self.owner[n]==unit['index']]
            counts=Counter(n.op.name for n in calls)
            if counts['nn.conv2d']!=int(unit['original_conv_index'] is not None):
                raise ValueError('unit absorbed an unintended convolution')
            unit.update(operator_counts=dict(counts),call_node_ids=[self.ids[n] for n in calls])

    def assign(self,node,index):
        if node in self.owner or isinstance(node,(relay.Var,relay.Constant)):return
        for arg in children(node):self.assign(arg,index)
        self.owner[node]=index

    def absorbable(self,node):
        if node in self.owner or isinstance(node,(relay.Var,relay.Constant)):return True
        if isinstance(node,relay.Call) and node.op.name not in POSTOPS:return False
        return all(self.absorbable(arg) for arg in children(node))

    def frontier(self,boundary):
        if not 0<=boundary<=len(self.units):raise ValueError('boundary outside unit domain')
        if boundary==0:return list(self.mod['main'].params)
        if boundary==len(self.units):return outputs(self.mod['main'].body)
        result=[]
        for node in self.nodes:
            if not isinstance(node,(relay.Var,relay.Call,relay.TupleGetItem)):continue
            if not isinstance(node.checked_type,relay.TensorType):continue
            if self.owner.get(node,-1)<boundary and any(self.owner.get(user,-1)>=boundary for user in self.users[node]):
                result.append(node)
        return result

    def stage(self,begin,end):
        if not 0<=begin<end<=len(self.units):raise ValueError('invalid stage range')
        incoming,outgoing=self.frontier(begin),self.frontier(end);visited=set()
        def check(node):
            if node in visited or node in incoming:return
            visited.add(node)
            if isinstance(node,relay.Call) and not begin<=self.owner[node]<end:
                raise ValueError('stage would recompute an operation outside its range')
            for arg in children(node):check(arg)
        for node in outgoing:check(node)
        params=[relay.var(f'edge{begin}_{i}',type_annotation=n.checked_type) for i,n in enumerate(incoming)]
        body=Replace(dict(zip(incoming,params))).visit(relay.Tuple(outgoing))
        if set(relay.analysis.free_vars(body))!=set(params):raise ValueError('incomplete frontier closure')
        return relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(params,body))),incoming,outgoing

    def recompose(self,ranges):
        values=list(self.mod['main'].params);position=0
        for begin,end in ranges:
            if begin!=position:raise ValueError('ranges do not form a continuous partition')
            mod,_,_=self.stage(begin,end);function=mod['main']
            body=Replace(dict(zip(function.params,values))).visit(function.body)
            values=list(body.fields);position=end
        expected=outputs(self.mod['main'].body)
        if position!=len(self.units) or len(values)!=len(expected) or any(
                not tvm.ir.structural_equal(a,b) for a,b in zip(values,expected)):
            raise ValueError('unit recomposition changed the canonical outputs')
        return True

def prepare(graph_directory,output):
    graph_directory,output=Path(graph_directory).resolve(),Path(output).resolve()
    source=graph_directory/'canonical_qir.json';mod=tvm.ir.load_json(source.read_text());graph=UnitGraph(mod)
    if len(graph.units)!=14:raise ValueError('expected 13 convolution units plus two-head decode')
    output.mkdir(parents=True,exist_ok=False);n=len(graph.units)
    graph.recompose([(i,i+1) for i in range(n)])
    # All interval frontiers are checked, including passthrough branch values.
    for begin in range(n):
        for end in range(begin+1,n+1):
            graph.recompose([(i,i+1) for i in range(begin)]+[(begin,end)]+[(i,i+1) for i in range(end,n)])
    prefixes=[set(CONV_ORDER[:i]) for i in range(14)]
    if any(set(prefix) not in prefixes for prefix in LEGACY_CONV_PREFIXES):raise ValueError('a historical convolution cut set disappeared')
    boundaries=[]
    for boundary in range(n+1):
        nodes=graph.frontier(boundary)
        boundaries.append({'before_unit':boundary,'tensor_count':len(nodes),
            'tensors':[{'node_id':graph.ids.get(node),'shape':list(map(int,node.checked_type.shape)),
                'dtype':str(node.checked_type.dtype)} for node in nodes]})
    write(output/'units_and_boundaries.json',{'units':graph.units,'boundaries':boundaries})
    for i in range(n):
        mod,_,_=graph.stage(i,i+1);(output/(f'unit_{i:02}.json')).write_text(tvm.ir.save_json(mod))
    write(output/'UNIT_STATUS.json',{'status':'STRUCTURALLY_VERIFIED_LOCAL','units':n,'convolutions':13,
        'source_sha256':file_hash(source),'unit_recomposition_exact':True,'all_105_interval_recompositions_exact':True,
        'terminal_output_count':len(outputs(graph.mod['main'].body)),
        'legacy_convolution_prefix_sets_retained':True,'conv_order':list(CONV_ORDER),
        'domain_limit':'fixed convolution/activation/pool blocks; routing is assigned to the consuming convolution; primitive internal cuts and other fusion variants are outside this frozen block domain',
        'extra_quantize_calls':0,'board_accessed':False,'ARM_correctness_verified':False,
        'multi_tensor_boundaries':[b['before_unit'] for b in boundaries if b['tensor_count']>1]})
    return {'units':n,'terminal_outputs':len(outputs(graph.mod['main'].body)),'multitensor_boundaries':[
        b['before_unit'] for b in boundaries if b['tensor_count']>1]}
