"""Read-only function ownership projection for full-graph LLVM cache binding.

Preserve declarations and function bodies verbatim. Follow actual references
from the packed entry, including compute helpers and initializer callbacks.
The existing per-function gate then checks handle/initializer/signature fields.
This is a certificate view; never compile the projected text.
"""
import re
from c3_official_window_llvm import _SYMBOL,_unquote,_without_comments,_function_bodies


DEFINITION=re.compile(r'^define\b[^\n]*?('+_SYMBOL+r')\([^\n]*\)[^\n]*\{\s*\n.*?^\}',re.M|re.S)
HANDLE=re.compile(r'^\s*('+_SYMBOL+r')\s*=')


def require(value,message):
    if not value:raise ValueError(message)


def project(llvm,symbol):
    clean=_without_comments(llvm);bodies=_function_bodies(clean)
    entry='@'+symbol
    require(entry in bodies,'GRAPH_LLVM_ENTRY_MISSING')
    reachable=set();pending=[entry]
    while pending:
        name=pending.pop()
        if name in reachable:continue
        reachable.add(name)
        for token in re.findall(_SYMBOL,bodies[name]):
            target=_unquote(token)
            if target in bodies and target not in reachable:pending.append(target)
    owned=set()
    for name in reachable:
        owned.update(_unquote(t) for t in re.findall(_SYMBOL,bodies[name])
                     if _unquote(t).startswith('@__tvm_static_handle'))
    records=list(DEFINITION.finditer(clean))
    require({_unquote(m[1]) for m in records}==set(bodies),'GRAPH_LLVM_DEFINITION_PARSE_MISMATCH')
    # Remove whole unrelated definitions first, never normalize their contents.
    result=DEFINITION.sub(lambda m:m[0] if _unquote(m[1]) in reachable else '',clean)
    lines=[];declared=[]
    for line in result.splitlines():
        match=HANDLE.match(line)
        if match and _unquote(match[1]).startswith('@__tvm_static_handle'):
            handle=_unquote(match[1])
            if handle not in owned:continue
            declared.append(handle)
        lines.append(line)
    require(set(declared)==owned and len(declared)==len(owned),'GRAPH_LLVM_HANDLE_DECLARATION_MISMATCH')
    return '\n'.join(lines),{'entry':entry,'reachable_functions':sorted(reachable),
                            'owned_handles_in_declaration_order':declared,
                            'projection_only_not_compiler_input':True}


def bind_graph(llvm,functions):
    """functions is the complete exact symbol -> final TIR function mapping."""
    from c3_official_window_llvm import llvm_alu_contract
    owners={};results={}
    for symbol,fn in functions.items():
        text,ownership=project(llvm,symbol)
        contract=llvm_alu_contract(fn,text)
        for handle in ownership['owned_handles_in_declaration_order']:
            require(handle not in owners,'GRAPH_CROSS_FUNCTION_CACHE_HANDLE_ALIAS')
            owners[handle]=symbol
        results[symbol]={'ownership':ownership,'initializer_contract':contract}
    all_handles=[]
    for line in _without_comments(llvm).splitlines():
        match=HANDLE.match(line)
        if match and _unquote(match[1]).startswith('@__tvm_static_handle'):
            all_handles.append(_unquote(match[1]))
    require(set(all_handles)==set(owners) and len(all_handles)==len(owners),
            'GRAPH_UNOWNED_OR_DUPLICATE_CACHE_HANDLE')
    return {'status':'FULL_GRAPH_LLVM_HANDLE_INITIALIZER_OWNERSHIP_PASSED',
            'functions':results,'distinct_handles':len(owners),
            'limitation':'address contents and actual runtime lifetimes still require native certification'}
