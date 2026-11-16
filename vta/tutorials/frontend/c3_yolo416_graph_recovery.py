"""One fresh graph attempt after a sealed CPU-only helper lifetime fix.

Adopts the separately successful 308-call operator stage; never adopts calls
from the failed whole-graph stage. Does not contact a board or retry a stage.
"""
import argparse
import ast
from pathlib import Path
import subprocess
import sys
import traceback
from c3_tps_sync_release_local import HERE, HOLD, PYTHON, read, write, sha, verify, require, env_for_worker
from c3_yolo416_graph_closeout import seal


def run(root, operators, source, old_prepared, prepared, qualified, toolfix):
    root.mkdir(exist_ok=False)
    try:
        require(not prepared.exists() and not qualified.exists(), 'RECOVERY_DIRECTORY_EXISTS')
        for folder in (operators, source, old_prepared, toolfix):
            m = read(folder/'completion_manifest.json')
            verify({str(folder/p):d for p,d in m['artifact_sha256'].items()})
        require(read(toolfix/'completion_manifest.json')['status']=='SEALED_CPU_ONLY_PLUGIN_LIFETIME_FIX',
                'TOOLFIX_NOT_QUALIFIED')
        require(read(operators/'completion_manifest.json')['all_certificates_passed']==308,
                'OPERATOR_QUALIFICATION_INCOMPLETE')
        functions = []
        for filename in ('c3_yolo416_graph_prepare.py','c3_yolo416_graph_prepare_v2.py'):
            tree = ast.parse((HERE/filename).read_text())
            functions.append(ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build'),
                                      include_attributes=False))
        require(functions[0]==functions[1], 'TARGET_BUILD_METHOD_CHANGED')
        write(root/'contract.json', {'operators':str(operators),'source':str(source),
            'old_prepared':str(old_prepared),'prepared':str(prepared),'qualified':str(qualified),
            'toolfix':str(toolfix),'adopted_operator_calls':308,'adopted_old_graph_calls':0,
            'new_graph_call_limit':24,'FPGA_calls':0,'target_build_AST_unchanged':True,
            'target_retries':0})
        for stage, script, args, folder in [
            ('prepare','c3_yolo416_graph_prepare_v2.py',
             ['prepare',str(prepared),'--source',str(source),'--qualified',str(operators)],prepared),
            ('qualify','c3_yolo416_graph_local_v2.py',
             ['run',str(qualified),'--prepared',str(prepared)],qualified)]:
            with (root/(stage+'.log')).open('x') as log:
                code = subprocess.call([str(PYTHON),str(HERE/script),*args],env=env_for_worker(),
                                       stdout=log,stderr=subprocess.STDOUT)
            ok = seal(folder,stage)
            write(root/(stage+'_result.json'), {'returncode':code,'sealed_success':ok,'output':str(folder)})
            require(code==0 and ok, stage.upper()+'_FAILED_NO_RETRY')
            if stage=='prepare':
                from c3_official_window_baseline import normalize_llvm
                proof=[]
                for v in ('TH','P'):
                    for name in ('codegen_input.json','graph.json','params.bin','CPU_manifest.json'):
                        require(sha(old_prepared/v/name)==sha(prepared/v/name),'TARGET_OR_GRAPH_INPUT_CHANGED:'+v+':'+name)
                    require(normalize_llvm((old_prepared/v/'module.ll').read_text())==
                            normalize_llvm((prepared/v/'module.ll').read_text()),'TARGET_LLVM_CHANGED:'+v)
                    proof.append({'variant':v,'same_codegen_input':True,'same_graph_parameters_CPU_manifest':True,
                                  'same_normalized_target_LLVM':True})
                write(root/'target_unchanged.json',proof)
        write(root/'result.json',{'status':'OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED',
              'FPGA_calls':0,'performance_samples':0,'new_graph_calls':24})
        print('OPERATORS_AND_WHOLE_GRAPHS_LOCAL_QUALIFIED',flush=True)
        return 0
    except Exception as e:
        write(root/'failure.json',{'type':type(e).__name__,'message':str(e),'FPGA_calls':0})
        traceback.print_exc()
        return 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    for name in ('operators','source','old-prepared','prepared','qualified','toolfix'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    sys.exit(run(*(x.resolve() for x in (a.output,a.operators,a.source,a.old_prepared,a.prepared,a.qualified,a.toolfix))))
