"""No file access and no pipeline throughput inputs."""
from copy import deepcopy
from .domain import canonical

def rank(pool,group_plan,measurements):
    if set(measurements)!=set(group_plan['groups']):raise ValueError('CPU group coverage incomplete')
    if set(group_plan['configuration_to_group'])!=set(pool):raise ValueError('candidate group mapping differs')
    scored=[]
    for name,row in pool.items():
        original=row['score_ns'];group=group_plan['configuration_to_group'][name];m=measurements[group]
        cpu=m['cpu_ii_ns']
        if (type(original)!=int or original<=0 or type(cpu)!=int or cpu<=0
                or m['completion_gaps']!=7):raise ValueError('invalid integer resource score')
        scored.append({**deepcopy(row),'original_score_ns':original,'cpu_group':group,
                       'cpu_group_ns':cpu,'score_ns':max(original,cpu),'score_ms':max(original,cpu)/1e6})
    scored.sort(key=lambda r:(r['score_ns'],canonical(r)))
    for i,row in enumerate(scored,1):row['rerank_position']=i
    return {'pipeline_feedback_used':False,'scope':'frozen DP candidate pool, not full-domain optimality',
            'rankings':scored,'top20':scored[:20]}
