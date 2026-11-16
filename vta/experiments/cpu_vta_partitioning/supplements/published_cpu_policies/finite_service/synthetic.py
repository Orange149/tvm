"""CPU-only held-out prediction uses requested CPU work and sleeps."""
from fractions import Fraction
from .scheduler import Scheduler


def mixed_actions():
    return [[('cpu',80_000_000)],
        [('sleep',1_000_000),('cpu',1_000_000)]*16,
        [('sleep',1_000_000),('cpu',30_000)]*16]


def run(actions,quanta,fluid=False):
    scheduler=Scheduler(quanta);time=Fraction(0);active={};delays={};cursor=[0]*len(actions)
    first={};end={};CPU=Fraction(0);calls=0
    def advance(actor):
        if cursor[actor]==len(actions[actor]):end[actor]=time;return
        op,amount=actions[actor][cursor[actor]];cursor[actor]+=1
        if amount<=0:raise ValueError('synthetic work must be positive')
        if op=='cpu':active[actor]=dict(actor=actor,core=0,remaining=Fraction(amount))
        elif op=='sleep':delays[actor]=time+amount
        else:raise ValueError('invalid synthetic action')
    for actor in range(len(actions)):advance(actor)
    while len(end)<len(actions):
        if fluid:
            rates={a:Fraction(1,len(active)) for a in active};expiry=None
        else:
            selected,expiry=scheduler.select(active);rates={a:Fraction(int(a in selected.values())) for a in active}
        candidates=[t['remaining']/rates[a] for a,t in active.items() if rates[a]>0]
        candidates += [t-time for t in delays.values()]
        if expiry is not None:candidates.append(expiry)
        if not candidates:raise ValueError('synthetic simulation stalled')
        delta=min(candidates)
        if not fluid:scheduler.consume(active,delta)
        done=[]
        for a,t in active.items():
            used=delta*rates[a];t['remaining']-=used;CPU+=used
            if used and a not in first:first[a]=time
            if t['remaining']==0:done.append(a)
        time+=delta
        for a in done:del active[a]
        wakes=[a for a,t in delays.items() if t==time]
        for a in wakes:del delays[a]
        for a in done+wakes:advance(a)
        calls+=1
        if calls>100_000:raise ValueError('synthetic simulation unbounded')
    return dict(wall_ns={str(a):int(t) for a,t in end.items()},
        first_service_ns={str(a):int(t) for a,t in first.items()},CPU_ns=int(CPU),events=calls,
        model='fluid' if fluid else 'finite_virtual_runtime',requested_work_only=True)
