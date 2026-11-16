"""Equal-weight virtual runtime with finite, empirically measured service.

One actor executes per core. Runtime persists between work segments. Sleeper
lag is capped at half an empirical service period; wake preemption requires
half a service granule. These are frozen approximations, not vendor CFS.
"""
from fractions import Fraction


class Scheduler:
    def __init__(self,quanta):
        self.quanta={int(n):Fraction(q) for n,q in quanta.items()}
        if set(self.quanta)!=set(range(2,7)) or any(q<=0 for q in self.quanta.values()):
            raise ValueError('positive platform granules for 2..6 actors required')
        self.vruntime={};self.last_core={};self.current={};self.left={};self.last_count={}
        self.minimum={c:Fraction(0) for c in range(4)};self.previous=set();self.choices={}

    def quantum(self,count):
        if count<2:return None
        if count not in self.quanta:raise ValueError('runnable count outside measured platform domain')
        return self.quanta[count]

    def select(self,tasks):
        by_core={c:[] for c in range(4)};by_actor={}
        for key,t in tasks.items():
            a=t['actor'];c=t['core']
            if c not in by_core or a in by_actor:raise ValueError('duplicate actor or invalid core')
            by_core[c].append(key);by_actor[a]=key
        wakes=set(by_actor)-self.previous
        for c,keys in by_core.items():
            n=len(keys);q=self.quantum(n)
            for key in keys:
                a=tasks[key]['actor'];old=self.last_core.get(a)
                if old is not None and old!=c:
                    self.vruntime[a]=self.vruntime[a]-self.minimum[old]+self.minimum[c]
                if a not in self.vruntime:self.vruntime[a]=self.minimum[c]
                elif a in wakes:
                    credit=(q*n/2) if q is not None else Fraction(0)
                    self.vruntime[a]=max(self.vruntime[a],self.minimum[c]-credit)
                self.last_core[a]=c
        chosen={};deadlines=[]
        for c,keys in by_core.items():
            if not keys:
                self.current.pop(c,None);self.left.pop(c,None);self.last_count[c]=0;continue
            n=len(keys);q=self.quantum(n)
            best=min(keys,key=lambda k:(self.vruntime[tasks[k]['actor']],k))
            old=self.current.get(c);valid=old in by_actor and by_actor[old] in keys
            wake_preempt=(valid and tasks[best]['actor'] in wakes and q is not None and
                          self.vruntime[old]-self.vruntime[tasks[best]['actor']]>q/2)
            expired=valid and self.left.get(c) is not None and self.left[c]<=0
            if not valid or expired or wake_preempt:
                actor=tasks[best]['actor'];self.current[c]=actor;self.left[c]=q
            else:
                actor=old
                if n==1:self.left[c]=None
                elif self.last_count.get(c,0)<2 or self.left.get(c) is None:self.left[c]=q
                else:self.left[c]=min(self.left[c],q)
            self.last_count[c]=n;chosen[c]=by_actor[actor]
            if self.left[c] is not None:deadlines.append(self.left[c])
        self.previous=set(by_actor);self.choices=chosen
        return chosen,(min(deadlines) if deadlines else None)

    def consume(self,tasks,duration):
        if duration<0:raise ValueError('negative scheduler progress')
        for c,key in self.choices.items():
            if key not in tasks:raise ValueError('scheduled actor disappeared')
            a=tasks[key]['actor'];self.vruntime[a]+=duration
            if self.left[c] is not None:self.left[c]-=duration
        for c in range(4):
            values=[self.vruntime[t['actor']] for t in tasks.values() if t['core']==c]
            if values:self.minimum[c]=max(self.minimum[c],min(values))
