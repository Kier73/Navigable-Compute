"""Independent recursive fleet worker tasks; source remains pinned."""
from os import getpid
from time import perf_counter_ns
from .fleet_worker import vocabulary,Source
from .recursive_fleet import Graph,describe,digest

_BASE=None


def source(n):
    if type(n) is not int or not 2<=n<=32 or n&(n-1):
        raise ValueError("power of two leaf count 2..32")
    return Source(128,tuple((0,3<<(3*i),5<<(3*i),6<<(3*i))
                            for i in range(n)),"recursive-fleet-"+str(n))


def init_worker(s):
    global _BASE
    _BASE=vocabulary(s)


def leaf(v,actor,task,i,k):
    r=v.rules[i]
    a=r[1]&-r[1]
    b=r[1]^a
    factor=(0,a,b,a|b)
    p=tuple(sorted(m|k for m in factor))
    return describe(v,actor,task,p,tuple(("base",i,m|k) for m in factor))


def join(v,actor,task,parents,ancestors):
    g=Graph(v)
    g.ingest(ancestors)
    p=set()
    for h in parents:
        p^=set(g.relation(h))
    return describe(v,actor,task,tuple(sorted(p)),
                    tuple(("dependency",h,0) for h in parents))


def run(task):
    t=perf_counter_ns()
    kind,actor,label,args=task
    if kind=="leaf":
        raw=leaf(_BASE,actor,label,*args)
    elif kind=="join":
        raw=join(_BASE,actor,label,*args)
    else:
        raise ValueError("unknown recursive worker task")
    return raw,getpid(),perf_counter_ns()-t
