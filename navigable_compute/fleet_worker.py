"""Portable independent fleet task descriptions and pure worker implementation."""
from dataclasses import dataclass
from time import perf_counter_ns
from os import getpid

from .intention_fibre import construct_fibre
from .fibre_refinement import refine
from .procedural_vocabulary import Vocabulary,Reduction
from .procedural_vocabulary_verify import verify_reduction
from .induced_transversal import induce,activate
from .coalescent_fleet import contribution


@dataclass(frozen=True)
class Source:
    width: int
    rows: tuple
    context: str


@dataclass(frozen=True)
class Task:
    actor: str
    task: str
    operation: str
    index: int


def vocabulary(s):
    p=construct_fibre(s.width,())
    return Vocabulary(p,s.rows,refine(p,s.rows),context=s.context)


def skill():
    a=vocabulary(Source(128,((0,3,5,6),),"skill-a"))
    b=vocabulary(Source(128,((0,24,40,48),),"skill-b"))
    a.train_symbolic()
    b.train_symbolic()
    return induce((a,b))


def fixture(n=8):
    if type(n) is not int or not 2<=n<=16:
        raise ValueError("fleet source count")
    return Source(128,tuple((0,3<<(3*i),5<<(3*i),6<<(3*i))
                            for i in range(n)),"fleet-shared-ground")


def intention(n):
    # XOR of one qualified zero factor per source, multiplied by disjoint
    # spectator coordinates. These were not worker targets.
    return tuple(sorted(m|(1<<(100+i)) for i in range(n)
                        for m in (0,1<<(3*i),2<<(3*i),3<<(3*i))))


def run_task(s,task):
    """Worker reconstitutes its own admitted source and exports descriptions."""
    start=perf_counter_ns()
    v=vocabulary(s)
    if (type(task) is not Task or type(task.index) is not int
            or not 0<=task.index<len(v.rules)):
        raise ValueError("task scope")
    if task.operation=="transversal":
        work=vocabulary(s)
        activate(skill(),work,task.index,max_arity=2)
        proofs=tuple(l.proof for l in work.learned)
    elif task.operation=="product":
        # A targeted product construction for a known task coordinate set.
        source=v.rules[task.index]
        mask=source[1]
        factor=(0,mask&-mask,mask^(mask&-mask),mask)
        p=Reduction(factor,(),tuple((task.index,m) for m in factor),
                    len(factor)*len(source))
        if not verify_reduction(v,factor,p):
            raise ValueError("worker product proof refused")
        proofs=(p,)
    else:
        raise ValueError("unknown task operation")
    packets=tuple(contribution(v,task.actor,task.task,p) for p in proofs)
    return packets,getpid(),perf_counter_ns()-start
