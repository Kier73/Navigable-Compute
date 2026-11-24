"""Measure a shape-selected heterogeneous route portfolio against exact refinement."""
from collections import Counter
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine,observation_rows
from navigable_compute.fibre_refinement_verify import verify_refinement
from navigable_compute.procedural_vocabulary import Vocabulary
from navigable_compute.induced_transversal import induce
from navigable_compute.intention_interpreter import Interpreter,Truth,Cube,Limits
from navigable_compute.intention_interpreter_verify import verify_result


def make(rows,context):
    f=construct_fibre(128,())
    return Vocabulary(f,rows,refine(f,rows),context=context)


def main():
    start=perf_counter_ns()
    train=[make(((0,3,5,6),),"train-a"),make(((0,24,40,48),),"train-b")]
    for v in train:
        v.train_symbolic()
    skill=induce(tuple(train))
    training_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    joint=make(((1,2),(2,4)),"joint")
    poly=make(((1,2),(0,12,20,24)),"polynomial")
    nonlinear=make(((1,2),(0,6)),"nonlinear")
    pairs=make(((0,)+tuple(3<<(2*i) for i in range(48)),),"pairs")
    free=make((),"free")
    suite=[]
    for j in range(16):
        k=1<<(96+j)
        z=sum(1<<(2*i+(j>>(i%4)&1)) for i in range(48))
        suite.extend(((pairs,Cube(k,z),Limits()),
                      (joint,Cube(1|k,4),Limits()),
                      (poly,Truth((k|1,k|2)),Limits()),
                      (poly,Truth((4,8,12,k|1,k|2)),Limits()),
                      (poly,Truth((k,)),Limits()),
                      (free,Cube(k,1),Limits(witness_samples=0)),
                      (nonlinear,Cube(k,1),Limits())))
    routes={id(v):Interpreter(v,skill) for v in (joint,poly,nonlinear,pairs,free)}
    preparation_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    results=[routes[id(v)].realize(q,limits=l) for v,q,l in suite]
    route_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    assert all(verify_result(v,q,r) for (v,q,_),r in zip(suite,results))
    reaudit_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    cold=[]
    for v,q,_ in suite:
        if type(q) is Truth:
            value,p=v.exact(q.target)
            cold.append("MIXED" if value is None else "ONE" if value else "ZERO")
        else:
            rows=v.observations+observation_rows(128,q.ones|q.zeros,q.ones)
            p=refine(v.parent,rows)
            assert verify_refinement(v.parent,rows,p)
            cold.append("POSSIBLE" if p.count else "EXCLUDED")
    cold_ns=perf_counter_ns()-start
    assert cold==[r.status for r in results]
    print("NCR019",dict(requests=len(suite),routes=dict(sorted(Counter(
          r.mechanism for r in results).items())),training_ns=training_ns,
          preparation_ns=preparation_ns,route_ns=route_ns,reaudit_ns=reaudit_ns,
          cold_ns=cold_ns,attempts=sum(len(r.attempts) for r in results)))


if __name__=="__main__":
    main()
