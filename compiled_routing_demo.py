"""NCR-021: compare exactly equal learned choices before and after compilation."""
from collections import Counter
from time import perf_counter_ns
from navigable_compute.induced_transversal import induce
from navigable_compute.intention_interpreter import Interpreter,Truth,Cube
from navigable_compute.intention_interpreter_verify import verify_result
from navigable_compute.route_policy import Trainer,realize
from navigable_compute.compiled_routing import CompiledPolicy
from reflective_routing_demo import make


def main():
    a,b=make(((0,3,5,6),),"train-a"),make(((0,24,40,48),),"train-b")
    a.train_symbolic();b.train_symbolic()
    skill=induce((a,b))
    def fixture(offset,phase):
        shift=lambda p:tuple(m<<offset if m else 0 for m in p)
        rows=(tuple(map(shift,((1,2),(2,4)))),
              (shift((0,)+tuple(3<<(2*i) for i in range(24))),),
              tuple(map(shift,((1,2),(0,12,20,24)))),())
        return tuple(Interpreter(make(r,phase+str(j)),skill) for j,r in enumerate(rows))
    def suite(engines,offset,start,n,shifted=False):
        joint,pair,poly,free=engines
        a,b,c=(1<<offset,2<<offset,4<<offset)
        for j in range(start,start+n):
            k=1<<(80+j)
            zeros=sum(1<<(offset+2*i+(j>>(i%3)&1)) for i in range(24))
            yield joint,Cube(a|k,c)
            yield pair,Cube(0,zeros)
            yield poly,Truth(tuple(sorted((k|a,k|b))))
            yield poly,Truth((k,))
            yield free,Cube(k,1 if shifted else 0)
            yield joint,Cube(k,0 if not shifted else a)
    t=Trainer("runner","one")
    train=fixture(0,"training")
    start=perf_counter_ns()
    for i,q in suite(train,0,0,6):
        assert t.observe(i,q)
    p=t.freeze()
    training_ns=perf_counter_ns()-start
    held=fixture(8,"heldout")
    start=perf_counter_ns()
    compiled=CompiledPolicy(p)
    bindings={id(i):compiled.bind(i) for i in held}
    compile_bind_ns=perf_counter_ns()-start
    print("NCR021_COMPILE",dict(training_ns=training_ns,compile_bind_ns=compile_bind_ns,
                              entries=len(compiled.table),packet_bytes=len(p.packet())))
    for name,shifted in (("matched",False),("shifted",True)):
        cases=tuple(suite(held,8,8,12,shifted))
        totals=[0,0,0]
        choices=Counter()
        for j,(i,q) in enumerate(cases):
            calls=(lambda:("baseline",i.realize(q)),
                   lambda:realize(p,i,q,actor="runner",segment="one"),
                   lambda:bindings[id(i)].realize(q,actor="runner",segment="one"))
            results={}
            # Rotate all three alternatives to reduce order effects.
            for k in (j%3,(j+1)%3,(j+2)%3):
                start=perf_counter_ns()
                s,r=calls[k]()
                totals[k]+=perf_counter_ns()-start
                assert verify_result(i.v,q,r)
                results[k]=(s,r)
            assert results[1][0]==results[2][0]
            assert results[0][1].status==results[1][1].status==results[2][1].status
            assert results[1][1].mechanism==results[2][1].mechanism
            choices[results[2][0]]+=1
        # Separate selection-only costs, same bound held-out cases, no proofs.
        selection=[]
        for method in (lambda i,q:p.choose(i.v,q),
                       lambda i,q:bindings[id(i)].choose(q,actor="runner",segment="one")):
            start=perf_counter_ns()
            for _ in range(100):
                for i,q in cases:
                    method(i,q)
            selection.append(perf_counter_ns()-start)
        print("NCR021",dict(case=name,requests=len(cases),baseline_ns=totals[0],
             interpreted_ns=totals[1],compiled_ns=totals[2],selection_calls=100*len(cases),
             interpreted_selection_ns=selection[0],compiled_selection_ns=selection[1],
             choices=dict(choices)))


if __name__=="__main__":
    main()
