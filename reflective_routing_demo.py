"""Paired observed schedule training with held-out matched and shifted intentions."""
import json
from collections import Counter
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine
from navigable_compute.procedural_vocabulary import Vocabulary
from navigable_compute.induced_transversal import induce
from navigable_compute.intention_interpreter import Interpreter,Truth,Cube
from navigable_compute.intention_interpreter_verify import verify_result
from navigable_compute.route_policy import Trainer,Policy,realize


def make(rows,context):
    f=construct_fibre(128,())
    return Vocabulary(f,rows,refine(f,rows),context=context)


def campaign(recorder=None):
    a,b=make(((0,3,5,6),),"train-a"),make(((0,24,40,48),),"train-b")
    a.train_symbolic()
    b.train_symbolic()
    skill=induce((a,b))
    # Training and evaluation have distinct coordinate placements and source labels.
    def fixture(offset,phase):
        shift=lambda p:tuple(m<<offset if m else 0 for m in p)
        joint=make(tuple(map(shift,((1,2),(2,4)))),phase+"-joint")
        pair=make((shift((0,)+tuple(3<<(2*i) for i in range(24))),),phase+"-pair")
        poly=make(tuple(map(shift,((1,2),(0,12,20,24)))),phase+"-poly")
        free=make((),phase+"-free")
        return tuple(Interpreter(v,skill) for v in (joint,pair,poly,free))
    train=fixture(0,"training")
    held=fixture(8,"heldout")

    def suite(engines,offset,start,n,shifted=False):
        joint,pair,poly,free=engines
        a,b,c=(1<<offset,2<<offset,4<<offset)
        for j in range(start,start+n):
            k=1<<(80+j)
            zeros=sum(1<<(offset+2*i+(j>>(i%3)&1)) for i in range(24))
            yield joint,Cube(a|k,c)  # joint span
            yield pair,Cube(0,zeros)  # direct cofactor
            yield poly,Truth(tuple(sorted((k|a,k|b))))  # rewrite
            yield poly,Truth((k,))  # mixed witness
            # Same observable feature class, but shifted assignments defeat
            # the deterministic first/last/midpoint/quartile witness samples.
            yield free,Cube(k,1 if shifted else 0)
            yield joint,Cube(k,0 if not shifted else a)  # changed overlap when shifted
    t=Trainer("runner","one",recorder=recorder)
    start=perf_counter_ns()
    for i,q in suite(train,0,0,6):
        assert t.observe(i,q)
    training_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    p=t.freeze()
    p=Policy.restore(p.packet(),actor="runner",segment="one")
    policy_ns=perf_counter_ns()-start
    results={}
    for label,shifted in (("matched",False),("shifted",True)):
        base_ns=learn_ns=0
        choices=Counter()
        routes=Counter()
        attempts=[0,0]
        count=0
        for j,(i,q) in enumerate(suite(held,8,8,12,shifted)):
            def baseline():
                return "baseline",i.realize(q)
            def learned():
                return realize(p,i,q,actor="runner",segment="one")
            measured={}
            for name,call in ((("baseline",baseline),("learned",learned)) if j%2==0 else
                              (("learned",learned),("baseline",baseline))):
                start=perf_counter_ns()
                choice,r=call()
                elapsed=perf_counter_ns()-start
                measured[name]=(choice,r,elapsed)
                assert verify_result(i.v,q,r)
            _,rb,bn=measured["baseline"]
            s,rl,ln=measured["learned"]
            assert rb.status==rl.status
            base_ns+=bn
            learn_ns+=ln
            choices[s]+=1
            routes[rl.mechanism]+=1
            attempts[0]+=len(rb.attempts)
            attempts[1]+=len(rl.attempts)
            count+=1
        results[label]=dict(requests=count,baseline_ns=base_ns,learned_ns=learn_ns,
                            choices=dict(choices),routes=dict(routes),attempts=attempts)
    report=dict(training_episodes=t.episodes,training_ns=training_ns,policy_ns=policy_ns,
                packet_bytes=len(p.packet()),stats=len(p.stats),
                contexts=len({s.context for s in p.stats}),results=results)
    print("NCR020",json.dumps(report,sort_keys=True))
    return report


if __name__=="__main__":
    campaign()
