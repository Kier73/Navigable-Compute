"""NCR-013 benchmark: separately charge proof catalogue, policy and warm navigation."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_attention import Offer,Prices,plan_attention
from navigable_compute.adaptive_attention import compile_policy,Runtime


def run(label,target,offers):
    f=construct_fibre(128,())
    start=perf_counter_ns()
    c=plan_attention(f,offers,target,Prices(10),max_packet=2)
    formation=perf_counter_ns()-start
    start=perf_counter_ns()
    p=compile_policy(c)
    compile_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    r=Runtime(c,p)
    admission=perf_counter_ns()-start
    start=perf_counter_ns()
    for i in range(1000):
        s=r.begin("fixture",str(i))
        while not r.view(s).terminal:
            name=r.view(s).decision.next_offer
            s=r.advance(s,name,i>>len(s.trace)&1)
        v=r.view(s)
        assert v.inferred is not None
        assert v.outcome.overlay.rank(v.outcome.overlay.unrank(0))==0
    warm=perf_counter_ns()-start
    print(label,dict(domain=f.count,fixed_numerator=min(q.score_numerator for q in c.candidates),
          adaptive_numerator=p.numerator,denominator=f.count,
          prepared_nodes=c.prepared_nodes,decisions=len(p.decisions),
          formation_ns=formation,compile_with_audit_ns=compile_ns,
          admission_ns=admission,thousand_sessions_ns=warm))


if __name__=="__main__":
    run("or",(1,2,3),(Offer("a",(1,),1),Offer("b",(2,),1)))
    run("mux",(3,4,5),tuple(Offer(n,(m,),1) for n,m in zip("abc",(1,2,4))))
