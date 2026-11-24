"""Measured NCR-012 exact packet selection; preparation and audit are paid."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_attention import Offer, Prices, plan_attention, select_outcome
from navigable_compute.fibre_attention_verify import verify_attention


def run(label, f, offers, target, prices, max_packet):
    start = perf_counter_ns()
    p = plan_attention(f,offers,target,prices,max_packet=max_packet)
    prepare_ns = perf_counter_ns()-start
    start = perf_counter_ns()
    assert verify_attention(f,offers,target,prices,p,max_packet=max_packet)
    audit_ns = perf_counter_ns()-start
    print(label, dict(domain=f.count, selected=p.selected, prepare_ns=prepare_ns,
          audit_ns=audit_ns, prepared_work=p.prepared_work, prepared_nodes=p.prepared_nodes))
    for c in p.candidates:
        print(dict(packet=c.names, score_numerator=c.score_numerator,
                   denominator=f.count, unresolved_pairs=c.unresolved_pairs,
                   target_pairs=c.target_pairs, fallback_mass=c.fallback_mass,
                   node_mass=c.node_mass, visit_mass=c.visit_mass))
    return p


def main():
    f = construct_fibre(128, ())
    lo,hi = 1,1<<127
    p = run("geometry",f,(Offer("a-low",(lo,),1),Offer("z-high",(hi,),1)),
            (lo,hi,lo|hi),Prices(20,1,1,1),1)
    assert p.selected == ("z-high",)
    o = select_outcome(p,(("z-high",1),))
    assert o.inferred == 1 and o.overlay.parent_address(10**30)==(1<<127)+10**30
    run("complementarity",f,(Offer("a",(1,),1),Offer("b",(2,),1)),
        (1,2),Prices(10),2)
    path = construct_fibre(128,tuple((3<<i,) for i in range(127)))
    run("coupled",path,(Offer("end",(1<<127,),1),Offer("middle",(1<<64,),1),
                        Offer("other-end",(1,),1)),
        (1<<126,1<<127),Prices(20,1,1,1),1)


if __name__ == "__main__":
    main()
