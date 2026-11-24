"""Compare cold induction and algebraic induction on the same unseen tasks."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine
from navigable_compute.procedural_vocabulary import Vocabulary
import random


def main():
    f=construct_fibre(128,())
    obs=((1,2),(0,12,20,24))
    overlay=refine(f,obs)
    cold=Vocabulary(f,obs,overlay,context="same")
    symbolic=Vocabulary(f,obs,overlay,context="same")
    t=perf_counter_ns()
    c=cold.train()
    cold_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    s,trials,work=symbolic.train_symbolic()
    symbolic_ns=perf_counter_ns()-t
    rng=random.Random(15015)
    targets=[]
    for i in range(48):
        terms={0} if i%2 else set()
        for _ in range(1+i%4):
            k=sum(1<<j for j in rng.sample(range(5,128),1+i%3))
            p=((0,4,8,12),(0,4,16,20),(0,8,16,24))[i%3]
            for m in p:
                terms^={k|m}
        targets.append(tuple(sorted(terms)))
    t=perf_counter_ns()
    actual=[symbolic.reduce(q).value for q in targets]
    warm_ns=perf_counter_ns()-t
    assert actual==[cold.reduce(q).value for q in targets]==[i%2 for i in range(48)]
    shifted=[tuple(sorted(set(q)^{1<<127})) for q in targets]
    assert all(symbolic.reduce(q).value is None for q in shifted)
    print("NCR016",dict(cold_rules=c,symbolic_rules=s,trial_products=trials,
          product_terms=work,cold_train_ns=cold_ns,symbolic_train_ns=symbolic_ns,
          warm_ns=warm_ns,resolved=len(actual),unknown_controls=len(shifted),
          cold_packet_bytes=len(cold.packet()),symbolic_packet_bytes=len(symbolic.packet())))


if __name__=="__main__":
    main()
