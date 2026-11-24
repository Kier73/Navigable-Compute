"""Measured discovery, exact cold baseline and learned held-out campaign."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine
from navigable_compute.procedural_vocabulary import Vocabulary
import random


def mul(p,m):
    z=set()
    for a in p:
        z^={a|m}
    return z


def main():
    start=perf_counter_ns()
    parent=construct_fibre(128,())
    obs=((1,2),(0,12,20,24))
    v=Vocabulary(parent,obs,refine(parent,obs),context="demo")
    admission=perf_counter_ns()-start
    # Predeclare unseen spectator-only multiplier grammar and a shifted
    # control. Training sees none of these targets.
    rng=random.Random(15015)
    targets=[]
    for i in range(48):
        z={0} if i%2 else set()
        for _ in range(1+i%4):
            mask=sum(1<<j for j in rng.sample(range(5,128),1+i%3))
            p=((0,4,8,12),(0,4,16,20),(0,8,16,24))[i%3]
            z^=mul(p,mask)
        targets.append(tuple(sorted(z)))
    shifted=[tuple(sorted(set(p)^{1<<127})) for p in targets]
    t=perf_counter_ns()
    base=[v.reduce(p) for p in targets]
    base_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    learned=v.train()
    train_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    warm=[v.reduce(p) for p in targets]
    warm_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    cold=[v.exact(p) for p in targets]
    cold_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    unknown=[v.reduce(p) for p in shifted]
    shifted_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    cold_shifted=[v.exact(p)[0] for p in shifted]
    shifted_cold_ns=perf_counter_ns()-t
    assert all(a.value==b[0]==i%2 for i,(a,b) in enumerate(zip(warm,cold)))
    assert all(a.value is b is None for a,b in zip(unknown,cold_shifted))
    t=perf_counter_ns()
    restored=Vocabulary.restore(v.parent,v.observations,v.overlay,context=v.context,payload=v.packet())
    restore_ns=perf_counter_ns()-t
    assert restored.reduce(targets[0])==warm[0]
    print("NCR015",dict(domain=v.overlay.count,proposals=len(v.proposals()),learned=learned,
        base_resolved=sum(x.value is not None for x in base),
        learned_resolved=sum(x.value is not None for x in warm),
        shifted_unresolved=sum(x.value is None for x in unknown),
        cold_overlay_nodes=sum(len(p.nodes) for _,p in cold),
        cold_overlay_work=sum(p.work for _,p in cold),
        warm_steps=sum(len(p.steps) for p in warm),
        warm_work=sum(p.work for p in warm),
        admission_ns=admission,base_ns=base_ns,train_ns=train_ns,warm_ns=warm_ns,cold_ns=cold_ns,
        shifted_ns=shifted_ns,shifted_cold_ns=shifted_cold_ns,restore_ns=restore_ns,
        packet_bytes=len(v.packet())))


if __name__=="__main__":
    main()
