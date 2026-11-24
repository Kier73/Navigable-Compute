"""Measure a learned proof-producing operation on novel mathematical sources."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine
from navigable_compute.procedural_vocabulary import Vocabulary
from navigable_compute.induced_transversal import induce,activate,Skill


def make(source,context):
    f=construct_fibre(128,())
    rows=(source,)
    return Vocabulary(f,rows,refine(f,rows),context=context)


def main():
    t=perf_counter_ns()
    examples=[make((0,3,5,6),"train-a"),make((0,24,40,48),"train-b")]
    for v in examples:
        v.train_symbolic()
    train_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    skill=induce(tuple(examples))
    skill=Skill.restore(skill.packet())
    induce_restore_ns=perf_counter_ns()-t
    for label,source in (("triple",(0,7,56,448)),
                         ("star48",(0,)+tuple(1|(1<<i) for i in range(1,48)))):
        t=perf_counter_ns()
        v=make(source,label)
        admission_ns=perf_counter_ns()-t
        old=v.train_symbolic()[0]
        t=perf_counter_ns()
        a=activate(skill,v,0,max_arity=3)
        activation_ns=perf_counter_ns()-t
        targets=[]
        for i,l in enumerate(v.learned):
            k=(1<<(70+i%40))|(1<<126)
            terms=set()
            for m in l.proof.target:
                terms^={k|m}
            if i%2:
                terms^={0}
            targets.append(tuple(sorted(terms)))
        t=perf_counter_ns()
        actual=[v.reduce(q).value for q in targets]
        warm_ns=perf_counter_ns()-t
        t=perf_counter_ns()
        exact=[v.exact(q)[0] for q in targets]
        cold_ns=perf_counter_ns()-t
        assert actual==exact==[i%2 for i in range(len(targets))]
        print("NCR017",dict(case=label,domain=v.overlay.count,old_rules=len(old),
              new_rules=len(a.admitted),search_work=a.search_work,product_terms=a.product_terms,
              skill_bytes=len(skill.packet()),training_ns=train_ns,
              induce_restore_ns=induce_restore_ns,admission_ns=admission_ns,
              activation_ns=activation_ns,warm_ns=warm_ns,cold_ns=cold_ns,targets=len(targets)))


if __name__=="__main__":
    main()
