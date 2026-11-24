"""Factored intention proof versus fresh exact coordinate intersection."""
from time import perf_counter_ns
from navigable_compute.intention_fibre import construct_fibre
from navigable_compute.fibre_refinement import refine,observation_rows
from navigable_compute.fibre_refinement_verify import verify_refinement
from navigable_compute.procedural_vocabulary import Vocabulary
from navigable_compute.induced_transversal import induce
from navigable_compute.intention_cofactor import prove_cube
from navigable_compute.intention_cofactor_verify import verify_cube


def make(source,context):
    f=construct_fibre(128,())
    obs=(source,)
    return Vocabulary(f,obs,refine(f,obs),context=context)


def main():
    t=perf_counter_ns()
    vs=[make((0,3,5,6),"train-a"),make((0,24,40,48),"train-b")]
    for v in vs:
        v.train_symbolic()
    skill=induce(tuple(vs))
    training_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    r=(0,)+tuple(3<<(2*i) for i in range(48))
    v=make(r,"pairs48")
    admission_ns=perf_counter_ns()-t
    # Every distinct choice of one zero coordinate per pair excludes the cube.
    # Test 32 such choices and 32 controls with one pair left unassigned.
    masks=tuple(sum(1<<(2*j+(i>> (j%5)&1)) for j in range(48)) for i in range(32))
    controls=tuple(z&~(3<<94) for z in masks)
    ones=1<<127
    t=perf_counter_ns()
    proofs=tuple(prove_cube(skill,v,ones,z) for z in masks)
    assert all(verify_cube(v,ones,z,p) for z,p in zip(masks,proofs))
    warm_ns=perf_counter_ns()-t
    t=perf_counter_ns()
    negative=tuple(prove_cube(skill,v,ones,z) for z in controls)
    shifted_ns=perf_counter_ns()-t
    assert all(p is None for p in negative)
    t=perf_counter_ns()
    counts=[]
    cold_work=0
    for z in masks+controls:
        rows=v.observations+observation_rows(128,ones|z,ones)
        o=refine(v.parent,rows)
        assert verify_refinement(v.parent,rows,o)
        counts.append(o.count)
        cold_work+=o.work
    cold_ns=perf_counter_ns()-t
    assert not any(counts[:32]) and all(counts[32:])
    print("NCR018",dict(domain=v.overlay.count,implicit_minimal_transversals=2**48,
          expanded_factor_terms=2**48,retained_rules=len(v.learned),proved=len(proofs),
          unknown=len(negative),certificate_bytes=len(proofs[0].packet()),
          training_ns=training_ns,admission_ns=admission_ns,warm_ns=warm_ns,
          shifted_ns=shifted_ns,cold_ns=cold_ns,cold_work=cold_work,
          certificate_source_terms=proofs[0].inspected_terms))


if __name__=="__main__":
    main()
