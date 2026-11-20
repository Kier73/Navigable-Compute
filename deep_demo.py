"""Reproducible NCA-001 experiments; reports actual counts, not simulated timings."""
from __future__ import annotations
import json
from navigable_compute import Instruction as I, Program
from navigable_compute.observability import ObservationAt
from navigable_compute.architect import Intention,realize


def main():
    epoch=10**33+123
    program=Program(8,tuple(I('add_const',i,2*i+1) for i in range(8)))
    seed=(0x2a,0x4d,0x66)
    q=(ObservationAt(epoch,0,1,(seed[0]+epoch)&1),
       ObservationAt(epoch,1,128,((seed[1]+3*epoch)&255)&128),
       ObservationAt(epoch,2,15,((seed[2]+5*epoch)&255)&15),
       ObservationAt(epoch+1,0,1,(seed[0]+epoch+1)&1))
    n=realize(Intention(8,q,program))
    c=n.certificate
    source_init=201
    delta=187
    far=10**43+255
    partial=tuple(ObservationAt(far+k,0,255,(source_init+delta*(far+k))%256)
                  for k in (0,64,128))
    full=partial+(ObservationAt(far+1,0,255,(source_init+delta*(far+1))%256),)
    p=realize(Intention(1,partial))
    f=realize(Intention(1,full))
    result={
        'schema':'navigable-compute.deep-experiments.v0.2',
        'machine':{'global_bits':8*program.size,'global_state_count_decimal':str(1<<(8*program.size)),
                   'coordinate_decimal':str(epoch),'observed_components':len(c.certificates),
                   'examined_local_states':c.examined_states,
                   'quotient_classes':[v.classes for v in c.certificates],
                   'map_products':c.products,
                   'compatible_initial_states_decimal':str(c.candidate_count),
                   'constructed_initial_hex':n.initial.hex(),
                   'independent_checker':n.independent_checker,
                   'receipt_sha256':c.witness_digest},
        'law_construction':{'grammar':'independent byte addition mod 256',
                            'observations_first':len(partial),
                            'law_candidates_after_first_decimal':str(p.certificate.law_solution_count),
                            'law_candidates_after_new_observation_decimal':str(f.certificate.law_solution_count),
                            'first_inferred_delta':p.program.instructions[0].b,
                            'final_inferred_delta':f.program.instructions[0].b,
                            'final_initial_byte':f.initial[0],
                            'independent_checker':f.independent_checker,
                            'receipt_sha256':f.certificate.witness_digest}}
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__': main()
