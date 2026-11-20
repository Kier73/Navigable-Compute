"""Deterministic NCI-001 demonstration; prints an evidence-scoped JSON report."""
import json
from hashlib import sha256
from math import log2

from navigable_compute.invariant_space import (
    InvariantDescription, construct_invariant_space, navigate_invariant_instance,
    construct_invariant_product, certify_invariant_obstruction)
from navigable_compute.invariant_verify import (
    verify_invariant_space, verify_invariant_product, verify_invariant_obstruction)
from navigable_compute.invariant_wire import (
    encode_invariant, decode_invariant, encode_invariant_obstruction,
    decode_invariant_obstruction)


def grid(side):
    coordinates = tuple((x, y) for y in range(side) for x in range(side))
    symmetry = tuple(x*side + side-1-y for x, y in coordinates)
    half = (side-1)/2
    goal = tuple(i for i, (x,y) in enumerate(coordinates)
                 if abs(x-half) < 1 and abs(y-half) < 1)
    return InvariantDescription(coordinates, symmetry, ('observation',)*len(coordinates),
                                tuple(range(len(coordinates))), goal, (), 1)


def run():
    desc=grid(7)
    space=construct_invariant_space(desc)
    other=navigate_invariant_instance(space, 499)
    assert verify_invariant_space(desc,space)
    assert verify_invariant_space(desc,other)
    wire=encode_invariant(desc,other)
    assert decode_invariant(wire,desc)==other

    combined=construct_invariant_product((desc,)*32,coordinate=10**50+17)
    assert verify_invariant_product((desc,)*32,combined)
    far=10**100+173
    assert combined.observe(far)==((24,),)*32

    base=grid(3)
    obstruction_spec=InvariantDescription(base.coordinates,base.symmetry,base.labels,
                                          (4,),(0,2,6,8),(),2)
    witness=certify_invariant_obstruction(obstruction_spec)
    assert verify_invariant_obstruction(obstruction_spec,witness)
    obstruction_wire=encode_invariant_obstruction(obstruction_spec,witness)
    assert decode_invariant_obstruction(obstruction_wire,obstruction_spec)==witness

    return {'schema':'navigable-compute.nci001.evidence.v0.1',
            'claims':'finite-exact-mathematical-only;no-physical-source-admission',
            'input_is_transition_table':False,
            'source_invariants':['L1 geometry','90-degree rotational equivariance',
                                 'observable-label preservation','safety','goal closure',
                                 'maximum one-step displacement'],
            'single_factor':{
                'states':len(desc.coordinates),'symmetry_orbits':len(space.orbits),
                'derived_shortest_convergence_steps':space.convergence_bound,
                'law_count_decimal':str(space.law_count),
                'two_instances_have_different_transition_laws':space.transition!=other.transition,
                'far_coordinate_decimal':str(far),
                'observed_goal_states':list(other.observe(far)),
                'verified':verify_invariant_space(desc,other),
                'wire_size_bytes':len(wire), 'wire_sha256':sha256(wire).hexdigest()},
            'factor_product':{
                'independent_factors':len(combined.factors),
                'explicit_local_states_analyzed':sum(len(f.description.coordinates) for f in combined.factors),
                'implicit_global_states_decimal':str(combined.global_state_count),
                'implicit_global_state_bits':combined.global_state_count.bit_length(),
                'independent_rank_optimal_laws_decimal':str(combined.law_count),
                'instance_coordinate_decimal':str(combined.coordinate),
                'worst_case_convergence_steps':combined.horizon,
                'all_32_observations_equal_goal':combined.observe(far)==((24,),)*32,
                'independent_verification':verify_invariant_product((desc,)*32,combined)},
            'impossibility':{'geometrically_near_but_symmetry_forbids_realization':True,
                             'trapped_initial':list(witness.trapped_initial),
                             'independent_verification':verify_invariant_obstruction(obstruction_spec,witness),
                             'wire_size_bytes':len(obstruction_wire),
                             'wire_sha256':sha256(obstruction_wire).hexdigest()},
            'scope':{'derived_transition_space':'finite, local and cyclic-group equivariant',
                     'law_family':'shortest-progress deterministic partial laws',
                     'unqualified_states':'explicitly undefined',
                     'external_realization':'not attempted',
                     'complexity':'source n^2 geometric+isometry work, factor-product not enumerated',
                     'RH':'motivating analogy only; no theorem imported'}}


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,indent=2))
