"""NCR-007 reproducible research: invariant-generated NONLINEAR source laws.

A 128-bit Boolean source is specified by tiered causal topology, conservation
and allowed polynomial degree, not a transition table. Independently chosen
architecture coordinates select two exact nonlinear transformations. A
source-observer Koopman Ground is then constructed without source-state
enumeration, checked independently and navigated to a distant coordinate.
"""
from __future__ import annotations

import json

from navigable_compute.nonlinear_koopman import (
    PolynomialInvariants, construct_polynomial_laws,
    construct_koopman_ground, navigate_nonlinear, PolynomialBudget)
from navigable_compute.nonlinear_koopman_verify import (
    verify_polynomial_laws,verify_koopman_ground,verify_nonlinear_result)
from navigable_compute.nonlinear_koopman_wire import (
    encode_nonlinear,decode_nonlinear)


def nonlinear_family(*, width: int = 128):
    if width != 128:
        raise ValueError('the declared multi-lane fixture is 128 bits')
    tiers = tuple([0,1,1,2,2,2,3,3] * 8 + [-1] * 64)
    inputs = []
    for start in range(0,64,8):
        inputs.extend((
            (start+1,start+2), (start+3,start+4),
            (start+3,start+4), (start+6,start+7),
            (start+6,start+7), (start+6,start+7), (), ()))
    inputs.extend([()] * 64)
    description = PolynomialInvariants(width,tiers,tuple(range(width)),
        tuple(inputs),2,((1,), (1<<64,)))
    # A zero-coordinate family reveals the *generated invariant-compatible*
    # coefficient orbits. The caller specifies no source transition matrix.
    prototype = construct_polynomial_laws(description,(0,))
    lookup = {candidate:index
              for index,orbit in enumerate(prototype.coefficient_orbits)
              for candidate in orbit}
    first = set()
    second = set()
    for start in range(0,64,8):
        first.update(((start+0,(1<<(start+1)) | (1<<(start+2))),
                      (start+1,1<<(start+3)),(start+2,1<<(start+4)),
                      (start+3,1<<(start+6)),(start+4,1<<(start+7))))
        second.update(((start+0,1<<(start+1)),
                       (start+1,1<<(start+4)),(start+2,1<<(start+3)),
                       (start+3,1<<(start+7)),(start+4,1<<(start+6)),
                       (start+5,(1<<(start+6)) | (1<<(start+7)))))
    coordinates = tuple(sum(1<<lookup[item] for item in selected)
                        for selected in (first,second))
    return construct_polynomial_laws(description,coordinates)


def long_shift_family(width: int = 14):
    description = PolynomialInvariants(width, tuple(range(width)),
        tuple(range(width)),
        tuple((i+1,) if i < width-1 else () for i in range(width)),
        1, ((1,),))
    prototype=construct_polynomial_laws(description,(0,))
    return construct_polynomial_laws(description,
        ((1<<len(prototype.coefficient_orbits))-1,))


def run() -> dict:
    family=nonlinear_family()
    assert verify_polynomial_laws(family)
    ground=construct_koopman_ground(family)
    assert verify_koopman_ground(family,ground)
    assert 4 < ground.dimension <= 16
    # Explicit nonlinear witness: output bit zero computes the AND of bits
    # one and two, not an affine or linear function of the source state.
    op=family.operators[0]
    x=1<<1
    y=1<<2
    assert ((op.apply(x)^op.apply(y)^op.apply(0))&1)==0
    assert op.apply(x|y)&1
    initial=sum(1<<i for i in (1,2,3,4,6,7,64))
    observed=[]
    for time in range(5):
        r=navigate_nonlinear(family,ground,initial,(0,),time)
        assert verify_nonlinear_result(family,ground,r)
        observed.append(int(r.observed_decimal))
    assert observed == [2,3,3,3,2]
    distant=10**100+173
    receipt=navigate_nonlinear(family,ground,initial,(0,),distant)
    assert verify_nonlinear_result(family,ground,receipt)
    assert int(receipt.observed_decimal)==2
    packet=encode_nonlinear(family,ground,receipt)
    assert decode_nonlinear(packet)==(family,ground,receipt)

    # Counterexample: a one-bit observer on a long causal shift eventually
    # depends on every source bit, so feature truncation must refuse.
    counter=long_shift_family()
    rejected=False
    try:
        construct_koopman_ground(counter,max_features=8)
    except PolynomialBudget:
        rejected=True
    assert rejected
    complete=construct_koopman_ground(counter)
    assert verify_koopman_ground(counter,complete)
    assert complete.dimension==14
    return {'schema':'navigable-compute.ncr007-invariant-nonlinear-evidence.v0.1',
            'synthetic_source':{
                'source_bit_width':family.description.width,
                'implicit_source_states_decimal':str(1<<family.description.width),
                'source_states_enumerated':0,
                'affine_source_assumed':False,
                'genuinely_quadratic_source_witness':True,
                'symmetry_equivariant_law_coefficient_orbits':len(family.coefficient_orbits),
                'possible_individual_source_laws_decimal':family.law_count_decimal,
                'constructed_generators':len(family.operators),
                'drain_horizon_all_source_states':family.description.drain_horizon,
                'observer_output_bits':len(family.description.observers),
                'exact_koopman_feature_dimension':ground.dimension,
                'linear_quotient_state_capacity_upper_bound_decimal':str(1<<ground.dimension),
                'realized_observer_state_classes_not_enumerated':True,
                'first_five_observations':observed,
                'distant_coordinate_decimal':receipt.repetitions_decimal,
                'distant_observation_decimal':receipt.observed_decimal,
                'distant_quotient_matrix_products':receipt.operator_products,
                'original_source_state_trajectory_replayed':False,
                'packet_bytes':len(packet),
                'mathematical_family_sha256':family.digest,
                'observer_ground_sha256':ground.digest},
            'negative_control':{
                'source_bits':counter.description.width,
                'minimum_linear_observer_features':complete.dimension,
                'eight_feature_budget_refused':rejected},
            'scope':'synthetic fully stated causal-invariant Boolean law families; not external system identification or physical execution'}


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,separators=(',',':')))
