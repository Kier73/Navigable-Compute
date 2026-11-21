"""NCR-006: 2^128 source states -> exact 3-bit future-stable processor.

A source-ground experiment: original bitmatrix values are stipulated, not
observations of a physical system, unknown neural model or external future.
"""
from __future__ import annotations
import json

from navigable_compute.symbolic_quotient import (
    AffineMap,SymbolicSource,construct_symbolic_quotient,navigate_symbolic)
from navigable_compute.symbolic_quotient_verify import (
    verify_symbolic_quotient,verify_symbolic_result)
from navigable_compute.symbolic_quotient_wire import encode_symbolic,decode_symbolic


def hidden_family(n=128,observed=3):
    generators=[]
    hidden=n-observed
    for g in range(observed):
        shift=(1,7,13)[g]%hidden
        rows=[1<<i for i in range(n)]
        for i in range(observed,n):
            rows[i]=1<<(observed+(i-observed+shift)%hidden)
        rows[observed]^=1<<((g+1)%observed)
        generators.append(AffineMap(tuple(rows),(1<<g)|(1<<(observed+shift))))
    return SymbolicSource(n,tuple(generators),tuple(1<<i for i in range(observed)),
                          'declared-hidden-operator-family')


def run():
    source=hidden_family()
    quotient=construct_symbolic_quotient(source)
    assert verify_symbolic_quotient(source,quotient)
    time=10**100+173
    result=navigate_symbolic(source,quotient,(1<<120)|4,(0,1,2,0,2),time)
    assert verify_symbolic_result(source,quotient,result)
    packet=encode_symbolic(source,quotient,result)
    assert decode_symbolic(packet)==(source,quotient,result)
    assert quotient.dimension==3 and int(result.observed_decimal)==6

    # A countermodel: observer bit zero eventually reveals all 48 source bits.
    width=48
    shifting=SymbolicSource(width,(
        AffineMap(tuple((1<<(i+1)) if i<width-1 else 0
                        for i in range(width)),0),),(1,),
        'declared-shift-counterexample')
    refused=False
    try:
        construct_symbolic_quotient(shifting,max_dimension=8)
    except ValueError:
        refused=True
    assert refused
    full=construct_symbolic_quotient(shifting,max_dimension=48)
    assert verify_symbolic_quotient(shifting,full)
    assert full.dimension==48

    return {'schema':'navigable-compute.ncr006-symbolic-quotient-evidence.v0.1',
            'large_source':{
                'source_bit_width':source.width,
                'implicit_source_state_count_decimal':str(1<<source.width),
                'source_states_enumerated':0,
                'source_operator_monoid_enumerated':False,
                'observer_dimensions':len(source.observer_rows),
                'derived_minimal_future_quotient_bits':quotient.dimension,
                'derived_quotient_state_count':1<<quotient.dimension,
                'generator_count':len(source.generators),
                'quotient_sha256':quotient.digest,
                'logical_coordinate_decimal':result.repetitions_decimal,
                'operator_products':result.operator_products,
                'observed_binary':format(int(result.observed_decimal),'03b'),
                'packet_bytes':len(packet),
                'independent_full_source_operator_audit':True,
                'source_states_replayed':0},
            'negative_control':{
                'source_bit_width':48,
                'required_future_quotient_bits':full.dimension,
                'requested_eight_bit_budget_refused':refused},
            'scope':'exact fully supplied GF(2) affine source laws only; no external physical realization'}


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,separators=(',',':')))
