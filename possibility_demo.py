"""NCP-001: describe a possibility space; construct a robust processor within it."""
from __future__ import annotations
import json
from navigable_compute.possibility import (FactorContract,PossibilityAction,construct_world,navigate_instance)
from navigable_compute.possibility_verify import verify_world
from navigable_compute.brief_verify import verify_brief_world


def fixture_brief():
    from navigable_compute.brief import EffectCase,ActionSketch,GroundBrief
    n=16
    return GroundBrief(n,tuple(range(15)),(0,),(15,),(
        ActionSketch('advance',(
            EffectCase(0,1,absolute=(0,)),
            EffectCase(1,15,deltas=(-1,)),
            EffectCase(15,16,absolute=(15,)),)),
        ActionSketch('skip',(
            EffectCase(0,2,absolute=(0,)),
            EffectCase(2,15,deltas=(-2,-1)),
            EffectCase(15,16,absolute=(15,)),)),
        ActionSketch('hazard',(
            EffectCase(0,4,absolute=(0,)),
            EffectCase(4,15,deltas=(-4,),absolute=(15,)),
            EffectCase(15,16,absolute=(15,)),))))


def fixture():
    from navigable_compute.brief import compile_brief
    return compile_brief(fixture_brief())


def run():
    from navigable_compute.brief import construct_from_briefs
    descriptions=(fixture_brief(),)*32
    worlds=tuple(fixture() for _ in range(32))
    result=construct_from_briefs(descriptions)
    assert verify_world(worlds,result)
    assert verify_brief_world(descriptions,result)
    alternative=navigate_instance(result,10**30+17)
    assert verify_world(worlds,alternative)
    assert verify_brief_world(descriptions,alternative)
    assert alternative.instance_coordinate!=result.instance_coordinate
    remote=(10**100)+173
    observations=result.observations_at(remote)
    assert observations==((0,),)*32
    assert result.observations_at(0)==(tuple(range(15)),)*32
    report={'experiment':'NCP-001','source':'declarative effect clauses and goals; no program supplied',
            'global_possible_states_decimal':str(result.global_states),
            'admissible_initial_states_decimal':str(result.initial_states),
            'factored_optimal_processor_laws_decimal':str(result.factored_optimal_laws),
            'states_examined_in_construction':sum(x.states for x in worlds),
            'worst_case_cycles_to_goal':result.convergence_bound,
            'target_coordinate_decimal':str(remote),
            'unique_guaranteed_observation':list(observations[0]),
            'all_factors_same_observation':len(set(observations))==1,
            'certificate_sha256':result.digest,
            'alternate_instance_coordinate_decimal':str(alternative.instance_coordinate),
            'alternate_certificate_sha256':alternative.digest,
            'alternate_worst_case_cycles':alternative.convergence_bound,
            'independent_checker':True,
            'provenance':'mathematical possibility contract, not observed physical host'}
    return report


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,indent=2))
