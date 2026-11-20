"""NCP-002: a processor chosen from intention when two components share an actuator."""
from __future__ import annotations

import json

from navigable_compute.brief import ActionSketch,EffectCase,GroundBrief,construct_from_briefs
from navigable_compute.coupled import (
    CoupledBrief,SharedResource,construct_coupled,navigate_coupled,
    CoupledNotRealizable,certify_coupled_obstruction,
)
from navigable_compute.coupled_verify import (
    verify_coupled,verify_coupled_obstruction,
)


def run()->dict:
    """Compare a false independence assumption with admitted shared capacity."""
    local=GroundBrief(3,(2,),(0,),(),(
        ActionSketch('advance',(EffectCase(1,3,deltas=(-1,)),)),
        ActionSketch('hold',(EffectCase(0,3,deltas=(0,)),)),
    ))
    shared=SharedResource('actuator',1,((0,'advance'),(1,'advance')))
    source=CoupledBrief((local,local),(shared,))
    isolated=construct_from_briefs((local,local))
    actual=construct_coupled(source)
    assert verify_coupled(source,actual)
    assert isolated.convergence_bound==2 and actual.worst_steps==4
    alternative=navigate_coupled(actual,1)
    assert verify_coupled(source,alternative)
    assert alternative.policy!=actual.policy
    distant=10**100+173
    assert actual.observations_at(distant)==((0,0),)
    assert alternative.observations_at(distant)==((0,0),)
    impossible=CoupledBrief((local,local),(
        SharedResource('actuator',0,((0,'advance'),(1,'advance'))),
    ))
    try:
        construct_coupled(impossible)
    except CoupledNotRealizable:
        obstruction=certify_coupled_obstruction(impossible)
        assert verify_coupled_obstruction(impossible,obstruction)
    else:
        raise AssertionError('an impossible shared-actuator demand was accepted')
    return {
        'experiment':'NCP-002',
        'source':'declared goals, possible effects, joint resource capacity; no program supplied',
        'joint_state_count':len(actual.states),
        'naive_independent_worst_steps':isolated.convergence_bound,
        'joint_qualified_worst_steps':actual.worst_steps,
        'qualified_joint_policies_decimal':str(actual.optimal_laws),
        'canonical_processor_digest':actual.digest,
        'alternative_processor_digest':alternative.digest,
        'alternative_coordinate_decimal':'1',
        'distant_time_decimal':str(distant),
        'guaranteed_distant_observation':[list(s) for s in actual.observations_at(distant)],
        'impossible_capacity_trap': [list(s) for s in obstruction.trap],
        'obstruction_digest':obstruction.digest,
        'independent_checker':True,
        'real_actuators_qualified':False,
    }


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,indent=2))
