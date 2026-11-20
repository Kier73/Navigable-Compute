"""NCP-002 independent checker for shared-resource coupled Grounds.

This checker never invokes the joint attractor constructor or its _graph.
It reconstructs source effects from original EffectCase clauses and validates
every Bellman rank, winning/losing state, selected policy coordinate and
adversarial obstruction. It verifies mathematical consistency, not actuators.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from math import prod
import json

from .coupled import CoupledBrief,CoupledGround,CoupledObstruction


def _sha(document:dict)->str:
    """Hash canonical JSON without trusting a supplied witness digest."""
    return sha256(json.dumps(document,sort_keys=True,separators=(',',':'),
                             ensure_ascii=True).encode()).hexdigest()


def _transitions(source:CoupledBrief,state:tuple[int,...]):
    """Independently expand original clauses and resource claims at this state."""
    per_factor=[]
    for i,(x,brief) in enumerate(zip(state,source.factors)):
        if x in brief.goal:
            per_factor.append(((-1,(x,)),))
            continue
        selections=[]
        for number,action in enumerate(brief.actions):
            matching=[case for case in action.cases if case.first<=x<case.stop]
            if len(matching)>1:
                return {}
            if not matching:
                continue
            case=matching[0]
            destinations=set(case.absolute)
            destinations.update(max(0,min(brief.states-1,x+d)) for d in case.deltas)
            if destinations:
                selections.append((number,tuple(sorted(destinations))))
        per_factor.append(tuple(selections))
    result={}
    for picked in product(*per_factor):
        indices=tuple(i for i,_ in picked)
        exceeded=False
        for shared in source.resources:
            used=sum((factor,source.factors[factor].actions[index].name)
                     in shared.claims for factor,index in enumerate(indices)
                     if index>=0)
            if used>shared.capacity:
                exceeded=True
                break
        if exceeded:
            continue
        successor=tuple(product(*(choices for _,choices in picked)))
        if successor:
            result[indices]=successor
    return result


def _shape(source:CoupledBrief):
    """Compute complete bounded joint-state basis and actual unsafe/goal classes."""
    if not isinstance(source,CoupledBrief):
        return None
    count=prod(b.states for b in source.factors)
    actions=prod(len(b.actions) for b in source.factors)
    if not 2<=count<=4096 or not 1<=actions<=4096:
        return None
    states=tuple(product(*(range(b.states) for b in source.factors)))
    safety={s for s in states if s not in source.unsafe_joint
            and all(x not in b.unsafe for x,b in zip(s,source.factors))}
    goals={s for s in safety if all(x in b.goal for x,b in zip(s,source.factors))}
    initial=set(product(*(b.initial for b in source.factors)))
    return states,safety,goals,initial


def verify_coupled(source:CoupledBrief, world:CoupledGround)->bool:
    """Verify an exact rank-optimal jointly observed controller without synthesis."""
    try:
        if not isinstance(world,CoupledGround) or world.source!=source:
            return False
        basis=_shape(source)
        if basis is None:
            return False
        states,safety,goals,initial=basis
        position={state:i for i,state in enumerate(states)}
        n=len(states)
        if (world.states!=states or len(world.rank)!=n or
            len(world.optimal)!=n or len(world.policy)!=n or
            any(type(x) is not int or not -1<=x<=n for x in world.rank)):
            return False
        if (type(world.instance_coordinate) is not int or
            type(world.optimal_laws) is not int or world.optimal_laws<1 or
            world.optimal_laws.bit_length()>4096 or
            not 0<=world.instance_coordinate<world.optimal_laws):
            return False
        remainder=world.instance_coordinate
        total=1
        for i,state in enumerate(states):
            depth=world.rank[i]
            if state in goals:
                if depth!=0 or world.optimal[i] or world.policy[i]:
                    return False
                continue
            if (state not in safety and depth!=-1) or (state in safety and depth==0):
                return False
            if not isinstance(world.optimal[i],tuple) or not isinstance(world.policy[i],tuple):
                return False
            relation=_transitions(source,state) if state in safety else {}
            if depth<0:
                if world.optimal[i] or world.policy[i]:
                    return False
                # A losing safe state cannot have even one universally winning action.
                if any(all(t in safety and world.rank[position[t]]>=0
                           for t in outcome)
                       for outcome in relation.values()):
                    return False
                continue
            choices=[]
            for action,outcomes in relation.items():
                if any(t not in safety or world.rank[position[t]]<0
                       for t in outcomes):
                    continue
                maximum=max(world.rank[position[t]] for t in outcomes)
                if maximum < depth-1:  # a strictly faster controller exists
                    return False
                if maximum==depth-1:
                    choices.append(action)
            if not choices or tuple(sorted(choices))!=world.optimal[i]:
                return False
            chosen=choices[remainder%len(choices)]
            remainder//=len(choices)
            if world.policy[i]!=chosen:
                return False
            total*=len(choices)
        if remainder or total!=world.optimal_laws:
            return False
        if any(world.rank[position[s]]<0 for s in initial):
            return False
        if world.worst_steps!=max(world.rank[position[s]] for s in initial):
            return False
        return world.digest==_sha(world.encode())
    except (ValueError,TypeError,IndexError,KeyError,AttributeError,OverflowError):
        return False


def verify_coupled_obstruction(source:CoupledBrief,witness:CoupledObstruction)->bool:
    """Check a source-bound adversarial trap without running an attractor."""
    try:
        if not isinstance(witness,CoupledObstruction):
            return False
        basis=_shape(source)
        if basis is None:
            return False
        states,safety,goals,initial=basis
        trap=witness.trap
        if (not isinstance(trap,tuple) or not trap or
            tuple(sorted(set(trap)))!=trap or
            any(s not in safety or s in goals for s in trap)):
            return False
        trapped=tuple(s for s in states if s in initial and s in trap)
        if not trapped or trapped!=witness.trapped_initial:
            return False
        for state in trap:
            for outcomes in _transitions(source,state).values():
                # If the environment cannot keep the controller in the
                # trap or an unsafe state, this cannot be a losing witness.
                if all(t in safety and t not in trap for t in outcomes):
                    return False
        return (witness.source_digest==_sha(source.encode()) and
                witness.digest==_sha(witness.encode()))
    except (ValueError,TypeError,IndexError,AttributeError,OverflowError):
        return False
