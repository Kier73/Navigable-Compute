"""NCB-001: construct a mathematical possibility space from a typed brief.

The brief describes locally available *possible effects*, not a learned or
already-written processor. Compilation expands mathematical clauses into an
explicit finite relation for NCP-001 to construct and independently check.
It does not assert that a real actuator or internet observer has these effects.
"""
from __future__ import annotations
from dataclasses import dataclass

from .possibility import (PossibilityAction,FactorContract,ConstructedGround,
                          construct_world)


@dataclass(frozen=True)
class EffectCase:
    """For states in [first,stop), possible successors are clipped s+delta
    and/or explicit absolute destinations. A state matching no case disables
    the action; cases in the same action cannot overlap.
    """
    first: int
    stop: int
    deltas: tuple[int,...] = ()
    absolute: tuple[int,...] = ()

    def __post_init__(self):
        if (type(self.first) is not int or type(self.stop) is not int or
            not 0 <= self.first < self.stop):
            raise ValueError('invalid case interval')
        for name in ('deltas','absolute'):
            values=getattr(self,name)
            if (not isinstance(values,tuple) or
                any(type(v) is not int for v in values) or
                tuple(sorted(set(values)))!=values):
                raise ValueError('effects must be ordered unique integer tuples')
        if not self.deltas and not self.absolute:
            raise ValueError('a case must have at least one possible effect')


@dataclass(frozen=True)
class ActionSketch:
    name: str
    cases: tuple[EffectCase,...]

    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name or len(self.name)>64:
            raise ValueError('action name required')
        if not isinstance(self.cases,tuple) or any(not isinstance(c,EffectCase) for c in self.cases):
            raise TypeError('cases must be typed')
        ordered=sorted(self.cases,key=lambda c:c.first)
        if any(a.stop>b.first for a,b in zip(ordered,ordered[1:])):
            raise ValueError('overlapping clauses make outcome semantics ambiguous')


@dataclass(frozen=True)
class GroundBrief:
    states: int
    initial: tuple[int,...]
    goal: tuple[int,...]
    unsafe: tuple[int,...]
    actions: tuple[ActionSketch,...]


def compile_brief(brief:GroundBrief)->FactorContract:
    """Construct possible-transition geometry from declarative effect clauses."""
    if not isinstance(brief,GroundBrief) or type(brief.states) is not int or not 2<=brief.states<=256:
        raise ValueError('qualified finite GroundBrief required')
    if not isinstance(brief.actions,tuple):
        raise TypeError('typed action sketches required')
    actions=[]
    for action in brief.actions:
        if not isinstance(action,ActionSketch):
            raise TypeError('typed action sketch required')
        if any(c.stop>brief.states for c in action.cases):
            raise ValueError('case interval exceeds source Ground')
        outcomes=[]
        for s in range(brief.states):
            case=next((c for c in action.cases if c.first<=s<c.stop),None)
            if case is None:
                outcomes.append(())  # disabled for this state
            else:
                destinations={min(brief.states-1,max(0,s+d)) for d in case.deltas}
                destinations.update(case.absolute)
                outcomes.append(tuple(sorted(destinations)))
        actions.append(PossibilityAction(action.name,tuple(outcomes)))
    return FactorContract(brief.states,brief.initial,brief.goal,brief.unsafe,tuple(actions))


def construct_from_briefs(briefs:tuple[GroundBrief,...],*,max_total_local_states:int=4096)->ConstructedGround:
    if not isinstance(briefs,tuple) or not briefs:
        raise ValueError('nonempty typed descriptions required')
    return construct_world(tuple(compile_brief(b) for b in briefs),
                           max_total_local_states=max_total_local_states)
