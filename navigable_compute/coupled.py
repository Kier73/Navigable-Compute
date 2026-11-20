"""NCP-002: construct a jointly realizable processor from coupled intentions.

NCP-001 constructs enormous product controllers under an explicit independence
assumption. This lane instead compiles *shared resource contention* and
forbidden joint states into one finite adversarial reachability Ground.

No executable controller, trajectory or training history is an input.
Construction enumerates the bounded joint space on purpose: an unsupported
independence assumption is never used as an invisible optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from math import prod
import json

from .brief import GroundBrief, compile_brief
from .possibility import FactorContract, NotRealizable


class CoupledNotRealizable(NotRealizable):
    """At least one admitted joint initial state cannot be forced to a goal."""


def _digest(document: dict) -> str:
    """Content address exact source clauses and candidate policy."""
    return sha256(json.dumps(document,sort_keys=True,separators=(',',':'),
                             ensure_ascii=True).encode()).hexdigest()


def _brief_record(brief: GroundBrief) -> dict:
    """Retain original declarative clauses rather than just their expansion."""
    return {
        'states':brief.states, 'initial':list(brief.initial),
        'goal':list(brief.goal), 'unsafe':list(brief.unsafe),
        'actions':[{
            'name':a.name,
            'cases':[{'first':c.first,'stop':c.stop,
                      'deltas':list(c.deltas),'absolute':list(c.absolute)}
                     for c in a.cases],
        } for a in brief.actions],
    }


@dataclass(frozen=True)
class SharedResource:
    """One action capacity spanning multiple explicitly named factors.

    A claim is (factor index, local action name). A simultaneous joint action
    consumes one slot for each matching claim. Capacity is a *stipulated*
    mathematical property, not proof of real actuator availability.
    """

    name: str
    capacity: int
    claims: tuple[tuple[int,str], ...]

    def __post_init__(self) -> None:
        """Reject ambiguous interface or negative/Boolean resource capacities."""
        if type(self.name) is not str or not 1<=len(self.name)<=64:
            raise ValueError('resource name must be bounded')
        if type(self.capacity) is not int or self.capacity < 0:
            raise ValueError('resource capacity must be a nonnegative integer')
        if (type(self.claims) is not tuple or not self.claims
                or tuple(sorted(set(self.claims))) != self.claims
                or any(type(x) is not tuple or len(x)!=2
                       or type(x[0]) is not int or x[0]<0
                       or type(x[1]) is not str or not x[1]
                       for x in self.claims)):
            raise ValueError('resource claims must be unique ordered factor/actions')

    def encode(self) -> dict:
        """Describe the exact shared resource without claiming its physical existence."""
        return {'name':self.name,'capacity':self.capacity,
                'claims':[[i,a] for i,a in self.claims]}


@dataclass(frozen=True)
class CoupledBrief:
    """An intention plus explicit joint-state and shared-actuator relations."""

    factors: tuple[GroundBrief, ...]
    resources: tuple[SharedResource, ...] = ()
    unsafe_joint: tuple[tuple[int,...], ...] = ()

    def __post_init__(self) -> None:
        """Require a nonambiguous, finite source contract."""
        if (type(self.factors) is not tuple or not 2<=len(self.factors)<=8
                or any(not isinstance(b,GroundBrief) for b in self.factors)):
            raise ValueError('2..8 typed component briefs required')
        if (type(self.resources) is not tuple
                or any(not isinstance(r,SharedResource) for r in self.resources)
                or len(set(r.name for r in self.resources))!=len(self.resources)):
            raise ValueError('resources must be typed with unique names')
        for r in self.resources:
            if any(i>=len(self.factors) or
                   name not in {a.name for a in self.factors[i].actions}
                   for i,name in r.claims):
                raise ValueError('resource claim references an unknown action')
        if (type(self.unsafe_joint) is not tuple or
                tuple(sorted(set(self.unsafe_joint)))!=self.unsafe_joint):
            raise ValueError('unsafe joint states must be ordered unique tuples')
        for state in self.unsafe_joint:
            if (type(state) is not tuple or len(state)!=len(self.factors)
                    or any(type(s) is not int or not 0<=s<b.states
                           for s,b in zip(state,self.factors))):
                raise ValueError('unsafe joint state outside declared product')
        for start in product(*(b.initial for b in self.factors)):
            if start in self.unsafe_joint:
                raise ValueError('admitted initial state violates joint safety')

    def encode(self) -> dict:
        """Bind the complete original brief and interface assumptions."""
        return {'schema':'navigable-compute.coupled-brief.v0.1',
                'factors':[_brief_record(b) for b in self.factors],
                'resources':[r.encode() for r in self.resources],
                'unsafe_joint':[list(s) for s in self.unsafe_joint]}


@dataclass(frozen=True)
class CoupledGround:
    """An addressed state-observed *joint* controller with exact Bellman ranks."""

    source: CoupledBrief
    states: tuple[tuple[int,...], ...]
    rank: tuple[int,...]
    optimal: tuple[tuple[tuple[int,...],...], ...]
    policy: tuple[tuple[int,...], ...]
    optimal_laws: int
    worst_steps: int
    instance_coordinate: int
    digest: str

    def encode(self) -> dict:
        """Expose source, all joint coordinates and a fully checkable controller."""
        return {'schema':'navigable-compute.coupled-world.v0.1',
                'source':self.source.encode(),
                'states':[list(s) for s in self.states],
                'rank':list(self.rank),
                'optimal':[[list(a) for a in row] for row in self.optimal],
                'policy':[list(a) for a in self.policy],
                'optimal_laws_decimal':str(self.optimal_laws),
                'worst_steps':self.worst_steps,
                'instance_coordinate_decimal':str(self.instance_coordinate)}

    def packet(self) -> dict:
        """Return an inert digest-bound mathematical certificate, not authority."""
        return {**self.encode(),'sha256':self.digest}

    def observations_at(self, coordinate:int) -> tuple[tuple[int,...], ...]:
        """Observe the exact set of joint states at a bounded or distant time.

        The separate checker is required before navigation. For a proven
        convergent controller only the bounded progress prefix is needed,
        even when the logical time coordinate is astronomically large.
        """
        from .coupled_verify import verify_coupled
        if not verify_coupled(self.source,self):
            raise ValueError('cannot navigate an unverified coupled controller')
        if type(coordinate) is not int or coordinate<0 or coordinate.bit_length()>4096:
            raise ValueError('invalid logical time coordinate')
        graph=_graph(
            self.source, max_states=max(512,len(self.states)),
            max_actions=max(256,prod(len(b.actions) for b in self.source.factors)),
        )
        options,successors,safe,goals,initial=graph
        present=set(initial)
        for _ in range(min(coordinate,self.worst_steps)):
            updated=set()
            for i in present:
                if i in goals:
                    updated.add(i)
                else:
                    action=self.policy[i]
                    updated.update(successors[i][action])
            present=updated
        return tuple(self.states[i] for i in sorted(present))


def _graph(source:CoupledBrief,*,max_states:int=512,max_actions:int=256):
    """Construct an explicit bounded product including actual resource conflicts."""
    if not isinstance(source,CoupledBrief):
        raise TypeError('typed coupled brief required')
    if (type(max_states) is not int or not 2<=max_states<=4096
            or type(max_actions) is not int or not 1<=max_actions<=4096):
        raise ValueError('invalid bounded joint construction budget')
    contracts=tuple(compile_brief(b) for b in source.factors)
    space=prod(b.states for b in source.factors)
    count=prod(len(c.actions) for c in contracts)
    if space>max_states or count>max_actions:
        raise ValueError('joint coupling exceeds explicit construction budget')
    states=tuple(product(*(range(b.states) for b in source.factors)))
    lookup={s:i for i,s in enumerate(states)}
    forbidden=set(source.unsafe_joint)
    safe=set(i for i,s in enumerate(states)
             if s not in forbidden and all(x not in c.unsafe for x,c in zip(s,contracts)))
    goals={i for i,s in enumerate(states) if i in safe and
           all(x in c.goal for x,c in zip(s,contracts))}
    initial=tuple(lookup[s] for s in product(*(c.initial for c in contracts)))
    options=[]
    successors=[]
    for i,state in enumerate(states):
        row={}
        if i in safe and i not in goals:
            local=tuple(
                (-1,) if x in c.goal else
                tuple(j for j,a in enumerate(c.actions) if a.outcomes[x])
                for x,c in zip(state,contracts)
            )
            for action in product(*local):
                if any(sum((j,a.name) in r.claims
                           for j,(choice,c) in enumerate(zip(action,contracts))
                           if choice>=0 for a in (c.actions[choice],)) > r.capacity
                       for r in source.resources):
                    continue
                results=tuple(tuple(choices) for choices in product(*(
                    (x,) if choice<0 else c.actions[choice].outcomes[x]
                    for x,c,choice in zip(state,contracts,action)
                )))
                if results:
                    row[action]=tuple(lookup[z] for z in results)
        options.append(tuple(sorted(row)))
        successors.append(row)
    return tuple(options),tuple(successors),safe,goals,initial


def construct_coupled(source:CoupledBrief,*,max_states:int=512,
                      max_actions:int=256)->CoupledGround:
    """Solve robust joint reachability, then address a controller without guessing.

    The environmental quantifiers are EXISTS one state-observed joint policy,
    FOR ALL admitted initial states and FOR ALL declared possible outcomes.
    No physical host capacity is inferred from a mathematical SharedResource.
    """
    options,successors,safe,goals,initial=_graph(
        source,max_states=max_states,max_actions=max_actions)
    n=len(options)
    rank=[-1]*n
    for goal in goals:
        rank[goal]=0
    for depth in range(1,n+1):
        new=[]
        for i in sorted(safe):
            if rank[i]>=0:
                continue
            if any(all(rank[j]>=0 and rank[j]<depth for j in successors[i][a])
                   for a in options[i]):
                new.append(i)
        if not new:
            break
        for i in new:
            rank[i]=depth
    if any(rank[i]<0 for i in initial):
        raise CoupledNotRealizable('joint resource/safety relation prevents guaranteed realization')
    optimum=[]
    chosen=[]
    laws=1
    for i in range(n):
        if rank[i]<=0:
            optimum.append(())
            chosen.append(())
            continue
        eligible=tuple(a for a in options[i]
                       if all(rank[j]>=0 for j in successors[i][a]) and
                       max(rank[j] for j in successors[i][a])==rank[i]-1)
        if not eligible:
            raise AssertionError('joint Bellman rank has no admissible action')
        optimum.append(eligible)
        chosen.append(eligible[0])
        laws*=len(eligible)
    if laws.bit_length()>4096:
        raise ValueError('controller instance coordinate exceeds qualified size')
    states=tuple(product(*(range(b.states) for b in source.factors)))
    prototype=CoupledGround(source,states,tuple(rank),tuple(optimum),tuple(chosen),
                            laws,max(rank[i] for i in initial),0,'')
    return CoupledGround(source,states,prototype.rank,prototype.optimal,
                         prototype.policy,laws,prototype.worst_steps,0,
                         _digest(prototype.encode()))


def navigate_coupled(world:CoupledGround,coordinate:int)->CoupledGround:
    """Navigate a selected rank-optimal joint policy by exact mixed-radix index."""
    from .coupled_verify import verify_coupled
    if not isinstance(world,CoupledGround) or not verify_coupled(world.source,world):
        raise ValueError('coupled source Ground is not independently verified')
    if (type(coordinate) is not int or coordinate<0 or
            coordinate>=world.optimal_laws or coordinate.bit_length()>4096):
        raise ValueError('joint processor coordinate outside certified instance space')
    rest=coordinate
    chosen=[]
    for options in world.optimal:
        if not options:
            chosen.append(())
        else:
            chosen.append(options[rest%len(options)])
            rest//=len(options)
    if rest:
        raise AssertionError('joint instance coordinate did not exhaust')
    proposed=CoupledGround(world.source,world.states,world.rank,world.optimal,
                           tuple(chosen),world.optimal_laws,world.worst_steps,
                           coordinate,'')
    return CoupledGround(proposed.source,proposed.states,proposed.rank,
                         proposed.optimal,proposed.policy,proposed.optimal_laws,
                         proposed.worst_steps,coordinate,_digest(proposed.encode()))


@dataclass(frozen=True)
class CoupledObstruction:
    """A safe adversarial trap excluding goal convergence for some initial state."""

    source_digest: str
    trap: tuple[tuple[int,...],...]
    trapped_initial: tuple[tuple[int,...],...]
    digest: str

    def encode(self) -> dict:
        """Expose all trapped joint coordinates for independent verification."""
        return {'schema':'navigable-compute.coupled-obstruction.v0.1',
                'source_digest':self.source_digest,
                'trap':[list(s) for s in self.trap],
                'trapped_initial':[list(s) for s in self.trapped_initial]}


def certify_coupled_obstruction(source:CoupledBrief,*,max_states:int=512,
                                max_actions:int=256)->CoupledObstruction:
    """Construct a closed losing set, never a failed-search timeout certificate."""
    options,successors,safe,goals,initial=_graph(
        source,max_states=max_states,max_actions=max_actions)
    win=set(goals)
    while True:
        added={i for i in safe-win if any(set(successors[i][a])<=win
                                          for a in options[i])}
        if not added:
            break
        win|=added
    trap=safe-win
    trapped=tuple(i for i in initial if i in trap)
    if not trapped:
        raise ValueError('all admitted joint initial states have a robust controller')
    states=tuple(product(*(range(b.states) for b in source.factors)))
    proposal=CoupledObstruction(
        _digest(source.encode()),
        tuple(states[i] for i in sorted(trap)),
        tuple(states[i] for i in trapped),'')
    return CoupledObstruction(proposal.source_digest,proposal.trap,
                              proposal.trapped_initial,_digest(proposal.encode()))
