"""NCP-001: intention-first construction from *possible* transitions.

There is no input program or previously observed trajectory. A contract defines
states, safe/goal sets, and the possible outcomes of each admissible operation.
The constructor synthesizes a stationary, state-observed processor law that
*guarantees* reaching the goal for every allowed initial state even when an
adversary chooses any declared successor. Product Grounds remain factored.

All claims are conditional on the declared possibility relation being faithful.
A mathematical law does not certify that a physical host can implement it.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod


class NotRealizable(ValueError):
    """The complete finite contract has no universally successful controller."""


@dataclass(frozen=True)
class PossibilityAction:
    name: str
    outcomes: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 64:
            raise ValueError('action name must be a short nonempty string')
        if not isinstance(self.outcomes, tuple):
            raise TypeError('action outcomes must be an ordered tuple')
        for choices in self.outcomes:
            if not isinstance(choices, tuple) or tuple(sorted(set(choices))) != choices:
                raise ValueError('each successor set must be a sorted unique tuple')
            if any(type(x) is not int for x in choices):
                raise TypeError('successor must be an integer')


@dataclass(frozen=True)
class FactorContract:
    """All local states observable; action outcomes can be nondeterministic.

    Empty outcome means an action is disabled, not a successful empty branch.
    Goals are terminal and persist; transitions from them are disregarded.
    """
    states: int
    initial: tuple[int, ...]
    goal: tuple[int, ...]
    unsafe: tuple[int, ...]
    actions: tuple[PossibilityAction, ...]

    def __post_init__(self):
        if type(self.states) is not int or not 2 <= self.states <= 256:
            raise ValueError('local state count must be 2..256')
        for name in ('initial', 'goal', 'unsafe'):
            items = getattr(self, name)
            if (not isinstance(items, tuple) or (not items and name != 'unsafe') or
                any(type(x) is not int or not 0 <= x < self.states for x in items) or
                tuple(sorted(set(items))) != items):
                raise ValueError(f'{name} must be a sorted unique in-range set')
        if set(self.goal) & set(self.unsafe) or set(self.initial) & set(self.unsafe):
            raise ValueError('initial/goal cannot contain unsafe states')
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ValueError('at least one action required')
        if len(self.actions) > 64 or len({a.name for a in self.actions}) != len(self.actions):
            raise ValueError('action names must be unique; maximum 64')
        for action in self.actions:
            if not isinstance(action, PossibilityAction) or len(action.outcomes) != self.states:
                raise ValueError('action table must cover all local states')
            if any(t < 0 or t >= self.states for row in action.outcomes for t in row):
                raise ValueError('successor outside declared local Ground')

    def encode(self):
        return {'states': self.states, 'initial': list(self.initial), 'goal': list(self.goal),
                'unsafe': list(self.unsafe),
                'actions': [{'name': a.name, 'outcomes': [list(row) for row in a.outcomes]}
                            for a in self.actions]}


@dataclass(frozen=True)
class ConstructedFactor:
    rank: tuple[int, ...]          # -1: no guaranteed goal, 0: goal
    action: tuple[int, ...]        # -1: terminal or not realizable
    optimal: tuple[tuple[int, ...], ...]
    optimal_laws: int
    worst_steps: int               # bound for initial states, not all states

    def encode(self):
        return {'rank': list(self.rank), 'action': list(self.action),
                'optimal': [list(row) for row in self.optimal],
                'optimal_laws_decimal': str(self.optimal_laws),
                'worst_steps': self.worst_steps}


@dataclass(frozen=True)
class ConstructedGround:
    contracts: tuple[FactorContract, ...]
    factors: tuple[ConstructedFactor, ...]
    global_states: int
    initial_states: int
    factored_optimal_laws: int
    convergence_bound: int
    instance_coordinate: int
    digest: str

    def encode(self):
        return {'schema':'navigable-compute.possibility-construction.v0.1',
                'semantics':'local-state-observed; adversarial-outcomes; goals-terminal',
                'contracts':[c.encode() for c in self.contracts],
                'factors':[c.encode() for c in self.factors],
                'global_states_decimal': str(self.global_states),
                'initial_states_decimal': str(self.initial_states),
                'factored_optimal_laws_decimal': str(self.factored_optimal_laws),
                'convergence_bound':self.convergence_bound,
                'instance_coordinate_decimal':str(self.instance_coordinate)}

    def packet(self):
        return {**self.encode(), 'sha256': self.digest}

    def observations_at(self, coordinate: int) -> tuple[tuple[int, ...], ...]:
        """Exact set of reachable local observations for every initial state.

        At large coordinates the certified rank bounds the required prefix;
        once the prefix reaches a terminal goal, future observations are fixed.
        This evaluates at most convergence_bound steps, not every distant tick.
        These are *sets*, not invented single outcomes under nondeterminism.
        """
        if type(coordinate) is not int or coordinate < 0 or coordinate.bit_length() > 4096:
            raise ValueError('invalid logical coordinate')
        steps = min(coordinate, self.convergence_bound)
        outputs = []
        for contract, factor in zip(self.contracts, self.factors):
            candidates = set(contract.initial)
            for _ in range(steps):
                updated = set()
                for state in candidates:
                    if state in contract.goal:
                        updated.add(state)
                    else:
                        a = factor.action[state]
                        if a < 0:
                            raise AssertionError('constructed controller reached losing state')
                        updated.update(contract.actions[a].outcomes[state])
                candidates = updated
            outputs.append(tuple(sorted(candidates)))
        return tuple(outputs)


def _canonical(doc:dict)->bytes:
    return json.dumps(doc,sort_keys=True,separators=(',', ':'),ensure_ascii=True).encode()


def _locally_safe(contract:FactorContract, state:int, action:PossibilityAction)->bool:
    choices=action.outcomes[state]
    return bool(choices) and all(t not in contract.unsafe for t in choices)


def _construct_factor(contract:FactorContract)->ConstructedFactor:
    """AND/OR attractor synthesis; no enumeration of candidate controllers.

    At level d, add a state if *some* action has *all* adversarial
    successors previously qualified at levels <d. Compute exact least
    worst-case rank; same-rank actions form a finite possibility family.
    """
    n=contract.states
    rank=[-1]*n
    for goal in contract.goal:
        rank[goal]=0
    for depth in range(1,n+1):
        newly=[]
        for s in range(n):
            if rank[s]>=0 or s in contract.unsafe:
                continue
            if any(_locally_safe(contract,s,act) and
                   all(rank[t]>=0 and rank[t]<depth for t in act.outcomes[s])
                   for act in contract.actions):
                newly.append(s)
        if not newly:
            break
        for s in newly:
            rank[s]=depth
    actions=[]
    optimal=[]
    laws=1
    for s in range(n):
        if rank[s]<=0:
            actions.append(-1)
            optimal.append(())
            continue
        choices=tuple(i for i,a in enumerate(contract.actions)
                      if _locally_safe(contract,s,a) and
                      all(rank[t]>=0 for t in a.outcomes[s]) and
                      max(rank[t] for t in a.outcomes[s]) == rank[s]-1)
        if not choices:
            raise AssertionError('rank lacks a achieving action')
        actions.append(choices[0]);optimal.append(choices)
        laws*=len(choices)
    if any(rank[s]<0 for s in contract.initial):
        failures=tuple(s for s in contract.initial if rank[s]<0)
        raise NotRealizable(f'initial states lack universal realization: {failures}')
    return ConstructedFactor(tuple(rank),tuple(actions),tuple(optimal),laws,
                             max(rank[s] for s in contract.initial))


def construct_world(contracts:tuple[FactorContract,...],*,max_total_local_states:int=4096)->ConstructedGround:
    """Construct a whole product processor from declarative possibility contracts.

    Independent factors are assumed by contract; their global product is not
    enumerated. No claim of cross-factor correlation or actuator availability.
    """
    if not isinstance(contracts,tuple) or not contracts or len(contracts)>128:
        raise ValueError('1..128 independent factor contracts required')
    if type(max_total_local_states) is not int or max_total_local_states<2:
        raise ValueError('invalid local-state construction budget')
    if any(not isinstance(c,FactorContract) for c in contracts):
        raise TypeError('invalid component contract')
    if sum(c.states for c in contracts)>max_total_local_states:
        raise ValueError('local-state construction budget exceeded')
    compiled=tuple(_construct_factor(c) for c in contracts)
    prototype=ConstructedGround(contracts,compiled,prod(c.states for c in contracts),
       prod(len(c.initial) for c in contracts),prod(f.optimal_laws for f in compiled),
       max(f.worst_steps for f in compiled),0,'')
    return ConstructedGround(contracts,compiled,prototype.global_states,
       prototype.initial_states,prototype.factored_optimal_laws,
       prototype.convergence_bound,0,sha256(_canonical(prototype.encode())).hexdigest())


@dataclass(frozen=True)
class ObstructionCertificate:
    """A closed adversarial trap intersecting the admitted initial states."""
    factor_index: int
    trap: tuple[int, ...]
    trapped_initial: tuple[int, ...]
    source_sha256: str
    witness_sha256: str

    def encode(self):
        return {'schema':'navigable-compute.possibility-obstruction.v0.1',
                'factor_index':self.factor_index,
                'trap':list(self.trap),
                'trapped_initial':list(self.trapped_initial),
                'source_sha256':self.source_sha256}

    def packet(self):
        return {**self.encode(),'sha256':self.witness_sha256}


def certify_nonrealizable(contracts:tuple[FactorContract,...])->ObstructionCertificate:
    """Return an impossibility *witness*, not a timed-out unsuccessful search.

    With goal terminal semantics, the complement of the least backward
    goal-attractor is an adversarially closed trap. Every possible law has
    an outcome that remains in it or violates the required safety condition.
    """
    if not isinstance(contracts,tuple) or not contracts or any(not isinstance(c,FactorContract) for c in contracts):
        raise ValueError('valid finite contracts required')
    for index,c in enumerate(contracts):
        good=set(c.goal)
        safe=set(range(c.states))-set(c.unsafe)
        while True:
            new={s for s in safe-good if any(a.outcomes[s] and
                        set(a.outcomes[s])<=good for a in c.actions)}
            if not new: break
            good |= new
        trap=tuple(sorted(safe-good))
        starting=tuple(s for s in c.initial if s in trap)
        if starting:
            source=sha256(_canonical(c.encode())).hexdigest()
            proposal=ObstructionCertificate(index,trap,starting,source,'')
            return ObstructionCertificate(index,trap,starting,source,
                                          sha256(_canonical(proposal.encode())).hexdigest())
    raise ValueError('all admitted initial states are realizable; no obstruction witness')



def navigate_instance(world:ConstructedGround,coordinate:int)->ConstructedGround:
    """Address a specific *realizable processor* without enumerating laws.

    The coordinate is mixed-radix over optimal action choices for every
    winning local state of every factor. No semantic equivalence between
    distinct policies is claimed; the count is the number of policies,
    including potentially observationally equivalent implementations.
    """
    from .possibility_verify import verify_world
    if not verify_world(world.contracts,world):
        raise ValueError('cannot navigate unverified possibility Ground')
    if (type(coordinate) is not int or coordinate<0 or
        coordinate>=world.factored_optimal_laws or coordinate.bit_length()>4096):
        raise ValueError('instance coordinate outside qualified possibility space')
    remainder=coordinate
    selected=[]
    for factor in world.factors:
        actions=[]
        for options in factor.optimal:
            if options:
                residue=remainder%len(options)
                remainder//=len(options)
                actions.append(options[residue])
            else:
                actions.append(-1)
        selected.append(ConstructedFactor(factor.rank,tuple(actions),factor.optimal,
                                          factor.optimal_laws,factor.worst_steps))
    if remainder:
        raise AssertionError('mixed-radix instance mapping did not exhaust coordinate')
    result=ConstructedGround(world.contracts,tuple(selected),world.global_states,
        world.initial_states,world.factored_optimal_laws,world.convergence_bound,
        coordinate,'')
    digest=sha256(_canonical(result.encode())).hexdigest()
    return ConstructedGround(result.contracts,result.factors,result.global_states,
        result.initial_states,result.factored_optimal_laws,result.convergence_bound,
        coordinate,digest)
