"""NCI-001 — construct transformation spaces from *invariants*, not an input law.

The caller provides a finite geometry, a symmetry, safe/goal states, preserved
observable labels and a maximum one-step geometric displacement. No transition
or action table is supplied. A certified quotient graph of symmetry orbits is
constructed; shortest progress ranks and the exact family of rank-optimal
commuting transition functions are *derived*.

The chosen law is mathematical. This module makes no assertion that any external
hardware, network path or actuator can physically implement its transitions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import prod

SCHEMA = 'navigable-compute.invariant-transformation-space.v0.1'
OBSTRUCTION_SCHEMA = 'navigable-compute.invariant-obstruction.v0.1'


class NoInvariantRealization(ValueError):
    """The stated invariants admit no converging transformation from an origin."""


def canonical(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def _exact_indices(items: tuple[int, ...], n: int, *, required=False) -> bool:
    return (type(items) is tuple and (not required or bool(items)) and
            all(type(x) is int and 0 <= x < n for x in items) and
            tuple(sorted(set(items))) == items)


@dataclass(frozen=True)
class InvariantDescription:
    """A finite system of possible states, WITHOUT its transition relation.

    The specified symmetry must be a permutation and an L1 isometry of every
    listed coordinate; goals, forbidden states and observable labels must be
    invariant under it. Legal candidate transitions preserve labels, remain
    within safe states and travel at most `max_displacement` in L1 distance.
    Their actual values and their ranking-to-goal are constructed, not given.
    """
    coordinates: tuple[tuple[int, int], ...]
    symmetry: tuple[int, ...]
    labels: tuple[str, ...]
    initial: tuple[int, ...]
    goal: tuple[int, ...]
    forbidden: tuple[int, ...]
    max_displacement: int

    def __post_init__(self) -> None:
        n = len(self.coordinates)
        if not 2 <= n <= 512 or type(self.coordinates) is not tuple:
            raise ValueError('require 2..512 explicitly named geometric states')
        if (any(type(p) is not tuple or len(p) != 2 or
                any(type(c) is not int or abs(c) > 100000 for c in p)
                for p in self.coordinates) or len(set(self.coordinates)) != n):
            raise ValueError('state coordinates must be unique bounded integer pairs')
        if (type(self.symmetry) is not tuple or len(self.symmetry) != n or
                any(type(s) is not int for s in self.symmetry) or
                set(self.symmetry) != set(range(n))):
            raise ValueError('symmetry must be a permutation of all state indices')
        if (type(self.labels) is not tuple or len(self.labels) != n or
                any(type(v) is not str or not v or len(v) > 48 or not v.isascii()
                    for v in self.labels)):
            raise ValueError('a short observed label is required for every state')
        for name in ('initial', 'goal', 'forbidden'):
            if not _exact_indices(getattr(self, name), n, required=name != 'forbidden'):
                raise ValueError(f'{name} must be a canonical finite state set')
        if (set(self.initial) & set(self.forbidden) or
                set(self.goal) & set(self.forbidden)):
            raise ValueError('initial and goal cannot contain forbidden states')
        if type(self.max_displacement) is not int or not 1 <= self.max_displacement <= 10000:
            raise ValueError('invalid geometric movement bound')
        # Check the symmetry is an actual isometry, not an arbitrary claimed chart.
        g = self.symmetry
        if any(self.labels[i] != self.labels[g[i]] or
               (i in self.goal) != (g[i] in self.goal) or
               (i in self.forbidden) != (g[i] in self.forbidden)
               for i in range(n)):
            raise ValueError('symmetry must preserve labels, goals and safety')
        for i, (x, y) in enumerate(self.coordinates):
            for j in range(i):
                a, b = self.coordinates[j]
                sx, sy = self.coordinates[g[i]]
                tx, ty = self.coordinates[g[j]]
                if abs(x-a) + abs(y-b) != abs(sx-tx) + abs(sy-ty):
                    raise ValueError('claimed symmetry is not an L1 isometry')

    def encode(self) -> dict:
        return {'coordinates': [list(p) for p in self.coordinates],
                'symmetry': list(self.symmetry), 'labels': list(self.labels),
                'initial': list(self.initial), 'goal': list(self.goal),
                'forbidden': list(self.forbidden),
                'max_displacement': self.max_displacement}


@dataclass(frozen=True)
class InvariantTransformationSpace:
    description: InvariantDescription
    orbits: tuple[tuple[int, ...], ...]     # symmetry cycles, canonicalized
    ranks: tuple[int, ...]                  # -1 means unqualified state
    choices: tuple[tuple[int, ...], ...]    # permissible target indices per orbit
    law_count: int                          # all shortest-progress equivariant laws
    convergence_bound: int
    coordinate: int
    transition: tuple[int, ...]             # -1 for forbidden or non-admitted states
    digest: str

    def encode(self) -> dict:
        return {'schema': SCHEMA, 'description': self.description.encode(),
                'orbits': [list(o) for o in self.orbits],
                'ranks': list(self.ranks),
                'choices': [list(v) for v in self.choices],
                'law_count_decimal': str(self.law_count),
                'convergence_bound': self.convergence_bound,
                'instance_coordinate_decimal': str(self.coordinate),
                'transition': list(self.transition),
                'claim': 'shortest-progress-equivariant-partial-law; finite-mathematical-only'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}

    def observe(self, logical_coordinate: int) -> tuple[int, ...]:
        if (type(logical_coordinate) is not int or logical_coordinate < 0 or
                logical_coordinate.bit_length() > 4096):
            raise ValueError('invalid logical coordinate')
        steps = min(logical_coordinate, self.convergence_bound)
        candidates = set(self.description.initial)
        for _ in range(steps):
            candidates = {self.transition[s] for s in candidates}
            if -1 in candidates:
                raise AssertionError('unqualified state reached from certified initial')
        return tuple(sorted(candidates))


def _orbits(shift: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    visited: set[int] = set()
    output: list[tuple[int, ...]] = []
    for origin in range(len(shift)):
        if origin in visited:
            continue
        cycle = []
        p = origin
        while p not in visited:
            visited.add(p)
            cycle.append(p)
            p = shift[p]
        if p != origin:
            raise AssertionError('symmetry was not a permutation')
        output.append(tuple(cycle))
    return tuple(output)


def _geometry(desc: InvariantDescription):
    """Build orbit adjacency solely from invariant constraints.

    An equivariant map from a source cycle of length m can land at target t
    iff g**m(t) = t. For a cyclic group this is equivalent to
    len(target_orbit) | m. Every representative choice determines an entire
    coherent mapping on its source orbit.
    """
    orbits = _orbits(desc.symmetry)
    index = [0] * len(desc.coordinates)
    for k, orbit in enumerate(orbits):
        for v in orbit:
            index[v] = k
    lengths = [len(orbits[index[i]]) for i in range(len(index))]
    forbidden = set(desc.forbidden)
    goals = set(desc.goal)
    options: list[tuple[int, ...]] = []
    for orbit in orbits:
        rep = orbit[0]
        if rep in forbidden:
            options.append(())
            continue
        if rep in goals:
            options.append((rep,))
            continue
        x, y = desc.coordinates[rep]
        allowed = []
        for t, (a, b) in enumerate(desc.coordinates):
            if t in forbidden or desc.labels[t] != desc.labels[rep]:
                continue
            distance = abs(x-a) + abs(y-b)
            if 1 <= distance <= desc.max_displacement and len(orbit) % lengths[t] == 0:
                allowed.append(t)
        options.append(tuple(allowed))
    return orbits, tuple(index), tuple(options)


def _ranks(orbit_indices, options, goal_orbits: set[int]) -> tuple[int, ...]:
    # Reverse graph reachability: each source is qualified at the shortest
    # number of geometric steps into the constructed goal set.
    preds = [set() for _ in options]
    for src, row in enumerate(options):
        for target in row:
            preds[orbit_indices[target]].add(src)
    ranks = [-1] * len(options)
    frontier = sorted(goal_orbits)
    for i in frontier:
        ranks[i] = 0
    depth = 0
    while frontier:
        depth += 1
        nxt = set()
        for target in frontier:
            for src in preds[target]:
                if ranks[src] == -1:
                    ranks[src] = depth
                    nxt.add(src)
        frontier = sorted(nxt)
    return tuple(ranks)


def _core(desc: InvariantDescription):
    orbits, idx, options = _geometry(desc)
    goals = {idx[s] for s in desc.goal}
    levels = _ranks(idx, options, goals)
    good = []
    for src, orbit in enumerate(orbits):
        rank = levels[src]
        if rank < 0:
            good.append(())
        elif rank == 0:
            good.append((orbit[0],))
        else:
            good.append(tuple(t for t in options[src] if levels[idx[t]] == rank - 1))
            if not good[-1]:
                raise AssertionError('positive rank has no descending transition')
    return orbits, idx, levels, tuple(good)


def _instantiate(desc, orbits, index, levels, choices, coordinate):
    if type(coordinate) is not int or coordinate < 0 or coordinate.bit_length() > 4096:
        raise ValueError('invalid architecture coordinate')
    count = prod(len(row) for k, row in enumerate(choices) if levels[k] > 0)
    if coordinate >= count:
        raise ValueError('architecture coordinate exceeds constructed possibility space')
    remainder = coordinate
    table = [-1] * len(desc.coordinates)
    for i, orbit in enumerate(orbits):
        if levels[i] == 0:
            for s in orbit:
                table[s] = s
        elif levels[i] > 0:
            row = choices[i]
            target = row[remainder % len(row)]
            remainder //= len(row)
            for s in orbit:
                table[s] = target
                target = desc.symmetry[target]
    if remainder:
        raise AssertionError('mixed-radix decomposition did not terminate')
    return table, count


def construct_invariant_space(desc: InvariantDescription, *, coordinate: int = 0) -> InvariantTransformationSpace:
    """Synthesize all shortest-progress equivariant laws from invariants alone.

    Reject if a declared initial state cannot reach goals while respecting the
    invariant geometry and symmetry. The count excludes unqualified/unsafe
    states and only includes shortest-progress laws, not every converging law.
    """
    if not isinstance(desc, InvariantDescription):
        raise TypeError('an exact invariant description is required')
    orbits, idx, levels, choices = _core(desc)
    if any(levels[idx[s]] < 0 for s in desc.initial):
        raise NoInvariantRealization('initial orbit lacks an equivariant safe path to the goal')
    table, count = _instantiate(desc, orbits, idx, levels, choices, coordinate)
    ranks = tuple(levels[idx[s]] for s in range(len(desc.coordinates)))
    horizon = max(ranks[s] for s in desc.initial)
    proto = InvariantTransformationSpace(desc, orbits, ranks, choices, count, horizon,
                                         coordinate, tuple(table), '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())


def navigate_invariant_instance(space: InvariantTransformationSpace, coordinate: int) -> InvariantTransformationSpace:
    if not isinstance(space, InvariantTransformationSpace):
        raise TypeError('invalid invariant space')
    orbits, idx, levels, options = _core(space.description)
    if orbits != space.orbits or options != space.choices:
        raise ValueError('space metadata does not match source invariants')
    table, count = _instantiate(space.description, orbits, idx, levels, options, coordinate)
    if count != space.law_count:
        raise ValueError('possibility count mismatch')
    obj = replace(space, coordinate=coordinate, transition=tuple(table), digest='')
    return replace(obj, digest=sha256(canonical(obj.encode())).hexdigest())


@dataclass(frozen=True)
class InvariantObstruction:
    description: InvariantDescription
    trapped_orbits: tuple[tuple[int, ...], ...]
    trapped_initial: tuple[int, ...]
    digest: str

    def encode(self) -> dict:
        return {'schema': OBSTRUCTION_SCHEMA, 'description': self.description.encode(),
                'trapped_orbits': [list(o) for o in self.trapped_orbits],
                'trapped_initial': list(self.trapped_initial),
                'claim': 'no-symmetry-equivariant-safe-geometric-exit-to-goal'}


def certify_invariant_obstruction(desc: InvariantDescription) -> InvariantObstruction:
    """Return a closed-trap witness of impossibility from the generated geometry.

    The trap contains all safe states unable to reach a goal in the quotient
    graph. Any legal equivariant transition remains in that trap (or is absent).
    A verifier can check closure directly without rerunning the rank builder.
    """
    if not isinstance(desc, InvariantDescription):
        raise TypeError('invalid invariant description')
    orbits, idx, levels, _ = _core(desc)
    trapped = tuple(orbit for k, orbit in enumerate(orbits)
                    if levels[k] < 0 and orbit[0] not in desc.forbidden)
    trapped_states = {s for orbit in trapped for s in orbit}
    initial = tuple(s for s in desc.initial if s in trapped_states)
    if not initial:
        raise ValueError('no obstruction affecting the admitted initial states')
    proto = InvariantObstruction(desc, trapped, initial, '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())


@dataclass(frozen=True)
class InvariantProduct:
    """Independent composition of exact mathematical processor spaces.

    This is not a claim about coupled components; inter-factor effects or
    observations require a distinct, independently validated construction.
    """
    factors: tuple[InvariantTransformationSpace, ...]
    global_state_count: int
    law_count: int
    coordinate: int
    horizon: int
    digest: str

    def encode(self) -> dict:
        return {'schema': 'navigable-compute.invariant-product.v0.1',
                'factors': [f.packet() for f in self.factors],
                'global_state_count_decimal': str(self.global_state_count),
                'law_count_decimal': str(self.law_count),
                'instance_coordinate_decimal': str(self.coordinate),
                'convergence_bound': self.horizon,
                'claim': 'independent-finite-product-mathematical-only'}

    def observe(self, logical_coordinate: int) -> tuple[tuple[int, ...], ...]:
        return tuple(f.observe(logical_coordinate) for f in self.factors)


def construct_invariant_product(descriptions: tuple[InvariantDescription, ...],
                                *, coordinate: int = 0,
                                max_total_states: int = 4096) -> InvariantProduct:
    if (type(descriptions) is not tuple or not 1 <= len(descriptions) <= 128 or
            any(not isinstance(x, InvariantDescription) for x in descriptions) or
            type(max_total_states) is not int or max_total_states < 2 or
            sum(len(x.coordinates) for x in descriptions) > max_total_states):
        raise ValueError('invalid or oversized independent invariant product')
    if type(coordinate) is not int or coordinate < 0 or coordinate.bit_length() > 4096:
        raise ValueError('invalid global architecture coordinate')
    basic = tuple(construct_invariant_space(d) for d in descriptions)
    count = prod(f.law_count for f in basic)
    if coordinate >= count:
        raise ValueError('architecture coordinate outside the product space')
    remainder = coordinate
    result = []
    for f in basic:
        result.append(navigate_invariant_instance(f, remainder % f.law_count))
        remainder //= f.law_count
    proto = InvariantProduct(tuple(result), prod(len(x.coordinates) for x in descriptions),
                             count, coordinate, max(f.convergence_bound for f in result), '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())
