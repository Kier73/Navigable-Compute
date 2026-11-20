"""Independent NCI-001 checker: validate a transformation space from invariants.

Does not import the constructive geometry, orbit-graph or rank algorithms.
Instead checks Bellman optimality at every state and derives the count of
symmetric representative choices from direct geometric and group constraints.
It also checks raw transition tables and complete closed-trap obstructions.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod

from .invariant_space import (InvariantDescription, InvariantTransformationSpace,
                              InvariantObstruction, InvariantProduct, canonical)


def _cycles(g):
    """Independent cycle decomposition using inverse-position assignment."""
    seen = set()
    result = []
    for s in range(len(g)):
        if s in seen:
            continue
        p, group = s, []
        while p not in seen:
            group.append(p)
            seen.add(p)
            p = g[p]
        if p != s:
            return None
        result.append(tuple(group))
    return tuple(result)


def _admissible(d, s, t, source_order, target_order):
    if t in d.forbidden or d.labels[s] != d.labels[t]:
        return False
    if source_order % target_order:
        return False
    x, y = d.coordinates[s]
    a, b = d.coordinates[t]
    distance = abs(x-a) + abs(y-b)
    return 1 <= distance <= d.max_displacement


def verify_invariant_space(desc: InvariantDescription, cert: InvariantTransformationSpace) -> bool:
    try:
        if not isinstance(desc, InvariantDescription) or not isinstance(cert, InvariantTransformationSpace):
            return False
        if cert.description != desc:
            return False
        n = len(desc.coordinates)
        cycles = _cycles(desc.symmetry)
        if cycles is None or cert.orbits != cycles:
            return False
        owner = [-1]*n
        for i, orbit in enumerate(cycles):
            for s in orbit:
                owner[s] = i
        sizes = [len(cycles[owner[s]]) for s in range(n)]
        if (len(cert.ranks) != n or len(cert.transition) != n or
            len(cert.choices) != len(cycles)):
            return False
        if any(type(r) is not int or not -1 <= r <= n for r in cert.ranks):
            return False
        if any(cert.ranks[s] != cert.ranks[desc.symmetry[s]] for s in range(n)):
            return False
        goal = set(desc.goal)
        forbidden = set(desc.forbidden)
        if any((cert.ranks[s] == 0) != (s in goal) for s in range(n)):
            return False
        if any(cert.ranks[s] != -1 for s in forbidden):
            return False
        if any(cert.ranks[s] < 0 for s in desc.initial):
            return False

        total_count = 1
        for index, orbit in enumerate(cycles):
            representative = orbit[0]
            rank = cert.ranks[representative]
            if rank < 0:
                if cert.choices[index] or any(cert.transition[s] != -1 for s in orbit):
                    return False
            elif rank == 0:
                if cert.choices[index] != (representative,):
                    return False
                if any(cert.transition[s] != s for s in orbit):
                    return False
            else:
                # Independent fixed-point test: g^|orbit|(t)=t.
                expected = []
                for t in range(n):
                    if _admissible(desc, representative, t, len(orbit), sizes[t]) and cert.ranks[t] == rank - 1:
                        expected.append(t)
                if not expected or cert.choices[index] != tuple(expected):
                    return False
                total_count *= len(expected)

        # Bellman fixed point proves shortest distances, including -1 regions.
        # No unreachable state may have an admissible edge to any reachable state.
        for s in range(n):
            if s in goal or s in forbidden:
                continue
            candidates = [cert.ranks[t] for t in range(n)
                          if _admissible(desc, s, t, sizes[s], sizes[t]) and cert.ranks[t] >= 0]
            expected = 1 + min(candidates) if candidates else -1
            if expected != cert.ranks[s]:
                return False

        if cert.law_count != total_count or type(cert.coordinate) is not int or not 0 <= cert.coordinate < total_count:
            return False
        if cert.coordinate.bit_length() > 4096:
            return False
        quotient_coordinate = cert.coordinate
        for orbit, options in zip(cycles, cert.choices):
            representative = orbit[0]
            rank = cert.ranks[representative]
            if rank > 0:
                expected = options[quotient_coordinate % len(options)]
                quotient_coordinate //= len(options)
                for s in orbit:
                    if cert.transition[s] != expected:
                        return False
                    expected = desc.symmetry[expected]
        if quotient_coordinate:
            return False
        if cert.convergence_bound != max(cert.ranks[s] for s in desc.initial):
            return False
        return sha256(canonical(cert.encode())).hexdigest() == cert.digest
    except (TypeError, ValueError, AttributeError, IndexError, KeyError, OverflowError):
        return False


def verify_invariant_obstruction(desc: InvariantDescription, witness: InvariantObstruction) -> bool:
    try:
        if not isinstance(desc, InvariantDescription) or not isinstance(witness, InvariantObstruction):
            return False
        if witness.description != desc:
            return False
        cycles = _cycles(desc.symmetry)
        if cycles is None:
            return False
        known = {cycle for cycle in cycles}
        if (not witness.trapped_orbits or
            tuple(sorted(set(witness.trapped_orbits))) != tuple(sorted(witness.trapped_orbits)) or
            any(orbit not in known for orbit in witness.trapped_orbits)):
            return False
        trapped = {s for orbit in witness.trapped_orbits for s in orbit}
        if trapped & (set(desc.goal) | set(desc.forbidden)):
            return False
        initial = tuple(s for s in desc.initial if s in trapped)
        if not initial or witness.trapped_initial != initial:
            return False
        orders = {s: len(c) for c in cycles for s in c}
        # A closed trap cannot have a geometrically legal equivariant exit.
        for s in trapped:
            for t in range(len(desc.coordinates)):
                if t not in trapped and _admissible(desc, s, t, orders[s], orders[t]):
                    return False
        return sha256(canonical(witness.encode())).hexdigest() == witness.digest
    except (TypeError, ValueError, AttributeError, IndexError, KeyError):
        return False


def verify_invariant_product(descriptions: tuple[InvariantDescription, ...], obj: InvariantProduct) -> bool:
    try:
        if (not isinstance(obj, InvariantProduct) or type(descriptions) is not tuple or
            len(descriptions) != len(obj.factors) or not descriptions):
            return False
        if any(not verify_invariant_space(d, f) for d, f in zip(descriptions, obj.factors)):
            return False
        total = prod(f.law_count for f in obj.factors)
        if (obj.law_count != total or obj.global_state_count != prod(len(d.coordinates) for d in descriptions) or
            obj.horizon != max(f.convergence_bound for f in obj.factors) or
            type(obj.coordinate) is not int or not 0 <= obj.coordinate < total):
            return False
        quotient = obj.coordinate
        for factor in obj.factors:
            if factor.coordinate != quotient % factor.law_count:
                return False
            quotient //= factor.law_count
        return quotient == 0 and sha256(canonical(obj.encode())).hexdigest() == obj.digest
    except (TypeError, ValueError, AttributeError, IndexError):
        return False
