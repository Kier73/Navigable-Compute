"""Independent semantic checker for NCS-001 generated instruction algebras.

The checker does not call the endogenous-algebra constructor or its interpreter.
It validates NCI source admission, exact generator law coordinates, independent
closure, multiplication, canonical minimal word descriptions, and checksum.
"""
from __future__ import annotations

from hashlib import sha256

from .invariant_space import InvariantDescription, canonical
from .invariant_verify import verify_invariant_space
from .self_space import EndogenousProcessor, AlgebraBudgetExceeded


def verify_endogenous_processor(desc: InvariantDescription, obj: EndogenousProcessor) -> bool:
    try:
        if not isinstance(desc, InvariantDescription) or not isinstance(obj, EndogenousProcessor):
            return False
        if not verify_invariant_space(desc, obj.source):
            return False
        if (type(obj.generator_limit) is not int or not 1 <= obj.generator_limit <= 32 or
                len(obj.operators) > 4096 or len(obj.operators) < 1):
            return False
        states = tuple(i for i,r in enumerate(obj.source.ranks) if r >= 0)
        if states != obj.states:
            return False
        n = len(states)
        canonical_coords = tuple(range(min(obj.source.law_count, obj.generator_limit)))
        if obj.generator_coordinates != canonical_coords:
            return False
        if not (len(obj.operators) == len(obj.normal_forms) == len(obj.multiply)):
            return False
        eye = tuple(range(n))
        if obj.operators[0] != eye or obj.normal_forms[0] != ():
            return False
        if len(set(obj.operators)) != len(obj.operators):
            return False
        if any(len(m) != n or any(type(x) is not int or not 0 <= x < n for x in m)
               for m in obj.operators):
            return False
        lookup = {s:i for i,s in enumerate(states)}
        # Construct each generator by a distinct, explicit mixed-radix decode;
        # do not invoke the producer's law navigation or the monoid builder.
        generators = []
        for coordinate in canonical_coords:
            q = coordinate
            mapping = {s:s for s in states if obj.source.ranks[s] == 0}
            for i, orbit in enumerate(obj.source.orbits):
                if obj.source.ranks[orbit[0]] < 0:
                    continue
                if obj.source.ranks[orbit[0]] == 0:
                    continue
                choices = obj.source.choices[i]
                if not choices:
                    return False
                goal = choices[q % len(choices)]
                q //= len(choices)
                for state in orbit:
                    mapping[state] = goal
                    goal = desc.symmetry[goal]
            if q or len(mapping) != n:
                return False
            generators.append(tuple(lookup[mapping[s]] for s in states))
        if len(obj.generator_opcodes) != len(generators):
            return False
        for g,i in zip(generators,obj.generator_opcodes):
            if type(i) is not int or not 0 <= i < len(obj.operators) or obj.operators[i] != g:
                return False
        # Independent fixed-point closure: grow the *set* of transformations,
        # not the producer's indexed breadth-first enumeration.
        closure = {eye}
        while True:
            additions = {tuple(g[f[x]] for x in range(n))
                         for f in closure for g in generators}
            if additions <= closure:
                break
            closure.update(additions)
            if len(closure) > 4096:
                return False
        if closure != set(obj.operators):
            return False
        idx = {v:i for i,v in enumerate(obj.operators)}
        # Every claimed operation must carry a word describing its OWN effect.
        for op,word in zip(obj.operators,obj.normal_forms):
            if type(word) is not tuple or len(word) > len(obj.operators)-1:
                return False
            current = eye
            for symbol in word:
                if type(symbol) is not int or not 0 <= symbol < len(generators):
                    return False
                g = generators[symbol]
                current = tuple(g[current[s]] for s in range(n))
            if current != op:
                return False
        # Prove producer's canonical shortlex words by a second, layered search.
        q = [eye]
        seen = {eye: ()}
        for mapping in q:
            path = seen[mapping]
            for i,g in enumerate(generators):
                new = tuple(g[mapping[s]] for s in range(n))
                if new not in seen:
                    seen[new] = path + (i,)
                    q.append(new)
        if any(seen[op] != word for op,word in zip(obj.operators,obj.normal_forms)):
            return False
        # The whole finite multiplication table is checked against direct
        # function composition. This also proves associativity by semantics.
        for a,row in zip(obj.operators,obj.multiply):
            if len(row) != len(obj.operators):
                return False
            for b,cell in zip(obj.operators,row):
                if type(cell) is not int or cell != idx[tuple(a[b[s]] for s in range(n))]:
                    return False
        return sha256(canonical(obj.encode())).hexdigest() == obj.digest
    except (TypeError,ValueError,IndexError,KeyError,AttributeError,OverflowError,MemoryError):
        return False
