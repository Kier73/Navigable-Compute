"""NCR-004 independent algebra audit, via Moore pair distinguishability.

Does not invoke the producer's reachable BFS, partition refiner, or closure
builder. Verifies a finite, stipulated system, not external environment truth.
"""
from __future__ import annotations

from hashlib import sha256
from heapq import heappush, heappop

from .invariant_space import canonical
from .preclosure_language import GeneratorObservation, PreclosureLanguage, LanguageRevision


def verify_preclosure_language(system: GeneratorObservation, design: PreclosureLanguage) -> bool:
    if not isinstance(system, GeneratorObservation) or not isinstance(design, PreclosureLanguage):
        return False
    try:
        if (design.dynamics_sha256 != system.dynamics_digest or
            design.observation_sha256 != system.observation_digest or
            design.digest != sha256(canonical(design.encode())).hexdigest()):
            return False
        n = len(system.states)
        known = [False] * n
        for state in system.initial:
            known[system.states.index(state)] = True
        while True:
            next_known = known[:]
            for i, included in enumerate(known):
                if included:
                    for step in system.generators:
                        next_known[step[i]] = True
            if next_known == known:
                break
            known = next_known
        reach = [i for i, value in enumerate(known) if value]
        if tuple(system.states[i] for i in reach) != design.reachable:
            return False

        # Independent *backwards distinguishability* fixed point: instead of
        # the producer's forward block refinement, mark differing pairs and
        # propagate disagreement back through each generator.
        distinguish = [[False] * n for _ in range(n)]
        for i in reach:
            for j in reach:
                distinguish[i][j] = system.outputs[i] != system.outputs[j]
        while True:
            changed = False
            for i in reach:
                for j in reach:
                    if not distinguish[i][j] and any(distinguish[g[i]][g[j]]
                            for g in system.generators):
                        distinguish[i][j] = True
                        distinguish[j][i] = True
                        changed = True
            if not changed:
                break
        blocks = []
        indexes = [-1] * n
        for i in reach:
            if indexes[i] != -1:
                continue
            group = tuple(j for j in reach if not distinguish[i][j])
            c = len(blocks)
            for j in group:
                if indexes[j] != -1:
                    return False
                indexes[j] = c
            blocks.append(tuple(system.states[j] for j in group))
        if (tuple(blocks) != design.state_classes or
            tuple(indexes) != design.state_class_of):
            return False
        for g in system.generators:
            for group in blocks:
                transitions = {indexes[g[system.states.index(s)]] for s in group}
                if len(transitions) != 1:
                    return False
        q = len(blocks)
        maps = tuple(tuple(indexes[g[system.states.index(group[0])]]
                           for group in blocks) for g in system.generators)
        if maps != design.quotient_generators:
            return False

        ident = tuple(range(q))
        if not design.operators or design.operators[0] != ident:
            return False
        if len(set(design.operators)) != len(design.operators):
            return False
        idx = {mapping: k for k, mapping in enumerate(design.operators)}
        if any(len(mapping) != q or any(type(s) is not int or not 0 <= s < q for s in mapping)
               for mapping in design.operators):
            return False
        if tuple(idx[g] for g in maps) != design.generator_opcodes:
            return False
        # Independently compute shortest lexicographic words using a priority
        # queue; do not use producer's breadth-first closure implementation.
        best = {ident: ()}
        heap = [(0, (), ident)]
        while heap:
            _, word, op = heappop(heap)
            if best.get(op) != word:
                continue
            for gidx, generator in enumerate(maps):
                new_map = tuple(generator[op[i]] for i in range(q))
                new_word = word + (gidx,)
                prev = best.get(new_map)
                if prev is None or (len(new_word), new_word) < (len(prev), prev):
                    best[new_map] = new_word
                    heappush(heap, (len(new_word), new_word, new_map))
            if len(best) > len(design.operators):
                return False
        if set(best) != set(idx):
            return False
        if tuple(best[op] for op in design.operators) != design.normal_forms:
            return False
        multiplication = tuple(tuple(idx[tuple(a[b[i]] for i in range(q))]
                                     for b in design.operators) for a in design.operators)
        if multiplication != design.multiplication:
            return False
        # Refinement count is a producer profile, not a mathematical property.
        # Validate by replaying only the counts of strict state splits in a
        # separate simultaneous color/refinement pass.
        color = [system.outputs[i] for i in reach]
        rounds = 0
        while True:
            lookup = {state: k for k, state in enumerate(reach)}
            sign = [(system.outputs[i], tuple(color[lookup[g[i]]] for g in system.generators))
                    for i in reach]
            palette = {}
            fresh = []
            for token in sign:
                if token not in palette:
                    palette[token] = len(palette)
                fresh.append(palette[token])
            old_count = len(set(color))
            new_count = len(set(fresh))
            if new_count == old_count:
                break
            if new_count < old_count or rounds >= len(reach):
                return False
            color = fresh
            rounds += 1
        if rounds != design.state_refinements:
            return False
        # Exact source-to-quotient semantics for all declared primitive steps.
        for i in reach:
            c = indexes[i]
            for k, g in enumerate(system.generators):
                if design.operators[design.generator_opcodes[k]][c] != indexes[g[i]]:
                    return False
        return True
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return False


def verify_language_revision(old_system: GeneratorObservation, old: PreclosureLanguage,
                             new_system: GeneratorObservation, new: PreclosureLanguage,
                             revision: LanguageRevision) -> bool:
    if not isinstance(revision, LanguageRevision):
        return False
    if not verify_preclosure_language(old_system, old) or not verify_preclosure_language(new_system, new):
        return False
    try:
        if (old_system.dynamics_digest != new_system.dynamics_digest or
            old.reachable != new.reachable or
            revision.prior_language_sha256 != old.digest or
            revision.new_language_sha256 != new.digest or
            revision.dynamics_sha256 != old.dynamics_sha256 or
            revision.digest != sha256(canonical(revision.encode())).hexdigest()):
            return False
        old_label = old_system.outputs
        new_label = new_system.outputs
        if any(new_label[i] == new_label[j] and old_label[i] != old_label[j]
               for i in range(len(old_label)) for j in range(len(old_label))):
            return False
        if len(revision.new_to_old_states) != len(new.state_classes):
            return False
        for c, group in enumerate(new.state_classes):
            if any(old.state_class_of[old_system.states.index(s)] != revision.new_to_old_states[c]
                   for s in group):
                return False
        if len(revision.new_to_old_opcodes) != len(new.operators):
            return False
        # Reconstruct mappings from each generated source word through the old
        # quotient, independently of the constructor's map implementation.
        for code, word in enumerate(new.normal_forms):
            source_code = 0
            for g in word:
                source_code = old.multiplication[old.generator_opcodes[g]][source_code]
            if revision.new_to_old_opcodes[code] != source_code:
                return False
        if set(revision.new_to_old_opcodes) != set(range(len(old.operators))):
            return False
        for a, row in enumerate(new.multiplication):
            for b, x in enumerate(row):
                if (revision.new_to_old_opcodes[x] !=
                    old.multiplication[revision.new_to_old_opcodes[a]][revision.new_to_old_opcodes[b]]):
                    return False
        witness = next(((a, b) for a, value in enumerate(revision.new_to_old_opcodes)
                        for b in range(a+1, len(revision.new_to_old_opcodes))
                        if value == revision.new_to_old_opcodes[b]), None)
        return witness == revision.lost_reverse_pair
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return False
