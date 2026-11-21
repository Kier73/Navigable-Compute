"""Independent NCR-005 verifier: checks algebra actions, output and provenance.

Does not import the NCR-005 constructor, coordinate finder, navigator or
readout implementation. Mathematical proof scope is finite and source-bound.
"""
from __future__ import annotations

from hashlib import sha256

from .invariant_space import canonical
from .preclosure_language import GeneratorObservation, PreclosureLanguage
from .preclosure_verify import verify_preclosure_language
from .observer_native import NativeAlgebra, NativeResult, FIELD, SCHEMA, RESULT_SCHEMA


def _canonical_character_basis(generators: tuple[tuple[int, ...], ...], q: int):
    """Independent orbit enumeration from selected involution generators."""
    discovered = {0: 0}
    chosen = []
    for gidx, g in enumerate(generators):
        if g[0] in discovered:
            continue
        new = {}
        for state, code in discovered.items():
            target = g[state]
            if target in discovered or target in new:
                return None
            new[target] = code | (1 << len(chosen))
        discovered.update(new)
        chosen.append(gidx)
    if len(discovered) != q:
        return None
    return tuple(chosen), tuple(discovered.get(i, -1) for i in range(q))


def verify_native_algebra(source: GeneratorObservation, language: PreclosureLanguage,
                          algebra: NativeAlgebra) -> bool:
    if not isinstance(algebra, NativeAlgebra) or not verify_preclosure_language(source, language):
        return False
    try:
        if (algebra.dynamics_sha256 != source.dynamics_digest or
            algebra.observation_sha256 != source.observation_digest or
            algebra.language_sha256 != language.digest or
            algebra.prime != FIELD or
            algebra.digest != sha256(canonical(algebra.encode())).hexdigest() or
            algebra.encode()['schema'] != SCHEMA):
            return False
        q = len(language.state_classes)
        if (type(algebra.dimension) is not int or
            type(algebra.basis_generators) is not tuple or
            any(type(x) is not int for x in algebra.basis_generators) or
            type(algebra.state_vectors) is not tuple or
            any(type(row) is not tuple or any(type(x) is not int for x in row)
                for row in algebra.state_vectors) or
            type(algebra.opcode_actions) is not tuple or
            any(type(row) is not tuple or any(type(x) is not int for x in row)
                for row in algebra.opcode_actions) or
            type(algebra.output_labels) is not tuple or
            any(type(x) is not str for x in algebra.output_labels) or
            type(algebra.readout) is not tuple or
            any(type(x) is not int for x in algebra.readout)):
            return False
        labels_by_state = tuple(source.outputs[source.states.index(group[0])]
                                for group in language.state_classes)
        alphabet = tuple(sorted(set(labels_by_state)))
        if algebra.output_labels != alphabet or len(algebra.state_vectors) != q:
            return False
        if len(algebra.opcode_actions) != len(language.operators):
            return False
        if algebra.kind == 'onehot':
            if algebra.dimension != q or algebra.basis_generators or len(algebra.readout) != q:
                return False
            for index, state in enumerate(algebra.state_vectors):
                if state != tuple(int(j == index) for j in range(q)):
                    return False
            for idx, op in enumerate(language.operators):
                if algebra.opcode_actions[idx] != op:
                    return False
            for i, label in enumerate(labels_by_state):
                if algebra.readout[i] != alphabet.index(label):
                    return False
            # Character fallback must not be disguised as canonical one-hot.
            # Check absence of a qualifying transitive commuting involution
            # group by direct source-map properties and reachability.
            generators = language.quotient_generators
            if q > 1 and (q & (q-1)) == 0:
                is_group = all(len(set(g)) == q and all(g[g[i]] == i for i in range(q))
                               for g in generators)
                commutes = all(all(a[b[i]] == b[a[i]] for i in range(q))
                               for a in generators for b in generators)
                if is_group and commutes and _canonical_character_basis(generators, q) is not None:
                    return False
        elif algebra.kind == 'character':
            d = algebra.dimension
            if (q < 2 or q != (1 << d) or d < 1 or
                len(algebra.readout) != q or
                len(algebra.basis_generators) != d):
                return False
            generators = language.quotient_generators
            if not all(len(set(g)) == q and all(g[g[i]] == i for i in range(q))
                       for g in generators):
                return False
            if not all(all(a[b[i]] == b[a[i]] for i in range(q))
                       for a in generators for b in generators):
                return False
            basis = _canonical_character_basis(generators, q)
            if basis is None or algebra.basis_generators != basis[0]:
                return False
            masks = basis[1]
            for i, vector in enumerate(algebra.state_vectors):
                expected = tuple(FIELD-1 if (masks[i] & (1 << bit)) else 1
                                 for bit in range(d))
                if vector != expected:
                    return False
            for op_idx, op in enumerate(language.operators):
                expected = tuple(FIELD-1 if (masks[op[0]] & (1 << bit)) else 1
                                 for bit in range(d))
                if algebra.opcode_actions[op_idx] != expected:
                    return False
                for state in range(q):
                    resulting = tuple((a*v) % FIELD for a,v in
                                      zip(expected, algebra.state_vectors[state]))
                    if resulting != algebra.state_vectors[op[state]]:
                        return False
            # Independent polynomial evaluation (bit-mask sign is obtained
            # from the source-derived orbit, not producer vector arithmetic).
            for state, mask in enumerate(masks):
                value = sum(coefficient * (-1 if (mask & subset).bit_count() % 2 else 1)
                            for subset, coefficient in enumerate(algebra.readout)) % FIELD
                if value != alphabet.index(labels_by_state[state]):
                    return False
        else:
            return False
        return True
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError):
        return False


def verify_native_result(source: GeneratorObservation, language: PreclosureLanguage,
                         algebra: NativeAlgebra, result: NativeResult) -> bool:
    if (not isinstance(result, NativeResult) or
        not verify_native_algebra(source, language, algebra)):
        return False
    try:
        if (result.language_sha256 != language.digest or
            result.algebra_sha256 != algebra.digest or
            result.digest != sha256(canonical(result.encode())).hexdigest() or
            result.encode()['schema'] != RESULT_SCHEMA or
            type(result.initial) is not int or result.initial not in source.initial or
            type(result.word) is not tuple or len(result.word) > 65536 or
            any(type(i) is not int or not 0 <= i < len(source.generators)
                for i in result.word) or
            type(result.repetitions_decimal) is not str or
            not result.repetitions_decimal.isascii() or
            not result.repetitions_decimal.isdecimal() or
            len(result.repetitions_decimal) > 1234):
            return False
        n = int(result.repetitions_decimal)
        if str(n) != result.repetitions_decimal or n.bit_length() > 4096:
            return False
        q = len(language.state_classes)
        identity = tuple(range(q))
        # Reconstruct the source word from the individually checked primitive
        # quotient maps, not the producer's opcode table / word_code method.
        word_map = identity
        for g_idx in result.word:
            g = language.quotient_generators[g_idx]
            word_map = tuple(g[word_map[i]] for i in range(q))
        # Independent repeated-square full state-map oracle: distinct from
        # character signs and one-hot native representation composition.
        selected_map = identity
        power = word_map
        remaining = n
        product_count = 0
        while remaining:
            if remaining & 1:
                selected_map = tuple(power[selected_map[i]] for i in range(q))
                product_count += 1
            remaining >>= 1
            if remaining:
                power = tuple(power[power[i]] for i in range(q))
                product_count += 1
        start = language.state_class_of[source.states.index(result.initial)]
        end = selected_map[start]
        label = source.outputs[source.states.index(language.state_classes[end][0])]
        return (result.source_class == start and result.result_class == end and
                result.output == label and type(result.operator_products) is int and
                result.operator_products == product_count)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError):
        return False
