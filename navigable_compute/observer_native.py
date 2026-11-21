"""NCR-005: construct exact observer-native state coordinates and actions.

Input is a separately verified NCR-004 quotient; NCI-001 may supply its
source generators without any source-monoid closure. Finite stationary laws
only. Nothing here authenticates observations or authorizes physical effects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from .invariant_space import canonical
from .preclosure_language import GeneratorObservation, PreclosureLanguage
from .preclosure_verify import verify_preclosure_language

SCHEMA = 'navigable-compute.observer-native-algebra.v0.1'
RESULT_SCHEMA = 'navigable-compute.observer-native-result.v0.1'
FIELD = 257  # odd prime, strictly greater than the maximum NCR-004 state count


@dataclass(frozen=True)
class NativeAlgebra:
    dynamics_sha256: str
    observation_sha256: str
    language_sha256: str
    kind: str  # character or onehot
    prime: int
    dimension: int
    basis_generators: tuple[int, ...]
    state_vectors: tuple[tuple[int, ...], ...]
    opcode_actions: tuple[tuple[int, ...], ...]  # diagonal signs or column targets
    output_labels: tuple[str, ...]
    readout: tuple[int, ...]  # Walsh coefficients or row of one-hot output map
    digest: str

    def encode(self) -> dict:
        return {
            'schema': SCHEMA, 'dynamics_sha256': self.dynamics_sha256,
            'observation_sha256': self.observation_sha256,
            'language_sha256': self.language_sha256, 'kind': self.kind,
            'prime': self.prime, 'dimension': self.dimension,
            'basis_generators': list(self.basis_generators),
            'state_vectors': [list(row) for row in self.state_vectors],
            'opcode_actions': [list(row) for row in self.opcode_actions],
            'output_labels': list(self.output_labels), 'readout': list(self.readout),
            'claim': 'finite-exact-operator-intertwining;not-physical-admission',
        }

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}

    def act(self, opcode: int, vector: tuple[int, ...]) -> tuple[int, ...]:
        if type(opcode) is not int or not 0 <= opcode < len(self.opcode_actions):
            raise ValueError('unknown observer-language opcode')
        if type(vector) is not tuple or vector not in self.state_vectors:
            raise ValueError('state outside the admitted quotient representation')
        action = self.opcode_actions[opcode]
        if self.kind == 'character':
            return tuple((a*v) % self.prime for a, v in zip(action, vector))
        idx = vector.index(1)
        return self.state_vectors[action[idx]]

    def read(self, vector: tuple[int, ...]) -> str:
        if type(vector) is not tuple or vector not in self.state_vectors:
            raise ValueError('state outside the admitted quotient representation')
        if self.kind == 'character':
            result = 0
            for mask, coeff in enumerate(self.readout):
                term = coeff
                for bit in range(self.dimension):
                    if (mask >> bit) & 1:
                        term = (term * vector[bit]) % self.prime
                result = (result + term) % self.prime
        else:
            result = sum(a*b for a, b in zip(self.readout, vector)) % self.prime
        if result >= len(self.output_labels):
            raise ValueError('invalid observed output code')
        return self.output_labels[result]


@dataclass(frozen=True)
class NativeResult:
    language_sha256: str
    algebra_sha256: str
    initial: int
    word: tuple[int, ...]
    repetitions_decimal: str
    source_class: int
    result_class: int
    output: str
    operator_products: int
    digest: str

    def encode(self) -> dict:
        return {'schema': RESULT_SCHEMA, 'language_sha256': self.language_sha256,
                'algebra_sha256': self.algebra_sha256, 'initial': self.initial,
                'word': list(self.word), 'repetitions_decimal': self.repetitions_decimal,
                'source_class': self.source_class, 'result_class': self.result_class,
                'output': self.output, 'operator_products': self.operator_products,
                'claim': 'exact-model-observation;not-external-future-evidence'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}


def _character_coordinates(language: PreclosureLanguage):
    """Return canonical group coordinates, or None for unsupported actions.

    Only a faithful transitive elementary abelian 2-group action is admitted.
    For that restricted group, basis-generator images uniquely identify each
    quotient state. No guessed symmetry or approximate diagonalization.
    """
    maps = language.quotient_generators
    q = len(language.state_classes)
    if q < 2 or q & (q-1):
        return None
    identity = tuple(range(q))
    if any(tuple(g[g[i]] for i in range(q)) != identity or
           len(set(g)) != q for g in maps):
        return None
    if any(tuple(a[b[i]] for i in range(q)) != tuple(b[a[i]] for i in range(q))
           for a in maps for b in maps):
        return None
    # Increasing original-generator order selects a deterministic independent
    # group basis. Every admitted new involution doubles the orbit of root 0.
    positions = {0: 0}
    bases = []
    for gindex, generator in enumerate(maps):
        if generator[0] in positions:
            continue
        bit = 1 << len(bases)
        grown = {generator[state]: mask | bit for state, mask in positions.items()}
        if len(grown) != len(positions) or set(grown) & set(positions):
            return None
        positions.update(grown)
        bases.append(gindex)
    if len(positions) != q or (1 << len(bases)) != q:
        return None
    masks = tuple(positions.get(state, -1) for state in range(q))
    for g in maps:
        shift = masks[g[0]]
        if any(masks[g[i]] != (masks[i] ^ shift) for i in range(q)):
            return None
    for op in language.operators:
        shift = masks[op[0]]
        if any(masks[op[i]] != (masks[i] ^ shift) for i in range(q)):
            return None
    return tuple(bases), masks


def construct_native_algebra(source: GeneratorObservation, language: PreclosureLanguage,
                             *, max_dimension: int = 64) -> NativeAlgebra:
    """Construct exact coordinates without source-operation monoid closure.

    Character mode is dimension-minimal among faithful affine representations
    of the whole transitive elementary abelian 2-group over odd fields.
    One-hot fallback is always exact, but makes no minimum-dimension claim.
    """
    if type(max_dimension) is not int or not 1 <= max_dimension <= 256:
        raise ValueError('invalid coordinate budget')
    if not verify_preclosure_language(source, language):
        raise ValueError('source observer-language certificate refused')
    q = len(language.state_classes)
    outputs = tuple(source.outputs[source.states.index(block[0])]
                    for block in language.state_classes)
    labels = tuple(sorted(set(outputs)))
    if len(labels) >= FIELD:
        raise ValueError('too many categorical outputs')
    codes = tuple(labels.index(value) for value in outputs)
    char = _character_coordinates(language)
    if char is not None:
        bases, masks = char
        dimension = len(bases)
        if dimension > max_dimension:
            raise ValueError('character dimension budget exceeded')
        vectors = tuple(tuple(FIELD - 1 if mask >> j & 1 else 1
                              for j in range(dimension)) for mask in masks)
        ops = tuple(tuple(FIELD - 1 if masks[op[0]] >> j & 1 else 1
                          for j in range(dimension)) for op in language.operators)
        by_mask = {mask: codes[index] for index, mask in enumerate(masks)}
        inverse_q = pow(q, -1, FIELD)
        readout = tuple(sum(by_mask[mask] * (-1 if (mask & subset).bit_count() % 2 else 1)
                            for mask in range(q)) * inverse_q % FIELD
                        for subset in range(q))
        kind = 'character'
    else:
        if q > max_dimension:
            raise ValueError('one-hot dimension budget exceeded')
        bases = ()
        dimension = q
        vectors = tuple(tuple(int(i == j) for j in range(q)) for i in range(q))
        ops = language.operators  # column-sparse matrix, column -> row index
        readout = codes
        kind = 'onehot'
    candidate = NativeAlgebra(source.dynamics_digest, source.observation_digest,
                              language.digest, kind, FIELD, dimension, bases,
                              vectors, ops, labels, readout, '')
    candidate = replace(candidate, digest=sha256(canonical(candidate.encode())).hexdigest())
    # Check complete quotient operator action and observable readout, not samples.
    for i, vector in enumerate(candidate.state_vectors):
        if candidate.read(vector) != outputs[i]:
            raise AssertionError('constructed observer readout disagrees')
        for opcode, op in enumerate(language.operators):
            if candidate.act(opcode, vector) != candidate.state_vectors[op[i]]:
                raise AssertionError('constructed operator intertwining failed')
    return candidate


def navigate_native(source: GeneratorObservation, language: PreclosureLanguage,
                    algebra: NativeAlgebra, initial: int, word: tuple[int, ...],
                    repetitions: int) -> NativeResult:
    """Power a generated language operation; no intermediate source states."""
    if (source.dynamics_digest != algebra.dynamics_sha256 or
        source.observation_digest != algebra.observation_sha256 or
        language.digest != algebra.language_sha256):
        raise ValueError('source, language or algebra identity changed')
    if type(initial) is not int or initial not in source.initial:
        raise ValueError('initial state outside admitted source set')
    if type(repetitions) is not int or repetitions < 0 or repetitions.bit_length() > 4096:
        raise ValueError('invalid logical coordinate')
    code = language.word_code(word)
    q = len(language.state_classes)
    # Representation-dependent exact composition, without a source trajectory.
    if algebra.kind == 'character':
        identity = (1,) * algebra.dimension
        result = identity
        power = algebra.opcode_actions[code]
        compose = lambda a, b: tuple(x*y % algebra.prime for x, y in zip(a, b))
    elif algebra.kind == 'onehot':
        identity = tuple(range(q))
        result = identity
        power = algebra.opcode_actions[code]
        compose = lambda a, b: tuple(a[b[i]] for i in range(q))
    else:
        raise ValueError('unknown representation kind')
    products = 0
    remaining = repetitions
    while remaining:
        if remaining & 1:
            result = compose(power, result)
            products += 1
        remaining >>= 1
        if remaining:
            power = compose(power, power)
            products += 1
    src_class = language.state_class_of[source.states.index(initial)]
    if algebra.kind == 'character':
        final_vector = tuple(x*y % algebra.prime for x, y in
                             zip(result, algebra.state_vectors[src_class]))
    else:
        final_vector = algebra.state_vectors[result[src_class]]
    try:
        end_class = algebra.state_vectors.index(final_vector)
    except ValueError as exc:
        raise ValueError('represented transition left the qualified state set') from exc
    proto = NativeResult(language.digest, algebra.digest, initial, word,
                         str(repetitions), src_class, end_class,
                         algebra.read(final_vector), products, '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())
