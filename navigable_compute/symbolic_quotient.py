"""NCR-006: symbolic observer quotient of affine binary source systems.

The source state space is F_2^n (up to 2^256 states), specified by n-bit
matrix rows, affine bias vectors, and linear observer readouts. The minimal
future-stable linear quotient is constructed by dual Krylov closure, without
enumerating source states or closing the source's operation monoid.

Exact finite mathematics conditional on the stipulated source; no external
sensor, neural, or authorized E-CPU execution is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

SOURCE_SCHEMA = 'navigable-compute.symbolic-affine-source.v0.1'
QUOTIENT_SCHEMA = 'navigable-compute.symbolic-observer-quotient.v0.1'
RESULT_SCHEMA = 'navigable-compute.symbolic-observer-result.v0.1'


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('ascii')


def parity(bits: int) -> int:
    return bits.bit_count() & 1


def apply_rows(rows: tuple[int, ...], state: int) -> int:
    return sum(parity(row & state) << i for i, row in enumerate(rows))


def row_times_matrix(row: int, matrix: tuple[int, ...]) -> int:
    """Multiply one GF(2) covector by a source matrix, without state visits."""
    result = 0
    while row:
        lsb = row & -row
        result ^= matrix[lsb.bit_length() - 1]
        row ^= lsb
    return result


@dataclass(frozen=True)
class AffineMap:
    rows: tuple[int, ...]
    bias: int

    def apply(self, state: int) -> int:
        return apply_rows(self.rows, state) ^ self.bias

    def encode(self) -> dict:
        return {'rows': [str(row) for row in self.rows], 'bias_decimal': str(self.bias)}


@dataclass(frozen=True)
class SymbolicSource:
    width: int
    generators: tuple[AffineMap, ...]
    observer_rows: tuple[int, ...]
    origin: str = 'declared-binary-affine-hypothesis'

    def __post_init__(self) -> None:
        n = self.width
        if type(n) is not int or not 1 <= n <= 256:
            raise ValueError('source bit width must be 1..256')
        if (type(self.generators) is not tuple or not 1 <= len(self.generators) <= 16 or
            any(not isinstance(g, AffineMap) or type(g.rows) is not tuple or
                len(g.rows) != n or
                any(type(r) is not int or not 0 <= r < 1 << n for r in g.rows) or
                type(g.bias) is not int or not 0 <= g.bias < 1 << n
                for g in self.generators)):
            raise ValueError('need complete bounded affine generator matrices')
        if (type(self.observer_rows) is not tuple or
            not 1 <= len(self.observer_rows) <= 32 or
            any(type(r) is not int or not 0 <= r < 1 << n
                for r in self.observer_rows)):
            raise ValueError('invalid exact linear observer readouts')
        if type(self.origin) is not str or not 1 <= len(self.origin) <= 180 or not self.origin.isascii():
            raise ValueError('invalid source provenance label')

    def encode(self) -> dict:
        return {'schema': SOURCE_SCHEMA, 'width': self.width,
                'generators': [g.encode() for g in self.generators],
                'observer_rows': [str(r) for r in self.observer_rows],
                'origin': self.origin,
                'claim': 'fully-declared-stationary-affine-source;not-external-observation'}

    @property
    def digest(self) -> str:
        return sha256(canonical(self.encode())).hexdigest()

    def observe(self, state: int) -> int:
        if type(state) is not int or not 0 <= state < 1 << self.width:
            raise ValueError('state outside declared source space')
        return apply_rows(self.observer_rows, state)


def canonical_basis(rows: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """High-pivot GF(2) reduced row-echelon basis, deterministic."""
    pivots: dict[int, int] = {}
    for value in rows:
        v = value
        for p in sorted(pivots, reverse=True):
            if v & (1 << p):
                v ^= pivots[p]
        if v:
            pivots[v.bit_length() - 1] = v
    for lower in sorted(pivots):
        for upper in sorted(pivots, reverse=True):
            if upper > lower and pivots[upper] & (1 << lower):
                pivots[upper] ^= pivots[lower]
    return tuple(pivots[p] for p in sorted(pivots, reverse=True))


def express(row: int, basis: tuple[int, ...]) -> int:
    value = row
    code = 0
    for i, b in enumerate(basis):
        pivot = 1 << (b.bit_length() - 1)
        if value & pivot:
            value ^= b
            code ^= 1 << i
    if value:
        raise ValueError('covector lies outside the admitted quotient row space')
    return code


@dataclass(frozen=True)
class SymbolicQuotient:
    source_sha256: str
    dimension: int
    basis_rows: tuple[int, ...]
    operators: tuple[AffineMap, ...]
    observer_decoder: tuple[int, ...]
    closure_rounds: int
    digest: str

    def encode(self) -> dict:
        return {'schema': QUOTIENT_SCHEMA, 'source_sha256': self.source_sha256,
                'dimension': self.dimension,
                'basis_rows': [str(r) for r in self.basis_rows],
                'operators': [x.encode() for x in self.operators],
                'observer_decoder': [str(x) for x in self.observer_decoder],
                'closure_rounds': self.closure_rounds,
                'claim': 'minimal-future-stable-linear-quotient-all-source-states'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}

    def project(self, source_state: int) -> int:
        return apply_rows(self.basis_rows, source_state)

    def read(self, quotient_state: int) -> int:
        return apply_rows(self.observer_decoder, quotient_state)


def construct_symbolic_quotient(source: SymbolicSource, *,
                                max_dimension: int = 64) -> SymbolicQuotient:
    """Compute the least generator-invariant dual row space containing O.

    For any source states x,y, equality of the projected Bx,By is equivalent
    to equality of every observable continuation O(A_w x + c_w) and
    O(A_w y + c_w), for all finite generator words w. The biases cancel in
    differences, so only the linear parts enter this dual closure.
    """
    if not isinstance(source, SymbolicSource):
        raise TypeError('exact declared symbolic source required')
    if type(max_dimension) is not int or not 0 <= max_dimension <= 256:
        raise ValueError('invalid certified quotient dimension budget')
    basis = canonical_basis(source.observer_rows)
    rounds = 0
    while True:
        images = [row_times_matrix(row, g.rows)
                  for row in basis for g in source.generators]
        expanded = canonical_basis([*basis, *images])
        if expanded == basis:
            break
        basis = expanded
        rounds += 1
        if len(basis) > max_dimension:
            raise ValueError('observer-future quotient exceeds declared dimension budget')
    if len(basis) > max_dimension:
        raise ValueError('observer-future quotient exceeds declared dimension budget')
    induced = []
    for g in source.generators:
        rows = tuple(express(row_times_matrix(b, g.rows), basis) for b in basis)
        bias = apply_rows(basis, g.bias)
        induced.append(AffineMap(rows, bias))
    decoder = tuple(express(row, basis) for row in source.observer_rows)
    proto = SymbolicQuotient(source.digest, len(basis), basis,
                             tuple(induced), decoder, rounds, '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())


def compose(after: AffineMap, before: AffineMap) -> AffineMap:
    """Exact affine composition after(before(x)) in the quotient space."""
    rows = tuple(row_times_matrix(row, before.rows) for row in after.rows)
    return AffineMap(rows, apply_rows(after.rows, before.bias) ^ after.bias)


def word_operator(operators: tuple[AffineMap, ...], word: tuple[int, ...],
                  dimension: int) -> AffineMap:
    op = AffineMap(tuple(1 << i for i in range(dimension)), 0)
    for index in word:
        op = compose(operators[index], op)
    return op


@dataclass(frozen=True)
class SymbolicResult:
    source_sha256: str
    quotient_sha256: str
    initial_decimal: str
    word: tuple[int, ...]
    repetitions_decimal: str
    source_projection_decimal: str
    final_projection_decimal: str
    observed_decimal: str
    operator_products: int
    digest: str

    def encode(self) -> dict:
        return {'schema': RESULT_SCHEMA, 'source_sha256': self.source_sha256,
                'quotient_sha256': self.quotient_sha256,
                'initial_decimal': self.initial_decimal,
                'word': list(self.word),
                'repetitions_decimal': self.repetitions_decimal,
                'source_projection_decimal': self.source_projection_decimal,
                'final_projection_decimal': self.final_projection_decimal,
                'observed_decimal': self.observed_decimal,
                'operator_products': self.operator_products,
                'claim': 'exact-symbolic-law-navigation;not-external-future-evidence'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}


def navigate_symbolic(source: SymbolicSource, quotient: SymbolicQuotient,
                      initial: int, word: tuple[int, ...], repetitions: int) -> SymbolicResult:
    if not isinstance(source, SymbolicSource) or not isinstance(quotient, SymbolicQuotient):
        raise ValueError('unqualified source or quotient')
    if quotient.source_sha256 != source.digest:
        raise ValueError('changed source Ground or observer')
    if type(initial) is not int or not 0 <= initial < 1 << source.width:
        raise ValueError('source state outside declared domain')
    if (type(word) is not tuple or len(word) > 65536 or
        any(type(i) is not int or not 0 <= i < len(source.generators)
            for i in word)):
        raise ValueError('invalid source-generator word')
    if type(repetitions) is not int or repetitions < 0 or repetitions.bit_length() > 4096:
        raise ValueError('invalid logical time coordinate')
    d = quotient.dimension
    op = word_operator(quotient.operators, word, d)
    result = AffineMap(tuple(1 << i for i in range(d)), 0)
    products = 0
    remaining = repetitions
    while remaining:
        if remaining & 1:
            result = compose(op, result)
            products += 1
        remaining >>= 1
        if remaining:
            op = compose(op, op)
            products += 1
    before = quotient.project(initial)
    after = result.apply(before)
    obs = quotient.read(after)
    proto = SymbolicResult(source.digest, quotient.digest, str(initial),
                           word, str(repetitions), str(before), str(after),
                           str(obs), products, '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())
