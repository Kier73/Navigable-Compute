"""NCR-007: generate nonlinear Boolean processors from invariants, then lift observers.

All computations are exact in the Boolean polynomial ring
F_2[x_0,...,x_{n-1}]/(x_i^2-x_i). Generator laws are *constructed* from a
ranked dependency graph, preserved registers, symmetry and polynomial degree,
rather than provided as transition tables. Koopman pullback closure then gives
an exact linear observer processor without enumerating the 2**n source states.

This is stipulated finite mathematics, never external law admission or a claim
about an unrestricted or approximately learned nonlinear system.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
import json


SCHEMA = 'navigable-compute.invariant-polynomial-law-family.v0.1'
KOOPMAN_SCHEMA = 'navigable-compute.nonlinear-observer-koopman.v0.1'
RESULT_SCHEMA = 'navigable-compute.nonlinear-observer-result.v0.1'


class PolynomialBudget(ValueError):
    """Exact source/feature complexity exceeded an explicit computation budget."""


def canonical(document: dict) -> bytes:
    return json.dumps(document, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':')).encode('ascii')


def _valid_polynomial(poly: object, width: int) -> bool:
    return (type(poly) is tuple and len(poly) <= 2048 and
            all(type(m) is int and 0 <= m < 1 << width for m in poly) and
            tuple(sorted(set(poly))) == poly)


def eval_poly(poly: tuple[int, ...], state: int) -> int:
    return sum((state & mask) == mask for mask in poly) & 1


def eval_observers(polys: tuple[tuple[int, ...], ...], state: int) -> int:
    return sum(eval_poly(poly, state) << i for i, poly in enumerate(polys))


def _permuted_mask(mask: int, symmetry: tuple[int, ...]) -> int:
    result = 0
    while mask:
        bit = mask & -mask
        result |= 1 << symmetry[bit.bit_length() - 1]
        mask ^= bit
    return result


@dataclass(frozen=True)
class PolynomialInvariants:
    width: int
    tiers: tuple[int, ...]                      # -1 is preserved, 0.. are drained
    symmetry: tuple[int, ...]                  # a single generating permutation
    allowed_inputs: tuple[tuple[int, ...], ...]  # directed causal/topological constraints
    degree: int                               # source nonlinear degree <= 3
    observers: tuple[tuple[int, ...], ...]      # ANF-polynomial output bits

    def __post_init__(self) -> None:
        n = self.width
        if type(n) is not int or not 2 <= n <= 128:
            raise ValueError('require 2..128 implicitly represented source bits')
        if (type(self.tiers) is not tuple or len(self.tiers) != n or
            any(type(t) is not int or not -1 <= t <= 8 for t in self.tiers)):
            raise ValueError('invalid causal rank tiers')
        active = {t for t in self.tiers if t >= 0}
        if active and active != set(range(max(active) + 1)):
            raise ValueError('causal rank tiers must be contiguous')
        if (type(self.symmetry) is not tuple or len(self.symmetry) != n or
            any(type(v) is not int for v in self.symmetry) or
            set(self.symmetry) != set(range(n))):
            raise ValueError('symmetry must be a bit permutation')
        if any(self.tiers[i] != self.tiers[self.symmetry[i]] for i in range(n)):
            raise ValueError('symmetry must preserve causal ranks')
        if (type(self.allowed_inputs) is not tuple or len(self.allowed_inputs) != n or
            any(type(row) is not tuple or
                any(type(j) is not int or not 0 <= j < n for j in row) or
                row != tuple(sorted(set(row))) for row in self.allowed_inputs)):
            raise ValueError('invalid canonical causal input graph')
        for i, row in enumerate(self.allowed_inputs):
            if self.tiers[i] < 0 and row:
                raise ValueError('preserved registers cannot accept source writes')
            if any(self.tiers[j] <= self.tiers[i] or self.tiers[j] < 0 for j in row):
                raise ValueError('active rows can depend only on strictly higher active tiers')
            if tuple(sorted(self.symmetry[j] for j in row)) != self.allowed_inputs[self.symmetry[i]]:
                raise ValueError('declared symmetry does not preserve causal input topology')
        if type(self.degree) is not int or not 1 <= self.degree <= 3:
            raise ValueError('degree must be a bounded positive polynomial degree')
        if (type(self.observers) is not tuple or not 1 <= len(self.observers) <= 16 or
            any(not _valid_polynomial(poly, n) for poly in self.observers)):
            raise ValueError('invalid exact Boolean observer polynomials')

    @property
    def drain_horizon(self) -> int:
        return max(self.tiers, default=-1) + 1

    def encode(self) -> dict:
        return {'width': self.width, 'tiers': list(self.tiers),
                'symmetry': list(self.symmetry),
                'allowed_inputs': [list(row) for row in self.allowed_inputs],
                'degree': self.degree,
                'observers': [[str(x) for x in row] for row in self.observers],
                'claim': 'rank-decreasing-topology-symmetry-conservation-invariants'}


def _templates(description: PolynomialInvariants, *, max_candidates: int = 8192):
    terms = []
    for i, inputs in enumerate(description.allowed_inputs):
        for d in range(1, min(len(inputs), description.degree) + 1):
            for group in combinations(inputs, d):
                mask = sum(1 << index for index in group)
                terms.append((i, mask))
                if len(terms) > max_candidates:
                    raise PolynomialBudget('candidate monomial budget exceeded')
    return tuple(sorted(terms))


def _coefficient_orbits(description: PolynomialInvariants):
    templates = set(_templates(description))
    visited = set()
    orbits = []
    for seed in sorted(templates):
        if seed in visited:
            continue
        orbit = []
        current = seed
        while current not in visited:
            if current not in templates:
                raise ValueError('symmetry does not preserve polynomial templates')
            visited.add(current)
            orbit.append(current)
            current = (description.symmetry[current[0]],
                       _permuted_mask(current[1], description.symmetry))
        if current != seed:
            raise AssertionError('invalid permutation-orbit partition')
        orbits.append(tuple(sorted(orbit)))
        if len(orbits) > 4096:
            raise PolynomialBudget('coefficient symmetry-orbit budget exceeded')
    return tuple(orbits)


@dataclass(frozen=True)
class BooleanOperator:
    rows: tuple[tuple[int, ...], ...]

    def apply(self, state: int) -> int:
        return eval_observers(self.rows, state)

    def encode(self) -> list[list[str]]:
        return [[str(mask) for mask in row] for row in self.rows]


@dataclass(frozen=True)
class PolynomialLawFamily:
    description: PolynomialInvariants
    coefficient_orbits: tuple[tuple[tuple[int, int], ...], ...]
    coordinates: tuple[int, ...]
    operators: tuple[BooleanOperator, ...]
    law_count_decimal: str
    digest: str

    def encode(self) -> dict:
        return {'schema': SCHEMA, 'description': self.description.encode(),
                'coefficient_orbits': [
                    [[i, str(mask)] for i,mask in orbit]
                    for orbit in self.coefficient_orbits],
                'coordinates': [str(k) for k in self.coordinates],
                'operators': [op.encode() for op in self.operators],
                'law_count_decimal': self.law_count_decimal,
                'drain_horizon': self.description.drain_horizon,
                'claim': 'symbolic-nonlinear-equivariant-draining-law-family'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}

    def project_observer(self, state: int) -> int:
        return eval_observers(self.description.observers, state)


def construct_polynomial_laws(description: PolynomialInvariants,
                              coordinates: tuple[int, ...]) -> PolynomialLawFamily:
    """Construct nonlinear law choices from invariant constraints only.

    Every symmetry orbit of permissible monomial coefficients is an
    independent Boolean choice. Neither source states nor source monoids are
    enumerated. All constructed laws preserve protected bits and drain active
    ranks within the declared horizon, for *every* source state and any word.
    """
    if not isinstance(description, PolynomialInvariants):
        raise TypeError('exact invariant specification required')
    orbits = _coefficient_orbits(description)
    dimension = len(orbits)
    if (type(coordinates) is not tuple or not 1 <= len(coordinates) <= 8 or
        any(type(x) is not int or x < 0 or x.bit_length() > dimension
            for x in coordinates)):
        raise ValueError('generator architecture coordinate outside qualified family')
    operators = []
    for coordinate in coordinates:
        rows = [({1 << i} if description.tiers[i] < 0 else set())
                for i in range(description.width)]
        for j, orbit in enumerate(orbits):
            if (coordinate >> j) & 1:
                for output, monomial in orbit:
                    rows[output].add(monomial)
        operators.append(BooleanOperator(tuple(tuple(sorted(row)) for row in rows)))
    proto = PolynomialLawFamily(description, orbits, coordinates,
                                tuple(operators), str(1 << dimension), '')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())


def _multiply(left: tuple[int, ...], right: tuple[int, ...],
              term_limit: int, product_limit: int) -> tuple[int, ...]:
    if len(left) * len(right) > product_limit:
        raise PolynomialBudget('polynomial multiplication operation budget exceeded')
    product = set()
    for a in left:
        for b in right:
            value = a | b  # x_i**2 == x_i on Boolean states
            if value in product:
                product.remove(value)
            else:
                product.add(value)
    if len(product) > term_limit:
        raise PolynomialBudget('expanded Boolean polynomial term budget exceeded')
    return tuple(sorted(product))


def pullback(polynomial: tuple[int, ...], operator: BooleanOperator,
             *, term_limit: int = 1024,
             product_limit: int = 16384) -> tuple[int, ...]:
    """Exact ANF of f(T(x)); XOR is symmetric difference of monomial masks."""
    result = set()
    for mask in polynomial:
        term = (0,)  # multiplicative identity
        while mask:
            bit = mask & -mask
            term = _multiply(term, operator.rows[bit.bit_length()-1],
                             term_limit, product_limit)
            mask ^= bit
            if not term:
                break
        result.symmetric_difference_update(term)
        if len(result) > term_limit:
            raise PolynomialBudget('observer feature polynomial too large')
    return tuple(sorted(result))


def _pivot_reduce(poly: tuple[int, ...], pivots: dict[int, tuple[int, tuple[int, ...]]]):
    rest = set(poly)
    code = 0
    for pivot in sorted(pivots, reverse=True):
        if pivot in rest:
            index, row = pivots[pivot]
            rest.symmetric_difference_update(row)
            code ^= 1 << index
    return tuple(sorted(rest)), code


@dataclass(frozen=True)
class NonlinearKoopmanGround:
    family_sha256: str
    basis: tuple[tuple[int, ...], ...]
    operator_rows: tuple[tuple[int, ...], ...]
    observer_decoder: tuple[int, ...]
    digest: str

    @property
    def dimension(self) -> int:
        return len(self.basis)

    def encode(self) -> dict:
        return {'schema': KOOPMAN_SCHEMA, 'family_sha256': self.family_sha256,
                'basis': [[str(mask) for mask in row] for row in self.basis],
                'operator_rows': [[str(v) for v in row] for row in self.operator_rows],
                'observer_decoder': [str(x) for x in self.observer_decoder],
                'claim': 'least-pullback-invariant-observer-feature-linear-ground'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}

    def project(self, state: int) -> int:
        return eval_observers(self.basis, state)

    def read(self, encoded: int) -> int:
        return sum(((row & encoded).bit_count() & 1) << i
                   for i, row in enumerate(self.observer_decoder))


def construct_koopman_ground(family: PolynomialLawFamily, *,
                             max_features: int = 128,
                             term_limit: int = 1024,
                             product_limit: int = 16384) -> NonlinearKoopmanGround:
    """Construct the LEAST GF(2)-linear pullback-invariant observer subspace.

    Unlike approximate Koopman truncations, a budget violation refuses
    instead of projecting a missing feature into the available span.
    """
    if not isinstance(family, PolynomialLawFamily):
        raise TypeError('constructed invariant law family required')
    if (type(max_features) is not int or not 0 <= max_features <= 256 or
        type(term_limit) is not int or not 1 <= term_limit <= 2048 or
        type(product_limit) is not int or not 1 <= product_limit <= 200000):
        raise ValueError('invalid exact feature construction budgets')
    pivots: dict[int, tuple[int, tuple[int, ...]]] = {}
    basis = []
    frontier = []
    def insert(poly: tuple[int, ...]):
        reduced, _ = _pivot_reduce(poly, pivots)
        if reduced:
            if len(reduced) > term_limit:
                raise PolynomialBudget('reduced feature exceeds term limit')
            if len(basis) >= max_features:
                raise PolynomialBudget('observer requires more exact future features')
            index = len(basis)
            pivots[max(reduced)] = (index, reduced)
            basis.append(reduced)
            frontier.append(reduced)
    for obs in family.description.observers:
        insert(obs)
    # Each basis function is a *function of the full source state*, not a
    # sampled state. Closing all generator pullbacks preserves any word.
    while frontier:
        previous = tuple(frontier)
        frontier.clear()
        for row in previous:
            for operator in family.operators:
                insert(pullback(row, operator, term_limit=term_limit,
                                product_limit=product_limit))
    output = []
    for operator in family.operators:
        rows = []
        for row in basis:
            reduced, code = _pivot_reduce(
                pullback(row,operator,term_limit=term_limit,
                         product_limit=product_limit),pivots)
            if reduced:
                raise AssertionError('unclosed producer Koopman subspace')
            rows.append(code)
        output.append(tuple(rows))
    decoder = []
    for obs in family.description.observers:
        reduced, code = _pivot_reduce(obs,pivots)
        if reduced:
            raise AssertionError('observer not present in generated subspace')
        decoder.append(code)
    proto = NonlinearKoopmanGround(family.digest,tuple(basis),
                                  tuple(output),tuple(decoder),'')
    return replace(proto, digest=sha256(canonical(proto.encode())).hexdigest())


def _matrix_after(after: tuple[int, ...], before: tuple[int, ...]) -> tuple[int, ...]:
    result = []
    for row in after:
        value = 0
        while row:
            bit = row & -row
            value ^= before[bit.bit_length()-1]
            row ^= bit
        result.append(value)
    return tuple(result)


def _apply_matrix(rows: tuple[int, ...], state: int) -> int:
    return sum(((row & state).bit_count() & 1) << i for i, row in enumerate(rows))


@dataclass(frozen=True)
class NonlinearResult:
    family_sha256: str
    ground_sha256: str
    initial_decimal: str
    word: tuple[int, ...]
    repetitions_decimal: str
    initial_features_decimal: str
    result_features_decimal: str
    observed_decimal: str
    operator_products: int
    digest: str

    def encode(self) -> dict:
        return {'schema': RESULT_SCHEMA, 'family_sha256': self.family_sha256,
                'ground_sha256': self.ground_sha256,
                'initial_decimal': self.initial_decimal,
                'word': list(self.word), 'repetitions_decimal': self.repetitions_decimal,
                'initial_features_decimal': self.initial_features_decimal,
                'result_features_decimal': self.result_features_decimal,
                'observed_decimal': self.observed_decimal,
                'operator_products': self.operator_products,
                'claim': 'exact-nonlinear-source-observer-navigation;mathematical-only'}

    def packet(self) -> dict:
        return {**self.encode(), 'sha256': self.digest}


def navigate_nonlinear(family: PolynomialLawFamily, ground: NonlinearKoopmanGround,
                       initial: int, word: tuple[int, ...], repetitions: int) -> NonlinearResult:
    if not isinstance(family, PolynomialLawFamily) or not isinstance(ground, NonlinearKoopmanGround):
        raise ValueError('unqualified source or observer ground')
    if ground.family_sha256 != family.digest:
        raise ValueError('changed source law or observer scope')
    if type(initial) is not int or not 0 <= initial < 1 << family.description.width:
        raise ValueError('initial state is not a declared source bit vector')
    if (type(word) is not tuple or len(word) > 65536 or
        any(type(g) is not int or not 0 <= g < len(family.operators) for g in word)):
        raise ValueError('unknown or unbounded generator word')
    if type(repetitions) is not int or repetitions < 0 or repetitions.bit_length() > 4096:
        raise ValueError('invalid logical coordinate')
    dimension = ground.dimension
    identity = tuple(1 << i for i in range(dimension))
    word_operator = identity
    for g in word:
        word_operator = _matrix_after(ground.operator_rows[g],word_operator)
    result = identity
    products = 0
    power = word_operator
    remaining = repetitions
    while remaining:
        if remaining & 1:
            result = _matrix_after(power,result)
            products += 1
        remaining >>= 1
        if remaining:
            power = _matrix_after(power,power)
            products += 1
    start = ground.project(initial)
    end = _apply_matrix(result,start)
    observed = ground.read(end)
    proto = NonlinearResult(family.digest,ground.digest,str(initial),word,
                            str(repetitions),str(start),str(end),str(observed),
                            products,'')
    return replace(proto,digest=sha256(canonical(proto.encode())).hexdigest())
