"""Independent NCR-007 audit using a distinct Boolean algebra and elimination path.

Never invokes the producer's law constructor, polynomial pullback, feature
closure, or source-free navigation. Reconstructs laws directly from invariants,
tests exact ANF identities, and uses a bounded original nonlinear source oracle.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json

from .nonlinear_koopman import (
    PolynomialInvariants, BooleanOperator, PolynomialLawFamily,
    NonlinearKoopmanGround, NonlinearResult, PolynomialBudget,
    SCHEMA, KOOPMAN_SCHEMA, RESULT_SCHEMA,
)


def _canonical(data: dict) -> bytes:
    return json.dumps(data,sort_keys=True,separators=(',',':'),
                      ensure_ascii=True).encode('ascii')


def _permutation_image(mask: int, permutation: tuple[int,...]) -> int:
    result = 0
    for index in range(len(permutation)):
        if mask & (1 << index):
            result |= 1 << permutation[index]
    return result


def _law_orbits(description: PolynomialInvariants):
    """Build independently via subset combinations and two-sided orbit BFS."""
    available = set()
    for target, inputs in enumerate(description.allowed_inputs):
        for degree in range(1, min(len(inputs),description.degree)+1):
            for group in combinations(inputs,degree):
                mask = 0
                for position in group:
                    mask |= 1 << position
                available.add((target,mask))
                if len(available) > 8192:
                    raise PolynomialBudget('too many independently checked law templates')
    inv = [0]*description.width
    for i,destination in enumerate(description.symmetry):
        inv[destination] = i
    accounted = set()
    orbits = []
    for seed in sorted(available):
        if seed in accounted:
            continue
        orbit = {seed}
        queue = [seed]
        for index,mask in queue:
            for perm in (description.symmetry,tuple(inv)):
                other=(perm[index],_permutation_image(mask,perm))
                if other not in available:
                    return None
                if other not in orbit:
                    orbit.add(other)
                    queue.append(other)
        if accounted & orbit:
            return None
        accounted.update(orbit)
        orbits.append(tuple(sorted(orbit)))
        if len(orbits)>4096:
            raise PolynomialBudget('law-orbit budget exceeded')
    return tuple(orbits)


def verify_polynomial_laws(family: PolynomialLawFamily) -> bool:
    if not isinstance(family, PolynomialLawFamily):
        return False
    try:
        description = family.description
        if not isinstance(description,PolynomialInvariants):
            return False
        # Re-run the full typed invariant validation, rather than trusting
        # a forged dataclass instance or just its digest.
        PolynomialInvariants(description.width,description.tiers,
            description.symmetry,description.allowed_inputs,description.degree,
            description.observers)
        if (family.digest != sha256(_canonical(family.encode())).hexdigest() or
            family.encode()['schema'] != SCHEMA or
            type(family.coordinates) is not tuple or
            not 1 <= len(family.coordinates) <= 8 or
            type(family.coefficient_orbits) is not tuple or
            len(family.operators) != len(family.coordinates)):
            return False
        discovered = _law_orbits(description)
        if discovered is None or family.coefficient_orbits != discovered:
            return False
        if family.law_count_decimal != str(1 << len(discovered)):
            return False
        for coordinate,operator in zip(family.coordinates,family.operators):
            if (type(coordinate) is not int or coordinate < 0 or
                coordinate.bit_length() > len(discovered) or
                not isinstance(operator,BooleanOperator) or
                type(operator.rows) is not tuple or
                len(operator.rows) != description.width):
                return False
            required = [({1 << i} if description.tiers[i] == -1 else set())
                        for i in range(description.width)]
            for orbit_idx,orbit in enumerate(discovered):
                if (coordinate >> orbit_idx) & 1:
                    for target,monomial in orbit:
                        required[target].add(monomial)
            if any(type(actual) is not tuple or actual != tuple(sorted(expected))
                   for actual,expected in zip(operator.rows,required)):
                return False
        return True
    except (ValueError,TypeError,IndexError,KeyError,AttributeError,OverflowError):
        return False


def _mul(left: tuple[int,...],right: tuple[int,...],
         term_limit: int,product_limit: int):
    if len(left)*len(right) > product_limit:
        raise PolynomialBudget('independent product operation budget exceeded')
    result = {}
    for a in left:
        for b in right:
            mask = a | b
            result[mask] = not result.get(mask,False)
    emitted = tuple(mask for mask,odd in sorted(result.items()) if odd)
    if len(emitted) > term_limit:
        raise PolynomialBudget('independent product feature expansion too large')
    return emitted


def _source_substitute(poly: tuple[int,...],operator: BooleanOperator,
                       term_limit: int,product_limit: int):
    """Independent Boolean polynomial evaluation via explicit term parities."""
    counts = {}
    for monomial in poly:
        factor = (0,)
        for bit_index in range(len(operator.rows)):
            if (monomial >> bit_index) & 1:
                factor = _mul(factor,operator.rows[bit_index],
                              term_limit,product_limit)
                if not factor:
                    break
        for produced in factor:
            counts[produced] = not counts.get(produced,False)
        if sum(counts.values()) > term_limit:
            raise PolynomialBudget('independent ANF explosion')
    return tuple(mask for mask,odd in sorted(counts.items()) if odd)


def _insert_low_poly(poly: tuple[int,...],pivots: dict[int,tuple[int,...]]):
    """Low-pivot elimination independent of producer's high-pivot algorithm."""
    rest = set(poly)
    for pivot in sorted(pivots):
        if pivot in rest:
            rest.symmetric_difference_update(pivots[pivot])
    if rest:
        reduced = tuple(sorted(rest))
        pivots[min(reduced)] = reduced
        return reduced
    return None


def _contains_low(poly: tuple[int,...],pivots: dict[int,tuple[int,...]]) -> bool:
    rest = set(poly)
    for pivot in sorted(pivots):
        if pivot in rest:
            rest.symmetric_difference_update(pivots[pivot])
    return not rest


def _source_future_space(family: PolynomialLawFamily,
                         max_features: int,term_limit: int,product_limit: int):
    pivots = {}
    frontier = []
    for obs in family.description.observers:
        r = _insert_low_poly(obs,pivots)
        if r is not None:
            frontier.append(r)
    while frontier:
        fresh = []
        for poly in frontier:
            for op in family.operators:
                produced = _source_substitute(poly,op,term_limit,product_limit)
                candidate = _insert_low_poly(produced,pivots)
                if candidate is not None:
                    fresh.append(candidate)
                    if len(pivots) > max_features:
                        raise PolynomialBudget('independent future-feature budget exceeded')
        frontier = fresh
    return pivots


def _xor_features(basis: tuple[tuple[int,...],...],bits: int) -> tuple[int,...]:
    parity = {}
    for i,row in enumerate(basis):
        if (bits >> i) & 1:
            for monomial in row:
                parity[monomial] = not parity.get(monomial,False)
    return tuple(monomial for monomial,odd in sorted(parity.items()) if odd)


def _evaluate(poly: tuple[int,...],source_state: int) -> int:
    return sum((source_state & mask)==mask for mask in poly) & 1


def _source_output(polynomials: tuple[tuple[int,...],...],state:int) -> int:
    result = 0
    for i,poly in enumerate(polynomials):
        result |= _evaluate(poly,state) << i
    return result


def verify_koopman_ground(family: PolynomialLawFamily,
                          ground: NonlinearKoopmanGround, *,
                          term_limit: int = 1024,
                          product_limit: int = 16384) -> bool:
    if not isinstance(ground,NonlinearKoopmanGround) or not verify_polynomial_laws(family):
        return False
    try:
        if (ground.family_sha256 != family.digest or
            ground.digest != sha256(_canonical(ground.encode())).hexdigest() or
            ground.encode()['schema'] != KOOPMAN_SCHEMA or
            type(ground.basis) is not tuple or
            ground.dimension > 256 or
            any(type(row) is not tuple or
                len(row) > 2048 or
                any(type(m) is not int or not 0 <= m < 1 << family.description.width
                    for m in row) or tuple(sorted(set(row))) != row
                for row in ground.basis) or
            type(ground.operator_rows) is not tuple or
            len(ground.operator_rows) != len(family.operators) or
            any(type(row) is not tuple or len(row) != ground.dimension or
                any(type(mask) is not int or not 0 <= mask < 1 << ground.dimension
                    for mask in row)
                for row in ground.operator_rows) or
            type(ground.observer_decoder) is not tuple or
            len(ground.observer_decoder) != len(family.description.observers) or
            any(type(mask) is not int or not 0 <= mask < 1 << ground.dimension
                for mask in ground.observer_decoder)):
            return False
        independent = _source_future_space(family, 256, term_limit, product_limit)
        if len(independent) != ground.dimension:
            return False
        submitted = {}
        for row in ground.basis:
            if not _contains_low(row,independent) or _insert_low_poly(row,submitted) is None:
                return False
        for obs,encoded in zip(family.description.observers,ground.observer_decoder):
            if _xor_features(ground.basis,encoded) != obs:
                return False
        for op,encoded_rows in zip(family.operators,ground.operator_rows):
            for basis_row,encoded in zip(ground.basis,encoded_rows):
                expected = _source_substitute(basis_row,op,term_limit,product_limit)
                if _xor_features(ground.basis,encoded) != expected:
                    return False
        return True
    except (TypeError,ValueError,KeyError,IndexError,AttributeError,OverflowError):
        return False


def _decimal(value,maximum_bits:int):
    if (type(value) is not str or not value or not value.isascii() or
        not value.isdecimal() or len(value) > maximum_bits*30103//100000+3):
        raise ValueError('not a canonical bounded decimal')
    number = int(value)
    if str(number)!=value or number.bit_length()>maximum_bits:
        raise ValueError('out-of-range noncanonical decimal')
    return number


def verify_nonlinear_result(family: PolynomialLawFamily,ground: NonlinearKoopmanGround,
                            result: NonlinearResult) -> bool:
    if not isinstance(result,NonlinearResult) or not verify_koopman_ground(family,ground):
        return False
    try:
        n=family.description.width
        r=ground.dimension
        if (result.family_sha256 != family.digest or
            result.ground_sha256 != ground.digest or
            result.digest != sha256(_canonical(result.encode())).hexdigest() or
            result.encode()['schema'] != RESULT_SCHEMA or
            type(result.word) is not tuple or len(result.word) > 65536 or
            any(type(i) is not int or not 0 <= i < len(family.operators)
                for i in result.word) or
            type(result.operator_products) is not int):
            return False
        source_state=_decimal(result.initial_decimal,n)
        count=_decimal(result.repetitions_decimal,4096)
        first=_decimal(result.initial_features_decimal,max(1,r))
        last=_decimal(result.result_features_decimal,max(1,r))
        observed=_decimal(result.observed_decimal,len(family.description.observers))
        actual_first=_source_output(ground.basis,source_state)
        if first != actual_first:
            return False

        # Independent original-source evaluation. Every generated source law
        # strictly moves active dependencies upward in tier and preserves all
        # protected bits: after at most the certified horizon of *any* word,
        # all active bits are zero. No approximate long-time extrapolation.
        steps=len(result.word)*count
        horizon=family.description.drain_horizon
        if steps >= horizon:
            protected=sum(1 << i for i,tier in enumerate(family.description.tiers)
                          if tier == -1)
            ending_state=source_state & protected
        else:
            ending_state=source_state
            # An empty word is identity at every logical coordinate, including
            # astronomically large repetition counts.
            if result.word:
                for _ in range(count):
                    for index in result.word:
                        ending_state=_source_output(family.operators[index].rows,ending_state)
        final_features=_source_output(ground.basis,ending_state)
        final_observer=_source_output(family.description.observers,ending_state)
        if last != final_features or observed != final_observer:
            return False
        # Verify the claimed count of quotient matrix products, not an
        # unmeasured elapsed time or number of original source transitions.
        expected=count.bit_count() + max(count.bit_length()-1,0)
        return result.operator_products == expected and (
            _source_output_from_feature_decode(ground.observer_decoder,last)
            == final_observer)
    except (TypeError,ValueError,KeyError,IndexError,AttributeError,OverflowError):
        return False


def _source_output_from_feature_decode(rows: tuple[int,...],z:int) -> int:
    output = 0
    for index,row in enumerate(rows):
        output |= ((row & z).bit_count() & 1) << index
    return output
