"""Independent NCR-006 verifier.

Does not use the symbolic quotient constructor, its linear basis reducer,
its matrix composition, or its navigator. Dual-span saturation checks
minimality; direct source-matrix exponentiation checks distant observations.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .symbolic_quotient import (
    SOURCE_SCHEMA, QUOTIENT_SCHEMA, RESULT_SCHEMA,
    AffineMap, SymbolicSource, SymbolicQuotient, SymbolicResult,
)

def _canonical(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':')).encode('ascii')


def _parity(x: int) -> int:
    return x.bit_count() % 2


def _value(rows: tuple[int, ...], state: int) -> int:
    answer = 0
    for i, row in enumerate(rows):
        answer ^= _parity(state & row) << i
    return answer


def _dual_image(row: int, matrix: tuple[int, ...]) -> int:
    """Independent, index-based GF(2) row-by-matrix calculation."""
    result = 0
    for j in range(len(matrix)):
        if (row >> j) & 1:
            result ^= matrix[j]
    return result


def _span_insert(value: int, pivots: dict[int,int]) -> int | None:
    """Low-pivot elimination, independent of producer's high-pivot basis."""
    v = value
    while v:
        pivot = (v & -v).bit_length() - 1
        if pivot not in pivots:
            pivots[pivot] = v
            return v
        v ^= pivots[pivot]
    return None


def _span_contains(value: int, pivots: dict[int,int]) -> bool:
    v = value
    while v:
        pivot = (v & -v).bit_length() - 1
        if pivot not in pivots:
            return False
        v ^= pivots[pivot]
    return True


def _independent_future_span(source: SymbolicSource) -> tuple[dict[int,int],int]:
    """Saturate O, OA_i, OA_iA_j, ... using a low-pivot work queue."""
    pivots: dict[int,int] = {}
    frontier = []
    for row in source.observer_rows:
        candidate = _span_insert(row, pivots)
        if candidate is not None:
            frontier.append(candidate)
    rounds = 0
    while frontier:
        next_frontier = []
        for row in frontier:
            for generator in source.generators:
                image = _dual_image(row, generator.rows)
                candidate = _span_insert(image, pivots)
                if candidate is not None:
                    next_frontier.append(candidate)
        if not next_frontier:
            break
        rounds += 1
        frontier = next_frontier
    return pivots, rounds


def verify_symbolic_quotient(source: SymbolicSource,
                             quotient: SymbolicQuotient) -> bool:
    if not isinstance(source, SymbolicSource) or not isinstance(quotient, SymbolicQuotient):
        return False
    try:
        if (quotient.source_sha256 != source.digest or
            quotient.digest != sha256(_canonical(quotient.encode())).hexdigest() or
            quotient.encode()['schema'] != QUOTIENT_SCHEMA or
            type(quotient.dimension) is not int or
            not 0 <= quotient.dimension <= source.width or
            type(quotient.basis_rows) is not tuple or
            len(quotient.basis_rows) != quotient.dimension or
            any(type(row) is not int or not 0 <= row < 1 << source.width
                for row in quotient.basis_rows) or
            type(quotient.operators) is not tuple or
            len(quotient.operators) != len(source.generators) or
            type(quotient.observer_decoder) is not tuple or
            len(quotient.observer_decoder) != len(source.observer_rows) or
            any(type(row) is not int or not 0 <= row < 1 << quotient.dimension
                for row in quotient.observer_decoder) or
            type(quotient.closure_rounds) is not int):
            return False
        pivots, rounds = _independent_future_span(source)
        if len(pivots) != quotient.dimension or quotient.closure_rounds != rounds:
            return False
        submitted: dict[int,int] = {}
        for row in quotient.basis_rows:
            if not _span_contains(row,pivots) or _span_insert(row,submitted) is None:
                return False
        # Equal dimension plus independence and membership establish equality
        # of the proposed and independently constructed entire row spaces.
        for observer, decoder in zip(source.observer_rows, quotient.observer_decoder):
            decoded = 0
            for k, row in enumerate(quotient.basis_rows):
                if (decoder >> k) & 1:
                    decoded ^= row
            if decoded != observer:
                return False
        for original, induced in zip(source.generators, quotient.operators):
            if (not isinstance(induced, AffineMap) or
                type(induced.rows) is not tuple or
                len(induced.rows) != quotient.dimension or
                any(type(x) is not int or not 0 <= x < 1 << quotient.dimension
                    for x in induced.rows) or
                type(induced.bias) is not int or
                not 0 <= induced.bias < 1 << quotient.dimension):
                return False
            for i, basis_row in enumerate(quotient.basis_rows):
                expected_source_row = _dual_image(basis_row, original.rows)
                lifted_induced_row = 0
                for j, other_row in enumerate(quotient.basis_rows):
                    if (induced.rows[i] >> j) & 1:
                        lifted_induced_row ^= other_row
                if expected_source_row != lifted_induced_row:
                    return False
                if _parity(basis_row & original.bias) != ((induced.bias >> i) & 1):
                    return False
        return True
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError):
        return False


def _source_compose(after: AffineMap, before: AffineMap) -> AffineMap:
    """Independent row-index implementation of affine operator composition."""
    out = []
    for row in after.rows:
        product = 0
        for index, encoded_row in enumerate(before.rows):
            if (row >> index) & 1:
                product ^= encoded_row
        out.append(product)
    offset = _value(after.rows,before.bias) ^ after.bias
    return AffineMap(tuple(out),offset)


def _canonical_decimal(raw: object, *, max_bits: int) -> int:
    if (type(raw) is not str or not raw or not raw.isascii() or
        not raw.isdecimal() or len(raw) > (max_bits * 30103 // 100000 + 3)):
        raise ValueError('invalid bounded decimal')
    number = int(raw)
    if str(number) != raw or number.bit_length() > max_bits:
        raise ValueError('noncanonical or out-of-range integer')
    return number


def verify_symbolic_result(source: SymbolicSource, quotient: SymbolicQuotient,
                           result: SymbolicResult) -> bool:
    if not isinstance(result, SymbolicResult) or not verify_symbolic_quotient(source, quotient):
        return False
    try:
        if (result.source_sha256 != source.digest or
            result.quotient_sha256 != quotient.digest or
            result.digest != sha256(_canonical(result.encode())).hexdigest() or
            result.encode()['schema'] != RESULT_SCHEMA or
            type(result.word) is not tuple or len(result.word) > 65536 or
            any(type(x) is not int or not 0 <= x < len(source.generators)
                for x in result.word) or
            type(result.operator_products) is not int):
            return False
        initial = _canonical_decimal(result.initial_decimal, max_bits=source.width)
        repetitions = _canonical_decimal(result.repetitions_decimal,max_bits=4096)
        before = _canonical_decimal(result.source_projection_decimal,
                                    max_bits=max(1,quotient.dimension))
        after = _canonical_decimal(result.final_projection_decimal,
                                   max_bits=max(1,quotient.dimension))
        observed = _canonical_decimal(result.observed_decimal,
                                      max_bits=len(source.observer_rows))
        if before != _value(quotient.basis_rows,initial):
            return False
        # Reconstruct entire source word and exponentiate it in F_2^n
        # WITHOUT relying on the quotient, its compiler or interpreter.
        n = source.width
        neutral = AffineMap(tuple(1 << i for i in range(n)),0)
        word_op = neutral
        for index in result.word:
            word_op = _source_compose(source.generators[index], word_op)
        current = neutral
        power = word_op
        remaining = repetitions
        products = 0
        while remaining:
            if remaining & 1:
                current = _source_compose(power,current)
                products += 1
            remaining >>= 1
            if remaining:
                power = _source_compose(power,power)
                products += 1
        final_source = _value(current.rows, initial) ^ current.bias
        return (after == _value(quotient.basis_rows,final_source) and
                observed == _value(source.observer_rows,final_source) and
                observed == _value(quotient.observer_decoder,after) and
                result.operator_products == products)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError):
        return False
