"""Independent residual/count checker for NCR-010 (no producer recursion)."""
from collections import Counter

from .intention_fibre import (
    Fibre, Node, TemporalFibre, MAX_NODES, MAX_WORK, _goals)
from .nonlinear_koopman_verify import verify_polynomial_laws


def _canonical(rows):
    distinct = sorted(set(rows) - {()})
    return ((0,),) if (0,) in distinct else tuple(distinct)


def _substitute(rows, k, value):
    # Independently split each polynomial into p = a XOR x_k*b,
    # then count parity in a or a XOR b.
    result = []
    for row in rows:
        a, b = [], []
        for m in row:
            if (m >> k) & 1:
                b.append(m - (1 << k))
            else:
                a.append(m)
        frequencies = Counter(a + (b if value else []))
        result.append(tuple(sorted(m for m, n in frequencies.items() if n % 2)))
    return _canonical(result)


def verify_fibre(width, equations, certificate):
    """Check every Shannon edge, count, state identity, and the supplied root."""
    try:
        if (type(width) is not int or not 1 <= width <= 128
                or type(equations) is not tuple or len(equations) > 1024
                or any(type(p) is not tuple or len(p) > 2048
                       or any(type(m) is not int or not 0 <= m < 2**width for m in p)
                       or tuple(sorted(set(p))) != p for p in equations)):
            return False
        expected = _canonical(equations)
        c = certificate
        if (type(c) is not Fibre or type(c.width) is not int or c.width != width
                or c.equations != expected or type(c.nodes) is not tuple
                or len(c.nodes) > MAX_NODES or type(c.root) is not int
                or not 0 <= c.root < len(c.nodes) + 2
                or type(c.work) is not int or not 0 <= c.work <= MAX_WORK):
            return False
        seen = {}
        identities = set()
        stack = [(c.root, width, expected)]
        work = 0
        while stack:
            node, k, rows = stack.pop()
            if rows == ((0,),):
                if node != 0:
                    return False
                continue
            if k == 0:
                if node != 1 or rows:
                    return False
                continue
            if node < 2:
                return False
            key = (k, rows)
            if node in seen:
                if seen[node] != key:
                    return False
                continue
            if key in identities:
                return False
            n = c.nodes[node - 2]
            if (type(n) is not Node or type(n.remaining) is not int
                    or n.remaining != k or n.equations != rows
                    or any(type(v) is not int or not 0 <= v < node
                           for v in (n.low, n.high))
                    or type(n.count) is not int or not 0 <= n.count <= 2**k):
                return False
            work += 1 + 2 * sum(1 + len(p) for p in rows)
            if work > MAX_WORK:
                return False
            counts = [v if v < 2 else c.nodes[v - 2].count for v in (n.low, n.high)]
            if any(type(v) is not int for v in counts) or n.count != sum(counts):
                return False
            seen[node] = key
            identities.add(key)
            stack.append((n.high, k - 1, _substitute(rows, k - 1, 1)))
            stack.append((n.low, k - 1, _substitute(rows, k - 1, 0)))
        return len(seen) == len(c.nodes) and work == c.work
    except (ValueError, TypeError, IndexError, AttributeError, OverflowError):
        return False


def verify_temporal_fibre(family, goals, certificate):
    """Recompile terminal ANFs with the independent NCR-007 substitution path."""
    try:
        return (type(certificate) is TemporalFibre
                and certificate.source_digest == family.digest
                and certificate.goals == goals and verify_polynomial_laws(family)
                and verify_fibre(family.description.width,
                                 _goals(family, goals, independent=True),
                                 certificate.fibre))
    except (ValueError, TypeError, IndexError, AttributeError, OverflowError):
        return False
