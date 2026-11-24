"""NCR-008: exact factorized image certificates for Boolean ANF feature maps.

The existing NCR-007 verifier must admit the source and feature Ground first.
This module certifies the image of its feature basis, not the source law.
"""
from dataclasses import dataclass
from math import prod


class ImageBudget(ValueError):
    """Exact local image construction exceeded the declared work allowance."""


def support(poly):
    """Return the bit mask of source coordinates appearing in an ANF."""
    mask = 0
    for term in poly:
        mask |= term
    return mask


def validate_basis(basis, width, max_cells):
    """Validate the bounded Boolean ANF input and work allowance."""
    if type(width) is not int or not 0 <= width <= 4096:
        raise ValueError("invalid source width")
    if type(max_cells) is not int or not 1 <= max_cells <= 10_000_000:
        raise ValueError("invalid cell allowance")
    if type(basis) is not tuple or len(basis) > 256:
        raise ValueError("invalid basis")
    for poly in basis:
        if (type(poly) is not tuple or len(poly) > 2048 or
                any(type(m) is not int or not 0 <= m < 1 << width for m in poly)
                or tuple(sorted(set(poly))) != poly):
            raise ValueError("noncanonical ANF")


def positions(mask):
    """List set-bit positions in increasing order."""
    return tuple(i for i in range(mask.bit_length()) if mask >> i & 1)


def embed(value, indices):
    """Scatter a compact bit vector into selected coordinates."""
    return sum(((value >> j) & 1) << i for j, i in enumerate(indices))


def evaluate(poly, state):
    """Evaluate an ANF exactly over GF(2)."""
    return sum((state & m) == m for m in poly) & 1


@dataclass(frozen=True)
class Factor:
    """Independent source support, feature indices and exact local image."""
    indices: tuple
    source_mask: int
    values: tuple


@dataclass(frozen=True)
class ImageCertificate:
    """Factored image; mixed-radix state count and charged local work."""
    factors: tuple
    cells: int

    @property
    def cardinality(self):
        """Exact number of realizable feature vectors."""
        return prod(len(f.values) for f in self.factors)

    @property
    def bits(self):
        """Minimum fixed-width encoding size for the certified image."""
        return (self.cardinality - 1).bit_length()


def construct_image(basis, width, *, max_cells=1_000_000):
    """Enumerate only connected feature-support components, not full states."""
    validate_basis(basis, width, max_cells)
    groups = []
    for i, poly in enumerate(basis):
        mask = support(poly)
        indices = [i]
        kept = []
        for other, other_mask in groups:
            if mask & other_mask:
                indices.extend(other)
                mask |= other_mask
            else:
                kept.append((other, other_mask))
        kept.append((indices, mask))
        groups = kept
    factors = []
    cells = 0
    for ids, mask in sorted(groups, key=lambda g: min(g[0])):
        ids = tuple(sorted(ids))
        pos = positions(mask)
        cost = (1 << len(pos)) * sum(1 + len(basis[i]) for i in ids)
        if cells + cost > max_cells:
            raise ImageBudget("local ANF enumeration exceeds allowance")
        cells += cost
        values = set()
        for a in range(1 << len(pos)):
            x = embed(a, pos)
            values.add(sum(evaluate(basis[i], x) << j for j, i in enumerate(ids)))
        factors.append(Factor(ids, mask, tuple(sorted(values))))
    return ImageCertificate(tuple(factors), cells)


def pack(certificate, feature_vector):
    """Encode an admitted feature image in its optimal fixed-width range."""
    n = sum(len(f.indices) for f in certificate.factors)
    if type(feature_vector) is not int or not 0 <= feature_vector < 1 << n:
        raise ValueError("feature vector outside declared width")
    code, stride = 0, 1
    for factor in certificate.factors:
        local = sum(((feature_vector >> i) & 1) << j
                    for j, i in enumerate(factor.indices))
        if local not in factor.values:
            raise ValueError("unrealizable feature vector")
        code += stride * factor.values.index(local)
        stride *= len(factor.values)
    return code


def unpack(certificate, code):
    """Recover all original feature coordinates from a valid packed state."""
    if type(code) is not int or not 0 <= code < certificate.cardinality:
        raise ValueError("code outside certified image")
    z = 0
    for factor in certificate.factors:
        code, digit = divmod(code, len(factor.values))
        z |= embed(factor.values[digit], factor.indices)
    return z


def verify_image(basis, width, certificate, *, max_cells=1_000_000):
    """Check disjoint supports and exact local images using separate evaluation.

    No reliance on the producer's component construction or bit embedding.
    """
    try:
        validate_basis(basis, width, max_cells)
        if type(certificate) is not ImageCertificate or type(certificate.factors) is not tuple:
            return False
        seen = set()
        used = set()
        cells = 0
        for f in certificate.factors:
            if (type(f) is not Factor or type(f.indices) is not tuple
                    or not f.indices or any(type(i) is not int or not 0 <= i < len(basis) for i in f.indices)
                    or tuple(sorted(set(f.indices))) != f.indices
                    or seen.intersection(f.indices)):
                return False
            seen.update(f.indices)
            # Independently represent ANF monomials as sets of source indices.
            polys = [[frozenset(k for k in range(m.bit_length()) if m >> k & 1)
                      for m in basis[i]] for i in f.indices]
            variables = sorted(set().union(*(term for p in polys for term in p)))
            if type(f.source_mask) is not int or f.source_mask != sum(1 << k for k in variables):
                return False
            if used.intersection(variables):
                return False
            used.update(variables)
            cost = (1 << len(variables)) * sum(1 + len(p) for p in polys)
            cells += cost
            if cells > max_cells:
                return False
            observed = set()
            for a in range(1 << len(variables)):
                true_vars = {v for j, v in enumerate(variables) if a >> j & 1}
                out = 0
                for j, polynomial in enumerate(polys):
                    if sum(term <= true_vars for term in polynomial) % 2:
                        out |= 1 << j
                observed.add(out)
            if (type(f.values) is not tuple or any(type(v) is not int for v in f.values)
                    or f.values != tuple(sorted(observed))):
                return False
        return (seen == set(range(len(basis))) and type(certificate.cells) is int
                and certificate.cells == cells)
    except (TypeError, ValueError, AttributeError):
        return False


def packed_step(ground, certificate, code, action):
    """Apply a qualified NCR-007 feature operator through the packed image."""
    if type(action) is not int or not 0 <= action < len(ground.operator_rows):
        raise ValueError("invalid action")
    z = unpack(certificate, code)
    next_z = sum(((row & z).bit_count() & 1) << i
                 for i, row in enumerate(ground.operator_rows[action]))
    return pack(certificate, next_z)
