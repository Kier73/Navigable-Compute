"""NCR-011: structure-selected coordinates for NCR-010 inverse fibres."""
from dataclasses import dataclass
from time import perf_counter_ns

from .intention_fibre import Fibre, FibreBudget, MAX_NODES, MAX_WORK, _input, construct_fibre
from .intention_fibre_verify import verify_fibre


def _order(width, order):
    if (type(order) is not tuple or len(order) != width
            or any(type(i) is not int for i in order)
            or set(order) != set(range(width))):
        raise ValueError("order must be a source-bit permutation")


def rename(value, order):
    """Permute source bit i into the position where order[position] == i."""
    if type(value) is not int or not 0 <= value < 1 << len(order):
        raise ValueError("invalid source bits")
    return sum(((value >> i) & 1) << j for j, i in enumerate(order))


def restore(value, order):
    """Invert rename."""
    if type(value) is not int or not 0 <= value < 1 << len(order):
        raise ValueError("invalid coordinate bits")
    return sum(((value >> j) & 1) << i for j, i in enumerate(order))


def rename_rows(rows, order):
    return tuple(tuple(sorted(rename(m, order) for m in p)) for p in rows)


def frontier_order(width, equations):
    """Greedy primal-incidence frontier; proposals alone carry no proof."""
    rows = _input(width, equations)
    neighbors = [0] * width
    for p in rows:
        support = 0
        for m in p:
            support |= m
        bits = support
        while bits:
            bit = bits & -bits
            neighbors[bit.bit_length() - 1] |= support ^ bit
            bits ^= bit
    remaining = (1 << width) - 1
    frontier = 0
    traversal = []
    while remaining:
        candidates = [i for i in range(width) if remaining >> i & 1]
        # Minimize still-unassigned vertices adjacent to assigned variables.
        # Prefer a frontier neighbor on ties; choose low remaining degree next.
        i = min(candidates, key=lambda j: (
            ((frontier | neighbors[j]) & (remaining ^ (1 << j))).bit_count(),
            not bool(frontier & (1 << j)),
            (neighbors[j] & remaining).bit_count(), j))
        remaining ^= 1 << i
        frontier = (frontier | neighbors[i]) & remaining
        traversal.append(i)
    # NCR-010 branches from highest coordinate bit to lowest.
    return tuple(reversed(traversal))


@dataclass(frozen=True)
class OrderedFibre:
    """Addresses are ordered by permuted coordinates, not source integers."""
    order: tuple[int, ...]
    fibre: Fibre

    @property
    def count(self):
        return self.fibre.count

    def unrank(self, address):
        return restore(self.fibre.unrank(address), self.order)

    def rank(self, state):
        return self.fibre.rank(rename(state, self.order))


def construct_ordered(width, equations, order, **budgets):
    rows = _input(width, equations)
    _order(width, order)
    return OrderedFibre(order, construct_fibre(width, rename_rows(rows, order), **budgets))


def verify_ordered(width, equations, certificate):
    """Check permutation and the original-source identity of every ANF."""
    try:
        rows = _input(width, equations)
        if type(certificate) is not OrderedFibre:
            return False
        order = certificate.order
        _order(width, order)
        inverse = {source: position for position, source in enumerate(order)}
        # Independent translation using set bits, rather than rename().
        translated = []
        for p in rows:
            translated.append(tuple(sorted(sum(2**inverse[i] for i in range(width)
                                                if m & (2**i)) for m in p)))
        return verify_fibre(width, tuple(translated), certificate.fibre)
    except (TypeError, ValueError, IndexError, AttributeError):
        return False


@dataclass(frozen=True)
class Attempt:
    label: str
    status: str
    elapsed_ns: int
    nodes: int | None
    work: int | None


def choose_ordered(width, equations, *, max_nodes=MAX_NODES, max_work=MAX_WORK):
    """Compare at most two bounded candidates; charge every attempted route."""
    rows = _input(width, equations)
    if (type(max_nodes) is not int or not 0 <= max_nodes <= MAX_NODES
            or type(max_work) is not int or not 0 <= max_work <= MAX_WORK):
        raise ValueError("invalid residual budget")
    start = perf_counter_ns()
    proposed = frontier_order(width, rows)
    planning_ns = perf_counter_ns() - start
    natural = tuple(range(width))
    candidates = [("frontier", proposed)]
    if natural != proposed:
        candidates.append(("natural", natural))
    attempts, admitted = [], []
    for label, order in candidates:
        start = perf_counter_ns()
        try:
            c = construct_ordered(width, rows, order, max_nodes=max_nodes, max_work=max_work)
            if not verify_ordered(width, rows, c):
                raise ValueError("independent coordinate audit failed")
            attempts.append(Attempt(label, "QUALIFIED", perf_counter_ns() - start,
                                    len(c.fibre.nodes), c.fibre.work))
            admitted.append(c)
        except FibreBudget:
            attempts.append(Attempt(label, "UNKNOWN", perf_counter_ns() - start, None, None))
    if not admitted:
        raise FibreBudget("no qualified offered ordering")
    if len({c.count for c in admitted}) != 1:
        raise ValueError("qualified cardinalities disagree")
    return min(admitted, key=lambda c: (len(c.fibre.nodes), c.fibre.work, c.order)), tuple(attempts), planning_ns
