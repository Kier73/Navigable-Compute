"""NCR-011: persistent intersections over an already admitted NCR-010 DAG.

Negative references reuse parent subtrees verbatim; zero is empty. Positive
references index overlay nodes. No parent residual is substituted again.
"""
from dataclasses import dataclass

from .intention_fibre import Fibre, FibreBudget, FALSE, MAX_NODES, MAX_WORK, _input, _restrict
from .intention_fibre_verify import verify_fibre


@dataclass(frozen=True)
class OverlayNode:
    parent_id: int
    equations: tuple
    low: int
    high: int
    count: int


@dataclass(frozen=True)
class Refinement:
    parent: Fibre
    equations: tuple
    nodes: tuple[OverlayNode, ...]
    root: int
    work: int

    def count_at(self, ref):
        if ref <= 0:
            return self.parent.count_at(-ref)
        return self.nodes[ref - 1].count

    @property
    def count(self):
        return self.count_at(self.root)

    def parent_address(self, address):
        """Embed a refined address in the old address space without decoding x."""
        if type(address) is not int or not 0 <= address < self.count:
            raise ValueError("invalid refined address")
        ref, offset = self.root, 0
        while ref > 0:
            n = self.nodes[ref - 1]
            left = self.count_at(n.low)
            if address < left:
                ref = n.low
            else:
                address -= left
                offset += self.parent.count_at(self.parent.nodes[n.parent_id - 2].low)
                ref = n.high
        if ref == 0:
            raise ValueError("invalid overlay path")
        return offset + address

    def refined_address(self, address):
        """Recover the new ordinal or refuse if the old member was excluded."""
        if type(address) is not int or not 0 <= address < self.parent.count:
            raise ValueError("invalid parent address")
        ref, offset = self.root, 0
        while ref > 0:
            n = self.nodes[ref - 1]
            left = self.parent.count_at(self.parent.nodes[n.parent_id - 2].low)
            if address < left:
                ref = n.low
            else:
                address -= left
                offset += self.count_at(n.low)
                ref = n.high
        if ref == 0:
            raise ValueError("parent member excluded by refinement")
        return offset + address

    def unrank(self, address):
        return self.parent.unrank(self.parent_address(address))

    def rank(self, state):
        return self.refined_address(self.parent.rank(state))


def refine(parent, equations, *, max_nodes=MAX_NODES, max_work=MAX_WORK):
    """Intersect additional ANFs; parent is immutable and independently admitted."""
    if type(parent) is not Fibre or not verify_fibre(parent.width, parent.equations, parent):
        raise ValueError("unqualified parent")
    equations = _input(parent.width, equations)
    if (type(max_nodes) is not int or not 0 <= max_nodes <= MAX_NODES
            or type(max_work) is not int or not 0 <= max_work <= MAX_WORK):
        raise ValueError("invalid refinement budget")
    nodes, memo = [], {}
    work = 0

    def count(ref):
        return parent.count_at(-ref) if ref <= 0 else nodes[ref - 1].count

    def build(p, rows):
        nonlocal work
        if rows == FALSE or not parent.count_at(p):
            return 0
        if not rows:
            return -p
        if p < 2:
            raise AssertionError("nonconstant residual at parent leaf")
        key = (p, rows)
        if key in memo:
            return memo[key]
        work += 1 + 2 * sum(1 + len(row) for row in rows)
        if work > max_work:
            raise FibreBudget("overlay work budget exceeded")
        n = parent.nodes[p - 2]
        bit = 1 << (n.remaining - 1)
        low = build(n.low, _restrict(rows, bit, 0))
        high = build(n.high, _restrict(rows, bit, 1))
        if len(nodes) >= max_nodes:
            raise FibreBudget("overlay node budget exceeded")
        ref = len(nodes) + 1
        nodes.append(OverlayNode(p, rows, low, high, count(low) + count(high)))
        memo[key] = ref
        return ref

    root = build(parent.root, equations)
    return Refinement(parent, equations, tuple(nodes), root, work)


def observation_rows(width, mask, value):
    """Convert a declared raw-coordinate observation into ANF equalities."""
    if (type(width) is not int or not 1 <= width <= 128 or type(mask) is not int
            or not 0 <= mask < 1 << width or type(value) is not int
            or value < 0 or value & ~mask):
        raise ValueError("invalid observation")
    return tuple((0, 1 << i) if value >> i & 1 else (1 << i,)
                 for i in range(width) if mask >> i & 1)
