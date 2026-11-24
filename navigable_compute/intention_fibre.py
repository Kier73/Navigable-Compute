"""NCR-010: addressable inverse observation fibres over exact Boolean Grounds.

A conjunction of ANF equations is compiled into a counted residual DAG.
Each address denotes one satisfying *source state*, in increasing integer
order. Formation can be exponential; explicit budgets refuse incomplete DAGs.
"""
from dataclasses import dataclass

from .complexity_navigation import _prefix, _terminal
from .nonlinear_koopman_verify import verify_polynomial_laws


class FibreBudget(ValueError):
    """The exact residual graph exceeded its declared formation allowance."""


FALSE = ((0,),)
MAX_NODES = 100_000
MAX_WORK = 20_000_000


def _normal(equations):
    """Canonical conjunction; each polynomial is required to equal zero."""
    result = tuple(sorted(set(p for p in equations if p)))
    return FALSE if (0,) in result else result


def _input(width, equations):
    if type(width) is not int or not 1 <= width <= 128:
        raise ValueError("source width must be 1..128")
    if (type(equations) is not tuple or len(equations) > 1024
            or any(type(p) is not tuple or len(p) > 2048
                   or any(type(m) is not int or not 0 <= m < 1 << width for m in p)
                   or tuple(sorted(set(p))) != p for p in equations)):
        raise ValueError("expected canonical ANF equations")
    return _normal(equations)


def _restrict(equations, bit, value):
    restricted = []
    for p in equations:
        terms = set()
        for m in p:
            if m & bit and not value:
                continue
            term = m & ~bit
            if term in terms:
                terms.remove(term)
            else:
                terms.add(term)
        restricted.append(tuple(sorted(terms)))
    return _normal(restricted)


@dataclass(frozen=True)
class Node:
    """Nonterminal residual at a specified remaining source width."""
    remaining: int
    equations: tuple
    low: int
    high: int
    count: int


@dataclass(frozen=True)
class Fibre:
    """Node IDs 0/1 denote false/true; other IDs index nodes plus two."""
    width: int
    equations: tuple
    nodes: tuple[Node, ...]
    root: int
    work: int

    def count_at(self, node):
        return node if node < 2 else self.nodes[node - 2].count

    @property
    def count(self):
        return self.count_at(self.root)

    def unrank(self, address):
        """Return the address-th satisfying state; use a previously audited DAG."""
        if type(address) is not int or not 0 <= address < self.count:
            raise ValueError("address outside the exact fibre")
        node, state = self.root, 0
        while node >= 2:
            n = self.nodes[node - 2]
            left = self.count_at(n.low)
            if address < left:
                node = n.low
            else:
                address -= left
                state |= 1 << (n.remaining - 1)
                node = n.high
        if node != 1:
            raise ValueError("invalid fibre")
        return state

    def rank(self, state):
        """Return the exact address of a member; refuse nonmembers."""
        if type(state) is not int or not 0 <= state < 1 << self.width:
            raise ValueError("invalid source state")
        node, address = self.root, 0
        while node >= 2:
            n = self.nodes[node - 2]
            if state & (1 << (n.remaining - 1)):
                address += self.count_at(n.low)
                node = n.high
            else:
                node = n.low
        if node != 1:
            raise ValueError("state does not satisfy intention")
        return address


def construct_fibre(width, equations, *, max_nodes=MAX_NODES, max_work=MAX_WORK):
    """Compile exactly; share identical residuals instead of visiting all states."""
    equations = _input(width, equations)
    if (type(max_nodes) is not int or not 0 <= max_nodes <= MAX_NODES
            or type(max_work) is not int or not 0 <= max_work <= MAX_WORK):
        raise ValueError("invalid residual budget")
    nodes, memo = [], {}
    work = 0

    def build(k, eqs):
        nonlocal work
        if eqs == FALSE:
            return 0
        if not k:
            if eqs:
                raise AssertionError("nonconstant residual after final bit")
            return 1
        key = (k, eqs)
        if key in memo:
            return memo[key]
        # Count equations and monomial visits in both restrictions, plus
        # one unit for the residual itself. This is not a CPU instruction count.
        work += 1 + 2 * sum(1 + len(p) for p in eqs)
        if work > max_work:
            raise FibreBudget("residual work budget exceeded")
        bit = 1 << (k - 1)
        low = build(k - 1, _restrict(eqs, bit, 0))
        high = build(k - 1, _restrict(eqs, bit, 1))
        if len(nodes) >= max_nodes:
            raise FibreBudget("residual node budget exceeded")
        count = sum(c if c < 2 else nodes[c - 2].count for c in (low, high))
        index = len(nodes) + 2
        nodes.append(Node(k, eqs, low, high, count))
        memo[key] = index
        return index

    root = build(width, equations)
    return Fibre(width, equations, tuple(nodes), root, work)


@dataclass(frozen=True)
class Goal:
    """Masked observer equality after repetitions of a declared action word."""
    word: tuple[int, ...]
    repetitions: int
    mask: int
    value: int


def _goals(family, goals, *, independent=False):
    if type(goals) is not tuple or not 1 <= len(goals) <= 64:
        raise ValueError("require 1..64 temporal intentions")
    equations = []
    size = len(family.description.observers)
    for g in goals:
        if (type(g) is not Goal or type(g.mask) is not int
                or not 0 < g.mask < 1 << size or type(g.value) is not int
                or g.value < 0 or g.value & ~g.mask):
            raise ValueError("invalid masked intention")
        polys = _terminal(family, _prefix(family, g.word, g.repetitions),
                          independent=independent)
        for i, p in enumerate(polys):
            if g.mask >> i & 1:
                # O_i(x) = value_i iff O_i(x) XOR value_i = 0.
                equations.append(tuple(sorted(set(p) ^ ({0} if g.value >> i & 1 else set()))))
    return tuple(equations)


@dataclass(frozen=True)
class TemporalFibre:
    """Source and intention identity are distinct from a residual graph."""
    source_digest: str
    goals: tuple[Goal, ...]
    fibre: Fibre


def construct_temporal_fibre(family, goals, **budgets):
    if not verify_polynomial_laws(family):
        raise ValueError("unqualified nonlinear source")
    equations = _goals(family, goals)
    return TemporalFibre(family.digest, goals,
                         construct_fibre(family.description.width, equations, **budgets))
