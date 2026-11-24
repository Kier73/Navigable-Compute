"""NCR-012: task-directed attention on persistent counted Boolean Grounds.

Uniform hidden-member prior is explicit. Candidate observations are mathematical
offers, not authenticated external measurements. Every possible outcome is
qualified before selection. No source-state enumeration is required.
"""
from dataclasses import dataclass
from itertools import combinations

from .intention_fibre import Fibre, MAX_NODES, MAX_WORK, _input
from .intention_fibre_verify import verify_fibre
from .fibre_refinement import Refinement, refine
from .fibre_refinement_verify import verify_refinement


@dataclass(frozen=True)
class Offer:
    name: str
    polynomial: tuple[int, ...]
    cost: int


@dataclass(frozen=True)
class Prices:
    fallback: int
    node: int = 0
    address_visit: int = 0
    address_queries: int = 0


@dataclass(frozen=True)
class Outcome:
    bits: tuple[int, ...]
    overlay: Refinement
    target_one: Refinement
    visits: int

    @property
    def count(self):
        return self.overlay.count

    @property
    def yes(self):
        return self.target_one.count

    @property
    def inferred(self):
        if not self.count:
            return None
        if not self.yes:
            return 0
        if self.yes == self.count:
            return 1
        return None


@dataclass(frozen=True)
class Candidate:
    names: tuple[str, ...]
    outcomes: tuple[Outcome, ...]
    acquisition: int
    unresolved_pairs: int
    target_pairs: int
    fallback_mass: int
    node_mass: int
    visit_mass: int
    score_numerator: int


@dataclass(frozen=True)
class Plan:
    parent: Fibre
    offers: tuple[Offer, ...]
    target: tuple[int, ...]
    prices: Prices
    max_packet: int
    max_nodes: int
    max_work: int
    candidates: tuple[Candidate, ...]
    selected: tuple[str, ...]

    @property
    def denominator(self):
        return self.parent.count

    @property
    def prepared_work(self):
        """Sum of charged overlay work across all prepared outcome proofs."""
        return sum(o.overlay.work + o.target_one.work
                   for c in self.candidates for o in c.outcomes)

    @property
    def prepared_nodes(self):
        return sum(len(o.overlay.nodes) + len(o.target_one.nodes)
                   for c in self.candidates for o in c.outcomes)


def _validate(parent, offers, target, prices, max_packet, max_nodes, max_work):
    if type(parent) is not Fibre or not verify_fibre(parent.width, parent.equations, parent):
        raise ValueError("unqualified parent")
    if not parent.count:
        raise ValueError("empty hidden-member domain")
    if type(offers) is not tuple or len(offers) > 6:
        raise ValueError("require at most six offers")
    for o in offers:
        if (type(o) is not Offer or type(o.name) is not str or not o.name
                or len(o.name) > 64 or type(o.cost) is not int or o.cost <= 0
                or o.cost.bit_length() > 64):
            raise ValueError("invalid offer")
        _input(parent.width, (o.polynomial,))
    if len({o.name for o in offers}) != len(offers):
        raise ValueError("duplicate offer identity")
    _input(parent.width, (target,))
    if (type(prices) is not Prices or any(type(v) is not int or v < 0 or v.bit_length() > 64
            for v in (prices.fallback, prices.node, prices.address_visit, prices.address_queries))
            or type(max_packet) is not int or not 0 <= max_packet <= 2
            or type(max_nodes) is not int or not 0 <= max_nodes <= MAX_NODES
            or type(max_work) is not int or not 0 <= max_work <= MAX_WORK):
        raise ValueError("invalid attention price or budget")
    return tuple(sorted(offers, key=lambda o: o.name))


def _equal(poly, bit):
    return tuple(sorted(set(poly) ^ ({0} if bit else set())))


def _visit_mass(overlay):
    """Sum of positive overlay-node visits over every refined address."""
    # Negative refs are unchanged parent subtrees: address embedding stops.
    # Recurrence counts a shared suffix once per distinct incoming prefix.
    visits = [0]
    for n in overlay.nodes:
        visits.append(n.count + (visits[n.low] if n.low > 0 else 0)
                      + (visits[n.high] if n.high > 0 else 0))
    return visits[overlay.root] if overlay.root > 0 else 0


def plan_attention(parent, offers, target, prices, *, max_packet=1,
                   max_nodes=MAX_NODES, max_work=MAX_WORK):
    """Exact choice over offered simultaneous packets of at most two bits.

    Each outcome has a persistent refinement and an independently verified
    target-positive intersection. Failure to qualify ANY candidate refuses the
    entire optimality claim. Budgets apply to each refinement, not the portfolio.
    """
    offers = _validate(parent, offers, target, prices, max_packet, max_nodes, max_work)
    budgets = dict(max_nodes=max_nodes, max_work=max_work)
    candidates = []
    for size in range(min(max_packet, len(offers)) + 1):
        for packet in combinations(offers, size):
            outcomes = []
            for coordinate in range(1 << size):
                bits = tuple((coordinate >> i) & 1 for i in range(size))
                rows = tuple(_equal(o.polynomial, b) for o, b in zip(packet, bits))
                target_rows = rows + (_equal(target, 1),)
                overlay = refine(parent, rows, **budgets)
                yes = refine(parent, target_rows, **budgets)
                if (not verify_refinement(parent, rows, overlay)
                        or not verify_refinement(parent, target_rows, yes)
                        or yes.count > overlay.count):
                    raise ValueError("unqualified attention outcome")
                outcomes.append(Outcome(bits, overlay, yes, _visit_mass(overlay)))
            outcomes = tuple(outcomes)
            if sum(o.count for o in outcomes) != parent.count:
                raise ValueError("observation partition disagrees")
            acquisition = sum(o.cost for o in packet)
            pairs = sum(o.count * (o.count - 1) // 2 for o in outcomes)
            task_pairs = sum(o.yes * (o.count - o.yes) for o in outcomes)
            fallback = sum(o.count for o in outcomes if 0 < o.yes < o.count)
            nodes = sum(o.count * len(o.overlay.nodes) for o in outcomes)
            visits = sum(o.visits for o in outcomes)
            score = (parent.count * acquisition + prices.fallback * fallback
                     + prices.node * nodes + prices.address_visit * prices.address_queries * visits)
            candidates.append(Candidate(tuple(o.name for o in packet), outcomes,
                                        acquisition, pairs, task_pairs, fallback, nodes, visits, score))
    selected = min(candidates, key=lambda c: (c.score_numerator, c.acquisition,
                                               len(c.names), c.names))
    return Plan(parent, offers, target, prices, max_packet, max_nodes, max_work,
                tuple(candidates), selected.names)


def select_outcome(plan, observations):
    """Use only after independent plan admission; values are caller-declared.

    The observation tuple must contain the selected names in packet order.
    A zero-mass outcome is a contradiction, not permission to change the law.
    No target bit is guessed on a mixed branch.
    """
    if type(plan) is not Plan or type(observations) is not tuple:
        raise ValueError("invalid attention outcome")
    if (len(observations) != len(plan.selected)
            or any(type(pair) is not tuple or len(pair) != 2
                   or type(pair[0]) is not str or type(pair[1]) is not int
                   or pair[1] not in (0, 1) for pair in observations)
            or tuple(pair[0] for pair in observations) != plan.selected):
        raise ValueError("observation identity or value mismatch")
    bits = tuple(pair[1] for pair in observations)
    candidate = next(c for c in plan.candidates if c.names == plan.selected)
    outcome = next(o for o in candidate.outcomes if o.bits == bits)
    if not outcome.count:
        raise ValueError("contradiction with admitted Ground")
    return outcome
