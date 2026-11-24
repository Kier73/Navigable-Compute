"""NCR-013: compile adaptive observation policies from qualified NCR-012 proofs."""
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType

from .fibre_attention import Plan
from .fibre_attention_verify import verify_attention


@dataclass(frozen=True)
class Decision:
    history: tuple
    count: int
    next_offer: str | None
    numerator: int


@dataclass(frozen=True)
class Policy:
    decisions: tuple[Decision, ...]

    @property
    def numerator(self):
        return next(d.numerator for d in self.decisions if not d.history)


def _index(catalogue):
    return {tuple(zip(c.names, o.bits)): o
            for c in catalogue.candidates for o in c.outcomes}


def _audit(c):
    return (type(c) is Plan and verify_attention(
        c.parent, c.offers, c.target, c.prices, c,
        max_packet=c.max_packet, max_nodes=c.max_nodes, max_work=c.max_work))


def _stop(outcome, prices):
    return (outcome.count * (prices.fallback * (0 < outcome.yes < outcome.count)
                             + prices.node * len(outcome.overlay.nodes))
            + prices.address_visit * prices.address_queries * outcome.visits)


def compile_policy(catalogue):
    """Reuse every existing outcome proof; no new residual construction.

    The finite Bellman objective charges acquisition on each reached state,
    and NCR-012 fallback/geometry only at the terminal branch. A tied stop
    wins, then lexically first probe. Impossible branches have zero charge.
    """
    if not _audit(catalogue):
        raise ValueError("unqualified attention catalogue")
    outcomes = _index(catalogue)
    costs = {o.name: o.cost for o in catalogue.offers}
    decisions = {}
    for h in sorted(outcomes, key=lambda h: (-len(h), h)):
        o = outcomes[h]
        options = [(_stop(o,catalogue.prices), 0, "")]
        if o.count and len(h) < catalogue.max_packet:
            seen = {name for name, _ in h}
            for name, cost in costs.items():
                if name in seen:
                    continue
                children = [tuple(sorted(h + ((name,b),))) for b in (0,1)]
                score = o.count*cost + sum(decisions[ch].numerator for ch in children)
                options.append((score,1,name))
        score, probe, name = min(options)
        decisions[h] = Decision(h, o.count, name if probe else None, score)
    result = Policy(tuple(decisions[h] for h in sorted(decisions)))
    if result.numerator > min(c.score_numerator for c in catalogue.candidates):
        raise AssertionError("adaptive policy worse than a fixed offered packet")
    return result


@dataclass(frozen=True)
class Session:
    epoch: str
    session_id: str
    trace: tuple[tuple[str,int], ...] = ()


@dataclass(frozen=True)
class View:
    decision: Decision
    outcome: object

    @property
    def terminal(self):
        return self.decision.next_offer is None

    @property
    def inferred(self):
        return self.outcome.inferred

    @property
    def fallback_required(self):
        return self.terminal and self.inferred is None


def _digest(body):
    return sha256(json.dumps(body,sort_keys=True,separators=(",",":"),
                             ensure_ascii=True).encode("ascii")).hexdigest()


def _identity(c, p):
    # The natural-coordinate parent equations uniquely specify its domain.
    # An admitted ordering is already embodied by these coordinate equations;
    # callers must keep their source epoch/coordinate mapping stable.
    return _digest(dict(schema="ncr013.policy.v1",width=c.parent.width,
        equations=c.parent.equations,target=c.target,
        offers=[(o.name,o.polynomial,o.cost) for o in c.offers],
        prices=(c.prices.fallback,c.prices.node,c.prices.address_visit,c.prices.address_queries),
        horizon=c.max_packet,budgets=(c.max_nodes,c.max_work),
        decisions=[(d.history,d.count,d.next_offer,d.numerator) for d in p.decisions]))


class Runtime:
    """Locally admitted immutable proof catalogue, reusable across sessions.

    This is not a security boundary against arbitrary Python code in-process.
    Receiver supplies the source epoch and a unique hidden-member session ID.
    No claimed observation is authenticated by the raw mathematical advance.
    """
    def __init__(self,catalogue,policy):
        from .adaptive_attention_verify import verify_policy
        if not verify_policy(catalogue,policy):
            raise ValueError("unqualified adaptive policy")
        self._catalogue = catalogue
        self._decisions = MappingProxyType({d.history:d for d in policy.decisions})
        self._outcomes = MappingProxyType(_index(catalogue))
        self._offers = MappingProxyType({o.name:o for o in catalogue.offers})
        self._identity = _identity(catalogue,policy)

    @property
    def identity(self):
        return self._identity

    def begin(self,epoch,session_id):
        s = Session(epoch,session_id)
        self.view(s)
        return s

    def view(self,s):
        if (type(s) is not Session or type(s.epoch) is not str or not 1 <= len(s.epoch) <= 256
                or type(s.session_id) is not str or not 1 <= len(s.session_id) <= 256
                or type(s.trace) is not tuple or len(s.trace) > self._catalogue.max_packet):
            raise ValueError("invalid session")
        h = ()
        for pair in s.trace:
            if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                    or type(pair[1]) is not int or pair[1] not in (0,1)
                    or pair[0] != self._decisions[h].next_offer):
                raise ValueError("observation is out of policy order")
            h = tuple(sorted(h+(pair,)))
            if not self._outcomes[h].count:
                raise ValueError("contradiction with admitted Ground")
        return View(self._decisions[h],self._outcomes[h])

    def advance(self,s,name,bit):
        """Mathematical declaration only; for finalized evidence use VGI bridge."""
        self.view(s)
        result = Session(s.epoch,s.session_id,s.trace+((name,bit),))
        self.view(result)
        return result

    def challenge(self,s):
        v = self.view(s)
        if v.terminal:
            raise ValueError("policy is terminal")
        offer = self._offers[v.decision.next_offer]
        return _digest(dict(schema="ncr013.observation.v1",policy=self.identity,
            epoch=s.epoch,session=s.session_id,trace=s.trace,
            offer=offer.name,polynomial=offer.polynomial))
