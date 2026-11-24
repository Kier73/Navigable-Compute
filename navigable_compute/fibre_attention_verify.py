"""NCR-012 independent finite-catalogue and conditional-cost audit."""
from .intention_fibre import Fibre, MAX_NODES, MAX_WORK
from .intention_fibre_verify import verify_fibre
from .fibre_refinement import Refinement
from .fibre_refinement_verify import verify_refinement
from .fibre_attention import Offer, Prices, Outcome, Candidate, Plan


def _poly(p, width):
    return (type(p) is tuple and len(p) <= 2048
            and all(type(m) is int and 0 <= m < 2**width for m in p)
            and tuple(sorted(set(p))) == p)


def _equation(p, bit):
    if not bit:
        return p
    return p[1:] if p and p[0] == 0 else (0,) + p


def _walk_mass(overlay):
    # Independent forward propagation of number of source prefixes reaching
    # each live overlay node. Charge node.count once for each prefix.
    prefixes = {}
    if overlay.root > 0:
        prefixes[overlay.root] = 1
    result = 0
    for ref in range(len(overlay.nodes), 0, -1):
        multiplicity = prefixes.get(ref, 0)
        if not multiplicity:
            continue
        n = overlay.nodes[ref-1]
        result += multiplicity * n.count
        for child in (n.low, n.high):
            if child > 0:
                prefixes[child] = prefixes.get(child, 0) + multiplicity
    return result


def verify_attention(parent, offers, target, prices, plan, *, max_packet=1,
                     max_nodes=MAX_NODES, max_work=MAX_WORK):
    """Audit every source-bound branch, exact rational cost and selected packet."""
    try:
        if (type(parent) is not Fibre or not verify_fibre(parent.width, parent.equations, parent)
                or not parent.count or type(offers) is not tuple or len(offers) > 6
                or not _poly(target, parent.width) or type(prices) is not Prices
                or any(type(v) is not int or not 0 <= v < 2**64 for v in
                       (prices.fallback, prices.node, prices.address_visit, prices.address_queries))
                or type(max_packet) is not int or not 0 <= max_packet <= 2
                or type(max_nodes) is not int or not 0 <= max_nodes <= MAX_NODES
                or type(max_work) is not int or not 0 <= max_work <= MAX_WORK):
            return False
        by_name = {}
        for o in offers:
            if (type(o) is not Offer or type(o.name) is not str or not 1 <= len(o.name) <= 64
                    or o.name in by_name or not _poly(o.polynomial, parent.width)
                    or type(o.cost) is not int or not 0 < o.cost < 2**64):
                return False
            by_name[o.name] = o
        names = sorted(by_name)
        if (type(plan) is not Plan or plan.parent != parent or plan.target != target
                or plan.prices != prices or plan.offers != tuple(by_name[n] for n in names)
                or any(type(v) is not int for v in (plan.max_packet, plan.max_nodes, plan.max_work))
                or (plan.max_packet, plan.max_nodes, plan.max_work) != (max_packet,max_nodes,max_work)
                or type(plan.candidates) is not tuple):
            return False
        expected = {tuple(n for i,n in enumerate(names) if mask >> i & 1)
                    for mask in range(1 << len(names)) if mask.bit_count() <= max_packet}
        ranks = []
        seen = set()
        for c in plan.candidates:
            if (type(c) is not Candidate or type(c.names) is not tuple
                    or c.names not in expected or c.names in seen
                    or type(c.outcomes) is not tuple or len(c.outcomes) != 2**len(c.names)):
                return False
            seen.add(c.names)
            total = pairs = task_pairs = fallback = nodes = visits = 0
            coordinates = set()
            for o in c.outcomes:
                if (type(o) is not Outcome or type(o.bits) is not tuple
                        or len(o.bits) != len(c.names) or o.bits in coordinates
                        or any(type(b) is not int or b not in (0,1) for b in o.bits)
                        or type(o.visits) is not int):
                    return False
                coordinates.add(o.bits)
                rows = tuple(_equation(by_name[n].polynomial, b) for n,b in zip(c.names,o.bits))
                if (not verify_refinement(parent, rows, o.overlay)
                        or not verify_refinement(parent, rows + (_equation(target,1),), o.target_one)
                        or any(len(f.nodes) > max_nodes or f.work > max_work
                               for f in (o.overlay, o.target_one))):
                    return False
                n, y = o.overlay.count, o.target_one.count
                v = _walk_mass(o.overlay)
                if not 0 <= y <= n or o.visits != v:
                    return False
                total += n
                pairs += n*(n-1)//2
                task_pairs += y*(n-y)
                fallback += n if y and y != n else 0
                nodes += n*len(o.overlay.nodes)
                visits += v
            if total != parent.count:
                return False
            acquisition = sum(by_name[n].cost for n in c.names)
            score = parent.count*acquisition + fallback*prices.fallback + nodes*prices.node
            score += visits*prices.address_queries*prices.address_visit
            actual = (c.acquisition,c.unresolved_pairs,c.target_pairs,c.fallback_mass,
                      c.node_mass,c.visit_mass,c.score_numerator)
            if any(type(v) is not int for v in actual) or actual != (
                    acquisition,pairs,task_pairs,fallback,nodes,visits,score):
                return False
            ranks.append((score,acquisition,len(c.names),c.names))
        return seen == expected and plan.selected == min(ranks)[-1]
    except (ValueError, TypeError, IndexError, AttributeError, KeyError, OverflowError):
        return False
