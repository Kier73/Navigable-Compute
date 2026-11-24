"""Independent NCR-013 finite decision-table checker."""
from .fibre_attention import Plan
from .fibre_attention_verify import verify_attention
from .adaptive_attention import Policy, Decision


def verify_policy(c,p):
    """Enumerate explicit stop/second choices, then compare first choices.

    No producer compilation, no residual formation, and no trust in claimed
    Bellman values. The independently checked NCR-012 catalogue is the source.
    """
    try:
        if (type(c) is not Plan or not verify_attention(
                c.parent,c.offers,c.target,c.prices,c,max_packet=c.max_packet,
                max_nodes=c.max_nodes,max_work=c.max_work)
                or type(p) is not Policy or type(p.decisions) is not tuple):
            return False
        expected = {}
        terminal_cost = {}
        for candidate in c.candidates:
            for o in candidate.outcomes:
                h = tuple(zip(candidate.names,o.bits))
                expected[h] = o.count
                terminal_cost[h] = (c.prices.node*len(o.overlay.nodes)*o.count
                    + c.prices.address_visit*c.prices.address_queries*o.visits
                    + (c.prices.fallback*o.count if 0 < o.yes < o.count else 0))
        table = {}
        for d in p.decisions:
            if (type(d) is not Decision or type(d.history) is not tuple
                    or d.history not in expected or d.history in table
                    or type(d.count) is not int or d.count != expected[d.history]
                    or type(d.numerator) is not int or d.numerator < 0
                    or (d.next_offer is not None and type(d.next_offer) is not str)):
                return False
            table[d.history] = d
        if set(table) != set(expected) or p.decisions != tuple(table[h] for h in sorted(table)):
            return False
        prices = {o.name:o.cost for o in c.offers}
        # Directly enumerate terminal alternatives at every singleton.
        best = {}
        for h,n in expected.items():
            if not h:
                continue
            options = [(terminal_cost[h],0,"")]
            if n and len(h) == 1 and c.max_packet == 2:
                for name,price in prices.items():
                    if name == h[0][0]:
                        continue
                    children = [key for key in expected
                                if len(key)==2 and h[0] in key
                                and name in dict(key)]
                    if len(children)!=2 or sum(expected[key] for key in children)!=n:
                        return False
                    options.append((n*price+sum(terminal_cost[key] for key in children),1,name))
            best[h] = min(options)
        # Enumerate every first probe with independently minimized continuation.
        options = [(terminal_cost[()],0,"")]
        if c.max_packet:
            for name,price in prices.items():
                keys = (((name,0),),((name,1),))
                if sum(expected[k] for k in keys)!=c.parent.count:
                    return False
                options.append((c.parent.count*price+sum(best[k][0] for k in keys),1,name))
        best[()] = min(options)
        for h,(score,probe,name) in best.items():
            if (table[h].numerator != score
                    or table[h].next_offer != (name if probe else None)):
                return False
        return best[()][0] <= min(x.score_numerator for x in c.candidates)
    except (ValueError,TypeError,IndexError,AttributeError,KeyError,OverflowError):
        return False
