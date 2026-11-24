"""Semantic verifier for heterogeneous intention routes.

Receiver must possess an independently admitted vocabulary. Advisory attempt
timing, chosen route optimality and resource consumption are not proof claims.
"""
from collections import Counter

from .intention_interpreter import (Truth,Cube,Result,SpanProof,Witness,RewriteExact,
                                     catalogue_id,validate_request,MAX_SPAN_WORK)
from .intention_cofactor_verify import verify_cube
from .procedural_vocabulary_verify import verify_reduction
from .procedural_vocabulary import _toggle
from .fibre_refinement import Refinement,observation_rows
from .fibre_refinement_verify import verify_refinement


def _bit(p,x):
    # Independent parity formulation.
    return len([m for m in p if (m|x)==x])%2


def _witnesses(v,p,n):
    if type(p) is not Witness or type(p.addresses) is not tuple or len(p.addresses)!=n:
        return None
    xs=[]
    for pair in p.addresses:
        if (type(pair) is not tuple or len(pair)!=2 or any(type(k) is not int for k in pair)
                or not 0<=pair[0]<v.overlay.count or not 0<=pair[1]<2**v.parent.width):
            return None
        j,x=pair
        if (any(_bit(row,x) for row in v.parent.equations+v.observations)
                or v.overlay.rank(x)!=j):
            return None
        xs.append(x)
    return xs


def _span(v,q,p):
    if (type(p) is not SpanProof or type(p.indices) is not tuple or not p.indices
            or any(type(i) is not int or not 0<=i<len(v.rules) for i in p.indices)
            or tuple(sorted(set(p.indices)))!=p.indices):
        return False
    rules=v.rules
    if sum(len(rules[i]) for i in p.indices)>MAX_SPAN_WORK:
        return False
    # Sum all original equations selected by provenance; independently
    # project their term occurrences using Counter parity.
    free=((1<<v.parent.width)-1)^(q.ones|q.zeros)
    c=Counter(m&free for i in p.indices for m in rules[i]
              if (m&(q.ones|q.zeros))==(m&q.ones))
    return {m for m,n in c.items() if n%2}=={0}


def _exact(v,target,p,status):
    if (type(p) is not Refinement or not verify_refinement(v.parent,
            v.observations+(_toggle(target,1),),p) or not 0<=p.count<=v.overlay.count):
        return False
    expected="ZERO" if p.count==0 else "ONE" if p.count==v.overlay.count else "MIXED"
    return expected==status


def verify_result(v,q,r):
    """Verify exact truth/existence claims. UNKNOWN is not a certificate."""
    try:
        validate_request(v,q)
        if (type(r) is not Result or type(r.request) is not type(q) or r.request!=q
                or r.ground!=v.identity or r.catalogue!=catalogue_id(v)):
            return False
        p=r.certificate
        if type(q) is Cube:
            if r.mechanism=="cofactor":
                return r.status=="EXCLUDED" and verify_cube(v,q.ones,q.zeros,p)
            if r.mechanism=="cofactor-span":
                return r.status=="EXCLUDED" and _span(v,q,p)
            if r.mechanism=="address-witness":
                xs=_witnesses(v,p,1)
                return (r.status=="POSSIBLE" and xs is not None
                        and xs[0]&q.ones==q.ones and not xs[0]&q.zeros)
            if r.mechanism=="residual":
                rows=v.observations+observation_rows(v.parent.width,q.ones|q.zeros,q.ones)
                return (type(p) is Refinement and verify_refinement(v.parent,rows,p)
                        and r.status==("POSSIBLE" if p.count else "EXCLUDED"))
            return False
        if r.mechanism=="rewrite":
            return (verify_reduction(v,q.target,p) and p.value in (0,1)
                    and r.status==("ONE" if p.value else "ZERO"))
        if r.mechanism=="address-witness":
            xs=_witnesses(v,p,2)
            return r.status=="MIXED" and xs is not None and [_bit(q.target,x) for x in xs]==[0,1]
        if r.mechanism=="rewrite+residual":
            return (type(p) is RewriteExact and verify_reduction(v,q.target,p.rewrite)
                    and _exact(v,p.rewrite.residual,p.refinement,r.status))
        if r.mechanism=="residual":
            return _exact(v,q.target,p,r.status)
        return False
    except (ValueError,TypeError,IndexError,AttributeError,KeyError,OverflowError):
        return False
