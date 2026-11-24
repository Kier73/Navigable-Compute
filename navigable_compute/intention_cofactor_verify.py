"""Independent exact parity check of a source equation on a signed cube."""
from collections import Counter
from hashlib import sha256

from .intention_cofactor import CubeProof
from .procedural_vocabulary import Vocabulary,_canonical


def verify_cube(v,ones,zeros,p):
    """Check receiver-bound intention, source identity and constant-one cofactor."""
    try:
        if (type(v) is not Vocabulary or type(p) is not CubeProof
                or any(type(m) is not int or not 0<=m<2**v.parent.width
                       for m in (ones,zeros,p.ones,p.zeros))
                or ones&zeros or (p.ones,p.zeros)!=(ones,zeros)
                or p.ground!=v.identity
                or type(p.rule_index) is not int or not 0<=p.rule_index<len(v.rules)
                or type(p.inspected_terms) is not int):
            return False
        r=v.rules[p.rule_index]
        if p.rule_digest!=sha256(_canonical(r)).hexdigest() or p.inspected_terms!=len(r):
            return False
        # Substitute simultaneously using explicit free-coordinate projection.
        free=((1<<v.parent.width)-1)^(ones|zeros)
        counts=Counter(m&free for m in r if (m&(ones|zeros))==(m&ones))
        return {m for m,n in counts.items() if n%2}=={0}
    except (ValueError,TypeError,AttributeError,IndexError,OverflowError):
        return False
