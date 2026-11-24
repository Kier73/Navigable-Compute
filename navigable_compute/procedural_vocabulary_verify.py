"""Independent algebraic replay of qualified rewrite certificates."""
from collections import Counter

from .procedural_vocabulary import (Vocabulary,Reduction,MAX_STEPS,
                                    MAX_REDUCTION_WORK,MAX_TERMS)


def verify_reduction(vocabulary,target,certificate):
    """Check the exact identity target XOR residual = XOR of rule multiples.

    The supplied Vocabulary must already be admitted by the receiver. This
    verifies semantics, not the producer's greedy order or claimed work metric.
    No residual DAG or state-space enumeration is performed here.
    """
    try:
        if (type(vocabulary) is not Vocabulary or type(certificate) is not Reduction
                or type(certificate.target) is not tuple or certificate.target!=target
                or type(certificate.residual) is not tuple
                or len(certificate.residual)>MAX_TERMS
                or type(certificate.steps) is not tuple or len(certificate.steps)>MAX_STEPS
                or type(certificate.work) is not int or not 0<=certificate.work<=MAX_REDUCTION_WORK):
            return False
        width=vocabulary.parent.width
        for p in (target,certificate.residual):
            if (type(p) is not tuple or any(type(m) is not int or not 0<=m<2**width for m in p)
                    or tuple(sorted(set(p)))!=p):
                return False
        parity=Counter(target)
        parity.update(certificate.residual)
        rules=vocabulary.rules
        charged=0
        for pair in certificate.steps:
            if (type(pair) is not tuple or len(pair)!=2
                    or any(type(x) is not int for x in pair)
                    or not 0<=pair[0]<len(rules) or not 0<=pair[1]<2**width):
                return False
            rule=rules[pair[0]]
            charged+=len(rule)
            if charged>MAX_REDUCTION_WORK:
                return False
            parity.update(m|pair[1] for m in rule)
        return not any(n%2 for n in parity.values())
    except (ValueError,TypeError,AttributeError,IndexError,OverflowError):
        return False
