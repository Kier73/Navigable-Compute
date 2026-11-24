"""Independent motif recognizer: structural conditions plus exact product parity."""
from collections import Counter
from .induced_transversal import Example, Skill, MAX_ARITY


def _poly(p):
    return (type(p) is tuple and 2<=len(p)<=2048
            and all(type(m) is int and 0<=m<2**128 for m in p)
            and tuple(sorted(set(p)))==p and p[0]==0)


def verify_example(e):
    try:
        if type(e) is not Example or not _poly(e.source) or not _poly(e.factor):
            return False
        support=e.factor[-1]
        n=support.bit_count()
        # Exactly the entire Boolean downset of support; not a guessed factor.
        if (not 1<=n<=MAX_ARITY or len(e.factor)!=2**n
                or any(m&support!=m for m in e.factor)
                or any(not m&support for m in e.source[1:])):
            return False
        product=Counter(a|b for a in e.factor for b in e.source)
        return tuple(sorted(m for m,c in product.items() if c%2))==e.factor
    except (TypeError,ValueError,AttributeError,IndexError,OverflowError):
        return False


def verify_skill(s):
    try:
        return (type(s) is Skill and type(s.examples) is tuple
                and 2<=len(s.examples)<=16
                and all(verify_example(e) for e in s.examples)
                and tuple(e.source for e in s.examples)==tuple(sorted({e.source for e in s.examples})))
    except (TypeError,ValueError,AttributeError):
        return False
