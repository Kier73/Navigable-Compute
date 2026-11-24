"""NCR-018: construct a proof for an intention directly in factored coordinates.

A signed cube specifies a simultaneous assignment of selected coordinates.
A vanishing admitted equation whose restriction is one rules out that cube.
No expansion of the cube indicator or enumeration of transversals is needed.
"""
from dataclasses import dataclass
from hashlib import sha256

from .intention_fibre import FibreBudget
from .procedural_vocabulary import Vocabulary,_canonical
from .induced_transversal_verify import verify_skill

MAX_WORK=1_000_000


@dataclass(frozen=True)
class CubeProof:
    ground: str
    rule_index: int
    rule_digest: str
    ones: int
    zeros: int
    inspected_terms: int

    def packet(self):
        return _canonical(dict(schema="ncr018.excluded-cube.v1",
            ground=self.ground,rule_index=self.rule_index,rule_digest=self.rule_digest,
            ones=str(self.ones),zeros=str(self.zeros),inspected_terms=self.inspected_terms))


def prove_cube(skill,vocabulary,ones,zeros,*,max_work=MAX_WORK):
    """Return a checked exclusion certificate, or None (UNKNOWN).

    Skill admission is separate from source-local semantic qualification.
    One-rule restriction is a sufficient condition, not a complete procedure
    for joint entailment from multiple equations. No new rules are retained.
    """
    from .intention_cofactor_verify import verify_cube
    if (not verify_skill(skill) or type(vocabulary) is not Vocabulary
            or any(type(m) is not int or not 0<=m<2**vocabulary.parent.width
                   for m in (ones,zeros)) or ones&zeros
            or type(max_work) is not int or not 0<=max_work<=MAX_WORK):
        raise ValueError("invalid cube intention or budget")
    work=0
    for i,rule in enumerate(vocabulary.rules):
        live=set()
        for m in rule:
            work+=1
            if work>max_work:
                raise FibreBudget("cube cofactor work")
            if not m&zeros:
                projected=m&~ones
                if projected in live:
                    live.remove(projected)
                else:
                    live.add(projected)
        if live=={0}:
            p=CubeProof(vocabulary.identity,i,sha256(_canonical(rule)).hexdigest(),
                        ones,zeros,len(rule))
            if not verify_cube(vocabulary,ones,zeros,p):
                raise ValueError("independent cofactor proof refused")
            return p
    return None
