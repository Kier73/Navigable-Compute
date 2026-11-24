"""NCR-017: induce a portable proof-construction operation from NCR-016 traces.

Training recognizes a supplied algebraic motif, not arbitrary program synthesis.
Activation derives fresh, independently checked rules from the receiving Ground.
"""
from dataclasses import dataclass
import json

from .intention_fibre import FibreBudget
from .procedural_vocabulary import (Vocabulary, Learned, Reduction, MAX_RULES,
                                    MAX_REDUCTION_WORK, _canonical, _toggle)
from .procedural_vocabulary_verify import verify_reduction

SCHEMA = "ncr017.transversal-skill.v1"
MAX_ARITY = 11
MAX_FRONTIER = 4096
MAX_WORK = 1_000_000


@dataclass(frozen=True)
class Example:
    source: tuple[int, ...]
    factor: tuple[int, ...]


@dataclass(frozen=True)
class Skill:
    examples: tuple[Example, ...]

    def packet(self):
        from .induced_transversal_verify import verify_skill
        if not verify_skill(self):
            raise ValueError("unqualified skill")
        return _canonical(dict(schema=SCHEMA,
            examples=[(e.source,e.factor) for e in self.examples]))

    @classmethod
    def restore(cls,payload):
        from .induced_transversal_verify import verify_skill
        if type(payload) is not bytes or not 0<len(payload)<=1_000_000:
            raise ValueError("invalid skill packet")
        body=json.loads(payload)
        if (type(body) is not dict or set(body)!={"schema","examples"}
                or body["schema"]!=SCHEMA or type(body["examples"]) is not list
                or not 2<=len(body["examples"])<=16):
            raise ValueError("invalid skill schema")
        examples=[]
        for pair in body["examples"]:
            if (type(pair) is not list or len(pair)!=2
                    or any(type(p) is not list for p in pair)):
                raise ValueError("invalid motif example")
            examples.append(Example(tuple(pair[0]),tuple(pair[1])))
        s=cls(tuple(examples))
        if not verify_skill(s) or s.packet()!=payload:
            raise ValueError("unqualified skill packet")
        return s


def induce(vocabularies):
    """Recognize the motif in actual earlier-rule NCR-016 product certificates.

    At least two distinct source polynomials must support the motif. This is
    a diversity check, not statistical independence. No learned bit transfers.
    """
    from .induced_transversal_verify import verify_example,verify_skill
    if type(vocabularies) is not tuple or not 1<=len(vocabularies)<=16:
        raise ValueError("invalid training set")
    found={}
    for v in vocabularies:
        if type(v) is not Vocabulary:
            raise ValueError("admitted Vocabulary required")
        base_count=len(v.rules)-len(v.learned)
        for j,learned in enumerate(v.learned):
            proof=learned.proof
            if type(proof) is not Reduction or proof.residual or not proof.steps:
                continue
            if not verify_reduction(v,proof.target,proof):
                continue
            if proof.target!=_toggle(learned.target,learned.bit):
                continue
            i=proof.steps[0][0]
            if (i>=base_count+j or tuple(proof.steps)!=tuple((i,m) for m in proof.target)):
                continue
            example=Example(v.rules[i],proof.target)
            if verify_example(example):
                found.setdefault(example.source,example)
    result=Skill(tuple(found[k] for k in sorted(found)[:16]))
    if not verify_skill(result):
        raise ValueError("insufficient qualified motif diversity")
    return result


def transversals(source,*,max_arity=MAX_ARITY,max_work=MAX_WORK):
    """All inclusion-minimal hitting sets of the source nonconstant terms
    within the declared arity. Intermediate antichain cap refuses, not truncates.
    """
    if (type(source) is not tuple or not 2<=len(source)<=2048
            or any(type(m) is not int or not 0<=m<2**128 for m in source)
            or tuple(sorted(set(source)))!=source or source[0]!=0
            or type(max_arity) is not int or not 1<=max_arity<=MAX_ARITY
            or type(max_work) is not int or not 0<=max_work<=MAX_WORK):
        raise ValueError("invalid constant-one source or transversal budget")
    current=(0,)
    work=0

    def charge():
        nonlocal work
        work+=1
        if work>max_work:
            raise FibreBudget("transversal search budget")

    # Process small edges first. Superset edges may be redundant, but the
    # same intersection update handles them without a separate simplifier.
    for edge in sorted(source[1:],key=lambda m:(m.bit_count(),m)):
        candidates=set()
        for s in current:
            charge()
            if s&edge:
                candidates.add(s)
            elif s.bit_count()<max_arity:
                bits=edge
                while bits:
                    charge()
                    b=bits&-bits
                    candidates.add(s|b)
                    bits^=b
                    if len(candidates)>MAX_FRONTIER:
                        raise FibreBudget("transversal candidate frontier")
        reduced=[]
        for s in sorted(candidates,key=lambda m:(m.bit_count(),m)):
            dominated=False
            for a in reduced:
                charge()
                if a&s==a:
                    dominated=True
                    break
            if not dominated:
                reduced.append(s)
        current=tuple(reduced)
        if not current:
            break
    return current,work


def _factor(s):
    """ANF of the indicator that all coordinates of s are zero."""
    terms=[]
    m=s
    while True:
        terms.append(m)
        if not m:
            break
        m=(m-1)&s
    return tuple(reversed(terms))


@dataclass(frozen=True)
class Activation:
    source: tuple[int, ...]
    masks: tuple[int, ...]
    admitted: tuple[tuple[int, ...], ...]
    search_work: int
    product_terms: int


def activate(skill,vocabulary,rule_index,*,max_arity=MAX_ARITY,
             max_work=MAX_WORK,max_product_terms=MAX_REDUCTION_WORK):
    """Compile new source-local rules; atomic on budget/proof refusal.

    The motif grants no source authority. Each product certificate uses an
    actual admitted source equation and the existing NCR-015 verifier.
    """
    from .induced_transversal_verify import verify_skill
    if (not verify_skill(skill) or type(vocabulary) is not Vocabulary
            or type(rule_index) is not int or not 0<=rule_index<len(vocabulary.rules)
            or type(max_product_terms) is not int
            or not 0<=max_product_terms<=MAX_REDUCTION_WORK):
        raise ValueError("invalid skill activation")
    source=vocabulary.rules[rule_index]
    masks,work=transversals(source,max_arity=max_arity,max_work=max_work)
    rules=set(vocabulary.rules)
    new=[]
    terms=0
    for s in masks:
        g=_factor(s)
        if g in rules:
            continue
        if len(vocabulary.learned)+len(new)>=MAX_RULES:
            raise FibreBudget("learned rule capacity")
        charge=len(g)*len(source)
        terms+=charge
        if terms>max_product_terms:
            raise FibreBudget("transversal product budget")
        proof=Reduction(g,(),tuple((rule_index,m) for m in g),charge)
        if not verify_reduction(vocabulary,g,proof):
            raise ValueError("fresh product proof refused")
        new.append(Learned(g[1:],1,proof))
        rules.add(g)
    vocabulary._learned+=tuple(new)
    return Activation(source,masks,tuple(x.target for x in new),work,terms)
