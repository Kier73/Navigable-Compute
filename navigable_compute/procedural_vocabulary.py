"""NCR-015: learn reusable vanishing ANFs, then navigate their rewrite geometry.

A rule is a polynomial known to vanish on one admitted conditioned fibre.
Its monomial multiples vanish too. Warm inference uses exact algebraic
certificates; an irreducible nonconstant is UNKNOWN, not a counterexample.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json

from .intention_fibre import FibreBudget, _input, MAX_NODES, MAX_WORK
from .intention_fibre_verify import verify_fibre
from .fibre_refinement import refine
from .fibre_refinement_verify import verify_refinement

MAX_RULES = 64
MAX_STEPS = 4096
MAX_REDUCTION_WORK = 1_000_000
MAX_TERMS = 16384


def _canonical(body):
    return json.dumps(body,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")


def _toggle(poly,bit):
    return tuple(sorted(set(poly)^({0} if bit else set())))


@dataclass(frozen=True)
class Learned:
    target: tuple[int,...]
    bit: int
    proof: object


@dataclass(frozen=True)
class Reduction:
    target: tuple[int,...]
    residual: tuple[int,...]
    steps: tuple[tuple[int,int],...]
    work: int

    @property
    def value(self):
        return 0 if not self.residual else 1 if self.residual==(0,) else None


class Vocabulary:
    """Receiver-local admitted mathematical rules, optionally VGI-derived.

    Parent and evidence equations are immutable by trusted Python convention.
    Learned facts are independently checked before they enter the rule set.
    This is not a security sandbox. The bound context is a caller-owned label.
    """
    def __init__(self,parent,observations,overlay,*,context):
        if (type(context) is not str or not 1<=len(context)<=256
                or not verify_fibre(parent.width,parent.equations,parent)
                or not verify_refinement(parent,observations,overlay)
                or not overlay.count):
            raise ValueError("unqualified vocabulary Ground")
        self.parent=parent
        self.observations=_input(parent.width,observations)
        self.overlay=overlay
        self.context=context
        self._learned=()
        self._base=tuple(sorted(set(parent.equations+self.observations)))
        self.identity=sha256(_canonical(dict(schema="ncr015.ground.v1",
            width=parent.width,equations=parent.equations,
            observations=self.observations,context=context))).hexdigest()

    @classmethod
    def from_memory(cls,memory):
        """Attach only to an actual receiver-admitted NCR-014 memory."""
        from .observation_memory import Memory,restore
        if type(memory) is not Memory:
            raise ValueError("observation memory required")
        s=memory.session
        raw=memory.snapshot()
        m=restore(memory.runtime,s.epoch,s.session_id,memory._sources,memory._quorum,raw)
        return cls(m.runtime.parent,m.runtime.observation_equations(m.session),
                   m.view.outcome.overlay,context=sha256(raw).hexdigest())

    @property
    def learned(self):
        return self._learned

    @property
    def rules(self):
        return self._base+tuple(_toggle(x.target,x.bit) for x in self._learned)

    def exact(self,target,*,max_nodes=MAX_NODES,max_work=MAX_WORK):
        """Cold exact comparison and independent residual proof audit."""
        _input(self.parent.width,(target,))
        rows=self.observations+(_toggle(target,1),)
        proof=refine(self.parent,rows,max_nodes=max_nodes,max_work=max_work)
        if not verify_refinement(self.parent,rows,proof) or proof.count>self.overlay.count:
            raise ValueError("unqualified cold target")
        bit=0 if not proof.count else 1 if proof.count==self.overlay.count else None
        return bit,proof

    def learn(self,target,*,max_nodes=MAX_NODES,max_work=MAX_WORK):
        """Retain a new equality only after cold qualification."""
        bit,proof=self.exact(target,max_nodes=max_nodes,max_work=max_work)
        if bit is None:
            return False
        row=_toggle(target,bit)
        if not row or row in self.rules:
            return False
        if len(self._learned)>=MAX_RULES:
            raise FibreBudget("learned rule capacity")
        self._learned+=(Learned(target,bit,proof),)
        return True

    def proposals(self):
        """Bounded grammar from the variable support of acquired observations.

        Individual variables, pair XOR, AND and OR. No held-out targets or
        target answers are used. Support >8 contributes no grammar candidates.
        """
        result=set()
        for p in self.observations:
            support=0
            for term in p:
                support|=term
            variables=[1<<i for i in range(self.parent.width) if support>>i&1]
            if len(variables)>8:
                continue
            result.update((v,) for v in variables)
            for a,b in combinations(variables,2):
                result.update(((a,b),(a|b,),(a,b,a|b)))
        return tuple(sorted(result))[:64]

    def train(self):
        """Try grammar proposals; unknowns are not admitted as relations."""
        return tuple(p for p in self.proposals() if self.learn(p))

    def train_symbolic(self,*,max_trials=4096,max_work=MAX_REDUCTION_WORK):
        """Discover vanishing rules by multiplication, with no cold queries.

        For g=q XOR b and any already admitted r, g*r=g implies g=0 on
        the fibre. Each successful identity is checked by the independent
        algebraic verifier before admission. Already resolved proposals are
        not retained. No completeness claim; budgets refuse partial training
        (already admitted rules remain valid).
        """
        from .procedural_vocabulary_verify import verify_reduction
        if (type(max_trials) is not int or not 0<=max_trials<=4096
                or type(max_work) is not int or not 0<=max_work<=MAX_REDUCTION_WORK):
            raise ValueError("invalid symbolic induction budget")
        added=[]
        trials=work=0
        for q in self.proposals():
            if self.reduce(q).value is not None:
                continue
            for b in (0,1):
                g=_toggle(q,b)
                if not g:
                    continue
                proof=None
                for i,r in enumerate(self.rules):
                    trials+=1
                    work+=len(g)*len(r)
                    if trials>max_trials or work>max_work:
                        raise FibreBudget("symbolic induction budget")
                    # Replay the entire product, not a sampled truth table.
                    candidate=Reduction(g,(),tuple((i,m) for m in g),len(g)*len(r))
                    if verify_reduction(self,g,candidate):
                        proof=candidate
                        break
                if proof is None:
                    continue
                if len(self._learned)>=MAX_RULES:
                    raise FibreBudget("learned rule capacity")
                self._learned+=(Learned(q,b,proof),)
                added.append(q)
                break
        return tuple(added),trials,work

    def reduce(self,target,*,max_steps=MAX_STEPS,max_work=MAX_REDUCTION_WORK):
        _input(self.parent.width,(target,))
        if (type(max_steps) is not int or not 0<=max_steps<=MAX_STEPS
                or type(max_work) is not int or not 0<=max_work<=MAX_REDUCTION_WORK):
            raise ValueError("invalid reduction budget")
        rules=self.rules
        live=set(target)
        steps=[]
        work=0
        # Always pick the largest reducible monomial. With multiplier disjoint
        # from the pivot, every other resulting monomial is strictly smaller.
        while True:
            match=None
            for m in sorted(live,reverse=True):
                if not m:
                    continue
                for i,r in enumerate(rules):
                    work+=1
                    if work>max_work:
                        raise FibreBudget("rewrite search budget")
                    pivot=r[-1]
                    if m&pivot==pivot:
                        match=(i,m^pivot)
                        break
                if match is not None:
                    break
            if match is None:
                break
            if len(steps)>=max_steps:
                raise FibreBudget("rewrite step budget")
            i,k=match
            work+=len(rules[i])
            if work>max_work:
                raise FibreBudget("rewrite arithmetic budget")
            for term in rules[i]:
                new=term|k
                if new in live:
                    live.remove(new)
                else:
                    live.add(new)
            if len(live)>MAX_TERMS:
                raise FibreBudget("rewrite term budget")
            steps.append(match)
        result=Reduction(target,tuple(sorted(live)),tuple(steps),work)
        from .procedural_vocabulary_verify import verify_reduction
        if not verify_reduction(self,target,result):
            raise ValueError("independent rewrite audit refused")
        return result

    def packet(self):
        """Persist learned proposals, not their unchecked authority."""
        return _canonical(dict(schema="ncr015.vocabulary.v1",ground=self.identity,
             learned=[(x.target,x.bit) for x in self.learned]))

    @classmethod
    def restore(cls,parent,observations,overlay,*,context,payload):
        """Readmit every learned rule by independent cold checks."""
        if type(payload) is not bytes or not 0<len(payload)<=1_000_000:
            raise ValueError("invalid vocabulary packet")
        body=json.loads(payload)
        v=cls(parent,observations,overlay,context=context)
        if (type(body) is not dict or set(body)!={"schema","ground","learned"}
                or body["schema"]!="ncr015.vocabulary.v1" or body["ground"]!=v.identity
                or type(body["learned"]) is not list or len(body["learned"])>MAX_RULES):
            raise ValueError("vocabulary Ground or schema mismatch")
        for pair in body["learned"]:
            if (type(pair) is not list or len(pair)!=2 or type(pair[0]) is not list
                    or type(pair[1]) is not int or pair[1] not in (0,1)):
                raise ValueError("invalid learned rule")
            if not v.learn(tuple(pair[0])) or v.learned[-1].bit!=pair[1]:
                raise ValueError("learned rule not qualified")
        if v.packet()!=payload:
            raise ValueError("noncanonical vocabulary")
        return v
