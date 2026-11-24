"""NCR-019: typed intentions form bounded routes across existing proof mechanisms."""
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter_ns

from .intention_fibre import FibreBudget,_input,MAX_NODES,MAX_WORK
from .fibre_refinement import refine,observation_rows
from .fibre_refinement_verify import verify_refinement
from .procedural_vocabulary import (Vocabulary,Reduction,_canonical,_toggle,
                                   MAX_STEPS,MAX_REDUCTION_WORK)
from .intention_cofactor import prove_cube
from .induced_transversal_verify import verify_skill

MAX_SPAN_WORK=1_000_000
MAX_SPAN_TERMS=16384


@dataclass(frozen=True)
class Truth:
    """Is this polynomial constant on the current nonempty Ground, or mixed?"""
    target: tuple[int,...]


@dataclass(frozen=True)
class Cube:
    """Does any member satisfy these simultaneous signed coordinates?"""
    ones: int
    zeros: int


@dataclass(frozen=True)
class Limits:
    cofactor_work: int=MAX_SPAN_WORK
    span_work: int=MAX_SPAN_WORK
    rewrite_steps: int=MAX_STEPS
    rewrite_work: int=MAX_REDUCTION_WORK
    witness_samples: int=5
    residual_nodes: int=MAX_NODES
    residual_work: int=MAX_WORK


@dataclass(frozen=True)
class SpanProof:
    indices: tuple[int,...]


@dataclass(frozen=True)
class Witness:
    addresses: tuple[tuple[int,int],...]  # (refined ordinal, source state)


@dataclass(frozen=True)
class RewriteExact:
    rewrite: Reduction
    refinement: object


@dataclass(frozen=True)
class Attempt:
    mechanism: str
    outcome: str
    elapsed_ns: int


@dataclass(frozen=True)
class Result:
    ground: str
    catalogue: str
    request: object
    status: str
    mechanism: str
    certificate: object
    attempts: tuple[Attempt,...]


def catalogue_id(v):
    return sha256(_canonical(v.rules)).hexdigest()


def validate_request(v,q):
    if type(v) is not Vocabulary:
        raise ValueError("admitted vocabulary required")
    if type(q) is Truth:
        _input(v.parent.width,(q.target,))
    elif type(q) is Cube:
        if (any(type(m) is not int or not 0<=m<2**v.parent.width for m in (q.ones,q.zeros))
                or q.ones&q.zeros):
            raise ValueError("invalid signed cube")
    else:
        raise ValueError("unknown intention type")


def _eval(poly,x):
    return sum((x&m)==m for m in poly)&1


def _restrict(poly,cube):
    terms=set()
    for m in poly:
        if not m&cube.zeros:
            n=m&~cube.ones
            if n in terms:
                terms.remove(n)
            else:
                terms.add(n)
    return terms


def combine_cofactors(v,cube,*,max_work=MAX_SPAN_WORK):
    """Find constant one in GF(2) span of restricted ANFs, or return None.

    Pivot elimination acts on sparse monomial coordinates, not source states.
    Provenance is a bit mask of the original equations. No multiplication is
    performed, so a negative result is not complete joint entailment.
    """
    validate_request(v,cube)
    if (type(cube) is not Cube or type(max_work) is not int
            or not 0<=max_work<=MAX_SPAN_WORK):
        raise ValueError("invalid span request or budget")
    pivots={}
    work=0
    for i,rule in enumerate(v.rules):
        work+=len(rule)
        if work>max_work:
            raise FibreBudget("cofactor span term budget")
        row=_restrict(rule,cube)
        prov=1<<i
        while row:
            if row=={0}:
                return SpanProof(tuple(j for j in range(i+1) if prov>>j&1))
            pivot=max(row)
            if pivot not in pivots:
                pivots[pivot]=(row,prov)
                break
            other,origin=pivots[pivot]
            work+=len(row)+len(other)
            if work>max_work:
                raise FibreBudget("cofactor span elimination budget")
            row=row^other
            if len(row)>MAX_SPAN_TERMS:
                raise FibreBudget("cofactor span intermediate term budget")
            prov^=origin
    return None


def _sample(v,n):
    total=v.overlay.count
    # Ordered deterministic probe addresses, no claim of uniform sampling.
    coords=(0,total-1,total//2,total//4,3*total//4)
    return tuple((j,v.overlay.unrank(j)) for j in dict.fromkeys(coords[:n]))


def _status(n,total):
    return "ZERO" if n==0 else "ONE" if n==total else "MIXED"


def _limits(l):
    if type(l) is not Limits:
        raise ValueError("typed route limits required")
    caps=(MAX_SPAN_WORK,MAX_SPAN_WORK,MAX_STEPS,MAX_REDUCTION_WORK,5,MAX_NODES,MAX_WORK)
    if any(type(x) is not int or not 0<=x<=cap for x,cap in zip(
            (l.cofactor_work,l.span_work,l.rewrite_steps,l.rewrite_work,
             l.witness_samples,l.residual_nodes,l.residual_work),caps)):
        raise ValueError("invalid route limits")


class Interpreter:
    """Shape-selected route portfolio, with checked composition and escalation.

    Not a global cost optimum. Timing is local advisory telemetry; an
    independent checker verifies the final claim, not the scheduling trace.
    """
    def __init__(self,vocabulary,skill):
        if type(vocabulary) is not Vocabulary or not verify_skill(skill):
            raise ValueError("admitted vocabulary and skill required")
        self.v=vocabulary
        self.skill=skill

    def realize(self,q,*,limits=Limits()):
        from .intention_interpreter_verify import verify_result
        v=self.v
        validate_request(v,q)
        _limits(limits)
        attempts=[]

        def attempt(name,call):
            t=perf_counter_ns()
            try:
                answer=call()
            except FibreBudget:
                attempts.append(Attempt(name,"BUDGET",perf_counter_ns()-t))
                return None
            attempts.append(Attempt(name,"UNKNOWN" if answer is None else "PRODUCED",
                                    perf_counter_ns()-t))
            return answer

        def finish(status,name,proof):
            result=Result(v.identity,catalogue_id(v),q,status,name,proof,tuple(attempts))
            if status!="UNKNOWN" and not verify_result(v,q,result):
                raise ValueError("independent intention verification refused")
            return result

        bounds=dict(max_nodes=limits.residual_nodes,max_work=limits.residual_work)
        if type(q) is Cube:
            proof=attempt("cofactor",lambda:prove_cube(self.skill,v,q.ones,q.zeros,
                                                     max_work=limits.cofactor_work))
            if proof is not None:
                return finish("EXCLUDED","cofactor",proof)
            proof=attempt("cofactor-span",lambda:combine_cofactors(v,q,max_work=limits.span_work))
            if proof is not None:
                return finish("EXCLUDED","cofactor-span",proof)

            def sample_cube():
                for j,x in _sample(v,limits.witness_samples):
                    if x&q.ones==q.ones and not x&q.zeros:
                        return Witness(((j,x),))
                return None

            proof=attempt("address-witness",sample_cube)
            if proof is not None:
                return finish("POSSIBLE","address-witness",proof)
            rows=v.observations+observation_rows(v.parent.width,q.ones|q.zeros,q.ones)
            proof=attempt("residual",lambda:refine(v.parent,rows,**bounds))
            if proof is not None:
                return finish("POSSIBLE" if proof.count else "EXCLUDED","residual",proof)
        else:
            reduction=attempt("rewrite",lambda:v.reduce(q.target,max_steps=limits.rewrite_steps,
                                                        max_work=limits.rewrite_work))
            if reduction is not None and reduction.value is not None:
                return finish("ONE" if reduction.value else "ZERO","rewrite",reduction)

            def sample_truth():
                by_bit={}
                for j,x in _sample(v,limits.witness_samples):
                    by_bit.setdefault(_eval(q.target,x),(j,x))
                    if len(by_bit)==2:
                        return Witness((by_bit[0],by_bit[1]))
                return None

            proof=attempt("address-witness",sample_truth)
            if proof is not None:
                return finish("MIXED","address-witness",proof)
            if reduction is not None and len(reduction.residual)<len(q.target):
                proof=attempt("rewrite+residual",lambda:v.exact(reduction.residual,**bounds)[1])
                if proof is not None:
                    return finish(_status(proof.count,v.overlay.count),"rewrite+residual",
                                  RewriteExact(reduction,proof))
            else:
                proof=attempt("residual",lambda:v.exact(q.target,**bounds)[1])
                if proof is not None:
                    return finish(_status(proof.count,v.overlay.count),"residual",proof)
        return finish("UNKNOWN","none",None)
