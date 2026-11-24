"""NCR-020: empirical paired route selection, never semantic authority."""
from dataclasses import dataclass
from fractions import Fraction
import json
from time import perf_counter_ns

from .intention_interpreter import Truth,Cube,Limits,Interpreter,validate_request,_limits,route_order
from .intention_interpreter_verify import verify_result
from .procedural_vocabulary import _canonical

SCHEMA="ncr020.route-policy.v1"
MAX_EPISODES=1024
MAX_NS=10**15
STRATEGIES=("baseline","witness-first","span-first","exact-first")


def offered(q):
    return STRATEGIES if type(q) is Cube else tuple(s for s in STRATEGIES if s!="span-first")


def features(v,q,limits=Limits()):
    """Structural features only: no source identity, answers, or probe execution.

    Include route budgets exactly, so a model trained with full allowances is
    never silently reused under a different caller's limits.
    """
    validate_request(v,q)
    _limits(limits)
    support,rule_count=rule_shape(v)
    return shape_features(v,q,limits,support,rule_count)


def rule_shape(v):
    rules=v.rules
    support=0
    for r in rules:
        for m in r:
            support|=m
    return support,len(rules)


def shape_features(v,q,limits,support,rule_count):
    """Shared feature projection; callers validate request and limits."""
    if type(q) is Cube:
        mask=q.ones|q.zeros
        kind="cube"
        size="small" if mask.bit_count()<=2 else "large"
    else:
        mask=0
        for m in q.target:
            mask|=m
        kind="truth"
        size="small" if len(q.target)<=1 else "large"
    overlap="disjoint" if not mask&support else "covered" if not mask&~support else "partial"
    # Support and source width are bucketed; unseen contexts default safely.
    rules="none" if not rule_count else "one" if rule_count==1 else "many"
    return (kind,size,overlap,rules,v.parent.width,
            limits.cofactor_work,limits.span_work,limits.rewrite_steps,
            limits.rewrite_work,limits.witness_samples,limits.residual_nodes,limits.residual_work)


@dataclass(frozen=True)
class Stat:
    context: tuple
    strategy: str
    n: int
    resolved: int
    elapsed_ns: int


@dataclass(frozen=True)
class Policy:
    actor: str
    segment: str
    stats: tuple[Stat,...]
    min_samples: int=3

    def choose(self,v,q,limits=Limits()):
        c=features(v,q,limits)
        candidates=[s for s in self.stats if s.context==c and s.n>=self.min_samples
                    and s.resolved==s.n and s.strategy in offered(q)]
        # Require paired coverage for the whole offered catalogue.
        if len(candidates)!=len(offered(q)):
            return "baseline"
        return min(candidates,key=lambda s:(Fraction(s.elapsed_ns,s.n),
                                            STRATEGIES.index(s.strategy))).strategy

    def packet(self):
        return _canonical(dict(schema=SCHEMA,actor=self.actor,segment=self.segment,
            min_samples=self.min_samples,stats=[(s.context,s.strategy,s.n,s.resolved,s.elapsed_ns)
                                               for s in self.stats]))

    @classmethod
    def restore(cls,payload,*,actor,segment):
        if type(payload) is not bytes or not 0<len(payload)<=1_000_000:
            raise ValueError("policy packet size")
        b=json.loads(payload)
        if (type(b) is not dict or set(b)!={"schema","actor","segment","min_samples","stats"}
                or b["schema"]!=SCHEMA or b["actor"]!=actor or b["segment"]!=segment
                or type(b["min_samples"]) is not int or not 1<=b["min_samples"]<=MAX_EPISODES
                or type(b["stats"]) is not list or len(b["stats"])>MAX_EPISODES*4):
            raise ValueError("policy schema or clock scope")
        rows=[]
        for row in b["stats"]:
            if type(row) is not list or len(row)!=5 or type(row[0]) is not list:
                raise ValueError("policy row")
            c,s,n,h,t=row
            if (len(c)!=12 or c[0] not in ("cube","truth") or c[1] not in ("small","large")
                    or c[2] not in ("disjoint","covered","partial") or c[3] not in ("none","one","many")
                    or type(c[4]) is not int or not 1<=c[4]<=128
                    or s not in STRATEGIES or (c[0]=="truth" and s=="span-first")
                    or any(type(x) is not int for x in (n,h,t))
                    or not 1<=n<=MAX_EPISODES or not 0<=h<=n or not 0<=t<=n*MAX_NS):
                raise ValueError("invalid policy statistics")
            _limits(Limits(*c[5:]))
            rows.append(Stat(tuple(c),s,n,h,t))
        keys=[(s.context,s.strategy) for s in rows]
        if keys!=sorted(set(keys)):
            raise ValueError("duplicate or unordered policy rows")
        groups={}
        for s in rows:
            groups.setdefault(s.context,[]).append(s)
        for c,rs in groups.items():
            expected=set(STRATEGIES if c[0]=="cube" else ("baseline","witness-first","exact-first"))
            if {s.strategy for s in rs}!=expected or len({s.n for s in rs})!=1:
                raise ValueError("unpaired policy statistics")
        if sum(rs[0].n for rs in groups.values())>MAX_EPISODES:
            raise ValueError("policy episode budget")
        p=cls(actor,segment,tuple(rows),b["min_samples"])
        if p.packet()!=payload:
            raise ValueError("noncanonical policy")
        return p


class Trainer:
    """Paired exhaustive schedule exploration on training episodes only.

    Measurement spans cover full realize(), including final semantic check.
    A timing record is admitted only when every offered schedule was measured
    for the same episode. UNKNOWN stays unresolved in the success counts.
    """
    def __init__(self,actor,segment,*,min_samples=3,recorder=None):
        if (any(type(x) is not str or not 1<=len(x)<=128 for x in (actor,segment))
                or type(min_samples) is not int or not 1<=min_samples<=MAX_EPISODES):
            raise ValueError("invalid training scope")
        self.actor,self.segment=actor,segment
        self.min_samples=min_samples
        self.recorder=recorder
        if recorder is not None and (recorder.clock_actor_id!=actor or recorder.clock_segment_id!=segment):
            raise ValueError("recorder clock scope mismatch")
        self.episodes=0
        self._stats={}

    def observe(self,interpreter,q,*,limits=Limits()):
        if self.episodes>=MAX_EPISODES:
            raise ValueError("training episode budget")
        if type(interpreter) is not Interpreter:
            raise ValueError("interpreter required")
        c=features(interpreter.v,q,limits)
        number=self.episodes
        self.episodes+=1
        choices=offered(q)
        # Rotate order to reduce systematic warm-cache bias.
        offset=number%len(choices)
        order=choices[offset:]+choices[:offset]
        pairs=[]
        outcomes=set()
        for s in order:
            call=lambda:interpreter.realize(q,limits=limits,strategy=s)
            if self.recorder is None:
                t=perf_counter_ns()
                result=call()
                duration=perf_counter_ns()-t
            else:
                from operation_spans import validate_operation_span
                before=len(self.recorder.spans)
                result=self.recorder.measure(attempt_id=f"ncr020-{number}",span_id=s,
                    operation_id="ncr020:"+":".join(map(str,c))+":"+s,
                    operation=call,context={"features":list(c),"strategy":s})
                recorded=self.recorder.spans[before:]
                duration=None
                if len(recorded)==1:
                    span=validate_operation_span(recorded[0])
                    if (span["clock_actor_id"]==self.actor and span["clock_segment_id"]==self.segment
                            and span["clock_status"]=="measured" and span["outcome"]=="succeeded"):
                        duration=span["elapsed_ns"]
            good=result.status!="UNKNOWN"
            if good:
                if not verify_result(interpreter.v,q,result):
                    raise ValueError("training claim failed independent audit")
                outcomes.add(result.status)
            pairs.append((s,duration,good))
        if len(outcomes)>1:
            raise ValueError("route semantic disagreement")
        if any(type(t) is not int or not 0<=t<=MAX_NS for _,t,_ in pairs):
            return False
        for s,t,good in pairs:
            key=(c,s)
            n,h,elapsed=self._stats.get(key,(0,0,0))
            self._stats[key]=(n+1,h+int(good),elapsed+t)
        return True

    def freeze(self):
        rows=tuple(Stat(c,s,*v) for (c,s),v in sorted(self._stats.items()))
        p=Policy(self.actor,self.segment,rows,self.min_samples)
        return Policy.restore(p.packet(),actor=self.actor,segment=self.segment)


def realize(policy,interpreter,q,*,actor,segment,limits=Limits()):
    """Stale clock scope or unseen structure falls back to the default route."""
    if type(policy) is not Policy:
        raise ValueError("typed policy required")
    choice=(policy.choose(interpreter.v,q,limits)
            if (actor,segment)==(policy.actor,policy.segment) else "baseline")
    return choice,interpreter.realize(q,limits=limits,strategy=choice)
