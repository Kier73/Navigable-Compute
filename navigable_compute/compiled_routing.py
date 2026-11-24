"""NCR-021: compile a frozen empirical policy; cache no semantic answers."""
from fractions import Fraction
from types import MappingProxyType

from .intention_interpreter import Interpreter,Limits,validate_request,_limits
from .route_policy import (Policy,STRATEGIES,rule_shape,shape_features,features)


class CompiledPolicy:
    """Index the exact same schedule choice as NCR-020 once per frozen policy."""
    def __init__(self,policy):
        if type(policy) is not Policy:
            raise ValueError("typed route policy required")
        self.policy=Policy.restore(policy.packet(),actor=policy.actor,segment=policy.segment)
        groups={}
        for stat in self.policy.stats:
            groups.setdefault(stat.context,[]).append(stat)
        selected={}
        for context,rows in groups.items():
            if all(s.n>=self.policy.min_samples and s.resolved==s.n for s in rows):
                selected[context]=min(rows,key=lambda s:(
                    Fraction(s.elapsed_ns,s.n),STRATEGIES.index(s.strategy))).strategy
        self.table=MappingProxyType(selected)

    def choose(self,v,q,limits=Limits()):
        return self.table.get(features(v,q,limits),"baseline")

    def bind(self,interpreter):
        return BoundRouter(self,interpreter)


class BoundRouter:
    """Cache the feature geometry of an already admitted rule catalogue.

    The owner must rebind after vocabulary changes. Old bindings safely
    default to baseline; the actual interpreter still checks every result.
    The trusted local Vocabulary object discipline is inherited from NCR-015.
    """
    def __init__(self,compiled,interpreter):
        if type(compiled) is not CompiledPolicy or type(interpreter) is not Interpreter:
            raise ValueError("compiled policy and interpreter required")
        self.compiled=compiled
        self.interpreter=interpreter
        v=interpreter.v
        self._v=v
        self._parent=v.parent
        self._observations=v.observations
        self._base=v._base
        self._learned=v.learned
        self._support,self._count=rule_shape(v)

    def current(self):
        v=self.interpreter.v
        return (v is self._v and v.parent is self._parent
                and v.observations is self._observations and v._base is self._base
                and v.learned is self._learned)

    def choose(self,q,*,actor,segment,limits=Limits()):
        v=self.interpreter.v
        validate_request(v,q)
        _limits(limits)
        p=self.compiled.policy
        if not self.current() or (actor,segment)!=(p.actor,p.segment):
            return "baseline"
        c=shape_features(v,q,limits,self._support,self._count)
        return self.compiled.table.get(c,"baseline")

    def realize(self,q,*,actor,segment,limits=Limits()):
        strategy=self.choose(q,actor=actor,segment=segment,limits=limits)
        return strategy,self.interpreter.realize(q,limits=limits,strategy=strategy)
