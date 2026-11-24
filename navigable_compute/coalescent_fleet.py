"""NCR-022: receiver-qualified convergence of independently constructed rules.

The inventory is a bounded grow-only set of canonical proof descriptions.
Every proof references the original admitted catalogue, never an arrival-
dependent learned index. Same-Ground mathematical facts coalesce; timing
and physical observer authority are deliberately not merged.
"""
from hashlib import sha256
import json

from .intention_fibre import FibreBudget,_input
from .procedural_vocabulary import (Vocabulary,Learned,Reduction,_canonical,
                                   MAX_RULES,MAX_STEPS,MAX_REDUCTION_WORK)
from .procedural_vocabulary_verify import verify_reduction

SCHEMA="ncr022.contribution.v1"
SNAPSHOT="ncr022.inventory.v1"
MAX_PACKET=65536
MAX_PACKETS=256
MAX_BYTES=8_000_000


def _digest(data):
    return sha256(data).hexdigest()


def _base(v):
    if type(v) is not Vocabulary or v.learned:
        raise ValueError("original admitted base vocabulary required")
    return _digest(_canonical(v.rules))


def contribution(v,actor,task,proof):
    """Export a vanishing equation proof, scoped to the original base only."""
    if type(proof) is not Reduction or proof.residual:
        raise ValueError("vanishing proof required")
    raw=_canonical(dict(schema=SCHEMA,ground=v.identity,catalogue=_base(v),
        actor=actor,task=task,relation=proof.target,steps=proof.steps))
    admit(v,raw)
    return raw


def admit(v,raw):
    """Independent receiver validation, no trust in sender's claimed answer."""
    catalogue=_base(v)
    if type(raw) is not bytes or not 0<len(raw)<=MAX_PACKET:
        raise ValueError("contribution size")
    b=json.loads(raw)
    if (type(b) is not dict or set(b)!={"schema","ground","catalogue","actor",
                "task","relation","steps"} or b["schema"]!=SCHEMA
            or b["ground"]!=v.identity or b["catalogue"]!=catalogue
            or any(type(b[k]) is not str or not 1<=len(b[k])<=128
                   for k in ("actor","task"))
            or type(b["relation"]) is not list or type(b["steps"]) is not list
            or not 1<=len(b["steps"])<=MAX_STEPS):
        raise ValueError("contribution identity or schema")
    relation=tuple(b["relation"])
    _input(v.parent.width,(relation,))
    if not relation or relation==(0,):
        raise ValueError("nontrivial relation required")
    steps=[]
    work=0
    rules=v.rules
    for pair in b["steps"]:
        if (type(pair) is not list or len(pair)!=2
                or any(type(x) is not int for x in pair)
                or not 0<=pair[0]<len(rules)
                or not 0<=pair[1]<2**v.parent.width):
            raise ValueError("invalid original-rule reference")
        steps.append(tuple(pair))
        work+=len(rules[pair[0]])
        if work>MAX_REDUCTION_WORK:
            raise FibreBudget("contribution proof budget")
    p=Reduction(relation,(),tuple(steps),work)
    if _canonical(b)!=raw or not verify_reduction(v,relation,p):
        raise ValueError("contribution proof refused")
    return p


class Replica:
    """Converges for identical valid packet sets within declared capacity."""
    def __init__(self,base):
        self.base=base
        self.catalogue=_base(base)
        self._packets={}

    @property
    def inventory(self):
        return tuple(sorted(self._packets))

    @property
    def packets(self):
        return tuple(self._packets[h] for h in self.inventory)

    def ingest(self,packets):
        """Atomic set union; invalid batches leave existing inventory unchanged."""
        if _base(self.base)!=self.catalogue:
            raise ValueError("base catalogue changed")
        if type(packets) is not tuple or len(packets)>MAX_PACKETS:
            raise ValueError("bounded packet tuple required")
        candidate=dict(self._packets)
        for raw in packets:
            if type(raw) is not bytes or not 0<len(raw)<=MAX_PACKET:
                raise ValueError("contribution size")
            h=_digest(raw)
            if h in candidate:
                if candidate[h]!=raw:
                    raise ValueError("digest collision")
                continue
            admit(self.base,raw)
            candidate[h]=raw
            if len(candidate)>MAX_PACKETS or sum(map(len,candidate.values()))>MAX_BYTES:
                raise FibreBudget("inventory capacity")
        # Qualify the complete union's materialization capacity before commit.
        relations={tuple(json.loads(raw)["relation"]) for raw in candidate.values()}
        if len(relations-set(self.base.rules))>MAX_RULES:
            raise FibreBudget("collective rule capacity")
        count=len(candidate)-len(self._packets)
        self._packets=candidate
        return count

    def missing(self,known):
        if type(known) is not tuple or len(known)>MAX_PACKETS:
            raise ValueError("bounded inventory required")
        return tuple(self._packets[h] for h in self.inventory if h not in known)

    def snapshot(self):
        return _canonical(dict(schema=SNAPSHOT,ground=self.base.identity,
            catalogue=self.catalogue,packets=[json.loads(p) for p in self.packets]))

    @classmethod
    def restore(cls,base,payload):
        if type(payload) is not bytes or not 0<len(payload)<=MAX_BYTES+65536:
            raise ValueError("snapshot size")
        b=json.loads(payload)
        r=cls(base)
        if (type(b) is not dict or set(b)!={"schema","ground","catalogue","packets"}
                or b["schema"]!=SNAPSHOT or b["ground"]!=base.identity
                or b["catalogue"]!=r.catalogue or type(b["packets"]) is not list):
            raise ValueError("snapshot identity")
        r.ingest(tuple(_canonical(p) for p in b["packets"]))
        if r.snapshot()!=payload:
            raise ValueError("snapshot not canonical")
        return r

    def materialize(self):
        """Deterministic union of semantics; redundant provenance stays in inventory."""
        v=Vocabulary(self.base.parent,self.base.observations,self.base.overlay,
                     context=self.base.context)
        if _base(v)!=self.catalogue or v.identity!=self.base.identity:
            raise ValueError("original source changed")
        selected={}
        for raw in self.packets:
            p=admit(v,raw)
            if p.target not in v.rules:
                selected.setdefault(p.target,p)
        # All proofs reference only the base, so canonical order is safe.
        v._learned=tuple(Learned(p.target,0,p) for _,p in sorted(selected.items()))
        return v
