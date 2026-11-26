"""NCR-023: content-addressed recursive mathematical contributions."""
from collections import Counter
from hashlib import sha256
from heapq import heappop,heappush
import json

from .intention_fibre import _input,FibreBudget
from .procedural_vocabulary import Vocabulary,Reduction,Learned,_canonical,MAX_RULES
from .procedural_vocabulary_verify import verify_reduction
from .coalescent_fleet import _base

SCHEMA="ncr023.recursive-proof.v1"
SNAPSHOT="ncr023.recursive-inventory.v1"
MAX_PACKETS=512
MAX_BYTES=16_000_000
MAX_PACKET=65536
MAX_STEPS=4096
MAX_WORK=16_000_000
MAX_DEPTH=64


def digest(raw):
    return sha256(raw).hexdigest()


def _hash(s):
    return type(s) is str and len(s)==64 and all(c in "0123456789abcdef" for c in s)


def describe(v,actor,task,relation,steps):
    """Steps are (base/dependency, immutable index/content hash, monomial)."""
    raw=_canonical(dict(schema=SCHEMA,ground=v.identity,catalogue=_base(v),
                       actor=actor,task=task,relation=relation,steps=steps))
    parse(v,raw)
    return raw


def parse(v,raw):
    if type(raw) is not bytes or not 0<len(raw)<=MAX_PACKET:
        raise ValueError("recursive packet size")
    b=json.loads(raw)
    if (type(b) is not dict or set(b)!={"schema","ground","catalogue","actor",
            "task","relation","steps"} or b["schema"]!=SCHEMA
            or b["ground"]!=v.identity or b["catalogue"]!=_base(v)
            or any(type(b[k]) is not str or not 1<=len(b[k])<=128
                   for k in ("actor","task"))
            or type(b["relation"]) is not list or type(b["steps"]) is not list
            or not 1<=len(b["steps"])<=MAX_STEPS):
        raise ValueError("recursive identity or shape")
    p=tuple(b["relation"])
    _input(v.parent.width,(p,))
    if not p or p==(0,):
        raise ValueError("nontrivial vanishing relation required")
    for s in b["steps"]:
        if (type(s) is not list or len(s)!=3 or type(s[2]) is not int
                or not 0<=s[2]<2**v.parent.width):
            raise ValueError("step shape")
        kind,ref,_=s
        if not ((kind=="base" and type(ref) is int and 0<=ref<len(v.rules))
                or (kind=="dependency" and _hash(ref))):
            raise ValueError("invalid immutable proof reference")
    if _canonical(b)!=raw:
        raise ValueError("noncanonical contribution")
    return b


def _closure(v,docs):
    """Deterministic topological audit. Missing dependencies stay pending."""
    deps={h:{s[1] for s in b["steps"] if s[0]=="dependency"} for h,b in docs.items()}
    children={}
    remaining={h:len(ds) for h,ds in deps.items()}
    ready=[]
    for h,ds in deps.items():
        if not ds:
            heappush(ready,h)
        for d in ds:
            children.setdefault(d,[]).append(h)
    state={}
    depth={}
    order=[]
    work=0
    rules=v.rules
    while ready:
        h=heappop(ready)
        b=docs[h]
        ds=deps[h]
        if any(state[d]!="VERIFIED" for d in ds):
            state[h]="INVALID"
        else:
            level=1+max((depth[d] for d in ds),default=0)
            if level>MAX_DEPTH:
                raise FibreBudget("recursive proof depth")
            # Independent parity of the entire algebraic identity.
            parity=Counter(b["relation"])
            for kind,ref,m in b["steps"]:
                r=rules[ref] if kind=="base" else docs[ref]["relation"]
                work+=len(r)
                if work>MAX_WORK:
                    raise FibreBudget("recursive audit work")
                parity.update(t|m for t in r)
            if any(n%2 for n in parity.values()):
                state[h]="INVALID"
            else:
                state[h]="VERIFIED"
                depth[h]=level
                order.append(h)
        for child in children.get(h,()):
            remaining[child]-=1
            if not remaining[child]:
                heappush(ready,child)
    for h in docs:
        state.setdefault(h,"PENDING")
    return state,depth,tuple(order),work


class Graph:
    """Bounded grow-only descriptions, separate from qualified relations."""
    def __init__(self,v):
        self.base=v
        self.catalogue=_base(v)
        self._packets={}
        self._docs={}
        self._state={}
        self._depth={}
        self._order=()
        self.work=0

    @property
    def inventory(self):
        return tuple(sorted(self._packets))

    @property
    def packets(self):
        return tuple(self._packets[h] for h in self.inventory)

    def status(self,h):
        return self._state.get(h,"MISSING")

    def relation(self,h):
        if self.status(h)!="VERIFIED":
            raise ValueError("dependency not qualified")
        return tuple(self._docs[h]["relation"])

    @property
    def depth(self):
        return max(self._depth.values(),default=0)

    def ingest(self,packets):
        if _base(self.base)!=self.catalogue:
            raise ValueError("base changed")
        if type(packets) is not tuple or len(packets)>MAX_PACKETS:
            raise ValueError("bounded packet tuple required")
        ps=dict(self._packets)
        docs=dict(self._docs)
        for raw in packets:
            if type(raw) is not bytes or not 0<len(raw)<=MAX_PACKET:
                raise ValueError("recursive packet size")
            h=digest(raw)
            if h in ps:
                if ps[h]!=raw:
                    raise ValueError("digest collision")
                continue
            docs[h]=parse(self.base,raw)
            ps[h]=raw
        if len(ps)>MAX_PACKETS or sum(map(len,ps.values()))>MAX_BYTES:
            raise FibreBudget("recursive inventory capacity")
        if len(ps)==len(self._packets):
            return 0
        state,depth,order,work=_closure(self.base,docs)
        n=len(ps)-len(self._packets)
        self._packets,self._docs,self._state,self._depth,self._order=ps,docs,state,depth,order
        self.work=work
        return n

    def ancestors(self,heads):
        """Return exact dependency closure, including the supplied heads."""
        seen=set()
        todo=list(heads)
        while todo:
            h=todo.pop()
            if h in seen:
                continue
            if self.status(h)!="VERIFIED":
                raise ValueError("unqualified closure")
            seen.add(h)
            todo.extend(s[1] for s in self._docs[h]["steps"] if s[0]=="dependency")
        return tuple(self._packets[h] for h in sorted(seen))

    def snapshot(self):
        return _canonical(dict(schema=SNAPSHOT,ground=self.base.identity,
            catalogue=self.catalogue,packets=[self._docs[h] for h in self.inventory]))

    @classmethod
    def restore(cls,v,payload):
        if type(payload) is not bytes or not 0<len(payload)<=MAX_BYTES+65536:
            raise ValueError("recursive snapshot size")
        b=json.loads(payload)
        g=cls(v)
        if (type(b) is not dict or set(b)!={"schema","ground","catalogue","packets"}
                or b["schema"]!=SNAPSHOT or b["ground"]!=v.identity
                or b["catalogue"]!=g.catalogue or type(b["packets"]) is not list):
            raise ValueError("recursive snapshot identity")
        g.ingest(tuple(_canonical(x) for x in b["packets"]))
        if g.snapshot()!=payload:
            raise ValueError("noncanonical snapshot")
        return g

    def materialize(self):
        """Recheck chronological certificates in the native NCR-015 kernel."""
        v=Vocabulary(self.base.parent,self.base.observations,self.base.overlay,
                     context=self.base.context)
        index={p:i for i,p in enumerate(v.rules)}
        refs={}
        for h in self._order:
            b=self._docs[h]
            p=tuple(b["relation"])
            steps=tuple((ref if kind=="base" else refs[ref],m)
                        for kind,ref,m in b["steps"])
            proof=Reduction(p,(),steps,sum(len(v.rules[i]) for i,_ in steps))
            if not verify_reduction(v,p,proof):
                raise ValueError("native recursive proof refused")
            if p not in index:
                if len(v.learned)>=MAX_RULES:
                    raise FibreBudget("native vocabulary capacity")
                index[p]=len(v.rules)
                v._learned+=(Learned(p,0,proof),)
            refs[h]=index[p]
        return v,refs
