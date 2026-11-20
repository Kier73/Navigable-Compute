"""NCQ-002: minimal equivalence under a finite, exact temporal intention.

Task classes need not be transition-stable and MUST NOT be treated as a
standalone machine valid for arbitrary continuation. Their scope is precisely
all (coordinate, address, mask) queries included in the signed-by-nobody
mathematical receipt. Different target coordinates may require new classes.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256

from .machine import Program, execute
from .synthesis import Constraint, Unrealizable
from .observability import ObservationAt
from .quotient import (QuotientBudgetExceeded, byte_components, component_program,
                       canonical_bytes, power_successor, _transition_digest)


@dataclass(frozen=True)
class TaskComponentCertificate:
    indices: tuple[int,...]
    class_of: tuple[int,...]
    labels: tuple[tuple[int,...],...]
    transition_sha256: str

    @property
    def classes(self):
        return len(self.labels)

    def encode(self):
        return {'indices':list(self.indices), 'class_of':list(self.class_of),
                'labels':[list(x) for x in self.labels],
                'transition_sha256':self.transition_sha256}


@dataclass(frozen=True)
class TaskTraceReceipt:
    initial: bytes
    observed: tuple[int,...]
    candidate_count: int
    global_bits: int
    examined_states: int
    products: int
    factors: tuple[tuple[int,...],...]
    certificates: tuple[TaskComponentCertificate,...]
    program_digest: str
    witness_digest: str

    def packet(self, queries:tuple[ObservationAt,...],fixed:tuple[Constraint,...]):
        return {'schema':'navigable-compute.task-quotient.v0.1',
                'scope':'only-the-declared-temporal-observations',
                'program_sha256':self.program_digest,
                'queries':[[str(q.cycle),q.address,q.mask,q.value] for q in queries],
                'fixed':[[c.index,c.mask,c.value] for c in fixed],
                'initial_hex':self.initial.hex(),
                'observed_masked':list(self.observed),
                'candidate_count_decimal':str(self.candidate_count),
                'global_bits':self.global_bits,
                'examined_states':self.examined_states,
                'power_products':self.products,
                'factors':[list(g) for g in self.factors],
                'certificates':[c.encode() for c in self.certificates]}


def _build_task_factor(program:Program, group:tuple[int,...],
                       queries:tuple[ObservationAt,...], *,max_states:int)->tuple[TaskComponentCertificate,int]:
    n=1<<(8*len(group))
    if n>max_states:
        raise QuotientBudgetExceeded('task quotient factor exceeds exhaustive budget')
    local=component_program(program,group)
    transitions=tuple(int.from_bytes(execute(local,s.to_bytes(local.size,'little'),1),'little')
                      for s in range(n))
    powers={}
    products=0
    for q in queries:
        if q.cycle not in powers:
            powers[q.cycle],p=power_successor(transitions,q.cycle)
            products+=p
    positions={a:i for i,a in enumerate(group)}
    signatures=tuple(tuple(((powers[q.cycle][s]>>(8*positions[q.address]))&q.mask)
                            for q in queries) for s in range(n))
    unique={}
    class_of=[]
    labels=[]
    for signature in signatures:
        if signature not in unique:
            unique[signature]=len(unique)
            labels.append(signature)
        class_of.append(unique[signature])
    return TaskComponentCertificate(group,tuple(class_of),tuple(labels),
                                    _transition_digest(transitions,local.size)),products


def synthesize_task_trace(program:Program,queries:tuple[ObservationAt,...], *,
                          fixed:tuple[Constraint,...]=(),max_states:int=4096,
                          max_coordinate_bits:int=4096)->TaskTraceReceipt:
    """Construct minimum-output-distinguishable task classes, then one input.

    For each component, exact class labels are the vector of observations at
    the *specified* coordinates, not all future outputs. This avoids any
    unqualified claim that the quotient supports a new next-state function.
    """
    if not queries:
        raise ValueError('nonempty queries required')
    if isinstance(max_states,bool) or not isinstance(max_states,int) or max_states<1:
        raise ValueError('invalid state budget')
    if isinstance(max_coordinate_bits,bool) or not isinstance(max_coordinate_bits,int) or max_coordinate_bits<1:
        raise ValueError('invalid coordinate budget')
    if any(q.address>=program.size for q in queries) or any(c.index>=program.size for c in fixed):
        raise ValueError('observation address outside machine')
    if any(q.cycle.bit_length()>max_coordinate_bits for q in queries):
        raise QuotientBudgetExceeded('coordinate encoding exceeds budget')
    factors=byte_components(program)
    chosen=bytearray(program.size)
    all_cert=[]
    outputs=[0]*len(queries)
    solutions=1
    examined=0
    products=0
    for group in factors:
        relevant=[(i,q) for i,q in enumerate(queries) if q.address in group]
        own_fixed=[c for c in fixed if c.index in group]
        if not relevant:
            masks={}
            values={}
            for c in own_fixed:
                prev=masks.get(c.index,0)
                if ((values.get(c.index,0)^c.value)&prev&c.mask)!=0:
                    raise Unrealizable('fixed input contradictory')
                values[c.index]=values.get(c.index,0)|c.value
                masks[c.index]=prev|c.mask
            solutions*=1<<(8*len(group)-sum(mask.bit_count() for mask in masks.values()))
            for a in group:
                chosen[a]=values.get(a,0)
            continue
        own=tuple(q for _,q in relevant)
        cert,p=_build_task_factor(program,group,own,max_states=max_states)
        all_cert.append(cert)
        examined+=len(cert.class_of)
        products+=p
        positions={a:i for i,a in enumerate(group)}
        matching_label=tuple(q.value for q in own)
        candidates=0
        best=None
        for state,cls in enumerate(cert.class_of):
            if cert.labels[cls]!=matching_label:
                continue
            if any(((state>>(8*positions[c.index]))&c.mask)!=c.value for c in own_fixed):
                continue
            candidates+=1
            if best is None:
                best=state
        if best is None:
            raise Unrealizable('no initial state realizes task observation')
        solutions*=candidates
        for i,a in enumerate(group):
            chosen[a]=(best>>(8*i))&255
        for i,q in relevant:
            outputs[i]=q.value
    temp=TaskTraceReceipt(bytes(chosen),tuple(outputs),solutions,program.size*8,
                          examined,products,factors,tuple(all_cert),program.digest,'')
    digest=sha256(canonical_bytes(temp.packet(queries,fixed))).hexdigest()
    return TaskTraceReceipt(temp.initial,temp.observed,temp.candidate_count,temp.global_bits,
                            temp.examined_states,temp.products,temp.factors,temp.certificates,
                            temp.program_digest,digest)
