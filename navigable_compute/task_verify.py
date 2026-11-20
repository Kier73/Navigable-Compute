"""NCQ-002 independent task-quotient checker via source enumeration + cycles.

No calls to task_quotient's constructor or function-power algorithm. A task
class is proven only for the exact list of requested output/time coordinates.
"""
from __future__ import annotations
from hashlib import sha256
from .machine import Program, Instruction, BINARY, execute
from .observability import ObservationAt
from .synthesis import Constraint
from .quotient import canonical_bytes
from .task_quotient import TaskTraceReceipt


def _components(program):
    components=[{i} for i in range(program.size)]
    for ins in program.instructions:
        if ins.op in BINARY:
            a=next(i for i,v in enumerate(components) if ins.a in v)
            b=next(i for i,v in enumerate(components) if ins.b in v)
            if a!=b:
                components[a]|=components[b]
                components.pop(b)
    return tuple(sorted((tuple(sorted(v)) for v in components),key=lambda v:v[0]))


def _local(program,group):
    positions={a:i for i,a in enumerate(group)}
    ins=tuple(Instruction(i.op,positions[i.a],positions[i.b] if i.op in BINARY else i.b)
              for i in program.instructions if i.a in positions)
    return Program(len(group),ins or (Instruction('xor_const',0,0),))


def _follow(one_step,start,n):
    # Functional graph cycle-finding, independent of builder's map squaring.
    positions={}
    trail=[]
    x=start
    while x not in positions:
        positions[x]=len(trail)
        trail.append(x)
        x=one_step[x]
    if n<len(trail): return trail[n]
    mu=positions[x]
    return trail[mu+(n-mu)%(len(trail)-mu)]


def verify_task_trace(program:Program,queries:tuple[ObservationAt,...],receipt:TaskTraceReceipt, *,
                      fixed:tuple[Constraint,...]=(),max_states:int=4096,
                      max_coordinate_bits:int=4096)->bool:
    try:
        if not queries or receipt.program_digest!=program.digest or len(receipt.initial)!=program.size:
            return False
        if any(q.address>=program.size or q.cycle.bit_length()>max_coordinate_bits for q in queries):
            return False
        if any(c.index>=program.size for c in fixed):
            return False
        groups=_components(program)
        if receipt.factors!=groups or receipt.global_bits!=8*program.size:
            return False
        certs={c.indices:c for c in receipt.certificates}
        if len(certs)!=len(receipt.certificates):
            return False
        products=0
        examined=0
        solution_count=1
        rendered=[0]*len(queries)
        for group in groups:
            own=[(i,q) for i,q in enumerate(queries) if q.address in group]
            own_fixed=[c for c in fixed if c.index in group]
            if not own:
                if group in certs: return False
                masks,values={},{}
                for c in own_fixed:
                    previous=masks.get(c.index,0)
                    if (values.get(c.index,0)^c.value)&previous&c.mask:
                        return False
                    masks[c.index]=previous|c.mask
                    values[c.index]=values.get(c.index,0)|c.value
                solution_count*=1<<(8*len(group)-sum(mask.bit_count() for mask in masks.values()))
                if any(receipt.initial[a]!=values.get(a,0) for a in group):
                    return False
                continue
            cert=certs.get(group)
            if cert is None:
                return False
            count=1<<(8*len(group))
            if count>max_states or len(cert.class_of)!=count:
                return False
            k=len(cert.labels)
            if not 1<=k<=count or len(set(cert.class_of))!=k or any(type(j)!=int or not 0<=j<k for j in cert.class_of):
                return False
            if any(len(label)!=len(own) for label in cert.labels):
                return False
            scoped={a:i for i,a in enumerate(group)}
            local=_local(program,group)
            next_state=[]
            hasher=sha256()
            for s in range(count):
                v=int.from_bytes(execute(local,s.to_bytes(local.size,'little'),1),'little')
                next_state.append(v)
                hasher.update(v.to_bytes(local.size,'little'))
            if cert.transition_sha256!=hasher.hexdigest():
                return False
            # Independently derive *all* exact task observation vectors for
            # every source state, then canonicalize to minimal output classes.
            seen={}
            ids=[]
            labels=[]
            for state in range(count):
                signature=tuple(((_follow(next_state,state,q.cycle)>>(8*scoped[q.address]))&q.mask)
                                for _,q in own)
                if signature not in seen:
                    seen[signature]=len(seen)
                    labels.append(signature)
                ids.append(seen[signature])
            if cert.class_of!=tuple(ids) or cert.labels!=tuple(labels):
                return False
            examined+=count
            for time in {q.cycle for _,q in own}:
                products+=time.bit_count()+max(0,time.bit_length()-1)
            goal=tuple(q.value for _,q in own)
            matching=[]
            for s,cls in enumerate(cert.class_of):
                if cert.labels[cls]!=goal: continue
                if any(((s>>(8*scoped[c.index]))&c.mask)!=c.value for c in own_fixed):
                    continue
                matching.append(s)
            if not matching:
                return False
            best=matching[0]
            solution_count*=len(matching)
            if any(receipt.initial[a]!=((best>>(8*j))&255) for j,a in enumerate(group)):
                return False
            for i,q in own:
                rendered[i]=q.value
        if set(certs)!={group for group in groups if any(q.address in group for q in queries)}:
            return False
        if receipt.candidate_count!=solution_count or receipt.examined_states!=examined or receipt.products!=products:
            return False
        if receipt.observed!=tuple(rendered):
            return False
        return receipt.witness_digest==sha256(canonical_bytes(receipt.packet(queries,fixed))).hexdigest()
    except (TypeError,ValueError,IndexError,KeyError,OverflowError):
        return False
