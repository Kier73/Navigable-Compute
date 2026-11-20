"""NCQ-001 independent reference checker, with no quotient-building imports.

Checks exhaustive concrete transitions against the standalone machine interpreter,
future-stable observation classes, minimality, goal/fixed witnesses, exact count,
and a bounded independent cycle-detection evaluator for distant coordinates.
The finite certificate is mathematical evidence, NOT external provenance/signature.
"""
from __future__ import annotations

from hashlib import sha256
from hmac import compare_digest
from .machine import Program, Instruction, execute, BINARY
from .synthesis import Constraint
from .observability import ObservationAt
from .quotient import QuotientTraceReceipt, canonical_bytes


def _factors(program: Program) -> tuple[tuple[int,...], ...]:
    groups = [set([i]) for i in range(program.size)]
    for op in program.instructions:
        if op.op in BINARY:
            ia=next(i for i,g in enumerate(groups) if op.a in g)
            ib=next(i for i,g in enumerate(groups) if op.b in g)
            if ia!=ib:
                groups[ia].update(groups[ib])
                groups.pop(ib)
    return tuple(sorted((tuple(sorted(g)) for g in groups),key=lambda g:g[0]))


def _local_program(program: Program, group: tuple[int,...]) -> Program:
    lookup = {a:i for i,a in enumerate(group)}
    code=[]
    for instruction in program.instructions:
        if instruction.a in lookup:
            b = lookup[instruction.b] if instruction.op in BINARY else instruction.b
            code.append(Instruction(instruction.op,lookup[instruction.a],b))
    return Program(len(group), tuple(code) or (Instruction('xor_const',0,0),))


def _validate_certificate(program, cert, masks, *, max_states) -> None:
    group = cert.indices
    n = 1 << (8*len(group))
    if not cert.observed_masks == tuple(sorted(masks.items())):
        raise ValueError('observation mask mismatch')
    if n > max_states or len(cert.class_of)!=n or not 1 <= len(cert.labels) <= n:
        raise ValueError('invalid finite state budget or class cardinality')
    k=len(cert.labels)
    if len(cert.successor)!=k or any(not isinstance(v,int) or not 0<=v<k for v in cert.successor):
        raise ValueError('invalid quotient transition function')
    if len(set(cert.class_of))!=k or any(not isinstance(v,int) or not 0<=v<k for v in cert.class_of):
        raise ValueError('invalid class assignment')
    if any(len(label)!=len(cert.observed_masks) or any(not isinstance(x,int) for x in label)
           for label in cert.labels):
        raise ValueError('invalid quotient output labels')
    local=_local_program(program,group)
    positions = {a:i for i,a in enumerate(group)}
    hasher=sha256()
    concrete_successors=[]
    concrete_observations=[]
    for state in range(n):
        successor=int.from_bytes(execute(local,state.to_bytes(local.size,'little'),1),'little')
        hasher.update(successor.to_bytes(local.size,'little'))
        concrete_successors.append(successor)
        group_class=cert.class_of[state]
        label=tuple(((state >> (8*positions[a])) & mask) for a,mask in cert.observed_masks)
        concrete_observations.append(label)
        if label!=cert.labels[group_class] or cert.class_of[successor]!=cert.successor[group_class]:
            raise ValueError('partition is not observation/transition congruent')
    if hasher.hexdigest()!=cert.transition_sha256:
        raise ValueError('concrete-transition digest does not match interpreter')
    # Re-derive the coarsest concrete-state partition independently, so the
    # submitted refinement count and class IDs are checkable, not metadata
    # accepted merely because the entire receipt rehashed correctly.
    def concrete_partition(signatures):
        ids = {}
        output = []
        for signature in signatures:
            if signature not in ids:
                ids[signature] = len(ids)
            output.append(ids[signature])
        return tuple(output)
    classes = concrete_partition(concrete_observations)
    actual_rounds = 0
    for _ in range(n+1):
        refined = concrete_partition(tuple((concrete_observations[i],classes[concrete_successors[i]])
                                           for i in range(n)))
        if refined == classes:
            break
        actual_rounds += 1
        classes = refined
    else:
        raise ValueError('concrete refinement failed to stabilize')
    if classes != cert.class_of or actual_rounds != cert.refinement_rounds:
        raise ValueError('certificate is not the canonical minimal partition')
    # Independent minimization of the *quotient* itself. All k blocks must
    # remain distinguishable by some finite future output; otherwise they can
    # be merged and the advertised minimality claim is false.
    def classify(keys):
        ids={}
        result=[]
        for key in keys:
            if key not in ids:
                ids[key]=len(ids)
            result.append(ids[key])
        return tuple(result)
    classes=classify(cert.labels)
    for _ in range(k):
        refined=classify(tuple((cert.labels[i],classes[cert.successor[i]]) for i in range(k)))
        if refined==classes:
            break
        classes=refined
    else:
        raise ValueError('nonconvergent quotient minimization')
    if len(set(classes))!=k:
        raise ValueError('certificate describes a nonminimal quotient')


def _follow(successor: tuple[int,...], start:int, n:int) -> int:
    """Independent eventual-cycle algorithm, not the builder's squaring."""
    sequence=[]
    seen={}
    x=start
    while x not in seen:
        seen[x]=len(sequence)
        sequence.append(x)
        x=successor[x]
    if n<len(sequence):
        return sequence[n]
    first=seen[x]
    return sequence[first+(n-first)%(len(sequence)-first)]


def verify_quotient_trace(program:Program, queries:tuple[ObservationAt,...],
                          receipt:QuotientTraceReceipt, *, fixed:tuple[Constraint,...]=(),
                          max_states:int=4096, max_coordinate_bits:int=4096)->bool:
    """Reconstruct and check from the program and submitted proof object.

    Can verify long-horizon claims without executing long trajectories. May cost
    O(sum 2^(8*component_bytes)) to verify finite factors, explicitly capped.
    """
    try:
        if not queries or receipt.program_digest!=program.digest:
            return False
        if len(receipt.initial)!=program.size or receipt.global_state_bits!=program.size*8:
            return False
        if any(q.address>=program.size or q.cycle.bit_length()>max_coordinate_bits for q in queries):
            return False
        if any(c.index>=program.size for c in fixed):
            return False
        factors = _factors(program)
        if receipt.factors!=factors:
            return False
        expected_certs={c.indices:c for c in receipt.certificates}
        if len(expected_certs)!=len(receipt.certificates):
            return False
        counts=1
        examined=0
        observed=[0]*len(queries)
        products=0
        for group in factors:
            scoped=[(qi,q) for qi,q in enumerate(queries) if q.address in group]
            scoped_fixed=tuple(c for c in fixed if c.index in group)
            positions={a:i for i,a in enumerate(group)}
            if not scoped:
                if group in expected_certs:
                    return False
                masks={}
                values={}
                for c in scoped_fixed:
                    overlap = masks.get(c.index,0)&c.mask
                    if (values.get(c.index,0)^c.value)&overlap:
                        return False
                    masks[c.index]=masks.get(c.index,0)|c.mask
                    values[c.index]=values.get(c.index,0)|c.value
                counts*=1<<(8*len(group)-sum(mask.bit_count() for mask in masks.values()))
                if any(receipt.initial[a]!=values.get(a,0) for a in group):
                    return False
                continue
            if group not in expected_certs:
                return False
            cert=expected_certs[group]
            masks={}
            for _,q in scoped:
                masks[q.address]=masks.get(q.address,0)|q.mask
            _validate_certificate(program,cert,masks,max_states=max_states)
            examined+=cert.state_count
            # Correct map composition count follows only from each coordinate's
            # binary length and popcount. This checks the reported metric too.
            for n in set(q.cycle for _,q in scoped):
                products+=n.bit_count()+max(0,n.bit_length()-1)
            field={a:j for j,(a,mask) in enumerate(cert.observed_masks)}
            valid=[]
            for state in range(cert.state_count):
                if any(((state>>(8*positions[c.index]))&c.mask)!=c.value for c in scoped_fixed):
                    continue
                local_values=[]
                for qi,q in scoped:
                    cls=_follow(cert.successor,cert.class_of[state],q.cycle)
                    value=cert.labels[cls][field[q.address]] & q.mask
                    if value!=q.value:
                        break
                    local_values.append((qi,value))
                else:
                    valid.append((state,local_values))
            if not valid:
                return False
            counts*=len(valid)
            best,values=valid[0]
            if any(receipt.initial[a] != ((best>>(8*i))&255) for i,a in enumerate(group)):
                return False
            for qi,value in values:
                observed[qi]=value
        if set(expected_certs) != {group for group in factors if any(q.address in group for q in queries)}:
            return False
        if receipt.candidate_count!=counts or receipt.examined_states!=examined or receipt.products!=products:
            return False
        if receipt.observed!=tuple(observed):
            return False
        expected=sha256(canonical_bytes(receipt.packet(queries,fixed))).hexdigest()
        return compare_digest(expected,receipt.witness_digest)
    except (TypeError,ValueError,IndexError,KeyError,OverflowError):
        return False
