"""NCQ-001: intention-relative, exact finite-state quotient Grounds.

The executable machine remains the independently coded byte interpreter. We
construct a minimal deterministic Moore quotient *for every future observation*
of selected byte masks, not merely an equivalence on sampled coordinates.
Independent byte components are qualified by a read/write dependency audit.
Finite exploration, refinement, and navigation are explicitly budgeted.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .machine import Program, Instruction, execute, BINARY
from .synthesis import Constraint, Unrealizable
from .observability import ObservationAt


class QuotientBudgetExceeded(ValueError):
    """An exact certificate would exceed the declared resource contract."""


def canonical_bytes(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def byte_components(program: Program) -> tuple[tuple[int, ...], ...]:
    """Static read/write component partition; no unproven dynamic alias claims."""
    parent = list(range(program.size))

    def root(a: int) -> int:
        while a != parent[a]:
            a = parent[a]
        return a

    for ins in program.instructions:
        if ins.op in BINARY:
            a, b = root(ins.a), root(ins.b)
            parent[max(a,b)] = min(a,b)
    groups: dict[int, list[int]] = {}
    for i in range(program.size):
        groups.setdefault(root(i), []).append(i)
    return tuple(sorted((tuple(g) for g in groups.values()), key=lambda g:g[0]))


def component_program(program: Program, indices: tuple[int, ...]) -> Program:
    """Projection valid only for a complete static component."""
    lookup = {a: i for i,a in enumerate(indices)}
    instructions = []
    for ins in program.instructions:
        if ins.a in lookup:
            if ins.op in BINARY and ins.b not in lookup:
                raise ValueError('component splits a read/write dependency')
            b = lookup[ins.b] if ins.op in BINARY else ins.b
            instructions.append(Instruction(ins.op, lookup[ins.a], b))
    if not instructions:
        # Identity is represented by a no-op register XOR with constant zero.
        instructions = [Instruction('xor_const', 0, 0)]
    return Program(len(indices), tuple(instructions))


def _transition_digest(values: tuple[int, ...], width: int) -> str:
    raw = b''.join(v.to_bytes(width, 'little') for v in values)
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class ComponentCertificate:
    """Finite proof object; contains all source-state class assignments."""
    indices: tuple[int, ...]
    observed_masks: tuple[tuple[int, int], ...]  # global byte index, mask
    class_of: tuple[int, ...]
    labels: tuple[tuple[int, ...], ...]
    successor: tuple[int, ...]
    transition_sha256: str
    refinement_rounds: int

    @property
    def classes(self) -> int:
        return len(self.labels)

    @property
    def state_count(self) -> int:
        return len(self.class_of)

    def encode(self) -> dict:
        return {'indices': list(self.indices),
                'observed_masks': [list(m) for m in self.observed_masks],
                'class_of': list(self.class_of),
                'labels': [list(x) for x in self.labels],
                'successor': list(self.successor),
                'transition_sha256': self.transition_sha256,
                'refinement_rounds': self.refinement_rounds}


def build_component_quotient(program: Program, indices: tuple[int, ...],
                             observed_masks: tuple[tuple[int, int], ...], *,
                             max_states: int = 4096,
                             max_refinement_rounds: int = 512) -> ComponentCertificate:
    """Coarsest future-observation congruence by exhaustive Moore refinement.

    Construction fully evaluates the declared finite factor, not a sample of it.
    It refuses instead of extrapolating if state or refinement budgets run out.
    """
    if indices not in byte_components(program):
        raise ValueError('not an independent static component')
    if not observed_masks or any(a not in indices or not 1 <= mask <= 255
                                 for a,mask in observed_masks):
        raise ValueError('nonempty observed masks must belong to the factor')
    if tuple(sorted(set(observed_masks))) != observed_masks or len({a for a,_ in observed_masks}) != len(observed_masks):
        raise ValueError('mask addresses must be unique and sorted')
    if isinstance(max_states, bool) or not isinstance(max_states,int) or max_states < 1:
        raise ValueError('max_states must be a positive integer')
    if isinstance(max_refinement_rounds, bool) or not isinstance(max_refinement_rounds,int) or max_refinement_rounds < 1:
        raise ValueError('max_refinement_rounds must be a positive integer')
    count = 1 << (8*len(indices))
    if count > max_states:
        raise QuotientBudgetExceeded(f'{count} concrete states exceed max_states={max_states}')
    local = component_program(program, indices)
    loc = {a:i for i,a in enumerate(indices)}
    transitions = tuple(int.from_bytes(execute(local, s.to_bytes(local.size,'little'), 1), 'little')
                        for s in range(count))
    observed = tuple(tuple((s >> (8*loc[a])) & mask for a,mask in observed_masks)
                     for s in range(count))
    def partition(signatures):
        ids = {}
        result = []
        for signature in signatures:
            if signature not in ids:
                ids[signature] = len(ids)
            result.append(ids[signature])
        return tuple(result)
    classes = partition(observed)
    rounds = 0
    while True:
        refined = partition(tuple((observed[s], classes[transitions[s]]) for s in range(count)))
        if refined == classes:
            break
        rounds += 1
        if rounds > max_refinement_rounds:
            raise QuotientBudgetExceeded('refinement budget exhausted; no unqualified quotient')
        classes = refined
    k = max(classes)+1
    labels: list[tuple[int, ...] | None] = [None] * k
    successor: list[int | None] = [None] * k
    for s in range(count):
        cls = classes[s]
        to = classes[transitions[s]]
        if labels[cls] is None:
            labels[cls] = observed[s]
            successor[cls] = to
        elif labels[cls] != observed[s] or successor[cls] != to:
            raise AssertionError('unstable refinement')
    return ComponentCertificate(indices, observed_masks, classes,
                                tuple(x for x in labels if x is not None),
                                tuple(x for x in successor if x is not None),
                                _transition_digest(transitions, local.size), rounds)


def power_successor(successor: tuple[int, ...], cycles: int) -> tuple[tuple[int, ...], int]:
    """Dense composition over quotient classes; count all map compositions."""
    if isinstance(cycles,bool) or not isinstance(cycles,int) or cycles < 0:
        raise ValueError('invalid cycle coordinate')
    result = tuple(range(len(successor)))
    base = successor
    n = cycles
    products = 0
    while n:
        if n & 1:
            result = tuple(base[result[i]] for i in range(len(result)))
            products += 1
        n >>= 1
        if n:
            base = tuple(base[base[i]] for i in range(len(base)))
            products += 1
    return result, products


def _fixed_allowed(s: int, indices: tuple[int, ...], fixed: tuple[Constraint, ...]) -> bool:
    pos = {a:i for i,a in enumerate(indices)}
    return all(((s >> (8*pos[c.index])) & c.mask) == c.value for c in fixed)


@dataclass(frozen=True)
class QuotientTraceReceipt:
    initial: bytes
    observed: tuple[int, ...]  # masked values in query order
    candidate_count: int
    global_state_bits: int
    examined_states: int
    products: int
    factors: tuple[tuple[int,...], ...]
    certificates: tuple[ComponentCertificate, ...]
    program_digest: str
    witness_digest: str

    def packet(self, queries: tuple[ObservationAt, ...], fixed: tuple[Constraint,...]) -> dict:
        return {'schema': 'navigable-compute.quotient-trace.v0.2',
                'program_sha256': self.program_digest,
                'queries': [[str(q.cycle),q.address,q.mask,q.value] for q in queries],
                'fixed': [[c.index,c.mask,c.value] for c in fixed],
                'initial_hex': self.initial.hex(),
                'rendered_masked': list(self.observed),
                'candidate_count_decimal': str(self.candidate_count),
                'global_state_bits': self.global_state_bits,
                'examined_states': self.examined_states,
                'quotient_products': self.products,
                'factors': [list(g) for g in self.factors],
                'certificates': [c.encode() for c in self.certificates]}


def synthesize_quotient_trace(program: Program, queries: tuple[ObservationAt, ...], *,
                              fixed: tuple[Constraint, ...] = (),
                              max_states: int = 4096,
                              max_refinement_rounds: int = 512,
                              max_coordinate_bits: int = 4096) -> QuotientTraceReceipt:
    """Goal-first synthesis for nonlinear, factored deterministic machines.

    Every allowed initial state is counted exactly as a product of finite-factor
    solution sets. Returns one canonical initial byte state. Unobserved factors
    cost O(1) and retain complete freedom except for specified input masks.
    """
    if not queries:
        raise ValueError('at least one observation required')
    if any(q.address >= program.size for q in queries) or any(c.index >= program.size for c in fixed):
        raise ValueError('constraint address outside machine')
    if isinstance(max_coordinate_bits,bool) or not isinstance(max_coordinate_bits,int) or max_coordinate_bits < 1:
        raise ValueError('invalid coordinate bit budget')
    if any(q.cycle.bit_length() > max_coordinate_bits for q in queries):
        raise QuotientBudgetExceeded('coordinate encoding exceeds bit budget')
    factors = byte_components(program)
    initial = bytearray(program.size)
    certificates=[]
    rendered=[0]*len(queries)
    candidate_count=1
    examined=0
    products=0
    for indices in factors:
        part_query = [(qi,q) for qi,q in enumerate(queries) if q.address in indices]
        part_fixed = tuple(c for c in fixed if c.index in indices)
        count = 1 << (8*len(indices))
        if not part_query:
            # Fixed-bit feasibility without enumerating the unconstrained domain.
            masks: dict[int,int] = {}
            values: dict[int,int] = {}
            for c in part_fixed:
                overlap = masks.get(c.index,0) & c.mask
                if (values.get(c.index,0) ^ c.value) & overlap:
                    raise Unrealizable('contradictory fixed input')
                masks[c.index] = masks.get(c.index,0) | c.mask
                values[c.index] = values.get(c.index,0) | c.value
            candidate_count *= 1 << (8*len(indices)-sum(m.bit_count() for m in masks.values()))
            for a in indices:
                initial[a] = values.get(a,0)
            continue
        combined: dict[int,int] = {}
        for _,q in part_query:
            combined[q.address] = combined.get(q.address,0) | q.mask
        masks = tuple(sorted(combined.items()))
        cert = build_component_quotient(program,indices,masks,max_states=max_states,
                                        max_refinement_rounds=max_refinement_rounds)
        certificates.append(cert)
        examined+=cert.state_count
        powered={}
        for _,q in part_query:
            if q.cycle not in powered:
                powered[q.cycle],p = power_successor(cert.successor,q.cycle)
                products += p
        observation_lookup = {a:i for i,(a,mask) in enumerate(cert.observed_masks)}
        valid=0
        best=None
        best_obs=None
        for state in range(count):
            if not _fixed_allowed(state,indices,part_fixed):
                continue
            current=[]
            for qi,q in part_query:
                cls = powered[q.cycle][cert.class_of[state]]
                value = cert.labels[cls][observation_lookup[q.address]] & q.mask
                if value != q.value:
                    break
                current.append((qi,value))
            else:
                valid+=1
                if best is None:
                    best=state
                    best_obs=current
        if best is None:
            raise Unrealizable('no local state realizes the requested future trace')
        candidate_count*=valid
        for i,a in enumerate(indices):
            initial[a]=(best >> (8*i)) & 255
        for qi,v in best_obs:
            rendered[qi]=v
    receipt=QuotientTraceReceipt(bytes(initial),tuple(rendered),candidate_count,
                                 program.size*8,examined,products,factors,
                                 tuple(certificates),program.digest,'')
    witness=sha256(canonical_bytes(receipt.packet(queries,fixed))).hexdigest()
    return QuotientTraceReceipt(receipt.initial,receipt.observed,receipt.candidate_count,
                                 receipt.global_state_bits,receipt.examined_states,receipt.products,
                                 receipt.factors,receipt.certificates,receipt.program_digest,witness)
