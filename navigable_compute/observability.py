"""Temporal observability: independent coordinate observations constrain one latent state."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from .machine import Program
from .ground import compile_ground
from .synthesis import Constraint, solve_gf2

@dataclass(frozen=True)
class ObservationAt:
    cycle: int
    address: int
    mask: int
    value: int

    def __post_init__(self):
        if not isinstance(self.cycle,int) or self.cycle<0: raise ValueError('invalid cycle coordinate')
        Constraint(self.address,self.mask,self.value)

@dataclass(frozen=True)
class TraceResult:
    initial: bytes
    rendered: tuple[int,...]
    rank: int
    free_bits: int
    products: int
    program_digest: str
    witness_digest: str


def _receipt(program:Program, queries:tuple[ObservationAt,...], fixed:tuple[Constraint,...],
             initial:bytes,rendered:tuple[int,...], rank:int, free_bits:int, products:int)->str:
    doc={
        'schema':'navigable-compute.temporal-observability.v0.1',
        'program_sha256':program.digest,
        'trace':[[str(q.cycle),q.address,q.mask,q.value] for q in queries],
        'fixed':[[q.index,q.mask,q.value] for q in fixed],
        'input_hex':initial.hex(),
        'rendered':rendered,
        'constraint_rank':rank,
        'unconstrained_input_bits':free_bits,
        'composition_products':products,
    }
    return sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def synthesize_trace(program: Program, queries: tuple[ObservationAt,...], *,
                     fixed: tuple[Constraint,...] = ()) -> TraceResult:
    """Construct one initial state compatible with observations at distinct times.

    Each observation is a linear constraint after a qualified GF(2) affine Ground.
    A longer chain need NOT add rank; the result measures its exact contribution.
    """
    if not queries: raise ValueError('at least one temporal observation required')
    if any(q.address >= program.size for q in queries) or any(c.index >= program.size for c in fixed):
        raise ValueError('invalid memory address')
    compiled=compile_ground(program)
    cache={}
    products=0
    equations=[]
    for q in queries:
        if q.cycle not in cache:
            powered=compiled.power(q.cycle)
            cache[q.cycle]=powered.ground
            products+=powered.products
        g=cache[q.cycle]
        for bit in range(8):
            if (q.mask>>bit)&1:
                j=8*q.address+bit
                equations.append((g.rows[j], ((q.value>>bit)&1) ^ ((g.offset>>j)&1)))
    for c in fixed:
        for bit in range(8):
            if (c.mask>>bit)&1:
                equations.append((1<<(c.index*8+bit),(c.value>>bit)&1))
    bits,rank=solve_gf2(equations,program.size*8)
    initial=bits.to_bytes(program.size,'little')
    rendered=tuple(cache[q.cycle].observe(initial,(q.address,))[0] for q in queries)
    for q,v in zip(queries,rendered):
        if v&q.mask != q.value:
            raise AssertionError('trace result fails its constraint')
    free=program.size*8-rank
    digest=_receipt(program,queries,fixed,initial,rendered,rank,free,products)
    return TraceResult(initial,rendered,rank,free,products,program.digest,digest)


def verify_trace(program:Program, queries:tuple[ObservationAt,...], receipt:TraceResult, *,
                 fixed:tuple[Constraint,...]=())->bool:
    """Full deterministic recomputation. No signature or external sensor authority."""
    if receipt.program_digest!=program.digest: return False
    try:
        expected=synthesize_trace(program,queries,fixed=fixed)
        return expected==receipt
    except (ValueError,TypeError,IndexError):
        return False


def observability_profile(program:Program, queries:tuple[ObservationAt,...])->tuple[int,...]:
    """Exact independent constraint rank as observations are added successively."""
    if not queries: return ()
    compiled=compile_ground(program)
    cache={}
    rows=[]
    ranks=[]
    pivots={}
    for q in queries:
        if q.address>=program.size: raise ValueError('invalid address')
        if q.cycle not in cache: cache[q.cycle]=compiled.power(q.cycle).ground
        g=cache[q.cycle]
        for bit in range(8):
            if (q.mask>>bit)&1:
                row=g.rows[8*q.address+bit]
                while row:
                    k=row.bit_length()-1
                    if k not in pivots:
                        pivots[k]=row
                        break
                    row^=pivots[k]
        ranks.append(len(pivots))
    return tuple(ranks)

