"""Intent-first construction: solve an exact future observation for an input state."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from .ground import compile_ground, parity
from .machine import Program

class Unrealizable(ValueError):
    pass

@dataclass(frozen=True)
class Constraint:
    """At byte address index, require (observed & mask) == value."""
    index: int
    mask: int
    value: int

    def __post_init__(self):
        if not all(isinstance(x,int) for x in (self.index,self.mask,self.value)):
            raise TypeError('constraint values must be integers')
        if self.index < 0 or not 0 <= self.mask <= 255 or self.value & ~self.mask:
            raise ValueError('invalid mask / value / address')

@dataclass(frozen=True)
class SynthesisResult:
    initial: bytes
    observed: bytes
    addresses: tuple[int,...]
    rank: int
    free_bits: int
    products: int
    cycles: int
    program_digest: str
    witness_digest: str


def solve_gf2(equations: list[tuple[int,int]], width: int) -> tuple[int,int]:
    """Exact high-pivot elimination. Free variables canonicalized to zero."""
    pivots: dict[int,tuple[int,int]] = {}
    for row, target in equations:
        if row < 0 or row >> width or target not in (0,1):
            raise ValueError('bad equation')
        while row:
            k = row.bit_length()-1
            if k not in pivots:
                pivots[k] = (row,target)
                break
            existing, existing_rhs = pivots[k]
            row ^= existing
            target ^= existing_rhs
        else:
            if target: raise Unrealizable('requested observation contradicts the Ground')
    answer = 0
    for k in sorted(pivots):
        row,rhs=pivots[k]
        answer |= (rhs ^ parity((row ^ (1<<k)) & answer)) << k
    for row,target in equations:
        if parity(row & answer) != target: raise AssertionError('internal solver mismatch')
    return answer,len(pivots)


def _validate(program: Program, constraints: tuple[Constraint,...]):
    for c in constraints:
        if c.index >= program.size: raise ValueError('constraint addresses exceed memory width')


def _witness(program: Program, cycles: int, initial: bytes, addresses: tuple[int,...],
             observed: bytes, goal: tuple[Constraint,...], fixed: tuple[Constraint,...]) -> str:
    packet = {
        'schema': 'navigable-compute.synthesis-receipt.v0.1',
        'program_sha256': program.digest,
        'cycles_decimal': str(cycles),
        'input_hex': initial.hex(),
        'observed_addresses': addresses,
        'observed_hex': observed.hex(),
        'goal': [[c.index,c.mask,c.value] for c in goal],
        'fixed': [[c.index,c.mask,c.value] for c in fixed],
    }
    return sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def synthesize(program: Program, cycles: int, goal: tuple[Constraint,...], *,
               fixed: tuple[Constraint,...] = ()) -> SynthesisResult:
    if not goal: raise ValueError('at least one output constraint is required')
    _validate(program,goal)
    _validate(program,fixed)
    p=compile_ground(program).power(cycles)
    g=p.ground
    equations: list[tuple[int,int]]=[]
    for c in goal:
        for bit in range(8):
            if (c.mask >> bit)&1:
                index=8*c.index+bit
                equations.append((g.rows[index], ((c.value>>bit)&1) ^ ((g.offset>>index)&1)))
    for c in fixed:
        for bit in range(8):
            if (c.mask >> bit)&1:
                equations.append((1 << (8*c.index+bit), (c.value>>bit)&1))
    input_bits,rank=solve_gf2(equations,8*program.size)
    initial=input_bits.to_bytes(program.size,'little')
    addresses=tuple(dict.fromkeys(c.index for c in goal))
    observed=g.observe(initial, addresses)
    for c in goal:
        if observed[addresses.index(c.index)] & c.mask != c.value:
            raise AssertionError('synthesized state violated requested intention')
    witness=_witness(program,cycles,initial,addresses,observed,goal,fixed)
    return SynthesisResult(initial,observed,addresses,rank,8*program.size-rank,
                           p.products,cycles,program.digest,witness)


def verify(program: Program, result: SynthesisResult, goal: tuple[Constraint,...], *,
           fixed: tuple[Constraint,...] = ()) -> bool:
    """Recheck witnessed state from the program; digest is integrity, NOT a signature."""
    if result.program_digest != program.digest or len(result.initial) != program.size:
        return False
    try:
        p = compile_ground(program).power(result.cycles)
        observed = p.ground.observe(result.initial,result.addresses)
        if observed != result.observed or p.products != result.products: return False
        if any(c.index not in result.addresses or
               (observed[result.addresses.index(c.index)] & c.mask) != c.value for c in goal):
            return False
        if any((result.initial[c.index] & c.mask) != c.value for c in fixed):
            return False
        if result.free_bits + result.rank != program.size*8: return False
        recomputed=synthesize(program,result.cycles,goal,fixed=fixed)
        return recomputed == result
    except (ValueError,TypeError,IndexError):
        return False

