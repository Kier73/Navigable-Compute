"""NCL-001: synthesize a processor law, not merely its starting state.

Given intended observations at temporal coordinates, construct a modular
additive byte processor from a declared parametric grammar. This is exact
congruence solving (powers-of-two CRT), not opaque search or learning a
physical law. The returned program was not supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import gcd

from .machine import Program, Instruction
from .observability import ObservationAt
from .quotient import canonical_bytes
from .synthesis import Unrealizable


@dataclass(frozen=True)
class LawCertificate:
    """Canonical modular-addition processor synthesized from exact intentions."""
    program: Program
    initial: bytes
    law_solution_count: int
    delta_residue_moduli: tuple[tuple[int,int], ...]  # delta == residue mod modulus
    observed: tuple[int,...]
    witness_digest: str

    def packet(self, queries:tuple[ObservationAt,...])->dict:
        return {'schema':'navigable-compute.law-construction.v0.1',
                'grammar':'independent-byte-additive-mod-256',
                'program':self.program.encode(),
                'program_sha256':self.program.digest,
                'initial_hex':self.initial.hex(),
                'queries':[[str(q.cycle),q.address,q.mask,q.value] for q in queries],
                'observed':list(self.observed),
                'solutions_decimal':str(self.law_solution_count),
                'delta_congruences':[list(pair) for pair in self.delta_residue_moduli]}


def _merge_pow2_congruences(r: int, m: int, s: int, n: int)->tuple[int,int]:
    """Intersect two residues mod powers of two, or disprove compatibility."""
    if (r-s)%min(m,n):
        raise Unrealizable('proposed temporal observations cannot share an additive law')
    return (r,m) if m>=n else (s,n)


def _infer_byte(observations:tuple[ObservationAt,...])->tuple[int,int,int,int]:
    """Return initial byte, canonical delta, delta residue modulus, count."""
    if not observations:
        return 0,0,1,65536
    anchor=observations[0]
    residue, modulus = 0,1
    for q in observations[1:]:
        a=(q.cycle-anchor.cycle)%256
        b=(q.value-anchor.value)%256
        g=gcd(a,256)
        if b%g:
            raise Unrealizable('incompatible modular time/value differences')
        m=256//g
        candidate=((b//g)*pow(a//g,-1,m))%m if m>1 else 0
        residue,modulus=_merge_pow2_congruences(residue,modulus,candidate,m)
    delta=residue
    x0=(anchor.value-(anchor.cycle%256)*delta)%256
    if any((x0+q.cycle*delta)%256 !=q.value for q in observations):
        raise AssertionError('modular law solver mismatch')
    return x0,delta,modulus,256//modulus


def synthesize_additive_law(size:int, queries:tuple[ObservationAt,...])->LawCertificate:
    """Construct a previously-unsupplied additive byte program from intentions.

    Every asserted observation must specify a full byte. An observation of only
    one bit would require a different constraint solver and is refused.
    """
    if isinstance(size,bool) or not isinstance(size,int) or not 1<=size<=128:
        raise ValueError('byte width out of range')
    if not queries:
        raise ValueError('at least one goal observation is required')
    if any(q.address>=size or q.mask!=255 for q in queries):
        raise ValueError('this exact law synthesizer requires full-byte observations')
    initial=[]
    deltas=[]
    moduli=[]
    count=1
    for addr in range(size):
        own=tuple(q for q in queries if q.address==addr)
        x0,d,mod,n=_infer_byte(own)
        initial.append(x0)
        deltas.append(d)
        moduli.append((d,mod))
        count*=n
    program=Program(size,tuple(Instruction('add_const',i,d) for i,d in enumerate(deltas)))
    receipt=LawCertificate(program,bytes(initial),count,tuple(moduli),
                           tuple(q.value for q in queries),'')
    witness=sha256(canonical_bytes(receipt.packet(queries))).hexdigest()
    return LawCertificate(receipt.program,receipt.initial,receipt.law_solution_count,
                          receipt.delta_residue_moduli,receipt.observed,witness)


def verify_additive_law(size:int, queries:tuple[ObservationAt,...],
                        receipt:LawCertificate)->bool:
    """Independent enumeration of each 8-bit delta, never the n-byte product.

    This deliberately does NOT call the symbolic congruence solver. Direct
    evaluation of x0 + t*delta mod 256 validates the synthesized law and exact
    number of solutions, with no large-coordinate replay.
    """
    try:
        if not 1<=size<=128 or not queries or len(receipt.initial)!=size:
            return False
        if any(q.address>=size or q.mask!=255 for q in queries):
            return False
        p=receipt.program
        if p.size!=size or len(p.instructions)!=size:
            return False
        if any(i.op!='add_const' or i.a!=a for a,i in enumerate(p.instructions)):
            return False
        count=1
        for addr in range(size):
            own=[q for q in queries if q.address==addr]
            if own:
                anchor=own[0]
                valid=[]
                for delta in range(256):
                    x0=(anchor.value - delta*anchor.cycle)%256
                    if all((x0+delta*q.cycle)%256==q.value for q in own):
                        valid.append((delta,x0))
                if not valid:
                    return False
                expected_delta,expected_initial=valid[0]
                # All satisfying deltas are one congruence modulo the inverse
                # of their number, because the modulus is a power of two.
                expected_modulus=256//len(valid)
                if receipt.delta_residue_moduli[addr]!=(expected_delta,expected_modulus):
                    return False
                if receipt.initial[addr]!=expected_initial or p.instructions[addr].b!=expected_delta:
                    return False
                count*=len(valid)
            else:
                if receipt.initial[addr]!=0 or p.instructions[addr].b!=0:
                    return False
                if receipt.delta_residue_moduli[addr]!=(0,1):
                    return False
                count*=65536
        if receipt.law_solution_count!=count or receipt.observed!=tuple(q.value for q in queries):
            return False
        return sha256(canonical_bytes(receipt.packet(queries))).hexdigest()==receipt.witness_digest
    except (ValueError,TypeError,IndexError,AttributeError,OverflowError):
        return False
