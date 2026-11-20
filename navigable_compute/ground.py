"""Exact GF(2) affine lift of byte-program semantics; no numerical approximation."""
from __future__ import annotations
from dataclasses import dataclass
from .machine import Program, AFFINE

class UnsupportedGround(ValueError):
    pass

def parity(x: int) -> int:
    return x.bit_count() & 1

@dataclass(frozen=True)
class Ground:
    """Each output bit j is parity(rows[j] & input) XOR offset[j]."""
    rows: tuple[int, ...]
    offset: int
    digest: str

    @property
    def width(self):
        return len(self.rows)

    @classmethod
    def identity(cls, width: int, digest: str):
        return cls(tuple(1 << j for j in range(width)), 0, digest)

    def apply_bits(self, bits: int) -> int:
        if bits < 0 or bits >> self.width: raise ValueError('state out of range')
        out = self.offset
        for j, row in enumerate(self.rows):
            out ^= parity(row & bits) << j
        return out

    def apply(self, initial: bytes) -> bytes:
        if 8*len(initial) != self.width: raise ValueError('state width mismatch')
        return self.apply_bits(int.from_bytes(initial, 'little')).to_bytes(self.width//8,'little')

    def observe(self, initial: bytes, addresses: tuple[int, ...]) -> bytes:
        """Only materialize requested output bytes, not the complete destination state."""
        if 8*len(initial) != self.width: raise ValueError('state width mismatch')
        bits = int.from_bytes(initial, 'little')
        out = bytearray()
        for a in addresses:
            if not 0 <= a < self.width//8: raise ValueError('bad output address')
            value = 0
            for b in range(8):
                j = 8*a+b
                value |= (parity(self.rows[j] & bits) ^ ((self.offset >> j)&1)) << b
            out.append(value)
        return bytes(out)

    def after(self, before: 'Ground') -> 'Ground':
        """Self ∘ before. Both maps are of equal width and same instruction provenance."""
        if self.width != before.width or self.digest != before.digest:
            raise ValueError('incompatible Ground composition')
        result_rows = []
        result_offset = self.offset
        for j, sources in enumerate(self.rows):
            row = 0
            while sources:
                bit = sources & -sources
                i = bit.bit_length()-1
                row ^= before.rows[i]
                sources ^= bit
            result_rows.append(row)
            result_offset ^= parity(self.rows[j] & before.offset) << j
        return Ground(tuple(result_rows), result_offset, self.digest)

    def power(self, cycles: int) -> 'PowerResult':
        if not isinstance(cycles,int) or cycles < 0: raise ValueError('cycles must be nonnegative integer')
        acc = Ground.identity(self.width, self.digest)
        base = self
        products = 0
        n = cycles
        while n:
            if n & 1:
                acc = base.after(acc)
                products += 1
            n >>= 1
            if n:
                base = base.after(base)
                products += 1
        return PowerResult(acc, products, cycles)

@dataclass(frozen=True)
class PowerResult:
    ground: Ground
    products: int
    cycles: int


def compile_ground(program: Program) -> Ground:
    """Symbolically lift each permitted instruction into GF(2) affine semantics."""
    width = program.size * 8
    rows = [1 << i for i in range(width)]
    off = [0]*width
    for ins in program.instructions:
        op, a, b = ins.op, ins.a, ins.b
        if op not in AFFINE:
            raise UnsupportedGround(f'{op} is not GF(2)-affine: exact Ground refused')
        sa = 8*a
        sb = 8*b if op in {'xor','copy','swap'} else -1
        if op == 'xor':
            for k in range(8):
                rows[sa+k] ^= rows[sb+k]
                off[sa+k] ^= off[sb+k]
        elif op == 'xor_const':
            for k in range(8): off[sa+k] ^= (b>>k)&1
        elif op == 'rotl':
            old_r, old_o = rows[sa:sa+8], off[sa:sa+8]
            for k in range(8):
                rows[sa+k] = old_r[(k-b)%8]
                off[sa+k] = old_o[(k-b)%8]
        elif op == 'swap':
            for k in range(8):
                rows[sa+k],rows[sb+k] = rows[sb+k], rows[sa+k]
                off[sa+k],off[sb+k] = off[sb+k],off[sa+k]
        elif op == 'copy':
            source_r,source_o = rows[sb:sb+8],off[sb:sb+8]
            rows[sa:sa+8] = source_r
            off[sa:sa+8] = source_o
        elif op == 'and_const':
            for k in range(8):
                if not (b>>k)&1:
                    rows[sa+k] = 0
                    off[sa+k] = 0
        elif op == 'set':
            for k in range(8):
                rows[sa+k] = 0
                off[sa+k] = (b>>k)&1
    offset = sum(v << i for i,v in enumerate(off))
    return Ground(tuple(rows),offset,program.digest)

